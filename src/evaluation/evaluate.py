from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

from dataclasses import replace

from configs.config import load_config
from data.datasets import load_and_format_data
from utils import format_metric, summarize_metrics

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError as exc:  # pragma: no cover
    raise SystemExit("vLLM is required. Install it with `pip install vllm`.") from exc


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation.")
    parser.add_argument("--config", required=True, help="Path to config JSON.")
    parser.add_argument("--run_dir", required=True, help="Run directory produced by training.")
    parser.add_argument("--method", help="Method name.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint name under run-dir.")
    parser.add_argument("--generations_path", help="Generations JSONL path.")
    parser.add_argument("--csv_path", help="Output CSV path.")
    parser.add_argument("--results_csv_path", help="Results summary CSV path.")
    return parser.parse_args()


def _read_metadata(run_dir: Path) -> Dict:
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return {}


def build_chat_prompts(tokenizer, prompts: List[str]) -> List[str]:
    formatted = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        if getattr(tokenizer, "chat_template", None):
            try:
                formatted.append(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
                continue
            except Exception:
                pass
        formatted.append(prompt)
    return formatted


def build_prompt_sets(data, tokenizer) -> Tuple[List[str], List[str], List[int]]:
    main_prompts = [item["prompt"] for item in data]
    main_prompts = build_chat_prompts(tokenizer, main_prompts)

    paraphrase_prompts = []
    paraphrase_indices = []
    for idx, item in enumerate(data):
        paraphrases = item.get("paraphrases") or []
        if paraphrases:
            paraphrase_prompts.append(paraphrases[0])
            paraphrase_indices.append(idx)

    paraphrase_prompts = build_chat_prompts(tokenizer, paraphrase_prompts)
    return main_prompts, paraphrase_prompts, paraphrase_indices


def generate_texts(llm: LLM, prompts: List[str], sampling_params: SamplingParams, lora_path: str | None) -> List[str]:
    if not prompts:
        return []
    lora_request = LoRARequest("adapter", 1, lora_path) if lora_path else None
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True, lora_request=lora_request)
    return [output.outputs[0].text for output in outputs]


def compute_metrics(data, outputs_main, outputs_para, para_indices, eval_new: bool) -> Dict[str, float]:
    rel = []
    gen = []

    para_map = dict(zip(para_indices, outputs_para))

    for idx, item in enumerate(data):
        target = item.get("target_new") if eval_new else item.get("target_old")
        target = (target or "").lower()
        out = (outputs_main[idx] or "").lower()
        rel.append(target in out)

        if idx in para_map:
            gen.append(target in (para_map[idx] or "").lower())

    return {
        "Reliability": float(np.mean(rel)) if rel else float("nan"),
        "Generality": float(np.mean(gen)) if gen else float("nan"),
    }


def initialize_prompt_rows(data, eval_new: bool):
    rows = {}
    for idx, item in enumerate(data):
        target = item.get("target_new") if eval_new else item.get("target_old")
        rows[(idx, "reliability")] = {
            "data_idx": idx,
            "label": "reliability",
            "prompt": item.get("prompt", ""),
            "ground_truth": target or "",
            "generations": {},
            "correctness": {},
        }
        paraphrases = item.get("paraphrases") or []
        if paraphrases:
            rows[(idx, "generality")] = {
                "data_idx": idx,
                "label": "generality",
                "prompt": paraphrases[0],
                "ground_truth": target or "",
                "generations": {},
                "correctness": {},
            }
    return rows


def evaluate_run():
    args = parse_args()
    config = load_config(args.config)
    if args.method:
        config = replace(config, method=replace(config.method, name=args.method))
    run_dir = Path(args.run_dir)

    metadata = _read_metadata(run_dir)
    method_name = metadata.get("method", config.method.name)
    base_model = metadata.get("model_name", config.model.name)

    dataset_cfg = config.dataset
    data = load_and_format_data(
        dataset_name=dataset_cfg.name,
        split=dataset_cfg.split,
        source=dataset_cfg.source,
        limit=dataset_cfg.limit,
    )

    lora_path = str(run_dir)
    if args.checkpoint:
        lora_path = str(run_dir / args.checkpoint)

    if method_name == "original":
        lora_path = None

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    main_prompts, para_prompts, para_indices = build_prompt_sets(data, tokenizer)

    llm = LLM(
        model=base_model,
        tokenizer=base_model,
        max_model_len=config.model.max_seq_length,
        tensor_parallel_size=config.evaluation.tensor_parallel_size,
        gpu_memory_utilization=config.evaluation.gpu_memory_utilization,
        dtype=config.evaluation.dtype,
        enable_lora=bool(lora_path),
        max_lora_rank=config.method.lora_rank,
    )

    suffix_results = "" if config.evaluation.eval_new else "_old"
    suffix_checkpoint = f"_ckpt_{args.checkpoint}" if args.checkpoint else ""

    generations_path = (
        Path(args.generations_path)
        if args.generations_path
        else run_dir / f"vllm_generations{suffix_results}{suffix_checkpoint}.jsonl"
    )
    csv_path = (
        Path(args.csv_path)
        if args.csv_path
        else run_dir / f"vllm_generations{suffix_results}{suffix_checkpoint}.csv"
    )
    results_csv_path = (
        Path(args.results_csv_path)
        if args.results_csv_path
        else run_dir / f"vllm_results{suffix_results}{suffix_checkpoint}.csv"
    )
    generations_path.parent.mkdir(parents=True, exist_ok=True)
    results_csv_path.parent.mkdir(parents=True, exist_ok=True)

    header = {
        "type": "metadata",
        "model_name": base_model,
        "dataset": {
            "name": dataset_cfg.name,
            "split": dataset_cfg.split,
            "source": dataset_cfg.source,
        },
        "eval_new": config.evaluation.eval_new,
        "max_new_tokens": config.evaluation.max_new_tokens,
        "temperature": config.evaluation.temperature,
        "top_p": config.evaluation.top_p,
    }

    ground_truths = [
        (item.get("target_new") if config.evaluation.eval_new else item.get("target_old"))
        or ""
        for item in data
    ]

    results = []
    prompt_rows = initialize_prompt_rows(data, config.evaluation.eval_new)
    generation_columns = {method_name}

    with generations_path.open("w") as gen_fp:
        gen_fp.write(json.dumps(header) + "\n")

        run_metrics = []
        for eval_idx in range(1, config.evaluation.eval_runs + 1):
            sampling_params = SamplingParams(
                temperature=config.evaluation.temperature,
                top_p=config.evaluation.top_p,
                max_tokens=config.evaluation.max_new_tokens,
                seed=config.training.seed + eval_idx,
            )
            outputs_main = generate_texts(llm, main_prompts, sampling_params, lora_path)
            outputs_para = generate_texts(llm, para_prompts, sampling_params, lora_path)

            metrics = compute_metrics(
                data,
                outputs_main,
                outputs_para,
                para_indices,
                config.evaluation.eval_new,
            )
            run_metrics.append(metrics)

            for idx, output in enumerate(outputs_main):
                row = prompt_rows[(idx, "reliability")]
                row["generations"][method_name] = output
                target = (row.get("ground_truth") or "").lower()
                row["correctness"][method_name] = bool(target) and target in (output or "").lower()

            for out_idx, output in zip(para_indices, outputs_para):
                key = (out_idx, "generality")
                if key in prompt_rows:
                    row = prompt_rows[key]
                    row["generations"][method_name] = output
                    target = (row.get("ground_truth") or "").lower()
                    row["correctness"][method_name] = bool(target) and target in (output or "").lower()

            for idx, item in enumerate(data):
                record = {
                    "type": "generation",
                    "baseline": method_name,
                    "run_dir": str(run_dir),
                    "eval_idx": eval_idx,
                    "data_idx": idx,
                    "prompt": item.get("prompt", ""),
                    "ground_truth": ground_truths[idx],
                    "generation": outputs_main[idx],
                }
                gen_fp.write(json.dumps(record) + "\n")

        results.append(summarize_metrics(run_metrics, method_name))

    base_columns = ["data_idx", "label", "prompt", "ground_truth"]
    generation_columns = sorted(generation_columns)
    ordered_columns = base_columns[:]
    for col in generation_columns:
        ordered_columns.append(col)
        ordered_columns.append(f"{col}_correct")

    with csv_path.open("w", newline="") as csv_fp:
        writer = csv.DictWriter(csv_fp, fieldnames=ordered_columns)
        writer.writeheader()
        for key in sorted(prompt_rows.keys()):
            row = prompt_rows[key]
            record = {col: row.get(col, "") for col in base_columns}
            generations = row.get("generations", {})
            correctness = row.get("correctness", {})
            for col in generation_columns:
                record[col] = generations.get(col, "")
                record[f"{col}_correct"] = "true" if correctness.get(col) else "false"
            writer.writerow(record)

    summary_lines = [
        "=" * 86,
        f"{'Editor':<25} | {'Reliability (mean+/-std)':<26} | {'Generality (mean+/-std)':<26} | {'Runs':<4}\n",
        "-" * 86,
    ]
    for res in results:
        rel = format_metric(res.get("Reliability_mean"), res.get("Reliability_std"))
        gen = format_metric(res.get("Generality_mean"), res.get("Generality_std"))
        summary_lines.append(
            f"{res['Editor']:<25} | {rel:<26} | {gen:<26} | {res.get('runs', 0):<4}"
        )
    summary_lines.append("=" * 86)

    print("\n" + summary_lines[0])
    for line in summary_lines[1:]:
        print(line)

    results_fields = ["Editor", "Reliability", "Generality", "runs"]
    with results_csv_path.open("w", newline="") as results_fp:
        writer = csv.DictWriter(results_fp, fieldnames=results_fields)
        writer.writeheader()
        for res in results:
            writer.writerow(
                {
                    "Editor": res.get("Editor", ""),
                    "Reliability": format_metric(res.get("Reliability_mean"), res.get("Reliability_std")),
                    "Generality": format_metric(res.get("Generality_mean"), res.get("Generality_std")),
                    "runs": res.get("runs", ""),
                }
            )
    print(f"Saved generations to {generations_path}")
    print(f"Saved CSV to {csv_path}")
    print(f"Saved summary CSV to {results_csv_path}")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    evaluate_run()
