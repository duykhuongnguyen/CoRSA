from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ModelConfig:
    name: str
    max_seq_length: int = 2048
    trust_remote_code: bool = True
    dtype: str = "auto"


@dataclass
class DatasetConfig:
    name: str
    split: str = "train"
    source: str = "hf"
    limit: Optional[int] = None


@dataclass
class MethodConfig:
    name: str
    lora_rank: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    sam_rho: float = 0.01
    sam_eps: float = 1e-12
    sft_weight: float = 1.0
    use_pcgrad: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 8
    grad_accum_steps: int = 4
    warmup_steps: int = 5
    epochs: int = 3
    learning_rate: float = 2e-4
    output_root: str = "outputs"
    save_steps: int = 200
    seed: int = 42
    gradient_checkpointing: bool = True


@dataclass
class EvaluationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 1.0
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    dtype: str = "auto"
    eval_runs: int = 1
    eval_new: bool = True


@dataclass
class RunConfig:
    model: ModelConfig
    dataset: DatasetConfig
    method: MethodConfig
    training: TrainingConfig
    evaluation: EvaluationConfig


def _coerce_section(section: Dict[str, Any], cls):
    if isinstance(section, cls):
        return section
    return cls(**section)


def load_config(path: str | Path) -> RunConfig:
    config_path = Path(path)
    raw = json.loads(config_path.read_text())
    return RunConfig(
        model=_coerce_section(raw["model"], ModelConfig),
        dataset=_coerce_section(raw["dataset"], DatasetConfig),
        method=_coerce_section(raw["method"], MethodConfig),
        training=_coerce_section(raw["training"], TrainingConfig),
        evaluation=_coerce_section(raw.get("evaluation", {}), EvaluationConfig),
    )


def save_config(config: RunConfig, path: str | Path) -> None:
    config_path = Path(path)
    payload = {
        "model": config.model.__dict__,
        "dataset": config.dataset.__dict__,
        "method": config.method.__dict__,
        "training": config.training.__dict__,
        "evaluation": config.evaluation.__dict__,
    }
    config_path.write_text(json.dumps(payload, indent=2))
