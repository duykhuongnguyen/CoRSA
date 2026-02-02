from __future__ import annotations

from typing import List

from datasets import Dataset, load_dataset


def _normalize_answer(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float)):
        return str(value)
    if isinstance(value, list):
        for item in value:
            normalized = _normalize_answer(item)
            if normalized:
                return normalized
    return None


def load_and_format_data(dataset_name: str = "zsre", split: str = "test", source: str = "hf", limit: int | None = None):
    if source != "hf":
        raise ValueError(f"Unsupported dataset source '{source}'.")

    dataset_name = dataset_name.lower()
    formatted_data = []
    if dataset_name == "zsre":
        ds = load_dataset("wangzn2001/zsre", split=split)
        n_data = 0
        for row in ds:
            if len(row["alternatives"]) > 0:
                formatted_data.append(
                    {
                        "subject": row["input"],
                        "prompt": row["input"],
                        "target_old": row["output"][0]["answer"],
                        "target_new": row["alternatives"][0],
                        "paraphrases": row["rephrases"],
                    }
                )
                n_data += 1
            if limit is not None and n_data >= limit:
                break
    elif dataset_name == "counterfact":
        ds = load_dataset("azhx/counterfact", split=split)
        for row in ds:
            d = row["requested_rewrite"]
            formatted_data.append(
                {
                    "subject": d["subject"],
                    "prompt": d["prompt"].format(d["subject"]),
                    "target_old": d["target_true"]["str"],
                    "target_new": d["target_new"]["str"],
                    "paraphrases": row["paraphrase_prompts"],
                }
            )
            if limit is not None and len(formatted_data) >= limit:
                break
    elif dataset_name in {"mquake", "mquake-remastered"}:
        ds = load_dataset("henryzhongsc/MQuAKE-Remastered", split=split)
        for row in ds:
            rewrite = row["requested_rewrite"][0]
            subject = rewrite["subject"]
            prompt = rewrite["prompt"].format(subject)
            paraphrases = [rewrite["question"]]
            target_old = rewrite["target_true_str"]
            target_new = rewrite["target_new_str"]

            formatted_data.append(
                {
                    "subject": subject,
                    "prompt": prompt,
                    "target_old": target_old,
                    "target_new": target_new,
                    "paraphrases": paraphrases,
                }
            )
            if limit is not None and len(formatted_data) >= limit:
                break
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}' for source='{source}'.")

    return formatted_data


def create_training_datasets(formatted_data: List[dict]):
    """Creates conversational datasets for SFTTrainer."""
    train_old = []
    train_new = []

    for item in formatted_data:
        prompt_messages = [{"role": "user", "content": item["prompt"]}]

        train_old.append(
            {
                "prompt": prompt_messages,
                "completion": [{"role": "assistant", "content": item["target_old"]}],
            }
        )

        train_new.append(
            {
                "prompt": prompt_messages,
                "completion": [{"role": "assistant", "content": item["target_new"]}],
            }
        )

    return Dataset.from_list(train_old), Dataset.from_list(train_new)


def create_dpo_training_datasets(formatted_data: List[dict]):
    train_data = []

    for item in formatted_data:
        train_data.append(
            {
                "prompt": [{"role": "user", "content": item["prompt"]}],
                "chosen": [{"role": "assistant", "content": item["target_new"]}],
                "rejected": [{"role": "assistant", "content": item["target_old"]}],
            }
        )

    return Dataset.from_list(train_data)
