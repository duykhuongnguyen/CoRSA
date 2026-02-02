import json
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np


def slugify(value: str) -> str:
    if not value:
        return "default"
    slug = value.replace("/", "-").replace("\\", "-")
    slug = re.sub(r"[^0-9a-zA-Z._-]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_.").lower()
    return slug or "default"


def build_run_dir(output_root: str, model_name: str, dataset_name: str, method_name: str) -> Path:
    base = Path(output_root)
    model_slug = slugify(model_name)
    dataset_slug = slugify(dataset_name)
    method_slug = slugify(method_name)
    return base / model_slug / dataset_slug / method_slug


def summarize_metrics(run_metrics, label: str):
    summary = {"Editor": label, "runs": len(run_metrics)}
    metrics = ("Reliability", "Generality", "Locality")
    for metric in metrics:
        values = np.array([float(m.get(metric, float("nan"))) for m in run_metrics], dtype=np.float32)
        if values.size == 0:
            summary[f"{metric}_mean"] = float("nan")
            summary[f"{metric}_std"] = float("nan")
            continue
        summary[f"{metric}_mean"] = float(np.nanmean(values))
        summary[f"{metric}_std"] = float(np.nanstd(values))
    return summary


def format_metric(mean_value: float, std_value: float) -> str:
    if mean_value is None or std_value is None or math.isnan(mean_value) or math.isnan(std_value):
        return "nan"
    return f"{mean_value:.4f} +/- {std_value:.4f}"


def serialize_metrics(metrics: dict) -> dict:
    serialized = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, np.integer)):
            serialized[key] = float(value)
        elif isinstance(value, (int, float, str)):
            serialized[key] = value
        else:
            serialized[key] = value
    return serialized


def cache_formatted_dataset(formatted_data, dataset_slug: str, output_root: str = "outputs/formatted_data") -> Path:
    # cache_path = None
    # if not args.no_cache_data:
    #     cache_path = cache_formatted_dataset(raw_data, dataset_cfg)
    #     print(f"Saved formatted dataset to {cache_path}")
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_path = output_path / f"{dataset_slug}.json"
    cache_path.write_text(json.dumps(formatted_data, indent=2))
    return cache_path
