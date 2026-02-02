from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from configs.config import load_config
from training.trainer import train_from_config
from utils import build_run_dir, slugify


def parse_args():
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument("--config", required=True, help="Path to config JSON.")
    parser.add_argument("--output_dir", help="Output directory (full path).")
    parser.add_argument("--method", help="Method name.")
    parser.add_argument("--methods", nargs="*", help="Train multiple methods.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    methods = args.methods or ([args.method] if args.method else [config.method.name])
    for method_name in methods:
        config = replace(config, method=replace(config.method, name=method_name))
        dataset_slug = slugify(f"{config.dataset.name}-{config.dataset.source}-{config.dataset.split}")
        base_dir = build_run_dir(
            config.training.output_root,
            config.model.name,
            dataset_slug,
            config.method.name,
        )

        if args.output_dir and len(methods) == 1:
            output_dir = Path(args.output_dir)
        else:
            output_dir = base_dir

        output_dir.mkdir(parents=True, exist_ok=True)
        train_from_config(config, output_dir)
        print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
