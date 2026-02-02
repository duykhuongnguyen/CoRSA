# CoRSA: Conflict-Resolving and Sharpness-Aware Minimization for Generalized Knowledge Editing with Multiple Updates

## Setup

```bash
uv venv .venv
source .venv/bin/activate
uv pip install torch transformers datasets peft trl vllm
```

## Configs for Training and Evaluation

The default config is at:

`src/configs/default.json`

Key fields:
- `model.name`: Model ID or local path.
- `dataset`: dataset name/split/source.
- `method.name`: method used for training (lora, sam, corsa).
- `training`: batch size, epochs, LR, output root, etc.
- `evaluation`: vLLM settings.

## Train

Example for training CoRSA on CounterFact:

```bash
python src/train.py --config src/configs/default.json
```

Training outputs under:

```
outputs/<model>/<dataset>/<method>/
```

Each run directory includes:
- `config.json`: full resolved run config
- `metadata.json`: method + dataset + model summary
- model weights / adapter artifacts

### Train CoRSA

```bash
python src/train.py --config src/configs/default.json --method corsa
```

## Evaluate

Evaluation uses vLLM and reads the run’s metadata to select the correct base model + LoRA adapter.

```bash
python src/evaluate.py --config src/configs/default.json --run_dir outputs/<model>/<dataset>/<method>/
```

Outputs are written to the run directory:
- `generations*.jsonl`: per-sample generations
- `generations*.csv`: generations and correctness
- `results*.csv`: summary table
