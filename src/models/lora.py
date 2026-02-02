from __future__ import annotations

from typing import Tuple

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.config import MethodConfig, ModelConfig


def load_base_model(model_cfg: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.name, trust_remote_code=model_cfg.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
        device_map="auto",
        torch_dtype=_resolve_dtype(model_cfg.dtype),
        trust_remote_code=model_cfg.trust_remote_code,
    )
    model.config.use_cache = False
    return model, tokenizer


def apply_lora(model, method_cfg: MethodConfig):
    config = LoraConfig(
        r=method_cfg.lora_rank,
        lora_alpha=method_cfg.lora_alpha,
        target_modules=list(method_cfg.target_modules),
        lora_dropout=method_cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def _resolve_dtype(dtype: str):
    if dtype == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    return dtype
