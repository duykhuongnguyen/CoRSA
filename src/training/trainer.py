from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from transformers import set_seed
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

from configs.config import MethodConfig, RunConfig
from data.datasets import (
    create_dpo_training_datasets,
    create_training_datasets,
    load_and_format_data,
)
from models.lora import apply_lora, load_base_model


class SAMTrainer(DPOTrainer):
    """SAMTrainer"""

    def __init__(
        self,
        *args,
        sft_weight: float = 1.0,
        lora_sam_rho: float = 0.0,
        lora_sam_eps: float = 1e-12,
        **kwargs,
    ):
        self.sft_weight = float(sft_weight)
        self.lora_sam_rho = float(lora_sam_rho)
        self.lora_sam_eps = float(lora_sam_eps)
        super().__init__(*args, **kwargs)

    def _compute_chosen_sft_loss(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.is_encoder_decoder:
            labels = batch["chosen_input_ids"].clone()
            labels[batch["chosen_attention_mask"] == 0] = self.label_pad_token_id
            outputs = model(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                labels=labels,
            )
            logits = outputs.logits
            loss_mask = batch["chosen_attention_mask"].bool()
            labels = labels.clone()
            labels[~loss_mask] = self.label_pad_token_id
            return F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=self.label_pad_token_id,
            )

        prompt_ids = batch["prompt_input_ids"]
        prompt_mask = batch["prompt_attention_mask"]
        completion_ids = batch["chosen_input_ids"]
        completion_mask = batch["chosen_attention_mask"]

        input_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        attention_mask = torch.cat((prompt_mask, completion_mask), dim=1)
        loss_mask = torch.cat((torch.zeros_like(prompt_mask), completion_mask), dim=1)

        for i in range(attention_mask.size(0)):
            first_one_idx = torch.nonzero(attention_mask[i])[0].item()
            input_ids[i] = torch.roll(input_ids[i], shifts=-first_one_idx)
            attention_mask[i] = torch.roll(attention_mask[i], shifts=-first_one_idx)
            loss_mask[i] = torch.roll(loss_mask[i], shifts=-first_one_idx)

        empty_cols = torch.sum(attention_mask, dim=0) == 0
        first_empty_col = (
            torch.nonzero(empty_cols)[0].item() if empty_cols.any() else attention_mask.size(1)
        )
        input_ids = input_ids[:, :first_empty_col]
        attention_mask = attention_mask[:, :first_empty_col]
        loss_mask = loss_mask[:, :first_empty_col]

        if self.args.max_length is not None:
            input_ids = input_ids[:, : self.args.max_length]
            attention_mask = attention_mask[:, : self.args.max_length]
            loss_mask = loss_mask[:, : self.args.max_length]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:].clone()
        loss_mask = loss_mask[:, 1:].bool()
        labels[~loss_mask] = self.label_pad_token_id

        return F.cross_entropy(
            torch.flatten(logits, end_dim=1),
            torch.flatten(labels, end_dim=1),
            ignore_index=self.label_pad_token_id,
        )

    def get_batch_loss_metrics(self, model, batch, train_eval: str = "train"):
        sft_loss = self._compute_chosen_sft_loss(model, batch)
        prefix = "eval_" if train_eval == "eval" else ""
        metrics = {f"{prefix}sft_loss": sft_loss.detach().float().cpu()}
        return sft_loss, metrics

    def _iter_lora_ab_params(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if (
                ("lora_A" in name)
                or ("lora_B" in name)
                or ("lora_embedding_A" in name)
                or ("lora_embedding_B" in name)
            ):
                yield name, p

    @torch.no_grad()
    def _lora_sam_perturb(self, model: torch.nn.Module, rho: float):
        sq_sum = None
        grads = []
        params = []

        for _, p in self._iter_lora_ab_params(model):
            if p.grad is None:
                continue
            g = p.grad
            if g.is_sparse:
                g = g.to_dense()
            grads.append(g)
            params.append(p)
            g2 = (g.float() * g.float()).sum()
            sq_sum = g2 if sq_sum is None else (sq_sum + g2)

        if sq_sum is None:
            return {}, torch.zeros((), device=model.device)

        grad_norm = torch.sqrt(sq_sum + self.lora_sam_eps)
        if not torch.isfinite(grad_norm) or grad_norm.item() == 0.0:
            return {}, grad_norm

        scale = (rho / grad_norm).to(grads[0].device)

        eps_dict = {}
        for p in params:
            g = p.grad
            if g.is_sparse:
                g = g.to_dense()
            eps = (g * scale).to(dtype=p.data.dtype)
            p.add_(eps)
            eps_dict[p] = eps

        return eps_dict, grad_norm.detach()

    @torch.no_grad()
    def _lora_sam_restore(self, eps_dict):
        for p, eps in eps_dict.items():
            p.sub_(eps)

    def training_step(self, model, inputs, num_items_in_batch=None, **kwargs):
        if self.lora_sam_rho <= 0.0:
            return super().training_step(model, inputs)

        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss_1, _ = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        if self.args.n_gpu > 1:
            loss_1 = loss_1.mean()
        loss_1 = loss_1 / self.args.gradient_accumulation_steps

        no_sync_ctx = nullcontext()
        try:
            if hasattr(self, "accelerator") and hasattr(self.accelerator, "no_sync"):
                no_sync_ctx = self.accelerator.no_sync(model)
        except Exception:
            no_sync_ctx = nullcontext()

        with no_sync_ctx:
            self.accelerator.backward(loss_1)

        try:
            scaler = getattr(self.accelerator, "scaler", None)
            if scaler is not None and self.optimizer is not None:
                scaler.unscale_(self.optimizer)
        except Exception:
            pass

        eps_dict, grad_norm = self._lora_sam_perturb(model, rho=self.lora_sam_rho)
        model.zero_grad(set_to_none=True)

        with self.compute_loss_context_manager():
            loss_2, metrics_2 = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        if self.args.n_gpu > 1:
            loss_2 = loss_2.mean()
        loss_2 = loss_2 / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss_2)
        self._lora_sam_restore(eps_dict)

        if hasattr(self, "store_metrics"):
            metrics_2 = dict(metrics_2)
            metrics_2["lora_sam_grad_norm"] = grad_norm.float().cpu()
            metrics_2["lora_sam_rho"] = float(self.lora_sam_rho)
            self.store_metrics(metrics_2, train_eval="train")

        return loss_2.detach()


class PCSAMTrainer(DPOTrainer):
    """PCSAMTrainer"""

    def __init__(
        self,
        *args,
        sft_weight: float = 1.0,
        lora_sam_rho: float = 0.0,
        lora_sam_eps: float = 1e-12,
        use_pcgrad: bool = False,
        pcgrad_eps: float = 1e-12,
        **kwargs,
    ):
        self.sft_weight = float(sft_weight)
        self.lora_sam_rho = float(lora_sam_rho)
        self.lora_sam_eps = float(lora_sam_eps)
        self.use_pcgrad = bool(use_pcgrad)
        self.pcgrad_eps = float(pcgrad_eps)
        super().__init__(*args, **kwargs)

    def _iter_lora_ab_params(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if (
                ("lora_A" in name)
                or ("lora_B" in name)
                or ("lora_embedding_A" in name)
                or ("lora_embedding_B" in name)
            ):
                yield name, p

    @torch.no_grad()
    def _lora_sam_perturb(self, model: torch.nn.Module, rho: float):
        sq_sum = None
        grads = []
        params = []

        for _, p in self._iter_lora_ab_params(model):
            if p.grad is None:
                continue
            g = p.grad
            if g.is_sparse:
                g = g.to_dense()
            grads.append(g)
            params.append(p)
            g2 = (g.float() * g.float()).sum()
            sq_sum = g2 if sq_sum is None else (sq_sum + g2)

        if sq_sum is None:
            return {}, torch.zeros((), device=model.device)

        grad_norm = torch.sqrt(sq_sum + self.lora_sam_eps)
        if not torch.isfinite(grad_norm) or grad_norm.item() == 0.0:
            return {}, grad_norm

        scale = (rho / grad_norm).to(grads[0].device)

        eps_dict = {}
        for p in params:
            g = p.grad
            if g.is_sparse:
                g = g.to_dense()
            eps = (g * scale).to(dtype=p.data.dtype)
            p.add_(eps)
            eps_dict[p] = eps

        return eps_dict, grad_norm.detach()

    @torch.no_grad()
    def _lora_sam_restore(self, eps_dict):
        for p, eps in eps_dict.items():
            p.sub_(eps)

    def _iter_trainable_params(self, model: torch.nn.Module):
        for p in model.parameters():
            if p.requires_grad:
                yield p

    def _snapshot_grads(self, params):
        snap = {}
        for p in params:
            if p.grad is None:
                snap[p] = None
            else:
                snap[p] = p.grad.detach().clone()
        return snap

    def _restore_grads(self, snap):
        for p, g in snap.items():
            if g is None:
                p.grad = None
            else:
                p.grad = g.clone()

    def _grad_increment(self, params, base_snap):
        inc = []
        for p in params:
            g0 = base_snap[p]
            g = p.grad
            if g0 is None and g is None:
                inc.append(torch.zeros_like(p.data))
            elif g0 is None and g is not None:
                inc.append(g.detach().clone())
            elif g0 is not None and g is None:
                inc.append(-g0.detach().clone())
            else:
                inc.append((g.detach() - g0.detach()))
        return inc

    def _dot(self, g_list_a, g_list_b):
        s = None
        for ga, gb in zip(g_list_a, g_list_b):
            ga_f = ga.float()
            gb_f = gb.float()
            v = (ga_f * gb_f).sum()
            s = v if s is None else (s + v)
        return s if s is not None else torch.zeros((), device=g_list_a[0].device)

    def _pcgrad_merge(self, task_grads):
        k = len(task_grads)
        if k == 1:
            return [g.clone() for g in task_grads[0]]

        proj = [[g.clone() for g in gs] for gs in task_grads]
        order = list(range(k))

        for i in range(k):
            for j in order:
                if i == j:
                    continue
                dot_ij = self._dot(proj[i], proj[j])
                if dot_ij < 0:
                    denom = self._dot(proj[j], proj[j]) + self.pcgrad_eps
                    coeff = dot_ij / denom
                    proj[i] = [gi - coeff * gj for gi, gj in zip(proj[i], proj[j])]

        merged = [torch.zeros_like(g) for g in proj[0]]
        for i in range(k):
            for p_idx in range(len(merged)):
                merged[p_idx] += proj[i][p_idx]
        merged = [g / k for g in merged]
        return merged

    def _compute_chosen_sft_loss(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.is_encoder_decoder:
            labels = batch["chosen_input_ids"].clone()
            labels[batch["chosen_attention_mask"] == 0] = self.label_pad_token_id
            outputs = model(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                labels=labels,
            )
            logits = outputs.logits
            loss_mask = batch["chosen_attention_mask"].bool()
            labels = labels.clone()
            labels[~loss_mask] = self.label_pad_token_id
            return F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=self.label_pad_token_id,
            )

        prompt_ids = batch["prompt_input_ids"]
        prompt_mask = batch["prompt_attention_mask"]
        completion_ids = batch["chosen_input_ids"]
        completion_mask = batch["chosen_attention_mask"]

        input_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        attention_mask = torch.cat((prompt_mask, completion_mask), dim=1)
        loss_mask = torch.cat((torch.zeros_like(prompt_mask), completion_mask), dim=1)

        for i in range(attention_mask.size(0)):
            first_one_idx = torch.nonzero(attention_mask[i])[0].item()
            input_ids[i] = torch.roll(input_ids[i], shifts=-first_one_idx)
            attention_mask[i] = torch.roll(attention_mask[i], shifts=-first_one_idx)
            loss_mask[i] = torch.roll(loss_mask[i], shifts=-first_one_idx)

        empty_cols = torch.sum(attention_mask, dim=0) == 0
        first_empty_col = (
            torch.nonzero(empty_cols)[0].item() if empty_cols.any() else attention_mask.size(1)
        )
        input_ids = input_ids[:, :first_empty_col]
        attention_mask = attention_mask[:, :first_empty_col]
        loss_mask = loss_mask[:, :first_empty_col]

        if self.args.max_length is not None:
            input_ids = input_ids[:, : self.args.max_length]
            attention_mask = attention_mask[:, : self.args.max_length]
            loss_mask = loss_mask[:, : self.args.max_length]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:].clone()
        loss_mask = loss_mask[:, 1:].bool()
        labels[~loss_mask] = self.label_pad_token_id

        return F.cross_entropy(
            torch.flatten(logits, end_dim=1),
            torch.flatten(labels, end_dim=1),
            ignore_index=self.label_pad_token_id,
        )

    def _compute_dpo_and_sft_losses(self, model, batch, train_eval="train"):
        dpo_loss, dpo_metrics = super().get_batch_loss_metrics(
            model, batch, train_eval=train_eval
        )
        sft_loss = self._compute_chosen_sft_loss(model, batch)
        return dpo_loss, sft_loss, dpo_metrics

    def training_step(self, model, inputs, num_items_in_batch=None, **kwargs):
        if self.lora_sam_rho <= 0.0:
            return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch, **kwargs)

        if not self.use_pcgrad:
            return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch, **kwargs)

        model.train()
        inputs = self._prepare_inputs(inputs)

        trainable_params = list(self._iter_trainable_params(model))
        base_snap = self._snapshot_grads(trainable_params)

        no_sync_ctx = nullcontext()
        try:
            if hasattr(self, "accelerator") and hasattr(self.accelerator, "no_sync"):
                no_sync_ctx = self.accelerator.no_sync(model)
        except Exception:
            no_sync_ctx = nullcontext()

        with no_sync_ctx:
            with self.compute_loss_context_manager():
                dpo_1, sft_1, _ = self._compute_dpo_and_sft_losses(model, inputs, train_eval="train")

            if self.args.n_gpu > 1:
                dpo_1 = dpo_1.mean()
                sft_1 = sft_1.mean()

            dpo_1 = dpo_1 / self.args.gradient_accumulation_steps
            sft_1 = (self.sft_weight * sft_1) / self.args.gradient_accumulation_steps

            self.accelerator.backward(dpo_1)
            g_dpo = self._grad_increment(trainable_params, base_snap)

            if self.sft_weight != 0.0:
                self.accelerator.backward(sft_1)
                g_total = self._grad_increment(trainable_params, base_snap)
                g_sft = [gt - gd for gt, gd in zip(g_total, g_dpo)]
                merged_probe = self._pcgrad_merge([g_dpo, g_sft])
            else:
                merged_probe = g_dpo

        self._restore_grads(base_snap)

        for p, g in zip(trainable_params, merged_probe):
            p.grad = g.to(dtype=p.data.dtype) if p.grad is None else g.to(dtype=p.grad.dtype)

        eps_dict, grad_norm = self._lora_sam_perturb(model, rho=self.lora_sam_rho)
        self._restore_grads(base_snap)

        with self.compute_loss_context_manager():
            dpo_2, sft_2, dpo_metrics_2 = self._compute_dpo_and_sft_losses(model, inputs, train_eval="train")

        if self.args.n_gpu > 1:
            dpo_2 = dpo_2.mean()
            sft_2 = sft_2.mean()

        dpo_2_scaled = dpo_2 / self.args.gradient_accumulation_steps
        sft_2_scaled = (self.sft_weight * sft_2) / self.args.gradient_accumulation_steps

        self.accelerator.backward(dpo_2_scaled)
        g_dpo_2 = self._grad_increment(trainable_params, base_snap)

        if self.sft_weight != 0.0:
            self.accelerator.backward(sft_2_scaled)
            g_total_2 = self._grad_increment(trainable_params, base_snap)
            g_sft_2 = [gt - gd for gt, gd in zip(g_total_2, g_dpo_2)]
            merged_update = self._pcgrad_merge([g_dpo_2, g_sft_2])
        else:
            merged_update = g_dpo_2

        for p, g_inc in zip(trainable_params, merged_update):
            g0 = base_snap[p]
            if g0 is None:
                p.grad = g_inc.to(dtype=p.grad.dtype if p.grad is not None else g_inc.dtype)
            else:
                out = g0 + g_inc.to(dtype=g0.dtype)
                p.grad = out

        self._lora_sam_restore(eps_dict)

        if hasattr(self, "store_metrics"):
            metrics_2 = dict(dpo_metrics_2)
            metrics_2["sft_loss"] = sft_2.detach().float().cpu()
            metrics_2["lora_sam_grad_norm"] = grad_norm.float().cpu()
            metrics_2["lora_sam_rho"] = float(self.lora_sam_rho)
            metrics_2["pcgrad_enabled"] = 1.0
            self.store_metrics(metrics_2, train_eval="train")

        total_loss_2 = dpo_2 + self.sft_weight * sft_2
        total_loss_2 = total_loss_2 / self.args.gradient_accumulation_steps
        return total_loss_2.detach()


def _build_sft_trainer(model, tokenizer, dataset, config: RunConfig, output_dir: Path):
    return SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=config.training.batch_size,
            gradient_accumulation_steps=config.training.grad_accum_steps,
            warmup_steps=config.training.warmup_steps,
            num_train_epochs=config.training.epochs,
            learning_rate=config.training.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            output_dir=str(output_dir),
            optim="adamw_torch",
            completion_only_loss=True,
            gradient_checkpointing=config.training.gradient_checkpointing,
            group_by_length=True,
            save_strategy="steps",
            save_steps=config.training.save_steps,
            report_to="none",
        ),
    )


def _build_dpo_trainer(model, tokenizer, dataset, config: RunConfig, output_dir: Path, trainer_cls):
    kwargs = {}
    if trainer_cls is PCSAMTrainer:
        kwargs["use_pcgrad"] = config.method.use_pcgrad
    return trainer_cls(
        model=model,
        ref_model=None,
        args=DPOConfig(
            per_device_train_batch_size=config.training.batch_size,
            gradient_accumulation_steps=config.training.grad_accum_steps,
            warmup_steps=config.training.warmup_steps,
            num_train_epochs=config.training.epochs,
            learning_rate=config.training.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=50,
            output_dir=str(output_dir),
            optim="adamw_torch",
            gradient_checkpointing=False,
            save_strategy="steps",
            save_steps=config.training.save_steps,
            report_to="none",
        ),
        processing_class=tokenizer,
        train_dataset=dataset,
        sft_weight=config.method.sft_weight,
        lora_sam_rho=config.method.sam_rho,
        lora_sam_eps=config.method.sam_eps,
        **kwargs,
    )


def train_from_config(config: RunConfig, output_dir: Path) -> Path:
    set_seed(config.training.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    formatted_data = load_and_format_data(
        dataset_name=config.dataset.name,
        split=config.dataset.split,
        source=config.dataset.source,
        limit=config.dataset.limit,
    )

    model, tokenizer = load_base_model(config.model)
    model = apply_lora(model, config.method)

    method = config.method.name.lower()
    if method == "lora":
        _, train_new = create_training_datasets(formatted_data)
        trainer = _build_sft_trainer(model, tokenizer, train_new, config, output_dir)
    elif method == "sam":
        dpo_dataset = create_dpo_training_datasets(formatted_data)
        trainer = _build_dpo_trainer(model, tokenizer, dpo_dataset, config, output_dir, SAMTrainer)
    elif method == "corsa":
        dpo_dataset = create_dpo_training_datasets(formatted_data)
        trainer = _build_dpo_trainer(model, tokenizer, dpo_dataset, config, output_dir, PCSAMTrainer)
    else:
        raise ValueError(f"Unsupported method '{config.method.name}'.")

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "method": config.method.name,
        "model_name": config.model.name,
        "dataset": {
            "name": config.dataset.name,
            "split": config.dataset.split,
            "source": config.dataset.source,
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "model": config.model.__dict__,
                "dataset": config.dataset.__dict__,
                "method": config.method.__dict__,
                "training": config.training.__dict__,
                "evaluation": config.evaluation.__dict__,
            },
            indent=2,
        )
    )

    return output_dir
