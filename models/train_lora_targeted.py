#!/usr/bin/env python3
"""
Unified LoRA fine-tuning for Mistral-Small-3.2-24B-Instruct, full and layer-targeted variants.
"""

import os
import glob
import json
import logging
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset

from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from transformers import (
    Trainer,
    TrainingArguments,
    Mistral3ForConditionalGeneration,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from peft import LoraConfig, get_peft_model

HF_CACHE = "/scratch/ca2627/huggingface"
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

#Mistral-Small-3.2-24B has 40 transformer blocks.
LAYER_RANGES: Dict[str, Optional[List[int]]] = {
    "targeted_l01": list(range(0, 24)),    # upstream probe (L1–L24, 24 blocks)
    "targeted_l14": list(range(13, 24)),   # mid-window ablation (L14–L24, 11 blocks)
    "targeted_l24": list(range(23, 40)),   # paper §4.2 primary method (L24–L40, 17 blocks)
    "targeted_l32": list(range(31, 40)),   # ablation (L32–L40, 9 blocks)
    "targeted_l35": list(range(34, 40)),   # late-layer probe (L35–L40, 6 blocks)
    "targeted_l01_34": list(range(0, 34)),   # CE-only ablation (L1–L34, 34 blocks)
    "full":         None,                  # full LoRA (all 40 blocks)
}

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

LETTER_SET = {"A", "B", "C", "D", "E", "F"}

SYSTEM_PROMPT = (
    "You are a medical expert answering multiple-choice exam questions. "
    "You will receive exactly ONE question followed by answer options labeled: "
    "A), B), C), D), E), and sometimes F). "
    "You must output exactly ONE line in this format: ANSWER: <LETTER> "
    "Rules: Output ONLY that line. Do NOT repeat or paraphrase the question. "
    "Do NOT translate anything. Do NOT explain your reasoning. "
    "Do NOT list the options."
)

logger = logging.getLogger(__name__)

def load_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def normalize_letter(x: Any) -> str:
    x = str(x).strip().upper()
    if x not in LETTER_SET:
        raise ValueError(f"Invalid answer letter: {x!r}")
    return x


def build_mcq_text(row: dict) -> str:
    stem = (row.get("question") or "").strip()
    if not stem:
        return ""
    lines = []
    for letter, key in zip("ABCDEF", ["opa", "opb", "opc", "opd", "ope", "opf"]):
        val = (row.get(key) or "").strip()
        if val:
            lines.append(f"{letter}) {val}")
    return stem + ("\n\n" + "\n".join(lines) if lines else "")


def load_tokenizer(model_name: str) -> MistralTokenizer:
    if os.path.isdir(model_name):
        tok_path = os.path.join(model_name, "tekken.json")
        return MistralTokenizer.from_file(tok_path)
    return MistralTokenizer.from_file("/scratch/ca2627/hf_models/mistral_small_3_2_24b_2506/tekken.json")


def _raw_encode(tokenizer: MistralTokenizer, text: str) -> List[int]:
    """encode plain text (no chat template) via the underlying sub-tokenizer"""
    text = text.strip()
    # mistral-common ≥0.4 path
    if hasattr(tokenizer, "instruct_tokenizer") and \
            hasattr(tokenizer.instruct_tokenizer, "tokenizer"):
        return tokenizer.instruct_tokenizer.tokenizer.encode(
            text, bos=False, eos=True)
    if hasattr(tokenizer, "tokenizer"):
        return tokenizer.tokenizer.encode(text, bos=False, eos=True)
    raise AttributeError(
        "Cannot find raw encode path on MistralTokenizer. "
        "Check your mistral-common version.")

def encode_example(
    tokenizer: MistralTokenizer,
    user_text: str,
    gold_letter: str,
) -> Dict[str, List[int]]:
    req = ChatCompletionRequest(messages=[
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user",   "content": [{"type": "text", "text": user_text}]},
    ])
    prompt_ids = list(tokenizer.encode_chat_completion(req).tokens)
    target_ids = _raw_encode(tokenizer, gold_letter)
    return {
        "input_ids": prompt_ids + target_ids,
        "labels":    [-100] * len(prompt_ids) + target_ids,
    }

class MCQDataset(Dataset):
    def __init__(
        self,
        rows: List[dict],
        tokenizer: MistralTokenizer,
        label_key: str = "answer",
        max_length: int = 2048,
    ):
        self.samples: List[Dict[str, List[int]]] = []
        dropped = 0
        for row in rows:
            try:
                gold = normalize_letter(row[label_key])
                user_text = build_mcq_text(row)
                if not user_text:
                    raise ValueError("Empty question/options")
                enc = encode_example(tokenizer, user_text, gold)
                if len(enc["input_ids"]) > max_length:
                    raise ValueError(
                        f"Sequence length {len(enc['input_ids'])} > {max_length}")
                self.samples.append(enc)
            except Exception as e:
                logger.warning("Dropping id=%s  %s: %s",
                               row.get("id", "?"), type(e).__name__, e)
                dropped += 1
        logger.info("Dataset ready: kept=%d  dropped=%d", len(self.samples), dropped)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

@dataclass
class DataCollatorForCausalLM:
    pad_token_id: int = 0

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        ids_batch, mask_batch, lab_batch = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            ids_batch.append(f["input_ids"] + [self.pad_token_id] * pad)
            mask_batch.append([1] * len(f["input_ids"]) + [0] * pad)
            lab_batch.append(f["labels"] + [-100] * pad)
        return {
            "input_ids":      torch.tensor(ids_batch,  dtype=torch.long),
            "attention_mask": torch.tensor(mask_batch, dtype=torch.long),
            "labels":         torch.tensor(lab_batch,  dtype=torch.long),
        }

def load_model(model_name: str, use_device_map: bool):
    kwargs: Dict[str, Any] = dict(
        torch_dtype=torch.bfloat16,
        cache_dir=HF_CACHE,
    )
    if use_device_map:
        kwargs["device_map"] = "auto"
    model = Mistral3ForConditionalGeneration.from_pretrained(model_name, **kwargs)
    return model


def apply_lora(
    model,
    mode: str,
    r: int,
    alpha: int,
    dropout: float,
) -> Any:
    layers_to_transform = LAYER_RANGES[mode]
    cfg_kwargs: Dict[str, Any] = dict(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    if layers_to_transform is not None:
        cfg_kwargs["layers_to_transform"] = layers_to_transform
        cfg_kwargs["layers_pattern"] = ["layers"]
        paper_lo = layers_to_transform[0] + 1
        logger.info(
            "LoRA: mode=%s  blocks=%s  (paper L%d–L40)",
            mode, layers_to_transform, paper_lo,
        )
    else:
        logger.info("LoRA: mode=full  all 40 blocks")

    model = get_peft_model(model, LoraConfig(**cfg_kwargs))
    model.print_trainable_parameters()
    return model

class ValLossLogger(TrainerCallback):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.records: List[dict] = []

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, float],
        **kwargs,
    ):
        if "eval_loss" not in metrics:
            return
        rec = {
            "epoch":     round(state.epoch or 0.0, 4),
            "step":      state.global_step,
            "eval_loss": round(metrics["eval_loss"], 6),
        }
        self.records.append(rec)
        path = os.path.join(self.output_dir, "val_loss_log.json")
        # Only rank-0 should write; in DDP Trainer calls callbacks on all ranks
        # but file writes from non-rank-0 are harmless (same content, race benign).
        with open(path, "w") as f:
            json.dump(self.records, f, indent=2)
        logger.info(
            "  eval  epoch=%.2f  step=%d  eval_loss=%.4f",
            state.epoch or 0.0, state.global_step, metrics["eval_loss"],
        )

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Targeted / full LoRA training on Mistral-Small-3.2-24B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("data")
    g.add_argument("--train_file", required=True,
                   help="Path to training JSON (list of MCQ dicts)")
    g.add_argument("--val_file",   default=None,
                   help="Path to validation JSON. Required for early stopping.")
    g.add_argument("--label_key",  default="answer",
                   help="Key for the gold answer letter in each JSON object")
    g.add_argument("--max_length", type=int, default=2048,
                   help="Max token length per example (longer are dropped)")

    g = p.add_argument_group("model")
    g.add_argument("--model_name", default="mistralai/Mistral-Small-3.2-24B-Instruct-2506")
    g.add_argument("--output_dir", required=True,
                   help="Directory for checkpoints, logs, and final adapter")

    g = p.add_argument_group("lora")
    g.add_argument("--lora_mode", default="targeted_l24",
                   choices=["targeted_l01", "targeted_l14", "targeted_l24", "targeted_l32", "targeted_l35", "full", "targeted_l01_34"],
                   help=(
                       "targeted_l24 → L24-L40 (paper §4.2, ~40M params); "
                       "targeted_l32 → L32-L40 (ablation); "
                       "full → all layers (~93M params)"
                   ))
    g.add_argument("--lora_r",       type=int,   default=16)
    g.add_argument("--lora_alpha",   type=int,   default=32)
    g.add_argument("--lora_dropout", type=float, default=0.05)

    g = p.add_argument_group("training")
    g.add_argument("--num_train_epochs",            type=float, default=5,
                   help="Use 5 for convergence check, 3 for random-search trials")
    g.add_argument("--per_device_train_batch_size", type=int,   default=1)
    g.add_argument("--per_device_eval_batch_size",  type=int,   default=2)
    g.add_argument("--gradient_accumulation_steps", type=int,   default=16,
                   help="Effective batch = per_device × n_gpus × grad_accum")
    g.add_argument("--learning_rate",               type=float, default=2e-4)
    g.add_argument("--warmup_ratio",                type=float, default=0.05,
                   help="Fraction of total steps for linear warmup (paper §4.5: 5%%)")
    g.add_argument("--weight_decay",                type=float, default=0.0)
    g.add_argument("--gradient_checkpointing",      action="store_true",
                   help="Enable gradient checkpointing to reduce activation memory")

    g = p.add_argument_group("eval_and_save")
    g.add_argument("--eval_steps",               type=int, default=200,
                   help="Evaluate (and potentially save) every N optimizer steps")
    g.add_argument("--save_steps",               type=int, default=200)
    g.add_argument("--save_total_limit",         type=int, default=3,
                   help="Keep at most N checkpoints (+ always best if early stopping)")
    g.add_argument("--logging_steps",            type=int, default=20)
    g.add_argument("--early_stopping_patience",  type=int, default=0,
                   help=(
                       "0 = disabled. "
                       "N = stop after N evaluations with no improvement on eval_loss. "
                       "Recommended: 1 for search runs, 2 for convergence check "
                       "(or 0 to let all 5 epochs run)."
                   ))

    g = p.add_argument_group("checkpoint")
    g.add_argument("--resume_from_checkpoint", type=str, default=None,
                   help=(
                       "Path to a checkpoint directory to resume from, "
                       "or 'auto' to resume from the latest checkpoint in output_dir."
                   ))

    g = p.add_argument_group("hardware")
    g.add_argument("--single_gpu", action="store_true",
                   help=(
                       "Run in single-process mode with device_map='auto'. "
                       "Default is DDP-compatible (launch with torchrun). "
                       "Use this flag if you are NOT using torchrun."
                   ))

    return p.parse_args()

def resolve_checkpoint(args: argparse.Namespace) -> Optional[str]:
    if args.resume_from_checkpoint is None:
        return None
    if args.resume_from_checkpoint == "auto":
        checkpoints = sorted(
            glob.glob(os.path.join(args.output_dir, "checkpoint-*")),
            key=lambda p: int(p.rsplit("-", 1)[-1]),
        )
        if checkpoints:
            logger.info("Auto-resuming from %s", checkpoints[-1])
            return checkpoints[-1]
        logger.info("No checkpoint found in %s; starting fresh.", args.output_dir)
        return None
    return args.resume_from_checkpoint

def main():
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_ddp = world_size > 1
    use_device_map = args.single_gpu or not is_ddp

    if local_rank == 0:
        if is_ddp:
            logger.info("DDP mode: WORLD_SIZE=%d  LOCAL_RANK=%d", world_size, local_rank)
        else:
            logger.info("Single-process mode (device_map=auto)")
        logger.info("lora_mode=%s  lr=%.2e  epochs=%g  output=%s",
                    args.lora_mode, args.learning_rate,
                    args.num_train_epochs, args.output_dir)

    logger.info("[rank%d] Loading tokenizer ...", local_rank)
    tokenizer = load_tokenizer(args.model_name)

    logger.info("[rank%d] Loading data ...", local_rank)
    train_rows = load_json(args.train_file)
    val_rows   = load_json(args.val_file) if args.val_file else None

    train_ds = MCQDataset(train_rows, tokenizer, args.label_key, args.max_length)
    if len(train_ds) == 0:
        raise RuntimeError("Training dataset is empty after preprocessing — check your data.")

    eval_ds: Optional[MCQDataset] = None
    if val_rows is not None:
        eval_ds = MCQDataset(val_rows, tokenizer, args.label_key, args.max_length)
        if len(eval_ds) == 0:
            logger.warning("Val dataset empty after preprocessing — disabling evaluation.")
            eval_ds = None

    if args.early_stopping_patience > 0 and eval_ds is None:
        raise ValueError(
            "--early_stopping_patience > 0 requires --val_file to be provided.")

    logger.info("[rank%d] Loading model %s ...", local_rank, args.model_name)
    model = load_model(args.model_name, use_device_map=use_device_map)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled.")

    logger.info("[rank%d] Applying LoRA ...", local_rank)
    model = apply_lora(model, args.lora_mode, args.lora_r, args.lora_alpha,
                       args.lora_dropout)

    use_early_stop = args.early_stopping_patience > 0 and eval_ds is not None

    training_args = TrainingArguments(
        output_dir=args.output_dir,

        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",          
        weight_decay=args.weight_decay,
        optim="adamw_torch",

        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps if eval_ds is not None else None,

        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=use_early_stop,
        metric_for_best_model="eval_loss"  if use_early_stop else None,
        greater_is_better=False            if use_early_stop else None,

        logging_steps=args.logging_steps,
        report_to="none",

        bf16=True,
        fp16=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,

        #adapters so they produce no gradients; DDP needs to know to skip them.
        ddp_find_unused_parameters=True,
    )

    callbacks = [ValLossLogger(args.output_dir)]
    if use_early_stop:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=1e-4,
        ))
        if local_rank == 0:
            logger.info("Early stopping enabled: patience=%d on eval_loss",
                        args.early_stopping_patience)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForCausalLM(pad_token_id=0),
        callbacks=callbacks,
    )

    checkpoint = resolve_checkpoint(args)
    if local_rank == 0:
        logger.info(
            "Starting training: mode=%s  lr=%.2e  epochs=%g%s",
            args.lora_mode,
            args.learning_rate,
            args.num_train_epochs,
            f"  (resuming from {checkpoint})" if checkpoint else "",
        )
    trainer.train(resume_from_checkpoint=checkpoint)

    if local_rank == 0:
        trainer.model.save_pretrained(args.output_dir)
        meta = {
            "base_model":        args.model_name,
            "lora_mode":         args.lora_mode,
            "targeted_blocks":   LAYER_RANGES[args.lora_mode],
            "paper_layer_range": (
                f"L{LAYER_RANGES[args.lora_mode][0]+1}–L{LAYER_RANGES[args.lora_mode][-1]+1}"
                if LAYER_RANGES[args.lora_mode] else "L1–L40"
            ),
            "lora_r":            args.lora_r,
            "lora_alpha":        args.lora_alpha,
            "target_modules":    TARGET_MODULES,
            "learning_rate":     args.learning_rate,
            "num_train_epochs":  args.num_train_epochs,
            "warmup_ratio":      args.warmup_ratio,
            "label_key":         args.label_key,
            "system_prompt":     SYSTEM_PROMPT,
        }
        with open(os.path.join(args.output_dir, "train_meta.json"), "w",
                  encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info("Adapter saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
