"""
train_lora_align.py: layer-targeted LoRA + cross-lingual distribution alignment.
"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    EarlyStoppingCallback,
    Mistral3ForConditionalGeneration,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

HF_CACHE = "/scratch/ca2627/huggingface"
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

LETTER_SET = {"A", "B", "C", "D", "E", "F"}

def load_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}")
    return data


def normalize_letter(x: Any) -> str:
    x = str(x).strip().upper()
    if x not in LETTER_SET:
        raise ValueError(f"Invalid answer letter: {x}")
    return x


def build_mcq_text(sample: dict) -> str:
    """Arabic MCQ prompt text (question + labelled options)."""
    stem = (sample.get("question") or "").strip()
    if not stem:
        return ""
    option_fields = {
        "A": sample.get("opa"), "B": sample.get("opb"),
        "C": sample.get("opc"), "D": sample.get("opd"),
        "E": sample.get("ope"), "F": sample.get("opf"),
    }
    lines = []
    for letter in ["A", "B", "C", "D", "E", "F"]:
        val = option_fields.get(letter)
        if val is None:
            continue
        val = str(val).strip()
        if val:
            lines.append(f"{letter}) {val}")
    return stem + "\n\n" + "\n".join(lines) if lines else stem


def load_tokenizer(model_name: str) -> MistralTokenizer:
    if os.path.isdir(model_name):
        tok_path = os.path.join(model_name, "tekken.json")
        return MistralTokenizer.from_file(tok_path)
    return MistralTokenizer.from_file("/scratch/ca2627/hf_models/mistral_small_3_2_24b_2506/tekken.json")


def _get_raw_tokenizer(tokenizer: MistralTokenizer):
    if hasattr(tokenizer, "instruct_tokenizer") and hasattr(tokenizer.instruct_tokenizer, "tokenizer"):
        return tokenizer.instruct_tokenizer.tokenizer
    if hasattr(tokenizer, "tokenizer"):
        return tokenizer.tokenizer
    raise AttributeError("Cannot locate raw tokenizer on MistralTokenizer.")


def tokenize_target_text(tokenizer: MistralTokenizer, text: str) -> List[int]:
    text = text.strip()
    raw = _get_raw_tokenizer(tokenizer)
    return raw.encode(text, bos=False, eos=True)


def tokenize_plain_text(tokenizer: MistralTokenizer, text: str, max_length: int) -> List[int]:
    text = text.strip()
    raw  = _get_raw_tokenizer(tokenizer)
    ids  = raw.encode(text, bos=True, eos=False)
    return ids[:max_length]


def encode_mcq_example(
    tokenizer: MistralTokenizer,
    system_prompt: str,
    user_text: str,
    gold_letter: str,
) -> Dict[str, List[int]]:
    req = ChatCompletionRequest(
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ]
    )
    prompt_tok  = tokenizer.encode_chat_completion(req)
    prompt_ids  = list(prompt_tok.tokens)
    target_ids  = tokenize_target_text(tokenizer, gold_letter)
    input_ids   = prompt_ids + target_ids
    labels      = [-100] * len(prompt_ids) + target_ids
    return {"input_ids": input_ids, "labels": labels}


class AlignMCQDataset(Dataset):
    def __init__(
        self,
        rows: List[dict],
        tokenizer: MistralTokenizer,
        system_prompt: str,
        label_key: str = "answer",
        max_length: int = 2048,
        align_max_length: int = 512,
    ):
        self.samples = []
        dropped = 0
        for row in rows:
            try:
                if label_key not in row:
                    raise KeyError(f"Missing label key '{label_key}'")

                user_text = build_mcq_text(row)
                if not user_text:
                    raise ValueError("Empty question/options")
                gold = normalize_letter(row[label_key])
                enc = encode_mcq_example(
                    tokenizer=tokenizer,
                    system_prompt=system_prompt,
                    user_text=user_text,
                    gold_letter=gold,
                )
                if len(enc["input_ids"]) > max_length:
                    raise ValueError(f"Sequence too long: {len(enc['input_ids'])} > {max_length}")

                ar_text = (row.get("full_text_ar") or "").strip()
                if not ar_text:
                    raise ValueError("Missing or empty 'full_text_ar' field")
                ar_ids = tokenize_plain_text(tokenizer, ar_text, align_max_length)
                if not ar_ids:
                    raise ValueError("Arabic alignment text tokenised to empty sequence")

                en_text = (row.get("full_text_en") or "").strip()
                if not en_text:
                    raise ValueError("Missing or empty 'full_text_en' field")
                en_ids = tokenize_plain_text(tokenizer, en_text, align_max_length)
                if not en_ids:
                    raise ValueError("English alignment text tokenised to empty sequence")

                self.samples.append({
                    "input_ids": enc["input_ids"],
                    "labels":    enc["labels"],
                    "ar_ids":    ar_ids,
                    "en_ids":    en_ids,
                })

            except Exception as e:
                print(f"[WARN] dropping id={row.get('id', 'N/A')}: {type(e).__name__}: {e}")
                dropped += 1

        print(f"[Dataset] kept={len(self.samples)}  dropped={dropped}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


@dataclass
class AlignDataCollator:
    pad_token_id: int = 0

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_ce = max(len(f["input_ids"]) for f in features)
        max_ar = max(len(f["ar_ids"])    for f in features)
        max_en = max(len(f["en_ids"])    for f in features)

        input_ids, attention_mask, labels   = [], [], []
        ar_ids_batch, ar_mask_batch         = [], []
        en_ids_batch, en_mask_batch         = [], []

        for f in features:
            ids  = f["input_ids"];  labs = f["labels"]
            pad  = max_ce - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)
            labels.append(labs + [-100] * pad)

            aids = f["ar_ids"]
            apad = max_ar - len(aids)
            ar_ids_batch.append(aids + [self.pad_token_id] * apad)
            ar_mask_batch.append([1] * len(aids) + [0] * apad)

            eids = f["en_ids"]
            epad = max_en - len(eids)
            en_ids_batch.append(eids + [self.pad_token_id] * epad)
            en_mask_batch.append([1] * len(eids) + [0] * epad)

        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels":         torch.tensor(labels,         dtype=torch.long),
            "ar_ids":         torch.tensor(ar_ids_batch,   dtype=torch.long),
            "ar_mask":        torch.tensor(ar_mask_batch,  dtype=torch.long),
            "en_ids":         torch.tensor(en_ids_batch,   dtype=torch.long),
            "en_mask":        torch.tensor(en_mask_batch,  dtype=torch.long),
        }

class ValLossLogger(TrainerCallback):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.log_path   = os.path.join(output_dir, "val_loss_log.json")
        self.entries: List[dict] = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        entry = {
            "step":      state.global_step,
            "epoch":     round(state.epoch, 4) if state.epoch is not None else None,
            "eval_loss": metrics.get("eval_loss"),
        }
        self.entries.append(entry)
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=2)


def load_model(model_name: str, use_qlora: bool):
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    device_map = {"": local_rank} if local_rank >= 0 else "auto"

    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_name, quantization_config=bnb_config,
            torch_dtype=torch.bfloat16, device_map=device_map, cache_dir=HF_CACHE,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
            device_map=device_map, cache_dir=HF_CACHE,
        )
    return model


def apply_lora(model, targeted_layers, paper_label, r=16, alpha=32, dropout=0.05):
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    print(f"[LoRA] Targeted mode: blocks {targeted_layers[0]}-{targeted_layers[-1]} "
          f"(paper notation {paper_label})")
    peft_config = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=target_modules,
        layers_to_transform=targeted_layers,
        layers_pattern=["layers"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model

def get_norm_and_lmhead(peft_model):
    if hasattr(peft_model, "module"):
        peft_model = peft_model.module
    base = peft_model.base_model.model

    if hasattr(base, "language_model"):
        lm = base.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "norm") and hasattr(lm, "lm_head"):
            return lm.model.norm, lm.lm_head
        if hasattr(lm, "norm") and hasattr(lm, "lm_head"):
            return lm.norm, lm.lm_head

    if hasattr(base, "model") and hasattr(base.model, "language_model"):
        lm = base.model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "norm") and hasattr(lm, "lm_head"):
            return lm.model.norm, lm.lm_head
        if hasattr(lm, "norm") and hasattr(lm, "lm_head"):
            return lm.norm, lm.lm_head

    if hasattr(base, "model") and hasattr(base.model, "norm") and hasattr(base, "lm_head"):
        return base.model.norm, base.lm_head

    if (hasattr(base, "model") and hasattr(base.model, "language_model")
            and hasattr(base.model.language_model, "norm") and hasattr(base, "lm_head")):
        return base.model.language_model.norm, base.lm_head

    def _tree(mod, prefix="", depth=3):
        if depth == 0:
            return
        for name, child in mod._modules.items():
            print(f"  {prefix}{name}: {type(child).__name__}")
            _tree(child, prefix + "  ", depth - 1)

    print("[LogitLens] ERROR — could not resolve norm/lm_head. Model tree (depth 3):")
    _tree(base)
    raise AttributeError("Cannot locate norm/lm_head.")

def calibrate_beta(
    model,
    train_dataset,
    collator,
    probe_layers: List[int],
    temperature: float = 1.0,
    n_batches: int = 8,
) -> float:
    
    from torch.utils.data import DataLoader, Subset

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Use the first n_batches samples (deterministic, no shuffle)
    n = min(n_batches, len(train_dataset))
    subset  = Subset(train_dataset, list(range(n)))
    loader  = DataLoader(subset, batch_size=1, collate_fn=collator, shuffle=False)

    norm, lm_head = get_norm_and_lmhead(model)

    ce_acc  = 0.0
    kl_acc  = 0.0
    counted = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            #CE pass 
            ce_out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                output_hidden_states=False,
            )
            ce_acc += ce_out.loss.item()

            #arabic align pass 
            ar_out = model(
                input_ids=batch["ar_ids"],
                attention_mask=batch["ar_mask"],
                output_hidden_states=True,
            )
            #english align pass 
            en_out = model(
                input_ids=batch["en_ids"],
                attention_mask=batch["en_mask"],
                output_hidden_states=True,
            )

            ar_hidden = ar_out.hidden_states
            en_hidden = en_out.hidden_states
            B = batch["labels"].shape[0]
            batch_idx    = torch.arange(B, device=device)
            ar_final_pos = (batch["ar_mask"].sum(dim=1) - 1).clamp(min=0)
            en_final_pos = (batch["en_mask"].sum(dim=1) - 1).clamp(min=0)

            kl_sum = 0.0
            for hs_idx in probe_layers:
                if hs_idx >= len(ar_hidden) or hs_idx >= len(en_hidden):
                    continue
                h_ar = ar_hidden[hs_idx][batch_idx, ar_final_pos, :].to(lm_head.weight.dtype)
                h_en = en_hidden[hs_idx][batch_idx, en_final_pos, :].to(lm_head.weight.dtype)

                logits_ar = lm_head(norm(h_ar)) / temperature
                logits_en = lm_head(norm(h_en)) / temperature

                ar_log_probs = F.log_softmax(logits_ar, dim=-1)
                en_probs     = F.softmax(logits_en,     dim=-1)

                kl = F.kl_div(ar_log_probs, en_probs, reduction="batchmean", log_target=False)
                kl_sum += kl.item()

            kl_acc  += kl_sum
            counted += 1

    model.train()

    if counted == 0 or kl_acc == 0.0:
        print("[Calib] WARNING: calibration failed (kl_acc=0); falling back to β=0.1")
        return 0.1

    ce_mean  = ce_acc  / counted
    kl_mean  = kl_acc  / counted
    beta_star = ce_mean / kl_mean

    print(f"[Calib] Calibration over {counted} samples:")
    print(f"[Calib]   L_CE_mean = {ce_mean:.4f}")
    print(f"[Calib]   L_KL_mean = {kl_mean:.4f}")
    print(f"[Calib]   β* = L_CE / L_KL = {beta_star:.6f}")

    return beta_star

class AlignTrainer(Trainer):
    """
    Extends HuggingFace Trainer with cross-lingual distribution alignment loss.
    """

    def __init__(
        self,
        *args,
        probe_layers: Tuple[int, ...] = (24,),
        alpha: float = 1.0,
        beta: float = 0.1,
        temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.probe_hs_indices = list(probe_layers)
        self.alpha            = alpha
        self.beta             = beta
        self.temperature      = temperature
        self._norm            = None
        self._lm_head         = None
        print(f"[Align] Probe layers: {list(probe_layers)}  α={alpha}  β={beta}  T={temperature}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels         = inputs["labels"]
        input_ids      = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        if not model.training:
            ce_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=False,
            )
            return (ce_outputs.loss, ce_outputs) if return_outputs else ce_outputs.loss

        ar_ids  = inputs["ar_ids"]
        ar_mask = inputs["ar_mask"]
        en_ids  = inputs["en_ids"]
        en_mask = inputs["en_mask"]

        #CE pass
        ce_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=False,
        )
        ce_loss = ce_outputs.loss

        #arabic align pass
        ar_align_outputs = model(
            input_ids=ar_ids,
            attention_mask=ar_mask,
            output_hidden_states=True,
        )
        ar_hidden = ar_align_outputs.hidden_states

        #english align pass
        with torch.no_grad():
            en_outputs = model(
                input_ids=en_ids,
                attention_mask=en_mask,
                output_hidden_states=True,
            )
        en_hidden = en_outputs.hidden_states

        #resolve logit-lens components
        if self._norm is None:
            self._norm, self._lm_head = get_norm_and_lmhead(model)
        norm, lm_head = self._norm, self._lm_head

        #final-token positions 
        batch_idx    = torch.arange(labels.shape[0], device=labels.device)
        ar_final_pos = (ar_mask.sum(dim=1) - 1).clamp(min=0)   # (B,)
        en_final_pos = (en_mask.sum(dim=1) - 1).clamp(min=0)   # (B,)

        l_align = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)

        for hs_idx in self.probe_hs_indices:
            if hs_idx >= len(ar_hidden) or hs_idx >= len(en_hidden):
                print(f"[WARN] probe index {hs_idx} out of range "
                      f"(ar_hidden len={len(ar_hidden)}, en_hidden len={len(en_hidden)}). Skipping.")
                continue

            h_ar = ar_hidden[hs_idx][batch_idx, ar_final_pos, :]          # (B, D)
            h_ar = h_ar.to(lm_head.weight.dtype)

            h_en = en_hidden[hs_idx][batch_idx, en_final_pos, :].detach() # (B, D)
            h_en = h_en.to(lm_head.weight.dtype)

            logits_ar    = lm_head(norm(h_ar)) / self.temperature          # (B, vocab)
            ar_log_probs = F.log_softmax(logits_ar, dim=-1)                # (B, vocab)

            with torch.no_grad():
                logits_en = lm_head(norm(h_en)) / self.temperature         # (B, vocab)
            en_probs = F.softmax(logits_en, dim=-1)                        # (B, vocab)

            kl = F.kl_div(ar_log_probs, en_probs, reduction="batchmean", log_target=False)
            l_align = l_align + kl

        #weighted total loss: α · L_CE + β · L_align 
        total_loss = self.alpha * ce_loss + self.beta * l_align

        if model.training and self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "loss_ce":    ce_loss.item(),
                "loss_align": l_align.item(),
                "loss_total": total_loss.item(),
            })

        return (total_loss, ce_outputs) if return_outputs else total_loss


def main():
    parser = argparse.ArgumentParser(
        description="Exp 3: Targeted LoRA + cross-lingual distribution alignment (L_align)."
    )
    parser.add_argument("--train_file",  required=True)
    parser.add_argument("--val_file",    default=None)
    parser.add_argument("--model_name",  default="mistralai/Mistral-Small-3.2-24B-Instruct-2506")
    parser.add_argument("--output_dir",  required=True)

    parser.add_argument("--label_key",        default="answer")
    parser.add_argument("--max_length",        type=int,   default=2048)
    parser.add_argument("--align_max_length",  type=int,   default=512)
    parser.add_argument("--use_qlora",         action="store_true")
    parser.add_argument("--lora_r",            type=int,   default=16)
    parser.add_argument("--lora_alpha",        type=int,   default=32)
    parser.add_argument("--lora_dropout",      type=float, default=0.05)

    parser.add_argument("--layer_start",  type=int, default=23,
                        help="First block to adapt (0-indexed). Default 23 = paper L24.")
    parser.add_argument("--layer_end",    type=int, default=39,
                        help="Last block to adapt (0-indexed, inclusive). Default 39 = paper L40.")

    parser.add_argument("--probe_layers", type=int, nargs="+", default=[24],
                        help=(
                            "Paper-notation layer indices at which to compute KL alignment. "
                            "hidden_states[k] = output of transformer block k-1. "
                            "Default: [24] (divergence source per Tuned Lens plots). "
                            "Previously used [38, 40] but that was too late."
                        ))
    parser.add_argument("--alpha",        type=float, default=1.0,
                        help="α: weight on L_CE.    L_total = α·L_CE + β·L_align")
    parser.add_argument("--beta",         type=float, default=0.1,
                        help="β: weight on L_align. Ignored when --auto_beta is set.")
    parser.add_argument("--auto_beta",    action="store_true",
                        help=(
                            "Derive β from data: β* = L_CE_init / L_KL_init on a "
                            "calibration batch.  This makes both terms contribute "
                            "equally at step 0 — fully principled, no manual tuning. "
                            "Overrides --beta."
                        ))
    parser.add_argument("--beta_multiplier", type=float, default=1.0,
                        help=(
                            "Dimensionless multiplier m applied to β*: actual β = m × β*. "
                            "Only used when --auto_beta is set.  Default 1.0 = use β* as-is. "
                            "Use values like 0.1, 0.5, 2.0, 10.0 for ablation sweeps."
                        ))
    parser.add_argument("--calib_batches", type=int, default=8,
                        help="Number of samples for β* calibration (--auto_beta only).")
    parser.add_argument("--temperature",  type=float, default=1.0,
                        help="Temperature T for logit-lens softmax.")

    parser.add_argument("--num_train_epochs",            type=float, default=10.0)
    parser.add_argument("--per_device_train_batch_size", type=int,   default=1)
    parser.add_argument("--per_device_eval_batch_size",  type=int,   default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int,   default=16)
    parser.add_argument("--learning_rate",               type=float, default=2.28e-05)
    parser.add_argument("--warmup_ratio",                type=float, default=0.05)
    parser.add_argument("--weight_decay",                type=float, default=0.0)
    parser.add_argument("--logging_steps",               type=int,   default=10)
    parser.add_argument("--save_steps",                  type=int,   default=200)
    parser.add_argument("--eval_steps",                  type=int,   default=200)
    parser.add_argument("--save_total_limit",            type=int,   default=1)
    parser.add_argument("--early_stopping_patience",     type=int,   default=2)
    parser.add_argument("--resume_from_checkpoint",      type=str,   default=None,
                        help="Path to checkpoint dir, or 'auto' to find latest in output_dir")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    targeted_layers = list(range(args.layer_start, args.layer_end + 1))
    paper_label     = f"L{args.layer_start + 1}–L{args.layer_end + 1}"

    print(f"[INFO] LoRA window  : {paper_label} (blocks {targeted_layers[0]}-{targeted_layers[-1]})")
    print(f"[INFO] Probe layers : {args.probe_layers}")
    print(f"[INFO] α={args.alpha}  β={args.beta}  T={args.temperature}")
    print(f"[INFO] LR={args.learning_rate}  warmup={args.warmup_ratio}  epochs={args.num_train_epochs}")

    print("[INFO] Loading tokenizer...")
    tokenizer = load_tokenizer(args.model_name)

    system_prompt = (
        "You are a medical expert answering multiple-choice exam questions. "
        "You will receive exactly ONE question followed by answer options labeled: A), B), C), D), E), and sometimes F). "
        "You must output exactly ONE line in this format: ANSWER: <LETTER>"
        "Rules: - Output ONLY that line. - Do NOT repeat or paraphrase the question. "
        "- Do NOT translate anything. - Do NOT explain your reasoning. - Do NOT list the options."
    )

    print("[INFO] Loading data...")
    train_rows = load_json(args.train_file)
    val_rows   = load_json(args.val_file) if args.val_file else None

    print("[INFO] Building datasets...")
    train_dataset = AlignMCQDataset(
        rows=train_rows, tokenizer=tokenizer, system_prompt=system_prompt,
        label_key=args.label_key, max_length=args.max_length,
        align_max_length=args.align_max_length,
    )
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty after preprocessing.")

    eval_dataset = None
    if val_rows is not None:
        eval_dataset = AlignMCQDataset(
            rows=val_rows, tokenizer=tokenizer, system_prompt=system_prompt,
            label_key=args.label_key, max_length=args.max_length,
            align_max_length=args.align_max_length,
        )
        if len(eval_dataset) == 0:
            print("[WARN] Validation dataset empty; disabling eval.")
            eval_dataset = None

    print("[INFO] Loading model...")
    model = load_model(args.model_name, use_qlora=args.use_qlora)

    print("[INFO] Applying LoRA...")
    model = apply_lora(
        model, targeted_layers=targeted_layers, paper_label=paper_label,
        r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
    )

    # ── β* auto-calibration (runs AFTER LoRA is applied, before training) ────
    beta_star = None
    if args.auto_beta:
        print(f"[INFO] Auto-calibrating β* over {args.calib_batches} samples...")
        beta_star  = calibrate_beta(
            model=model,
            train_dataset=train_dataset,
            collator=AlignDataCollator(pad_token_id=0),
            probe_layers=args.probe_layers,
            temperature=args.temperature,
            n_batches=args.calib_batches,
        )
        args.beta = beta_star * args.beta_multiplier
        print(f"[INFO] β* = {beta_star:.6f}  ×  multiplier {args.beta_multiplier}"
              f"  →  β = {args.beta:.6f}")
    else:
        print(f"[INFO] Using manual β = {args.beta}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        load_best_model_at_end=True if eval_dataset is not None else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=args.save_total_limit,
        bf16=True,
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        ddp_find_unused_parameters=True,
    )

    callbacks = [ValLossLogger(args.output_dir)]
    if eval_dataset is not None and args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=1e-4,
            )
        )

    trainer = AlignTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=AlignDataCollator(pad_token_id=0),
        callbacks=callbacks,
        probe_layers=tuple(args.probe_layers),
        alpha=args.alpha,
        beta=args.beta,
        temperature=args.temperature,
    )

    print("[INFO] Starting training...")
    resume_ckpt = args.resume_from_checkpoint
    if resume_ckpt == "auto":
        from transformers.trainer_utils import get_last_checkpoint
        last = get_last_checkpoint(args.output_dir)
        resume_ckpt = last  # None if no checkpoint exists yet (fine — starts fresh)
        if last:
            print(f"[INFO] Resuming from checkpoint: {last}")
        else:
            print("[INFO] No checkpoint found in output_dir — starting from scratch")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    print("[INFO] Saving adapter...")
    trainer.model.save_pretrained(args.output_dir)

    meta = {
        "experiment":         "Exp3_CrossLingualAlign",
        "base_model":         args.model_name,
        "system_prompt":      system_prompt,
        "label_key":          args.label_key,
        "max_length":         args.max_length,
        "align_max_length":   args.align_max_length,
        "use_qlora":          args.use_qlora,
        "lora_window_blocks": targeted_layers,
        "lora_window_paper":  paper_label,
        "probe_layers_paper": args.probe_layers,
        "alpha":              args.alpha,
        "beta":               args.beta,
        "auto_beta":          args.auto_beta,
        "beta_star":          beta_star,
        "beta_multiplier":    args.beta_multiplier if args.auto_beta else None,
        "temperature":        args.temperature,
        "learning_rate":      args.learning_rate,
        "warmup_ratio":       args.warmup_ratio,
    }
    with open(os.path.join(args.output_dir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
