"""
per-layer cross-lingual KL divergence profile on the base model Mistral
"""

import os
import json
import argparse
import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from transformers import Mistral3ForConditionalGeneration

from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

HF_CACHE = "/scratch/ca2627/huggingface"
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def load_tokenizer(model_name: str) -> MistralTokenizer:
    if os.path.isdir(model_name):
        return MistralTokenizer.from_file(os.path.join(model_name, "tekken.json"))
    return MistralTokenizer.from_hf_hub(model_name)


def _get_raw_tokenizer(tok: MistralTokenizer):
    if hasattr(tok, "instruct_tokenizer") and hasattr(tok.instruct_tokenizer, "tokenizer"):
        return tok.instruct_tokenizer.tokenizer
    if hasattr(tok, "tokenizer"):
        return tok.tokenizer
    raise AttributeError("Cannot locate raw tokenizer.")


def tokenize_plain(tok: MistralTokenizer, text: str, max_len: int = 512) -> List[int]:
    raw = _get_raw_tokenizer(tok)
    ids = raw.encode(text.strip(), bos=True, eos=False)
    return ids[:max_len]

def get_norm_and_lmhead(model):
    """Resolve (RMSNorm, lm_head) from the raw (non-PEFT) model."""
    base = model

    if (hasattr(base, "model") and hasattr(base.model, "language_model")
            and hasattr(base.model.language_model, "norm") and hasattr(base, "lm_head")):
        return base.model.language_model.norm, base.lm_head

    if hasattr(base, "language_model"):
        lm = base.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "norm") and hasattr(lm, "lm_head"):
            return lm.model.norm, lm.lm_head
        if hasattr(lm, "norm") and hasattr(lm, "lm_head"):
            return lm.norm, lm.lm_head

    if hasattr(base, "model") and hasattr(base.model, "norm") and hasattr(base, "lm_head"):
        return base.model.norm, base.lm_head

    # fallback: walk the tree
    def _tree(mod, prefix="", depth=3):
        if depth == 0:
            return
        for name, child in mod._modules.items():
            print(f"  {prefix}{name}: {type(child).__name__}")
            _tree(child, prefix + "  ", depth - 1)

    print("[LogitLens] ERROR — could not resolve norm/lm_head. Model tree:")
    _tree(base)
    raise AttributeError("Cannot locate norm/lm_head.")

def load_pairs(data_file: str, n_examples: int, seed: int = 42) -> List[dict]:
    with open(data_file, "r", encoding="utf-8") as f:
        rows = json.load(f)

    valid = []
    for r in rows:
        ar = (r.get("full_text_ar") or "").strip()
        en = (r.get("full_text_en") or "").strip()
        if ar and en:
            valid.append({"ar": ar, "en": en, "id": r.get("id", "?")})

    print(f"[Data] {len(valid)} valid parallel pairs found in {data_file}")

    rng = random.Random(seed)
    rng.shuffle(valid)
    selected = valid[:n_examples]
    print(f"[Data] Using {len(selected)} examples (seed={seed})")
    return selected

@torch.no_grad()
def compute_kl_profile(
    model,
    norm,
    lm_head,
    pairs: List[dict],
    tokenizer: MistralTokenizer,
    device: torch.device,
    max_len: int = 512,
    dtype=torch.bfloat16,
) -> np.ndarray:
    model.eval()

    n_pairs = len(pairs)
    n_layers = None
    kl_accum = None

    for i, pair in enumerate(pairs):
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n_pairs}] processing...")

        ar_ids = tokenize_plain(tokenizer, pair["ar"], max_len)
        en_ids = tokenize_plain(tokenizer, pair["en"], max_len)

        ar_tensor = torch.tensor([ar_ids], dtype=torch.long, device=device)
        en_tensor = torch.tensor([en_ids], dtype=torch.long, device=device)

        ar_mask = torch.ones_like(ar_tensor)
        en_mask = torch.ones_like(en_tensor)

        ar_out = model(input_ids=ar_tensor, attention_mask=ar_mask, output_hidden_states=True)
        en_out = model(input_ids=en_tensor, attention_mask=en_mask, output_hidden_states=True)

        ar_hidden = ar_out.hidden_states   # tuple: (embedding, L1, ..., L40)
        en_hidden = en_out.hidden_states   # same

        if n_layers is None:
            # hidden_states[0] = embedding output (before any transformer layer)
            # hidden_states[k] = output of transformer block k  (k = 1..L)
            # We profile transformer blocks only: indices 1..L
            n_layers = len(ar_hidden) - 1   # = 40 for Mistral-Small-24B
            kl_accum = np.zeros(n_layers, dtype=np.float64)

        #the final token positions
        ar_pos = ar_tensor.shape[1] - 1
        en_pos = en_tensor.shape[1] - 1

        for ℓ in range(n_layers):
            hs_idx = ℓ + 1   # skip embedding (index 0)

            h_ar = ar_hidden[hs_idx][0, ar_pos, :].unsqueeze(0).to(dtype)
            h_en = en_hidden[hs_idx][0, en_pos, :].unsqueeze(0).to(dtype)

            logits_ar = lm_head(norm(h_ar))           # (1, vocab)
            logits_en = lm_head(norm(h_en))           # (1, vocab)

            ar_log_probs = F.log_softmax(logits_ar, dim=-1)
            en_probs     = F.softmax(logits_en,     dim=-1)

            kl = F.kl_div(ar_log_probs, en_probs, reduction="batchmean", log_target=False)
            kl_accum[ℓ] += kl.item()

        # free memory each step
        del ar_out, en_out, ar_hidden, en_hidden
        torch.cuda.empty_cache()

    kl_profile = kl_accum / n_pairs
    return kl_profile

def derive_window(kl_profile: np.ndarray, l_patch: int) -> dict:
    mu  = float(np.mean(kl_profile))
    std = float(np.std(kl_profile))
    tau = mu + std

    L_kl = None
    for i, kl in enumerate(kl_profile):
        if kl > tau:
            L_kl = i + 1   # convert to 1-indexed paper notation
            break

    if L_kl is None:
        print("[WARN] No layer exceeded threshold — using L1 as fallback.")
        L_kl = 1

    return {
        "tau":         tau,
        "mu":          mu,
        "std":         std,
        "L_kl":        L_kl,
        "L_patch":     l_patch,
        "lora_window": [L_kl, l_patch],
        "kl_probe":    l_patch,
    }

def make_plot(kl_profile: np.ndarray, window: dict, output_path: str):
    n = len(kl_profile)
    layers = np.arange(1, n + 1)   # L1..L40

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(layers, kl_profile, color="#2c7bb6", linewidth=1.8, label="KL divergence")

    #threshold line
    ax.axhline(window["tau"], color="#d7191c", linestyle="--", linewidth=1.2,
               label=f"τ = μ + σ = {window['tau']:.3f}")

    L_kl   = window["L_kl"]
    L_patch = window["L_patch"]

    ax.axvline(L_kl, color="#fdae61", linestyle=":", linewidth=1.5,
               label=f"L_kl = L{L_kl}  (divergence onset)")
    ax.axvline(L_patch, color="#1a9641", linestyle=":", linewidth=1.5,
               label=f"L_patch = L{L_patch}  (causal boundary)")

    ax.axvspan(L_kl, L_patch, alpha=0.12, color="#fdae61",
               label=f"LoRA window L{L_kl}–L{L_patch}")

    ax.set_xlabel("Transformer Layer", fontsize=12)
    ax.set_ylabel("KL Divergence (nats)", fontsize=12)
    ax.set_title(
        "Cross-Lingual KL Divergence Profile (Arabic vs. English, Logit Lens)\n"
        "Mistral-Small-3.2-24B · Base Model · MedAraBench val",
        fontsize=11,
    )
    ax.set_xlim(1, n)
    ax.set_xticks(np.arange(0, n + 1, 5))
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Compute per-layer cross-lingual KL divergence profile (no training)."
    )
    parser.add_argument("--data_file",   required=True,
                        help="JSON file with full_text_ar + full_text_en fields.")
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--model_name",  default="mistralai/Mistral-Small-3.2-24B-Instruct-2506")
    parser.add_argument("--n_examples",  type=int, default=300,
                        help="Number of parallel pairs to average over.")
    parser.add_argument("--max_len",     type=int, default=512,
                        help="Max token length per text (Arabic or English).")
    parser.add_argument("--l_patch",     type=int, default=24,
                        help="L_patch from activation patching (1-indexed paper notation). "
                             "Default 24 = Mistral patching onset.")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    print("[INFO] Loading tokenizer...")
    tokenizer = load_tokenizer(args.model_name)

    pairs = load_pairs(args.data_file, args.n_examples, args.seed)
    if not pairs:
        raise ValueError("No valid parallel pairs found. Check full_text_ar / full_text_en fields.")

    print("[INFO] Loading model (base, no LoRA)...")
    model = Mistral3ForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=HF_CACHE,
    )
    model.eval()

    norm, lm_head = get_norm_and_lmhead(model)
    print(f"[INFO] Model loaded. Profiling {len(pairs)} examples across all layers...")

    kl_profile = compute_kl_profile(
        model=model,
        norm=norm,
        lm_head=lm_head,
        pairs=pairs,
        tokenizer=tokenizer,
        device=device,
        max_len=args.max_len,
        dtype=torch.bfloat16,
    )

    print("\n[Results] Per-layer KL divergence:")
    for i, kl in enumerate(kl_profile):
        bar = "█" * int(kl * 20)
        print(f"  L{i+1:>2}  {kl:.4f}  {bar}")

    window = derive_window(kl_profile, args.l_patch)
    print(f"\n[Window] τ = {window['tau']:.4f}  (μ={window['mu']:.4f}, σ={window['std']:.4f})")
    print(f"[Window] L_kl    = L{window['L_kl']}  (first layer exceeding τ)")
    print(f"[Window] L_patch = L{window['L_patch']}  (from activation patching)")
    print(f"[Window] → LoRA window : L{window['lora_window'][0]}–L{window['lora_window'][1]}")
    print(f"[Window] → KL probe    : L{window['kl_probe']}")

    result = {
        "model":      args.model_name,
        "n_examples": len(pairs),
        "seed":       args.seed,
        "kl_profile": kl_profile.tolist(),
        "window":     window,
    }
    json_path = os.path.join(args.output_dir, "kl_profile.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[Saved] {json_path}")

    plot_path = os.path.join(args.output_dir, "kl_profile.png")
    make_plot(kl_profile, window, plot_path)


if __name__ == "__main__":
    main()
