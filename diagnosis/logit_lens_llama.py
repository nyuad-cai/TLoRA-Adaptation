"""
logit lens analysis for Llama 3.3 70B on MedAraBench.
"""

import os
import csv
import re
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--csv",        required=True,  help="Sampled quadrants CSV (100/quadrant)")
parser.add_argument("--model_path", required=True,  help="Path to Llama 3.3 70B Instruct snapshot dir")
parser.add_argument("--out_dir",    default="./logit_lens_llama_out")
parser.add_argument("--batch_size", type=int, default=1,   help="Keep at 1 for 70B on 80GB")
parser.add_argument("--max_len",    type=int, default=512)
parser.add_argument("--load_in_8bit", action="store_true", default=True,
                    help="8-bit quantization (required for 70B on ≤2×40GB GPU)")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

# Probe layers — 80 transformer layers total for Llama 3.3 70B
# hidden_states[0] = embedding, hidden_states[L] = output of transformer layer L-1
PROBE_LAYERS = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 79, 80]
#note: hidden_states[80] = output of the final transformer layer (layer 79),

QUAD_ORDER  = ["both_correct", "access_gap", "arabic_only", "both_wrong"]
QUAD_COLORS = {
    "both_correct": "#2DC653",
    "access_gap":   "#E63946",
    "arabic_only":  "#F4A261",
    "both_wrong":   "#ADB5BD",
}
QUAD_LABELS = {
    "both_correct": "En✓ Ar✓  (shared knowledge)",
    "access_gap":   "En✓ Ar✗  (access gap)",
    "arabic_only":  "En✗ Ar✓  (Arabic-only)",
    "both_wrong":   "En✗ Ar✗  (both wrong)",
}

SYSTEM_PROMPT = (
    "You are a medical expert. "
    "Answer the following multiple choice question "
    "by responding with only the letter of the correct option: A, B, C, or D. "
    "Do not explain your answer."
)

def extract_letter(val):
    if not val or not isinstance(val, str): return ""
    s = val.strip().upper()
    m = re.search(r'\bANSWER\s*:\s*([A-F])\b', s)
    if m: return m.group(1)
    m = re.search(r'\b([A-F])\b', s)
    return m.group(1) if m else ""

print(f"Loading {args.csv}...")
rows = []
with open(args.csv, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        rows.append(row)
print(f"  {len(rows)} rows")

en_texts   = [r["input_english"] for r in rows]
ar_texts   = [r["input_arabic"]  for r in rows]
quadrants  = np.array([r["quadrant"]     for r in rows])
gt_letters = [extract_letter(r["ground_truth"]) for r in rows]
N = len(rows)

quad_counts = defaultdict(int)
for q in quadrants: quad_counts[q] += 1
print("  Quadrant breakdown:")
for q in QUAD_ORDER:
    print(f"    {q}: {quad_counts[q]}")

print("\nLoading Llama 3.3 70B...")
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if args.load_in_8bit:
    print("  Using 8-bit quantization (LLM.int8())...")
    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_cfg,
        device_map={"": 0},          # force all layers onto GPU 0
        max_memory={0: "78GiB"},     # leave ~2GB headroom
        local_files_only=True,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )

model.eval()
print("  Model loaded.")
print(f"  Transformer layers: {len(model.model.layers)}")   # should be 80

final_norm = model.model.norm
lm_head    = model.lm_head
print(f"  final_norm: {type(final_norm).__name__}")
print(f"  lm_head   : {type(lm_head).__name__}  ({lm_head.weight.shape})")

answer_token_ids = {
    "A": 32,
    "B": 33,
    "C": 34,
    "D": 35,
}
print(f"  Answer token IDs (bare letters): {answer_token_ids}")

for letter, tid in answer_token_ids.items():
    decoded = tokenizer.decode([tid]).strip()
    if decoded.upper() != letter:
        print(f"  [WARN] Token {tid} decodes as {repr(decoded)}, not '{letter}'")

def tokenize_batch(texts):
    all_input_ids = []
    for t in texts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": t},
        ]
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors=None,
        )
        if not isinstance(ids, list):
            try:
                ids = ids['input_ids']   # BatchEncoding → dict-style access
            except (KeyError, TypeError):
                ids = ids.ids            # tokenizers.Encoding → .ids attribute
        all_input_ids.append(ids[:args.max_len])

    bs     = len(texts)
    max_l  = max(len(x) for x in all_input_ids)
    pad_id = tokenizer.pad_token_id

    input_ids = torch.full((bs, max_l), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((bs, max_l), dtype=torch.long)
    for j, ids in enumerate(all_input_ids):
        sl = len(ids)
        # left-pad
        input_ids[j, max_l - sl:] = torch.tensor(ids)
        attn_mask[j, max_l - sl:] = 1

    seq_lens = attn_mask.sum(dim=1) - 1   # index of last real token
    return input_ids, attn_mask, seq_lens

def run_logit_lens(texts, gt_letters_batch, label=""):
    n = len(texts)
    all_ranks = np.zeros((n, len(PROBE_LAYERS)), dtype=np.int32)
    all_probs = np.zeros((n, len(PROBE_LAYERS)), dtype=np.float32)

    for i in range(0, n, args.batch_size):
        batch_texts = texts[i : i + args.batch_size]
        batch_gt    = gt_letters_batch[i : i + args.batch_size]
        bs = len(batch_texts)

        input_ids, attn_mask, seq_lens = tokenize_batch(batch_texts)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids.to(model.device),
                attention_mask=attn_mask.to(model.device),
                output_hidden_states=True,
            )

        for li, layer_idx in enumerate(PROBE_LAYERS):
            hs   = outputs.hidden_states[layer_idx]           # (bs, seq, d)
            last = hs[torch.arange(bs), seq_lens]             # (bs, d)

            # Apply final norm + lm_head (standard logit lens)
            normed = final_norm(last.to(final_norm.weight.dtype))
            logits = lm_head(normed).float().cpu()            # (bs, vocab)

            probs_all = torch.softmax(logits, dim=-1)

            for j, gt in enumerate(batch_gt):
                if gt not in answer_token_ids:
                    all_probs[i+j, li] = 0.0
                    all_ranks[i+j, li] = -1
                    continue
                tok_id = answer_token_ids[gt]
                rank   = (logits[j] > logits[j, tok_id]).sum().item()
                prob   = probs_all[j, tok_id].item()
                all_ranks[i+j, li] = rank
                all_probs[i+j, li] = prob

        if i % 20 == 0:
            print(f"  [{label}] {min(i+bs, n)}/{n}")

    return all_ranks, all_probs

print(f"\nRunning logit lens — English ({N} questions)...")
en_ranks, en_probs = run_logit_lens(en_texts, gt_letters, label="EN")

print(f"\nRunning logit lens — Arabic ({N} questions)...")
ar_ranks, ar_probs = run_logit_lens(ar_texts, gt_letters, label="AR")

out_npz = os.path.join(args.out_dir, "logit_lens_results.npz")
np.savez(out_npz,
         en_ranks=en_ranks, en_probs=en_probs,
         ar_ranks=ar_ranks, ar_probs=ar_probs,
         quadrants=quadrants, layers=np.array(PROBE_LAYERS))
print(f"\nSaved → {out_npz}")

print("\n── Mean P(correct) at each probe layer ──")
header = "  ".join([f"L{l:2d}" for l in PROBE_LAYERS])
print(f"{'Quadrant / Lang':<25} {header}")
for q in QUAD_ORDER:
    mask = quadrants == q
    if not mask.any(): continue
    for lang, probs in [("English", en_probs), ("Arabic", ar_probs)]:
        vals = "  ".join([f"{probs[mask, li].mean():.3f}" for li in range(len(PROBE_LAYERS))])
        print(f"  {q[:15]:<15} {lang:<8} {vals}")
    print()

print("Generating figure...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

for ax, (metric_en, metric_ar, ylabel, title_suffix, invert) in zip(axes, [
    (en_probs, ar_probs,
     "Mean probability of correct answer letter",
     "Probability of Correct Answer at Each Layer", False),
    (en_ranks, ar_ranks,
     "Mean rank of correct answer letter\n(lower = better)",
     "Rank of Correct Answer at Each Layer", True),
]):
    for q in QUAD_ORDER:
        mask = quadrants == q
        if not mask.any(): continue

        en_vals = metric_en[mask].mean(axis=0)
        ar_vals = metric_ar[mask].mean(axis=0)
        en_err  = metric_en[mask].std(axis=0) / np.sqrt(mask.sum())
        ar_err  = metric_ar[mask].std(axis=0) / np.sqrt(mask.sum())

        ax.plot(PROBE_LAYERS, en_vals, color=QUAD_COLORS[q], linewidth=2.5,
                linestyle="-", marker="o", markersize=6)
        ax.fill_between(PROBE_LAYERS, en_vals - en_err, en_vals + en_err,
                        color=QUAD_COLORS[q], alpha=0.12)
        ax.plot(PROBE_LAYERS, ar_vals, color=QUAD_COLORS[q], linewidth=2.5,
                linestyle="--", marker="^", markersize=6)
        ax.fill_between(PROBE_LAYERS, ar_vals - ar_err, ar_vals + ar_err,
                        color=QUAD_COLORS[q], alpha=0.08)

    ax.set_xlabel("Layer depth", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title_suffix, fontsize=12, fontweight="bold")
    ax.set_xticks(PROBE_LAYERS)
    ax.set_xticklabels(
        ["L0\n(emb.)" if l == 0 else ("L79\n(final)" if l == 80 else f"L{l}")
         for l in PROBE_LAYERS], fontsize=10
    )
    if invert: ax.invert_yaxis()
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

quad_handles  = [mlines.Line2D([], [], color=QUAD_COLORS[q], linewidth=2,
                                label=QUAD_LABELS[q]) for q in QUAD_ORDER]
style_handles = [
    mlines.Line2D([], [], color="grey", linewidth=2, linestyle="-",
                  marker="o", markersize=6, label="English  ●"),
    mlines.Line2D([], [], color="grey", linewidth=2, linestyle="--",
                  marker="^", markersize=6, label="Arabic  ▲"),
]
fig.legend(handles=quad_handles + style_handles,
           loc="lower center", ncol=3, fontsize=10,
           frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.suptitle(
    "Logit Lens: Correct Answer Emergence Across Layers\n"
    "Llama 3.3 70B  ·  MedAraBench",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
for ext in [".pdf", ".png"]:
    out_fig = os.path.join(args.out_dir, f"fig_logit_lens_llama{ext}")
    plt.savefig(out_fig, bbox_inches="tight", dpi=150)
    print(f"Saved → {out_fig}")
plt.close()
print("\nDone.")
