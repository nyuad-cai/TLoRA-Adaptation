"""
Causal activation patching for Llama 3.3 70B on MedAraBench access_gap questions
"""

import os
import csv
import re
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--csv",        required=True)
parser.add_argument("--model_path", required=True)
parser.add_argument("--out_dir",    default="./activation_patching_llama_out")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--max_len",    type=int, default=512)
parser.add_argument("--load_in_8bit", action="store_true", default=False)
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

# Probe layers for patching — near the end of Llama 3.3 70B (80 layers)
# hidden_states[L] = output of transformer layer L-1
# Hook registered on model.model.layers[L-1]
PROBE_LAYERS = [8, 16, 24, 32, 40, 48, 56, 60, 64, 68, 72, 76, 79]

# Patch configs: which layers to inject simultaneously
PATCH_CONFIGS = {
    "patch_L8":     [8],
    "patch_L16":    [16],
    "patch_L24":    [24],
    "patch_L32":    [32],
    "patch_L40":    [40],
    "patch_L48":    [48],
    "patch_L56":    [56],
    "patch_L60":    [60],
    "patch_L64":    [64],
    "patch_L68":    [68],
    "patch_L72":    [72],
    "patch_L76":    [76],
    "patch_L79":    [79],
    "patch_L72_79": [72, 76, 79],
    "patch_L64_79": [64, 68, 72, 76, 79],
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

ag_rows    = [r for r in rows if r["quadrant"] == "access_gap"]
en_texts   = [r["input_english"] for r in ag_rows]
ar_texts   = [r["input_arabic"]  for r in ag_rows]
gt_letters = [extract_letter(r["ground_truth"]) for r in ag_rows]
N = len(ag_rows)
print(f"  Total rows: {len(rows)} | access_gap: {N}")

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
        device_map="auto",
        local_files_only=True,
    )
else:
    # 2-GPU bfloat16: device_map="auto" splits ~140GB across cuda:0 + cuda:1
    print("  Using bfloat16 across 2 GPUs (device_map=auto) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
model.eval()

transformer_layers = model.model.layers
n_layers = len(transformer_layers)
print(f"  Model has {n_layers} transformer layers.")
assert n_layers == 80, f"Expected 80 layers for Llama 3.3 70B, got {n_layers}"

for L in PROBE_LAYERS:
    assert 1 <= L <= n_layers, f"Probe layer {L} out of range [1, {n_layers}]"

ANSWER_TOKEN_IDS = {"A": 32, "B": 33, "C": 34, "D": 35, "E": 36, "F": 37}
print(f"  Answer token IDs: {ANSWER_TOKEN_IDS}")
for letter, tid in ANSWER_TOKEN_IDS.items():
    decoded = tokenizer.decode([tid]).strip()
    if decoded.upper() != letter:
        print(f"  [WARN] Token {tid} decodes as {repr(decoded)}, not '{letter}'")

def tokenize_batch(texts):
    all_ids = []
    for t in texts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": t},
        ]
        # Two-step to avoid tokenizers.Encoding dtype issue in some HF versions
        chat_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        ids = tokenizer.encode(chat_str, add_special_tokens=False)
        if hasattr(ids, "ids"):
            ids = ids.ids
        ids = list(ids)
        all_ids.append(ids[:args.max_len])

    bs    = len(texts)
    max_l = max(len(x) for x in all_ids)
    pad   = tokenizer.pad_token_id
    input_ids = torch.full((bs, max_l), pad, dtype=torch.long)
    attn_mask = torch.zeros((bs, max_l), dtype=torch.long)
    for j, ids in enumerate(all_ids):
        sl = len(ids)
        input_ids[j, max_l - sl:] = torch.tensor(ids)
        attn_mask[j, max_l - sl:] = 1
    seq_lens = attn_mask.sum(dim=1) - 1
    return input_ids, attn_mask, seq_lens

def correct_probs(logits, seq_lens, gt_batch):
    """logits: (bs, seq, vocab)"""
    bs = logits.shape[0]
    probs_all = torch.softmax(logits.float(), dim=-1)
    result = np.zeros(bs, dtype=np.float32)
    for j, gt in enumerate(gt_batch):
        if gt not in ANSWER_TOKEN_IDS: continue
        tid      = ANSWER_TOKEN_IDS[gt]
        result[j] = probs_all[j, seq_lens[j], tid].item()
    return result

print(f"\nPhase 1: English forward pass — caching layers {PROBE_LAYERS}...")
en_hs        = {L: [] for L in PROBE_LAYERS}
en_base_prob = []

for i in range(0, N, args.batch_size):
    batch_en = en_texts[i : i + args.batch_size]
    batch_gt = gt_letters[i : i + args.batch_size]
    bs       = len(batch_en)

    input_ids, attn_mask, seq_lens = tokenize_batch(batch_en)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.to("cuda:0"),
            attention_mask=attn_mask.to("cuda:0"),
            output_hidden_states=True,
        )

    for L in PROBE_LAYERS:
        hs   = outputs.hidden_states[L]                     # (bs, seq, d)
        last = hs[torch.arange(bs), seq_lens].float().cpu() # (bs, d)
        en_hs[L].append(last.numpy())

    probs = correct_probs(outputs.logits.cpu(), seq_lens, batch_gt)
    en_base_prob.extend(probs.tolist())

    if i % 20 == 0:
        print(f"  [EN] {min(i+bs, N)}/{N}")

for L in PROBE_LAYERS:
    en_hs[L] = np.concatenate(en_hs[L], axis=0)   # (N, hidden_dim)
en_base_prob = np.array(en_base_prob)
print(f"  English baseline mean P(correct): {en_base_prob.mean():.3f}")

print(f"\nPhase 2: Arabic baseline (no patching)...")
ar_base_prob = []

for i in range(0, N, args.batch_size):
    batch_ar = ar_texts[i : i + args.batch_size]
    batch_gt = gt_letters[i : i + args.batch_size]
    bs       = len(batch_ar)
    input_ids, attn_mask, seq_lens = tokenize_batch(batch_ar)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.to("cuda:0"),
            attention_mask=attn_mask.to("cuda:0"),
        )
    probs = correct_probs(outputs.logits.cpu(), seq_lens, batch_gt)
    ar_base_prob.extend(probs.tolist())
    if i % 20 == 0:
        print(f"  [AR base] {min(i+bs, N)}/{N}")

ar_base_prob = np.array(ar_base_prob)
print(f"  Arabic baseline mean P(correct): {ar_base_prob.mean():.3f}")

patch_results = {}

for config_name, patch_layers in PATCH_CONFIGS.items():
    print(f"\nPhase 3 [{config_name}]: patching at layers {patch_layers}...")
    config_probs = []

    for i in range(0, N, args.batch_size):
        batch_ar = ar_texts[i : i + args.batch_size]
        batch_gt = gt_letters[i : i + args.batch_size]
        bs       = len(batch_ar)

        input_ids, attn_mask, seq_lens = tokenize_batch(batch_ar)

        batch_en_hs = {
            L: torch.tensor(en_hs[L][i : i + bs])   # (bs, d)
            for L in patch_layers
        }

        handles = []
        for L in patch_layers:
            layer_idx = L - 1   # 0-indexed
            en_h = batch_en_hs[L]
            ar_sl = seq_lens

            def make_hook(en_h_=en_h, ar_sl_=ar_sl, bs_=bs):
                def hook_fn(module, inp, output):
                    if isinstance(output, tuple):
                        hs_out = output[0].clone()
                    else:
                        hs_out = output.clone()
                    for j in range(bs_):
                        hs_out[j, ar_sl_[j]] = (
                            en_h_[j].to(hs_out.device).to(hs_out.dtype)
                        )
                    if isinstance(output, tuple):
                        return (hs_out,) + output[1:]
                    return hs_out
                return hook_fn

            handle = transformer_layers[layer_idx].register_forward_hook(make_hook())
            handles.append(handle)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids.to("cuda:0"),
                attention_mask=attn_mask.to("cuda:0"),
            )
        for h in handles:
            h.remove()

        probs = correct_probs(outputs.logits.cpu(), seq_lens, batch_gt)
        config_probs.extend(probs.tolist())
        if i % 20 == 0:
            print(f"  [{config_name}] {min(i+bs, N)}/{N}")

    patch_results[config_name] = np.array(config_probs)
    print(f"  Mean P(correct): {patch_results[config_name].mean():.3f}")

out_npz = os.path.join(args.out_dir, "patching_results.npz")
save_dict = dict(en_base_prob=en_base_prob, ar_base_prob=ar_base_prob,
                 gt_letters=np.array(gt_letters))
save_dict.update(patch_results)
np.savez(out_npz, **save_dict)
print(f"\nSaved → {out_npz}")

ar_mean = ar_base_prob.mean()
en_mean = en_base_prob.mean()
gap     = en_mean - ar_mean

def recovery(val):
    return 100 * (val - ar_mean) / gap if gap > 0 else 0.0

print("\n══ Activation Patching Results ═══════════════════════════")
print(f"  {'Config':<22} {'Mean P(correct)':>16}  {'Recovery':>10}")
print(f"  {'ar_base':<22} {ar_mean:>16.3f}  {'(0% baseline)':>10}")
for c in PATCH_CONFIGS:
    m = patch_results[c].mean()
    print(f"  {c:<22} {m:>16.3f}  {recovery(m):>9.1f}%")
print(f"  {'en_base':<22} {en_mean:>16.3f}  {'(100%)':>10}")

all_configs = ["ar_base"] + list(PATCH_CONFIGS.keys()) + ["en_base"]
all_means   = ([ar_base_prob.mean()] +
               [patch_results[c].mean() for c in PATCH_CONFIGS] +
               [en_base_prob.mean()])
all_sems    = ([ar_base_prob.std() / np.sqrt(N)] +
               [patch_results[c].std() / np.sqrt(N) for c in PATCH_CONFIGS] +
               [en_base_prob.std() / np.sqrt(N)])

n_patches = len(PATCH_CONFIGS)
cmap = plt.cm.RdYlGn(np.linspace(0.2, 0.85, n_patches))
colors = ["#ADB5BD"] + list(cmap) + ["#2DC653"]

x_labels = (["Arabic\n(baseline)"] +
             [c.replace("patch_", "patch\n") for c in PATCH_CONFIGS] +
             ["English\n(baseline)"])

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(range(len(all_configs)), all_means, color=colors,
              yerr=all_sems, capsize=4, alpha=0.88,
              edgecolor="white", linewidth=0.5)

for bar, m in zip(bars, all_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{m:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

for ci, c in enumerate(PATCH_CONFIGS.keys()):
    xi  = ci + 1
    rec = recovery(patch_results[c].mean())
    ax.text(xi, -0.018, f"{rec:.0f}%", ha="center", va="top",
            fontsize=8.5, color="grey")
ax.text(-0.5, -0.018, "Recovery:", ha="left", va="top", fontsize=8.5, color="grey")

ax.axhline(ar_mean, color="#ADB5BD", linewidth=1.5, linestyle="--", alpha=0.7,
           label=f"Arabic baseline ({ar_mean:.3f})")
ax.axhline(en_mean, color="#2DC653", linewidth=1.5, linestyle="--", alpha=0.7,
           label=f"English baseline ({en_mean:.3f})")

ax.set_xticks(range(len(all_configs)))
ax.set_xticklabels(x_labels, fontsize=9.5)
ax.set_ylabel("Mean P(correct answer letter) at model output", fontsize=11)
ax.set_title(
    "Activation Patching: Injecting English Hidden States into Arabic Forward Pass\n"
    "Llama 3.3 70B  ·  MedAraBench  ·  access_gap quadrant (En✓ Ar✗)",
    fontsize=12, fontweight="bold"
)
ax.legend(fontsize=10, frameon=False, loc="upper left")
ax.set_ylim(-0.03, max(all_means) * 1.25)
ax.grid(axis="y", alpha=0.2)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
for ext in [".pdf", ".png"]:
    out_fig = os.path.join(args.out_dir, f"fig_patching_llama{ext}")
    plt.savefig(out_fig, bbox_inches="tight", dpi=150)
    print(f"Saved → {out_fig}")
plt.close()
print("\nDone.")
