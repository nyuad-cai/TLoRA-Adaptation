"""
Reverse causal activation patching for Mistral-Small-3.2-24B on MedAraBench
"""

import os, csv, re, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

parser = argparse.ArgumentParser()
parser.add_argument("--csv",        required=True)
parser.add_argument("--model_path", required=True)
parser.add_argument("--out_dir",    default="./activation_patching_mistral_reverse_out")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--max_len",    type=int, default=512)
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

PROBE_LAYERS = [4, 8, 12, 16, 20, 24, 28, 32, 34, 36, 38, 40]

PATCH_CONFIGS = {
    "patch_L4":     [4],
    "patch_L8":     [8],
    "patch_L12":    [12],
    "patch_L16":    [16],
    "patch_L20":    [20],
    "patch_L24":    [24],
    "patch_L28":    [28],
    "patch_L32":    [32],
    "patch_L34":    [34],
    "patch_L36":    [36],
    "patch_L38":    [38],
    "patch_L40":    [40],
    "patch_L36_40": [36, 38, 40],
    "patch_L34_40": [34, 36, 38, 40],
}

ANSWER_TOKEN_IDS = {
    "A": 1065, "B": 1066, "C": 1067,
    "D": 1068, "E": 1069, "F": 1070,
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

print("\nLoading Mistral-Small-3.2-24B...")
from transformers import AutoTokenizer, Mistral3ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = Mistral3ForConditionalGeneration.from_pretrained(
    args.model_path,
    dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)
model.eval()
print(f"  Model type: {type(model).__name__}")

def get_transformer_layers(model):
    candidates = [
        ("model.model.language_model.model.layers", lambda m: m.model.language_model.model.layers),
        ("model.model.language_model.layers",        lambda m: m.model.language_model.layers),
        ("model.model.layers",                       lambda m: m.model.layers),
    ]
    for name, fn in candidates:
        try:
            layers = fn(model)
            if layers is not None and len(layers) > 0:
                print(f"  Transformer layers at: {name}  (n={len(layers)})")
                return layers
        except AttributeError:
            continue
    raise RuntimeError("Cannot find transformer layers.")

transformer_layers = get_transformer_layers(model)
n_layers = len(transformer_layers)
print(f"  n_layers: {n_layers}")

for L in PROBE_LAYERS:
    assert 1 <= L <= n_layers, f"Probe layer {L} out of range [1, {n_layers}]"

print("  Verifying answer token IDs ...")
for letter, tid in ANSWER_TOKEN_IDS.items():
    decoded = tokenizer.decode([tid]).strip()
    status = "✓" if decoded.upper() == letter else f"[WARN] decodes as {repr(decoded)}"
    print(f"    {letter} → {tid}  {status}")

def tokenize_batch(texts):
    all_ids = []
    for t in texts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": t},
        ]
        chat_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(chat_str)
        ids = encoded["input_ids"]
        if hasattr(ids, "ids"): ids = ids.ids
        all_ids.append(list(ids)[:args.max_len])

    bs    = len(texts)
    max_l = max(len(x) for x in all_ids)
    pad   = tokenizer.pad_token_id
    input_ids = torch.full((bs, max_l), pad, dtype=torch.long)
    attn_mask = torch.zeros((bs, max_l), dtype=torch.long)
    for j, ids in enumerate(all_ids):
        sl = len(ids)
        input_ids[j, max_l - sl:] = torch.tensor(ids)
        attn_mask[j, max_l - sl:] = 1
    seq_lens = torch.full((bs,), max_l - 1, dtype=torch.long)
    return input_ids, attn_mask, seq_lens

def correct_probs(logits, seq_lens, gt_batch):
    bs = logits.shape[0]
    probs_all = torch.softmax(logits.float(), dim=-1)
    result = np.zeros(bs, dtype=np.float32)
    for j, gt in enumerate(gt_batch):
        if gt not in ANSWER_TOKEN_IDS: continue
        result[j] = probs_all[j, seq_lens[j], ANSWER_TOKEN_IDS[gt]].item()
    return result

print(f"\nPhase 1: Arabic forward pass — caching layers {PROBE_LAYERS} ...")
ar_hs        = {L: [] for L in PROBE_LAYERS}
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
            output_hidden_states=True,
        )
    for L in PROBE_LAYERS:
        hs   = outputs.hidden_states[L]
        last = hs[torch.arange(bs), seq_lens].float().cpu()
        ar_hs[L].append(last.numpy())
    probs = correct_probs(outputs.logits.cpu(), seq_lens, batch_gt)
    ar_base_prob.extend(probs.tolist())
    if i % 20 == 0:
        print(f"  [AR] {min(i+bs, N)}/{N}")

for L in PROBE_LAYERS:
    ar_hs[L] = np.concatenate(ar_hs[L], axis=0)
ar_base_prob = np.array(ar_base_prob)
print(f"  Arabic baseline mean P(correct): {ar_base_prob.mean():.3f}")

print(f"\nPhase 2: English baseline (no patching) ...")
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
        )
    probs = correct_probs(outputs.logits.cpu(), seq_lens, batch_gt)
    en_base_prob.extend(probs.tolist())
    if i % 20 == 0:
        print(f"  [EN base] {min(i+bs, N)}/{N}")

en_base_prob = np.array(en_base_prob)
print(f"  English baseline mean P(correct): {en_base_prob.mean():.3f}")

patch_results = {}

for config_name, patch_layers in PATCH_CONFIGS.items():
    print(f"\nPhase 3 [{config_name}]: injecting Arabic → English at layers {patch_layers} ...")
    config_probs = []

    for i in range(0, N, args.batch_size):
        batch_en = en_texts[i : i + args.batch_size]   # ← English is the recipient
        batch_gt = gt_letters[i : i + args.batch_size]
        bs       = len(batch_en)
        input_ids, attn_mask, seq_lens = tokenize_batch(batch_en)

        #arabic activations injected into english forward pass
        batch_ar_hs = {L: torch.tensor(ar_hs[L][i : i + bs]) for L in patch_layers}

        handles = []
        for L in patch_layers:
            ar_h  = batch_ar_hs[L]
            en_sl = seq_lens

            def make_hook(ar_h_=ar_h, en_sl_=en_sl, bs_=bs):
                def hook_fn(module, inp, output):
                    hs_out = output[0].clone() if isinstance(output, tuple) else output.clone()
                    for j in range(bs_):
                        hs_out[j, en_sl_[j]] = ar_h_[j].to(hs_out.device).to(hs_out.dtype)
                    return (hs_out,) + output[1:] if isinstance(output, tuple) else hs_out
                return hook_fn

            handle = transformer_layers[L - 1].register_forward_hook(make_hook())
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

out_npz = os.path.join(args.out_dir, "patching_results_reverse.npz")
save_dict = dict(en_base_prob=en_base_prob, ar_base_prob=ar_base_prob,
                 gt_letters=np.array(gt_letters))
save_dict.update(patch_results)
np.savez(out_npz, **save_dict)
print(f"\nSaved → {out_npz}")

ar_mean = ar_base_prob.mean()
en_mean = en_base_prob.mean()
gap     = en_mean - ar_mean

def degradation(val):
    """% drop toward Arabic baseline. 0% = no effect; 100% = fully degraded."""
    return 100 * (en_mean - val) / gap if gap > 0 else 0.0

print("\n══ Reverse Patching Results (Arabic → English injection) ════════")
print(f"  {'Config':<22} {'Mean P':>10}  {'Degradation':>12}")
print(f"  {'en_base':<22} {en_mean:>10.3f}  {'(0%)':>12}")
for c in PATCH_CONFIGS:
    m = patch_results[c].mean()
    print(f"  {c:<22} {m:>10.3f}  {degradation(m):>11.1f}%")
print(f"  {'ar_base':<22} {ar_mean:>10.3f}  {'(100%)':>12}")

PALETTE = {
    'Base': '#87CEEB',
    'CoT':  '#2A9D8F',
    'IR':   '#0D3349',
    'AP':   '#F4C430',
    'FS':   '#F07C00',
    'MP':   '#C0392B',
}
_PATCH_CMAP = LinearSegmentedColormap.from_list(
    "patch_grad",
    [PALETTE['IR'], PALETTE['AP'], PALETTE['FS'], PALETTE['MP']],
)

n_patches = len(PATCH_CONFIGS)
patch_cols = [_PATCH_CMAP(i / (n_patches - 1)) for i in range(n_patches)]

all_configs = ["en_base"] + list(PATCH_CONFIGS.keys()) + ["ar_base"]
all_means   = [en_base_prob.mean()] + [patch_results[c].mean() for c in PATCH_CONFIGS] + [ar_base_prob.mean()]
all_sems    = ([en_base_prob.std() / np.sqrt(N)] +
               [patch_results[c].std() / np.sqrt(N) for c in PATCH_CONFIGS] +
               [ar_base_prob.std() / np.sqrt(N)])
colors      = [PALETTE['CoT']] + patch_cols + [PALETTE['Base']]

x_labels = (["English\n(base)"] +
             [c.replace("patch_", "") for c in PATCH_CONFIGS] +
             ["Arabic\n(base)"])

fig, ax = plt.subplots(figsize=(13, 6))
bars = ax.bar(range(len(all_configs)), all_means, color=colors,
              yerr=all_sems, capsize=4, alpha=0.88,
              edgecolor="white", linewidth=0.5)

for bar, m in zip(bars, all_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{m:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Degradation % below patch bars
y_ann = -max(all_means) * 0.07
ax.text(-0.5, y_ann, "Degrad.:", ha="left", va="top", fontsize=8, color="#555555")
for ci, c in enumerate(PATCH_CONFIGS.keys()):
    ax.text(ci + 1, y_ann, f"{degradation(patch_results[c].mean()):.0f}%",
            ha="center", va="top", fontsize=8, color="#555555")

ax.axhline(en_mean, color=PALETTE['CoT'], linewidth=1.5, linestyle="--", alpha=0.8,
           label=f"English baseline ({en_mean:.3f})")
ax.axhline(ar_mean, color=PALETTE['Base'], linewidth=1.5, linestyle="--", alpha=0.8,
           label=f"Arabic baseline ({ar_mean:.3f})")

ax.set_xticks(range(len(all_configs)))
ax.set_xticklabels(x_labels, fontsize=9.5)
ax.set_ylabel("Mean P(correct answer letter) at model output", fontsize=11)
ax.set_title(
    "Reverse Activation Patching: Arabic → English Hidden State Injection\n"
    "Mistral-Small-3.2-24B  ·  MedAraBench  ·  access_gap (En✓ Ar✗)",
    fontsize=12, fontweight="bold")
ax.legend(fontsize=10, frameon=False, loc="upper right")
ax.set_ylim(y_ann * 1.6, max(all_means) * 1.3)
ax.grid(axis="y", alpha=0.2)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
for ext in [".pdf", ".png"]:
    out_fig = os.path.join(args.out_dir, f"fig_patching_mistral_reverse{ext}")
    plt.savefig(out_fig, bbox_inches="tight", dpi=150)
    print(f"Saved → {out_fig}")
plt.close()
print("\nDone.")
