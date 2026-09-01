"""
plot_patching_panel.py
-----------------------
2×2 panel figure from activation patching results for all four models.
Reads patching_results.npz from each model's output directory.

Usage:
  python plot_patching_panel.py \
      --llama_dir    ./activation_patching_llama_out \
      --mistral_dir  ./activation_patching_mistral_out \
      --allam_dir    ./activation_patching_allam_out \
      --medgemma_dir ./activation_patching_medgemma_out \
      --out_dir      ./activation_patching_panel_out
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

parser = argparse.ArgumentParser()
parser.add_argument("--llama_dir",    default="./activation_patching_llama_out")
parser.add_argument("--mistral_dir",  default="./activation_patching_mistral_out")
parser.add_argument("--allam_dir",    default="./activation_patching_allam_out")
parser.add_argument("--medgemma_dir", default="./activation_patching_medgemma_out")
parser.add_argument("--out_dir",      default="./activation_patching_panel_out")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

# ── Palette (consistent with tuned-lens panel) ─────────────────────────────────
PALETTE = {
    'Base': '#87CEEB',   # sky blue  — Arabic baseline
    'CoT':  '#2A9D8F',   # teal      — English baseline
    'IR':   '#0D3349',   # navy
    'AP':   '#F4C430',   # mango
    'FS':   '#F07C00',   # tangerine
    'MP':   '#C0392B',   # crimson
}

AR_COLOR   = PALETTE['Base']
EN_COLOR   = PALETTE['CoT']
ANNO_COLOR = "#555555"

# Smooth gradient for patch bars: navy (early/low) → mango → tangerine → crimson (late/high)
# No cycling — each bar gets a unique shade along the continuum.
_PATCH_CMAP = LinearSegmentedColormap.from_list(
    "patch_grad",
    [PALETTE['IR'], PALETTE['AP'], PALETTE['FS'], PALETTE['MP']],
)

def patch_colors(n):
    """Return n colours spread evenly across the navy→crimson gradient."""
    if n == 1:
        return [PALETTE['MP']]
    return [_PATCH_CMAP(i / (n - 1)) for i in range(n)]

# ── Model registry ─────────────────────────────────────────────────────────────
MODELS = [
    ("Llama-3.3-70B",          args.llama_dir),
    ("Mistral-Small-3.2-24B",  args.mistral_dir),
    ("ALLaM-7B",               args.allam_dir),
    ("MedGemma-27B",           args.medgemma_dir),
]

# ── Load helper ────────────────────────────────────────────────────────────────
def load_npz(out_dir):
    path = os.path.join(out_dir, "patching_results.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    en_base  = d["en_base_prob"]
    ar_base  = d["ar_base_prob"]
    N        = len(en_base)
    # All keys except the fixed ones are patch configs
    patch_keys = [k for k in d.files
                  if k not in ("en_base_prob", "ar_base_prob", "gt_letters")]
    patch_data = {k: d[k] for k in patch_keys}
    return dict(en_base=en_base, ar_base=ar_base, N=N, patch_data=patch_data)

# ── Recovery helper ────────────────────────────────────────────────────────────
def recovery(val, ar_mean, en_mean):
    gap = en_mean - ar_mean
    return 100 * (val - ar_mean) / gap if gap > 0 else 0.0

# ── Draw one subplot ───────────────────────────────────────────────────────────
def draw_panel(ax, data, model_name, show_ylabel, show_xlabel):
    if data is None:
        ax.set_facecolor("#f5f5f5")
        ax.text(0.5, 0.5, "Pending", ha="center", va="center",
                fontsize=15, color="#aaaaaa", transform=ax.transAxes)
        ax.set_title(model_name, fontsize=15, fontweight="bold", pad=8)
        ax.axis("off")
        return

    en_base   = data["en_base"]
    ar_base   = data["ar_base"]
    patch_data = data["patch_data"]
    N          = data["N"]

    ar_mean = ar_base.mean()
    en_mean = en_base.mean()

    patch_keys  = list(patch_data.keys())
    patch_means = [patch_data[k].mean() for k in patch_keys]
    patch_sems  = [patch_data[k].std() / np.sqrt(N) for k in patch_keys]

    all_means = [ar_mean]  + patch_means  + [en_mean]
    all_sems  = [ar_base.std()/np.sqrt(N)] + patch_sems + [en_base.std()/np.sqrt(N)]

    n_patches = len(patch_keys)
    cmap      = patch_colors(n_patches)
    colors    = [AR_COLOR] + list(cmap) + [EN_COLOR]

    x_labels = (["Ar\n(base)"] +
                [k.replace("patch_", "") for k in patch_keys] +
                ["En\n(base)"])

    xs = np.arange(len(all_means))
    bars = ax.bar(xs, all_means, color=colors,
                  yerr=all_sems, capsize=3, alpha=0.88,
                  edgecolor="white", linewidth=0.4)

    # Value labels above bars
    for bar, m in zip(bars, all_means):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(all_means)*0.02,
                f"{m:.2f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    # Recovery % below patch bars (skip ar_base at index 0 and en_base at last)
    y_ann = -max(all_means) * 0.07
    ax.text(xs[0], y_ann, "Rec:", ha="center", va="top",
            fontsize=9, color=ANNO_COLOR)
    for ci, m in enumerate(patch_means):
        ax.text(xs[ci + 1], y_ann,
                f"{recovery(m, ar_mean, en_mean):.0f}%",
                ha="center", va="top", fontsize=9, color=ANNO_COLOR)

    ax.axhline(ar_mean, color=AR_COLOR, linewidth=1.2, linestyle="--", alpha=0.7)
    ax.axhline(en_mean, color=EN_COLOR, linewidth=1.2, linestyle="--", alpha=0.7)

    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, fontsize=11, rotation=0)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylim(y_ann * 1.6, max(all_means) * 1.3)
    ax.set_title(model_name, fontsize=15, fontweight="bold", pad=8)

    if show_ylabel:
        ax.set_ylabel("Mean P(correct letter)", fontsize=14)
    if show_xlabel:
        ax.set_xlabel("Patch config", fontsize=14)

    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)

# ── Build 2×2 panel ────────────────────────────────────────────────────────────
loaded = [(name, load_npz(d)) for name, d in MODELS]

fig = plt.figure(figsize=(22, 13))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.22, wspace=0.28)

positions = [(0,0), (0,1), (1,0), (1,1)]
for idx, ((model_name, data), (row, col)) in enumerate(zip(loaded, positions)):
    ax = fig.add_subplot(gs[row, col])
    show_ylabel  = (col == 0)
    show_xlabel  = (row == 1)
    draw_panel(ax, data, model_name, show_ylabel, show_xlabel)

# ── Shared legend ──────────────────────────────────────────────────────────────
from matplotlib.patches import Patch
# Build a mini gradient swatch strip to represent the continuous shading
_grad_handles = [
    Patch(facecolor=_PATCH_CMAP(v), edgecolor="none", label="" if i > 0 else "Patch layers (early → late)")
    for i, v in enumerate(np.linspace(0, 1, 8))
]
legend_handles = [
    Patch(facecolor=AR_COLOR, label="Arabic baseline"),
    Patch(facecolor=EN_COLOR, label="English baseline"),
] + _grad_handles

fig.legend(handles=legend_handles, loc="lower center",
           ncol=len(legend_handles), fontsize=13, frameon=False,
           handlelength=1.2, handletextpad=0.5, columnspacing=0.8,
           bbox_to_anchor=(0.5, 0.02))

fig.suptitle(
    "Causal Activation Patching: English → Arabic Hidden-State Injection\n",
    fontsize=16, fontweight="bold", y=0.95
)

for ext in [".pdf", ".png"]:
    out_path = os.path.join(args.out_dir, f"fig_patching_panel{ext}")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved → {out_path}")

plt.close()
print("Done.")