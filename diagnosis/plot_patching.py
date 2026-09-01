"""
plot_patching.py  —  regenerate activation patching figure from saved .npz
"""
import argparse, os
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--npz", required=True)
parser.add_argument("--out", default=".")
args = parser.parse_args()

d = np.load(args.npz, allow_pickle=True)
en_base_prob = d["en_base_prob"]
ar_base_prob = d["ar_base_prob"]

PATCH_CONFIGS = [
    "patch_L56", "patch_L60", "patch_L64", "patch_L68",
    "patch_L72", "patch_L76", "patch_L79", "patch_L72_79", "patch_L64_79",
]
patch_results = {c: d[c] for c in PATCH_CONFIGS}
N = len(ar_base_prob)

PALETTE = {
    'Base': '#87CEEB',
    'CoT':  '#2A9D8F',
    'IR':   '#0D3349',
    'AP':   '#F4C430',
    'FS':   '#F07C00',
    'MP':   '#C0392B',
}

ar_mean = ar_base_prob.mean()
en_mean = en_base_prob.mean()
gap     = en_mean - ar_mean

def recovery(val):
    return 100 * (val - ar_mean) / gap if gap > 0 else 0.0

all_configs = ["ar_base"] + PATCH_CONFIGS + ["en_base"]
all_means   = ([ar_mean] +
               [patch_results[c].mean() for c in PATCH_CONFIGS] +
               [en_mean])
all_sems    = ([ar_base_prob.std() / np.sqrt(N)] +
               [patch_results[c].std() / np.sqrt(N) for c in PATCH_CONFIGS] +
               [en_base_prob.std() / np.sqrt(N)])

n_patches = len(PATCH_CONFIGS)
colors = [PALETTE['IR']] + [PALETTE['AP']] * n_patches + [PALETTE['CoT']]

x_labels = (["Arabic\n(baseline)"] +
             [c.replace("patch_", "patch\n") for c in PATCH_CONFIGS] +
             ["English\n(baseline)"])

plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(range(len(all_configs)), all_means, color=colors,
              yerr=all_sems, capsize=4, alpha=0.88,
              edgecolor="white", linewidth=0.5)

for bar, m in zip(bars, all_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{m:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

for ci, c in enumerate(PATCH_CONFIGS):
    rec = recovery(patch_results[c].mean())
    ax.text(ci + 1, -0.018, f"{rec:.0f}%", ha="center", va="top",
            fontsize=8.5, color="grey")
ax.text(-0.5, -0.018, "Recovery:", ha="left", va="top", fontsize=8.5, color="grey")

ax.axhline(ar_mean, color=PALETTE['IR'], linewidth=1.5, linestyle="--", alpha=0.7,
           label=f"Arabic baseline ({ar_mean:.3f})")
ax.axhline(en_mean, color=PALETTE['CoT'], linewidth=1.5, linestyle="--", alpha=0.7,
           label=f"English baseline ({en_mean:.3f})")

ax.set_xticks(range(len(all_configs)))
ax.set_xticklabels(x_labels, fontsize=9.5)
ax.set_ylabel("Mean P(correct answer letter) at model output")
ax.set_title(
    "Activation Patching: Injecting English Hidden States into Arabic Forward Pass\n"
    "Llama 3.3 70B  ·  MedAraBench  ·  access\_gap quadrant (En✓ Ar✗)",
    fontweight="bold"
)
ax.legend(frameon=False, loc="upper left")
ax.set_ylim(-0.03, max(all_means) * 1.25)
ax.grid(axis="y", alpha=0.2)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out = os.path.join(args.out, "fig_patching_llama.pdf")
plt.savefig(out, bbox_inches="tight")
print(f"Saved → {out}")
plt.close()
