"""
plot_logit_lens.py  —  regenerate logit lens figure from saved .npz
"""
import argparse, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

parser = argparse.ArgumentParser()
parser.add_argument("--npz", required=True)
parser.add_argument("--out", default=".")
args = parser.parse_args()

d = np.load(args.npz, allow_pickle=True)
en_probs  = d["en_probs"]
ar_probs  = d["ar_probs"]
en_ranks  = d["en_ranks"]
ar_ranks  = d["ar_ranks"]
quadrants = d["quadrants"]
PROBE_LAYERS = d["layers"].tolist()

PALETTE = {
    'Base': '#87CEEB',
    'CoT':  '#2A9D8F',
    'IR':   '#0D3349',
    'AP':   '#F4C430',
    'FS':   '#F07C00',
    'MP':   '#C0392B',
}
QUAD_ORDER  = ["both_correct", "access_gap", "arabic_only", "both_wrong"]
QUAD_COLORS = {
    "both_correct": PALETTE['CoT'],
    "access_gap":   PALETTE['MP'],
    "arabic_only":  PALETTE['FS'],
    "both_wrong":   PALETTE['IR'],
}
QUAD_LABELS = {
    "both_correct": "En✓ Ar✓  (shared knowledge)",
    "access_gap":   "En✓ Ar✗  (access gap)",
    "arabic_only":  "En✗ Ar✓  (Arabic-only)",
    "both_wrong":   "En✗ Ar✗  (both wrong)",
}

plt.rcParams.update({"font.family": "serif"})
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

    ax.set_xlabel("Layer depth")
    ax.set_ylabel(ylabel)
    ax.set_title(title_suffix, fontweight="bold")
    ax.set_xticks(PROBE_LAYERS)
    ax.set_xticklabels(
        ["L0\n(emb.)" if l == 0 else ("L79\n(final)" if l == 80 else f"L{l}")
         for l in PROBE_LAYERS]
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
           loc="lower center", ncol=3,
           frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.suptitle(
    "Logit Lens: Correct Answer Emergence Across Layers\n"
    "Llama 3.3 70B  ·  MedAraBench",
    fontweight="bold"
)
plt.tight_layout()
out = os.path.join(args.out, "fig_logit_lens_llama.pdf")
plt.savefig(out, bbox_inches="tight")
print(f"Saved → {out}")
plt.close()
