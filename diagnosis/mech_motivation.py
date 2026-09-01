"""
Three-panel mechanistic motivation figure for Mistral-Small-3.2-24B only: 
(a) Tuned Lens   : Mean P(correct answer) by quadrant × language
(b) Activation Patching : Single-layer recovery % (English → Arabic)
(c) KL Divergence Profile : Cross-lingual D_KL per layer (logit lens)
"""

import os
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

parser = argparse.ArgumentParser()
parser.add_argument("--tl_npz",    required=True,
                    help="tuned_lens_results.npz from tuned_lens_mistral.py")
parser.add_argument("--patch_npz", required=True,
                    help="patching_results.npz from activation_patching_mistral.py")
parser.add_argument("--kl_json",   required=True,
                    help="kl_profile.json from probe_kl_profile.py")
parser.add_argument("--out_dir",   default="./mech_motivation_panel")
parser.add_argument("--L_patch",   type=int, default=24,
                    help="Causal boundary layer (τ_patch=0.5 threshold)")
parser.add_argument("--L_kl",      type=int, default=34,
                    help="KL divergence onset layer (τ=μ+σ threshold)")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

L_PATCH = args.L_patch
L_KL    = args.L_kl

FS_AXIS  = 10      # axis label
FS_TICK  = 9      # tick label
FS_LEG   = 10      # legend
FS_ANNOT = 7.5    # in-panel annotations
LW       = 1.5    # primary line width
LW_REF   = 1.1    # reference / threshold lines
MS       = 3.5    # marker size

QUAD_ORDER  = ["both_correct", "access_gap", "arabic_only", "both_wrong"]
QUAD_COLORS = {
    "both_correct": "#2A9D8F",   # teal
    "access_gap":   "#C0392B",   # crimson
    "arabic_only":  "#F4C430",   # mango
    "both_wrong":   "#0D3349",   # navy
}
QUAD_LABELS = {
    "both_correct": "En✓ Ar✓",   # En✓ Ar✓
    "access_gap":   "En✓ Ar×",   # En✓ Ar×
    "arabic_only":  "En× Ar✓",   # En× Ar✓
    "both_wrong":   "En× Ar×",   # En× Ar×
}

PATCH_LINE  = "#C0392B"   # crimson — recovery line
EN_COLOR    = "#2A9D8F"   # teal    — English reference
AR_COLOR    = "#0D3349"   # navy    — Arabic reference
KL_COLOR    = "#1f77b4"   # blue    — KL divergence line
TAU_COLOR   = "#C0392B"   # crimson — τ threshold
LP_COLOR    = "#2E7D32"   # green   — L_patch vertical
LK_COLOR    = "#D4831A"   # orange  — L_kl vertical
WIN_COLOR   = "#FBE9D0"   # pale orange — LoRA window fill

def draw_tuned_lens(ax, npz_path):
    d         = np.load(npz_path, allow_pickle=True)
    en_probs  = d["en_probs"]               
    ar_probs  = d["ar_probs"]               
    quadrants = d["quadrants"].astype(str)  
    layers    = d["layers"]                 

    for q in QUAD_ORDER:
        mask = quadrants == q
        if not mask.any():
            continue
        ev = en_probs[mask].mean(0)
        av = ar_probs[mask].mean(0)
        ee = en_probs[mask].std(0) / np.sqrt(mask.sum())
        ae = ar_probs[mask].std(0) / np.sqrt(mask.sum())
        c  = QUAD_COLORS[q]

        ax.plot(layers, ev, color=c, lw=LW, ls="-",
                marker="o", ms=MS, zorder=3)
        ax.fill_between(layers, ev - ee, ev + ee,
                        color=c, alpha=0.12, zorder=2)
        ax.plot(layers, av, color=c, lw=LW, ls="--",
                marker="^", ms=MS, zorder=3)
        ax.fill_between(layers, av - ae, av + ae,
                        color=c, alpha=0.07, zorder=2)

    ax.axvline(L_PATCH, color=LP_COLOR, lw=LW_REF, ls=":", zorder=5)
    ax.axvline(L_KL,    color=LK_COLOR, lw=LW_REF, ls=":", zorder=5)

    ax.set_xlabel("Transformer layer", fontsize=FS_AXIS)
    ax.set_ylabel("Mean P(correct answer)", fontsize=FS_AXIS)
    # Zoom to where the signal lives; flat early layers visible from L16 onward
    ax.set_xlim(15.5, 41)
    ax.set_ylim(bottom=0)
    ax.set_xticks([16, 20, 24, 28, 32, 36, 40])
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(axis="y", alpha=0.20, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    quad_h = [
        mlines.Line2D([], [], color=QUAD_COLORS[q], lw=1.4,
                      label=QUAD_LABELS[q])
        for q in QUAD_ORDER
    ]
    style_h = [
        mlines.Line2D([], [], color="grey", lw=1.4, ls="-",
                      marker="o", ms=3, label="English"),
        mlines.Line2D([], [], color="grey", lw=1.4, ls="--",
                      marker="^", ms=3, label="Arabic"),
    ]
    return quad_h + style_h

def draw_patching(ax, npz_path):
    d       = np.load(npz_path, allow_pickle=True)
    en_base = d["en_base_prob"]   # (N,)
    ar_base = d["ar_base_prob"]   # (N,)
    N       = len(en_base)

    ar_mean = ar_base.mean()
    en_mean = en_base.mean()
    gap     = en_mean - ar_mean

    def rec(val):
        return 100.0 * (val - ar_mean) / gap if gap > 0 else 0.0

    def sem_rec(arr):
        return 100.0 * arr.std() / (np.sqrt(N) * gap) if gap > 0 else 0.0

    single = []
    for k in d.files:
        m = re.match(r"^patch_L(\d+)$", k)
        if m:
            single.append((int(m.group(1)), k))
    single.sort(key=lambda x: x[0])

    layers_p = np.array([l  for l, _ in single])
    recs     = np.array([rec(d[k].mean()) for _, k in single])
    sems     = np.array([sem_rec(d[k])    for _, k in single])

    Y_MIN, Y_MAX = -30, 130
    band_lo = np.clip(recs - sems, Y_MIN, Y_MAX)
    band_hi = np.clip(recs + sems, Y_MIN, Y_MAX)

    ax.plot(layers_p, recs, color=PATCH_LINE, lw=LW,
            marker="o", ms=MS, zorder=3, label="Single-layer recovery %")
    ax.fill_between(layers_p, band_lo, band_hi,
                    color=PATCH_LINE, alpha=0.10, zorder=2)
    ax.axhline(100, color=EN_COLOR, lw=LW_REF, ls="--", alpha=0.85,
               label="English baseline (100%)")
    ax.axhline(0,   color=AR_COLOR, lw=LW_REF, ls="--", alpha=0.85,
               label="Arabic baseline (0%)")
    ax.axvline(L_PATCH, color=LP_COLOR, lw=LW_REF, ls=":", zorder=5)

    # Annotate if peak exceeds y-axis ceiling
    peak = recs.max()
    if peak > Y_MAX:
        pidx = int(recs.argmax())
        ax.annotate(
            f"peak\n{peak:.0f}%",
            xy=(layers_p[pidx], Y_MAX),
            xytext=(max(layers_p[pidx] - 6, layers_p[0]), Y_MAX - 22),
            fontsize=FS_ANNOT, color=PATCH_LINE, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=PATCH_LINE, lw=0.8),
        )

    ax.text(L_PATCH + 0.7, Y_MAX - 6,
            f"$L_{{\\mathrm{{patch}}}}\\!=\\!{L_PATCH}$",
            fontsize=FS_ANNOT, color=LP_COLOR, va="top")

    ax.set_xlabel("Patch layer", fontsize=FS_AXIS)
    ax.set_ylabel("Recovery (% of En–Ar gap)", fontsize=FS_AXIS)
    # Zoom to where recovery diverges from baseline
    ax.set_xlim(15.5, 42)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xticks([16, 20, 24, 28, 32, 36, 40])
    ax.tick_params(labelsize=FS_TICK, axis="x")
    ax.tick_params(labelsize=FS_TICK, axis="y")
    ax.grid(axis="y", alpha=0.20, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    return [
        mlines.Line2D([0], [0], color=PATCH_LINE, lw=LW,
                      marker="o", ms=MS, label="Single-layer recovery %"),
        mlines.Line2D([0], [0], color=EN_COLOR, lw=LW_REF, ls="--",
                      label="English baseline (100%)"),
        mlines.Line2D([0], [0], color=AR_COLOR, lw=LW_REF, ls="--",
                      label="Arabic baseline (0%)"),
    ]

def draw_kl_profile(ax, json_path):
    with open(json_path) as f:
        data = json.load(f)

    kl_mean = np.array(data["kl_profile"], dtype=np.float64)
    layers  = np.arange(1, len(kl_mean) + 1)
    kl_se   = np.zeros_like(kl_mean)   # per-example SE not saved by probe_kl_profile.py

    w   = data["window"]
    tau = float(w["tau"])

    ax.axvspan(L_PATCH, L_KL, color=WIN_COLOR, alpha=0.85, zorder=0)
    ax.plot(layers, kl_mean, color=KL_COLOR, lw=LW, zorder=3,
            label="KL divergence")
    ax.fill_between(layers, kl_mean - kl_se, kl_mean + kl_se,
                    color=KL_COLOR, alpha=0.15, zorder=2)

    ax.axhline(tau, color=TAU_COLOR, lw=LW_REF, ls="--", zorder=4,
               label=f"$\\tau = \\mu + \\sigma = {tau:.3f}$")
    ax.axvline(L_PATCH, color=LP_COLOR, lw=LW_REF, ls=":", zorder=5,
               label=f"$L_{{\\mathrm{{patch}}}}={L_PATCH}$ (causal boundary)")
    ax.axvline(L_KL,    color=LK_COLOR, lw=LW_REF, ls=":", zorder=5,
               label=f"$L_{{\\mathrm{{KL}}}}={L_KL}$ (divergence onset)")

    win_handle = mpatches.Patch(facecolor=WIN_COLOR, edgecolor="none",
                                label=f"LoRA window $L_{{{L_PATCH}}}$–$L_{{{L_KL}}}$")

    ax.set_xlabel("Transformer layer", fontsize=FS_AXIS)
    ax.set_ylabel("KL divergence (nats)", fontsize=FS_AXIS)
    # Zoom to where divergence rises; aligns x-axis with panels (a) and (b)
    ax.set_xlim(15.5, float(layers.max()) + 0.5)
    ax.set_ylim(bottom=0)
    ax.set_xticks([16, 20, 24, 28, 32, 36, 40])
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(axis="y", alpha=0.20, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    kl_handles, _ = ax.get_legend_handles_labels()
    return [win_handle] + kl_handles

fig, axes = plt.subplots(
    1, 3,
    figsize=(12.0, 3.8),
    gridspec_kw=dict(wspace=0.22, left=0.06, right=0.99,
                     top=0.93, bottom=0.18),
)

h_tl    = draw_tuned_lens (axes[0], args.tl_npz)
h_patch = draw_patching   (axes[1], args.patch_npz)
h_kl    = draw_kl_profile (axes[2], args.kl_json)

# Sub-panel labels — small, normal weight, top-left of each axes
for ax, lbl in zip(axes, ["(a)", "(b)", "(c)"]):
    ax.text(-0.02, 1.01, lbl,
            transform=ax.transAxes,
            fontsize=7, fontweight="normal",
            va="bottom", ha="left")

fig.legend(
    handles=h_tl + h_patch + h_kl,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.07),
    bbox_transform=fig.transFigure,
    ncol=7,
    fontsize=FS_LEG,
    frameon=False,
    handlelength=1.6,
    labelspacing=0.20,
    borderpad=0.3,
    columnspacing=1.0,
)

for ext in [".pdf", ".png"]:
    out = os.path.join(args.out_dir, f"fig_mech_motivation{ext}")
    plt.savefig(out, bbox_inches="tight", dpi=200)
    print(f"Saved → {out}")

plt.close()
print("Done.")
