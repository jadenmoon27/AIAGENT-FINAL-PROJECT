"""Generate figures for the paper."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

BLUE   = "#3A7EC6"
ORANGE = "#E07B39"
GREEN  = "#4CAF7D"
GRAY   = "#AAAAAA"
RED    = "#C94040"


# ── Figure 1: Identification accuracy curve ───────────────────────────────────
# Naive WMs vs ego-matched WMs (other_rel), both on coord_env

with open("outputs/results/crossplay_identification_results.json") as f:
    id_data = json.load(f)

K = [1, 2, 3, 5, 8, 10, 15, 20]

naive_full  = [id_data["full_obs"][str(k)]["correct"] / id_data["full_obs"][str(k)]["total"] * 100 for k in K]
naive_rel   = [id_data["other_rel"][str(k)]["correct"] / id_data["other_rel"][str(k)]["total"] * 100 for k in K]
ego_full    = [id_data["cross_pop_full"][str(k)]["correct"] / id_data["cross_pop_full"][str(k)]["total"] * 100 for k in K]
ego_rel     = [id_data["cross_pop_rel"][str(k)]["correct"] / id_data["cross_pop_rel"][str(k)]["total"] * 100 for k in K]

fig, ax = plt.subplots(figsize=(6.5, 4.2))

ax.axhline(50, color=GRAY, linewidth=1.2, linestyle=":", label="Chance (50%)")
ax.plot(K, naive_full,  color=GRAY,   linewidth=1.5, linestyle="--", marker="o", markersize=4, label="Naive WMs (full obs)")
ax.plot(K, naive_rel,   color=GRAY,   linewidth=1.5, linestyle="-",  marker="s", markersize=4, label="Naive WMs (partner rel.)")
ax.plot(K, ego_full,    color=BLUE,   linewidth=2.0, linestyle="--", marker="o", markersize=5, label="Ego-matched (full obs)")
ax.plot(K, ego_rel,     color=ORANGE, linewidth=2.5, linestyle="-",  marker="s", markersize=6, label="Ego-matched (partner rel.)")

# annotate K=10 peak
ax.annotate("91.6% at K=10", xy=(10, 91.6), xytext=(11.5, 86),
            fontsize=9, color=ORANGE,
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))

ax.set_xlabel("Steps observed (K)")
ax.set_ylabel("Partner identification accuracy (%)")
ax.set_xlim(0, 21)
ax.set_ylim(44, 102)
ax.set_xticks(K)
ax.legend(fontsize=9, loc="upper left")
ax.set_title("Partner Identification Accuracy vs. Observation Window")
fig.tight_layout()
fig.savefig(OUT / "fig1_identification_curve.pdf", dpi=300)
fig.savefig(OUT / "fig1_identification_curve.png", dpi=300)
print("Saved fig1_identification_curve")
plt.close()


# ── Figure 2: Adaptive coordination — mean return by condition ────────────────

with open("outputs/results/adaptive_identification_results.json") as f:
    adapt = json.load(f)

conditions  = ["Always-A", "Always-B", "Oracle", "K=1", "K=3", "K=5", "K=10", "K=15"]
means = [
    adapt["always_A"]["mean"],
    adapt["always_B"]["mean"],
    adapt["oracle"]["mean"],
    adapt["adaptive"]["1"]["mean"],
    adapt["adaptive"]["3"]["mean"],
    adapt["adaptive"]["5"]["mean"],
    adapt["adaptive"]["10"]["mean"],
    adapt["adaptive"]["15"]["mean"],
]
stds = [
    adapt["always_A"]["std"],
    adapt["always_B"]["std"],
    adapt["oracle"]["std"],
    adapt["adaptive"]["1"]["std"],
    adapt["adaptive"]["3"]["std"],
    adapt["adaptive"]["5"]["std"],
    adapt["adaptive"]["10"]["std"],
    adapt["adaptive"]["15"]["std"],
]

colors = [RED, GRAY, GREEN] + [BLUE] * 5
n = 500
sems = [s / np.sqrt(n) for s in stds]

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(conditions))
bars = ax.bar(x, means, yerr=sems, color=colors, alpha=0.85, capsize=4, width=0.65, error_kw={"linewidth": 1.5})

# highlight K=10 bar
bars[6].set_edgecolor("black")
bars[6].set_linewidth(2)

# zoom y-axis to where the differences are visible
ax.set_ylim(-40, -27)

# significance bracket at top of the zoomed range
y_top = -28.2
h = 0.4
ax.plot([0, 0, 6, 6], [y_top, y_top+h, y_top+h, y_top], color="black", linewidth=1.2)
ax.text(3, y_top + h + 0.15, "p = 0.0002", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(conditions, rotation=15, ha="right")
ax.set_ylabel("Mean episode return")
ax.set_title("Adaptive Policy Switching vs. Baselines (n=500)")

legend_patches = [
    mpatches.Patch(color=RED,   label="Always-A (no adaptation)"),
    mpatches.Patch(color=GRAY,  label="Always-B"),
    mpatches.Patch(color=GREEN, label="Oracle (upper bound)"),
    mpatches.Patch(color=BLUE,  label="Adaptive (K-step warmup)"),
]
ax.legend(handles=legend_patches, fontsize=8.5, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "fig2_adaptive_returns.pdf", dpi=300)
fig.savefig(OUT / "fig2_adaptive_returns.png", dpi=300)
print("Saved fig2_adaptive_returns")
plt.close()


# ── Figure 3: Benefit vs K curve (gap vs Always-A and vs Oracle) ──────────────

K_adapt = [1, 3, 5, 10, 15]
gap_vs_A = [adapt["adaptive"][str(k)]["gap_vs_always_A"] for k in K_adapt]
gap_vs_oracle = [adapt["adaptive"][str(k)]["gap_vs_oracle"] for k in K_adapt]
p_vs_A = [adapt["adaptive"][str(k)]["p_vs_always_A"] for k in K_adapt]

fig, ax = plt.subplots(figsize=(5.5, 3.8))

ax.axhline(0, color=GRAY, linewidth=1, linestyle=":")
ax.plot(K_adapt, gap_vs_A,     color=BLUE,  linewidth=2.2, marker="o", markersize=6, label="Gap vs Always-A")
ax.plot(K_adapt, gap_vs_oracle, color=ORANGE, linewidth=2.0, marker="s", markersize=5, linestyle="--", label="Gap vs Oracle")

# mark significant points
for k, g, p in zip(K_adapt, gap_vs_A, p_vs_A):
    if p < 0.05:
        ax.plot(k, g, "*", color=BLUE, markersize=10, zorder=5)

ax.annotate("* p < 0.05", xy=(1, 1.77), xytext=(2.5, 0.8),
            fontsize=8.5, color=BLUE,
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1))

ax.set_xlabel("Warmup length K (steps)")
ax.set_ylabel("Return gap")
ax.set_xticks(K_adapt)
ax.set_title("Coordination Benefit by Identification Window")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig3_benefit_curve.pdf", dpi=300)
fig.savefig(OUT / "fig3_benefit_curve.png", dpi=300)
print("Saved fig3_benefit_curve")
plt.close()


# ── Figure 4: Two-environment comparison bar ─────────────────────────────────
# Divergence ratio and identification accuracy side by side

fig, axes = plt.subplots(1, 2, figsize=(9, 4))

# Left: divergence ratio
envs = ["simple_spread\n(no coupling)", "coord_env\n(coupled)"]
ratios = [1.01, 152]
bar_colors = [GRAY, ORANGE]
ax = axes[0]
bars = ax.bar(envs, ratios, color=bar_colors, alpha=0.85, width=0.5)
ax.set_ylabel("Cross-pop / self-divergence ratio")
ax.set_title("Convention Divergence Score")
ax.set_yscale("log")
ax.set_ylim(0.5, 500)
for bar, val in zip(bars, ratios):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
            f"{val}×", ha="center", fontsize=10, fontweight="bold")

# Right: identification accuracy at K=10 and K=20
ax = axes[1]
x = np.array([0, 1])
width = 0.3

# simple_spread: ~50% both K
ss_k10 = 50.0
ss_k20 = 48.3
# coord_env_v2 ego-matched other_rel
ce_k10 = 91.6
ce_k20 = 98.6

bars1 = ax.bar(x - width/2, [ss_k10, ce_k10], width, label="K=10", color=BLUE, alpha=0.85)
bars2 = ax.bar(x + width/2, [ss_k20, ce_k20], width, label="K=20", color=ORANGE, alpha=0.85)
ax.axhline(50, color=GRAY, linewidth=1.2, linestyle=":", label="Chance")
ax.set_xticks(x)
ax.set_xticklabels(envs)
ax.set_ylabel("Partner ID accuracy (%)")
ax.set_title("Identification Accuracy (Ego-Matched)")
ax.set_ylim(30, 110)
ax.legend(fontsize=9)

fig.suptitle("Two-Environment Comparison", fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "fig4_two_env_comparison.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig4_two_env_comparison.png", dpi=300, bbox_inches="tight")
print("Saved fig4_two_env_comparison")
plt.close()

print("\nAll figures saved to outputs/figures/")
