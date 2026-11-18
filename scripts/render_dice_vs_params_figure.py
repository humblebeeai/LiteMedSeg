"""Figure 8 of the paper: Dice vs. number of parameters (all datasets, 11 models).
Dice = mean +/- std over the 3 training seeds from results/final_protocol/campaign_summary.csv; parameter counts are the values of Table 2.
Run from the repository root:  python scripts/render_dice_vs_params_figure.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# model -> (params_millions, color)
MODELS = {
    "LiteMedSeg (ours)":     (0.643,  "crimson"),
    "U-Net":                 (7.76,   "green"),
    "UNet++":                (9.16,   "royalblue"),
    "ResUNet":                (16.16,  "darkorange"),
    "ResUNet++":              (27.13,  "purple"),
    "Attention U-Net":       (7.85,   "orchid"),
    "DeepLabV3+":            (26.08,  "deepskyblue"),
    "SCA-InceptionUNeXt":    (10.32,  "darkred"),
    "UNeXt":                 (1.469,  "yellowgreen"),
    "Fast-SCNN":             (1.135,  "teal"),
    "MobileViT-Seg":         (1.047,  "saddlebrown"),
}

# Dice: mean and std over 3 training seeds from the final leakage-safe protocol (fixed split 0).
import csv, collections, statistics as st
_KEY = {"LiteMedSeg (ours)": "LiteMedSeg", "U-Net": "UNet", "UNet++": "UNet++", "ResUNet": "ResUNet",
        "ResUNet++": "ResUNet++", "Attention U-Net": "AttentionUNet", "DeepLabV3+": "DeepLabV3+",
        "SCA-InceptionUNeXt": "SCA-InceptionUNet", "UNeXt": "UNeXt", "Fast-SCNN": "Fast-SCNN",
        "MobileViT-Seg": "MobileViT-Seg"}
_raw = collections.defaultdict(list)
for r in csv.DictReader(open("results/final_protocol/campaign_summary.csv")):
    if r["status"] == "ok":
        _raw[(r["model"], r["dataset"])].append(float(r["dice"]))
DICE = {m: {d: (st.mean(_raw[(k, d)]), st.stdev(_raw[(k, d)])) for d in ("CVC-ClinicDB", "Kvasir-SEG", "PolypGen")}
        for m, k in _KEY.items()}
for m in DICE:
    assert all(len(_raw[(_KEY[m], d)]) == 3 for d in DICE[m]), m

DATASET_MARKERS = {"CVC-ClinicDB": "o", "Kvasir-SEG": "s", "PolypGen": "^"}

fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)

for model, (params, color) in MODELS.items():
    for dataset, marker in DATASET_MARKERS.items():
        dice, sd = DICE[model][dataset]
        ax.errorbar(params, dice, yerr=sd, fmt="none", ecolor=color, elinewidth=1.0, capsize=2.5, zorder=2)
        ax.scatter(params, dice, marker=marker, color=color, s=110,
                   edgecolors="black", linewidths=0.6, zorder=3)

ax.set_xlabel("Parameters (Millions)", fontsize=12)
ax.set_ylabel("Dice coefficient", fontsize=12)
ax.grid(True, alpha=0.3)

dataset_handles = [Line2D([0], [0], marker=m, color="black", linestyle="", markersize=9,
                          markerfacecolor="white", label=d)
                   for d, m in DATASET_MARKERS.items()]
model_handles = [Line2D([0], [0], marker="o", color="white", linestyle="", markersize=9,
                        markerfacecolor=color, markeredgecolor="black", label=model)
                 for model, (_, color) in MODELS.items()]

legend1 = ax.legend(handles=dataset_handles, loc="upper left", bbox_to_anchor=(0.0, -0.10),
                    ncol=1, fontsize=10, frameon=False, title="Dataset", title_fontsize=10)
ax.add_artist(legend1)
ax.legend(handles=model_handles, loc="upper left", bbox_to_anchor=(0.27, -0.10),
          ncol=3, fontsize=10, frameon=False, title="Model", title_fontsize=10)

plt.tight_layout()
plt.savefig("results/dice_vs_params_figure_final.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig("results/dice_vs_params_figure_final.pdf", bbox_inches="tight", facecolor="white")
print("Saved dice_vs_params_figure_final.png / .pdf")
