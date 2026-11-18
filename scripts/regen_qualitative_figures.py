"""Regenerate the qualitative figures (manuscript Figures 9 and 10) from the FINAL-protocol checkpoints (seed 0, fixed split, random state 0).
Nothing is hand-picked: Figure 9 shows, per dataset, the test images at fixed quantiles of LiteMedSeg's per-image Dice;
Figure 10 selects failure categories by explicit, scripted rules over LiteMedSeg's predictions on all three test sets.
Outputs (working dir = LiteMedSeg/): results/qualitative/preds_<dataset>.npz, fig9_qualitative.png, fig10_failures.png, results/qualitative/selection.json"""
import glob, json, sys
from pathlib import Path
import cv2, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
sys.path.insert(0, ".")
from scripts.train import build_model, build_data

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path("results/qualitative"); OUT.mkdir(exist_ok=True)
DATASETS = ["CVC-ClinicDB", "Kvasir-SEG", "PolypGen"]
MODELS = ["LiteMedSeg", "UNet", "UNet++", "ResUNet", "ResUNet++", "AttentionUNet", "DeepLabV3+", "SCA-InceptionUNet", "UNeXt", "Fast-SCNN", "MobileViT-Seg"]
TITLE = {"LiteMedSeg": "LiteMedSeg\n(ours)", "UNet": "U-Net", "UNet++": "UNet++", "ResUNet": "ResUNet", "ResUNet++": "ResUNet++", "AttentionUNet": "Attention\nU-Net",
         "DeepLabV3+": "DeepLabV3+", "SCA-InceptionUNet": "SCA-Incep.\nUNeXt", "UNeXt": "UNeXt", "Fast-SCNN": "Fast-SCNN", "MobileViT-Seg": "MobileViT\n-Seg"}


def dice(p, g):
    return float((2 * (p * g).sum() + 1e-7) / (p.sum() + g.sum() + 1e-7))


def predict_all(ds_name):
    cache = OUT / f"preds_{ds_name}.npz"
    ds = build_data(ds_name, 8, 0, "phash")
    _, _, tl = ds.dataloaders()
    names = [ds.image_paths[i].name for i in ds.test_idx]
    if cache.exists():
        z = np.load(cache, allow_pickle=True); return {k: z[k] for k in z.files}, names
    X, Y = [], []
    for x, y in tl: X.append(x); Y.append(y)
    X = torch.cat(X); Y = torch.cat(Y)
    res = {"image": (X.numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8), "gt": (Y.numpy()[:, 0] > 0.5).astype(np.uint8)}
    for m in MODELS:
        f = sorted(glob.glob(f"results/final_protocol/models/best_{m.replace('-', '_')}_seed0_model_{ds_name.replace('-', '_')}_*.pth"))[-1]
        net = build_model(m, base_channels=12).to(dev)
        net.load_state_dict({k: v for k, v in torch.load(f, map_location="cpu").items() if "total_ops" not in k and "total_params" not in k}); net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 8):
                o = net(X[i:i + 8].to(dev)); o = o[0] if isinstance(o, (tuple, list)) else o
                out.append((torch.sigmoid(o) > 0.5).cpu().numpy()[:, 0].astype(np.uint8))
        res[m] = np.concatenate(out)
        print(ds_name, m, "mean dice", np.mean([dice(res[m][i], res["gt"][i]) for i in range(len(X))]).round(4), flush=True)
    np.savez_compressed(cache, **res)
    return res, names


def spec_frac(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV); return float(((hsv[..., 2] > 235) & (hsv[..., 1] < 40)).mean())


def overlay(img, gt, pred):
    o = img.copy()
    c1, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); c2, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(o, c1, -1, (0, 200, 0), 2); cv2.drawContours(o, c2, -1, (230, 30, 30), 2); return o


R = {}; NAMES = {}
for d in DATASETS: R[d], NAMES[d] = predict_all(d)
sel = {"fig9": [], "fig10": []}


def style(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_linewidth(0.5); s.set_color("#999999")


def gray(m): return np.stack([m * 255] * 3, -1).astype(np.uint8)


# ------------------------------ Figure 9: fixed quantiles of LiteMedSeg Dice
QS = [0.10, 0.30, 0.50, 0.70, 0.90]
rows = []
for d in ["Kvasir-SEG", "CVC-ClinicDB"]:
    ld = np.array([dice(R[d]["LiteMedSeg"][i], R[d]["gt"][i]) for i in range(len(R[d]["gt"]))])
    order = np.argsort(ld, kind="stable")
    for q in QS:
        i = int(order[int(round(q * (len(order) - 1)))]); rows.append((d, i)); sel["fig9"].append(dict(dataset=d, file=NAMES[d][i], index=i, quantile=q, lite_dice=float(ld[i])))
cols = ["Image", "Mask"] + MODELS
fig, axes = plt.subplots(len(rows), len(cols) + 1, figsize=(7.0, 0.5 * len(rows) + 0.35), dpi=450, gridspec_kw={"width_ratios": [0.3] + [1] * len(cols)})
for r, (d, i) in enumerate(rows):
    a = axes[r, 0]; a.axis("off")
    if r in (0, len(QS)): a.text(0.5, 0.5 if r == 0 else 0.5, d, rotation=90, ha="center", va="center", fontsize=6.5, fontweight="bold", transform=a.transAxes)
    panels = [R[d]["image"][i], gray(R[d]["gt"][i])] + [gray(R[d][m][i]) for m in MODELS]
    for c, p in enumerate(panels):
        ax = axes[r, c + 1]; ax.imshow(p); style(ax)
        if r == 0: ax.set_title(("Image", "Mask")[c] if c < 2 else TITLE[MODELS[c - 2]], fontsize=4.6, fontweight="bold", pad=3.5)
        if c >= 2: ax.text(3, 14, f"{dice(R[d][MODELS[c - 2]][i], R[d]['gt'][i]):.2f}", color="yellow", fontsize=4.2, fontweight="bold", bbox=dict(facecolor="black", alpha=0.55, pad=0.4, lw=0))
plt.tight_layout(pad=0.15, h_pad=0.15, w_pad=0.15); plt.savefig("results/qualitative/fig9_qualitative.png", dpi=450, bbox_inches="tight", facecolor="white"); plt.close()

# ------------------------------ Figure 10: scripted failure categories (LiteMedSeg, seed 0)
rec = []
for d in DATASETS:
    n = len(R[d]["gt"])
    for i in range(n):
        g, p, im = R[d]["gt"][i], R[d]["LiteMedSeg"][i], R[d]["image"][i]
        rec.append(dict(d=d, i=i, dice=dice(p, g), gt=int(g.sum()), pred=int(p.sum()), spec=spec_frac(im), contrast=float(im.std()),
                        fg_bg=float(abs(im[g > 0].mean() - im[g == 0].mean())) if 0 < g.sum() < g.size else 0.0))
cats = []
def pick(label, cond, key, rev=False):
    c = [r for r in rec if cond(r)]
    if not c: print("NO CANDIDATE for", label); return
    r = sorted(c, key=key, reverse=rev)[0]; cats.append((label, r))
pick("Small polyp\n(missed)", lambda r: r["gt"] > 0 and r["pred"] == 0, lambda r: r["gt"])
pick("Atypical background\n(missed)", lambda r: r["gt"] > 0 and r["pred"] == 0 and all(r is not x[1] for x in cats), lambda r: r["fg_bg"])
pick("Specular highlight\n(poor overlap)", lambda r: r["gt"] > 0 and r["dice"] < 0.3 and all(r is not x[1] for x in cats), lambda r: r["spec"], True)
pick("Under-segmentation\n(partial overlap)", lambda r: r["gt"] > 0 and 0.3 <= r["dice"] <= 0.6 and r["pred"] > 0 and all(r is not x[1] for x in cats), lambda r: r["gt"], True)
pick("False positive\n(empty mask)", lambda r: r["gt"] == 0 and r["pred"] > 0, lambda r: r["pred"], True)
pick("Large polyp,\nlargely missed", lambda r: r["gt"] > 0 and r["dice"] < 0.5 and all(r is not x[1] for x in cats), lambda r: r["gt"], True)
fig, axes = plt.subplots(len(cats), 5, figsize=(4.3 * 4 + 8.0, len(cats) * 5.6), dpi=100, gridspec_kw={"width_ratios": [1.86, 1, 1, 1, 1]})
ct = ["Input\nImage", "Ground\nTruth", "LiteMedSeg\nPrediction", "Overlay\n(GT = green,\nPred = red)"]
for r, (label, x) in enumerate(cats):
    d, i = x["d"], x["i"]; im, g, p = R[d]["image"][i], R[d]["gt"][i], R[d]["LiteMedSeg"][i]
    sel["fig10"].append(dict(category=label.replace("\n", " "), dataset=d, file=NAMES[d][i], index=i, dice=x["dice"], gt_area=x["gt"], pred_area=x["pred"], specular=x["spec"], fg_bg_contrast=x["fg_bg"]))
    axes[r, 0].axis("off"); axes[r, 0].text(0.98, 0.5, f"{label}\n{d}\nDice = {x['dice']:.2f}", fontsize=34.5, va="center", ha="right", transform=axes[r, 0].transAxes, linespacing=1.6)
    for c, pn in enumerate([im, gray(g), gray(p), overlay(im, g, p)]):
        ax = axes[r, c + 1]; ax.imshow(pn); style(ax)
        if r == 0: ax.set_title(ct[c], fontsize=36, fontweight="bold", pad=16, linespacing=1.3)
fig.legend(handles=[mpatches.Patch(color=(0, 200 / 255, 0), label="Ground truth boundary"), mpatches.Patch(color=(230 / 255, 30 / 255, 30 / 255), label="Predicted boundary")], loc="lower center", ncol=2, fontsize=33, frameon=False, bbox_to_anchor=(0.55, -0.012))
plt.tight_layout(rect=[0, 0.02, 1, 1]); plt.savefig("results/qualitative/fig10_failures.png", dpi=100, bbox_inches="tight", facecolor="white"); plt.close()
json.dump(sel, open(OUT / "selection.json", "w"), indent=2)
print(json.dumps(sel["fig10"], indent=1))
