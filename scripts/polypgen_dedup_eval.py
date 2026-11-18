"""Re-evaluate the saved PolypGen checkpoints (results/final_protocol) on (a) the full official test split and (b) the subset of test images
that have NO near-duplicate (average-hash, 16x16, Hamming <= 6) among the train+val images. Output: results/final_protocol/polypgen_dedup_eval.csv"""
import csv, glob, sys, statistics as st
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, ".")
from scripts.train import build_model, build_data
from segmentation_models.data.grouping import average_hash, hamming_distance
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds = build_data("PolypGen", 8, 0, "phash")
_, _, test_loader = ds.dataloaders()
names = [ds.image_paths[i].name for i in ds.test_idx]
H = {i: average_hash(str(ds.image_paths[i]), hash_size=16) for i in list(ds.train_idx) + list(ds.val_idx) + list(ds.test_idx)}
trv = list(ds.train_idx) + list(ds.val_idx)
mind = np.array([min(hamming_distance(H[t], H[j]) for j in trv) for t in ds.test_idx])
clean = mind > 6
print("test", len(names), "clean (no near-dup in train/val):", int(clean.sum()))
MODELS = ["LiteMedSeg", "UNet", "UNet++", "ResUNet", "ResUNet++", "AttentionUNet", "DeepLabV3+", "SCA-InceptionUNet", "UNeXt", "Fast-SCNN", "MobileViT-Seg"]
X, Y = [], []
for x, y in test_loader: X.append(x); Y.append(y)
X = torch.cat(X); Y = torch.cat(Y)
out = []
for m in MODELS:
    per = {"all": [], "clean": []}
    for s in (0, 1, 2):
        fs = sorted(glob.glob(f"results/final_protocol/models/best_{m.replace('-', '_')}_seed{s}_model_PolypGen_*.pth"))
        net = build_model(m, base_channels=12).to(dev)
        sd = {k: v for k, v in torch.load(fs[-1], map_location="cpu").items() if "total_ops" not in k and "total_params" not in k}
        net.load_state_dict(sd); net.eval(); d = []
        with torch.no_grad():
            for i in range(0, len(X), 8):
                o = net(X[i:i+8].to(dev)); o = o[0] if isinstance(o, (tuple, list)) else o
                p = (torch.sigmoid(o) > 0.5).float().cpu(); g = (Y[i:i+8] > 0.5).float()
                inter = (p * g).sum((1, 2, 3)); den = p.sum((1, 2, 3)) + g.sum((1, 2, 3))
                d += list(((2 * inter + 1e-6) / (den + 1e-6)).numpy())   # smooth 1e-6; empty/empty -> 1
        d = np.array(d); per["all"].append(d.mean()); per["clean"].append(d[clean].mean())
    out.append([m] + [f"{st.mean(per[k]):.4f}" for k in ("all", "clean")] + [f"{st.stdev(per[k]):.4f}" for k in ("all", "clean")])
    print(out[-1], flush=True)
with open("results/final_protocol/polypgen_dedup_eval.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model", "dice_all153", "dice_clean%d" % clean.sum(), "std_all", "std_clean"]); w.writerows(out)
