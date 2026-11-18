"""Evaluate the released LiteMedSeg checkpoints on the fixed test split of each dataset.

Usage (from the repository root, datasets placed under data/ as described in docs/DATA.md):
    python scripts/evaluate.py                       # all datasets, seeds 0-2
    python scripts/evaluate.py --datasets Kvasir-SEG --seeds 0

Metrics are computed per test image on the binarised prediction (sigmoid > 0.5) and averaged over images (macro average).
The numbers should match results/final_protocol/table3_main.csv (LiteMedSeg rows) up to rounding.
"""
import argparse
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import build_data, build_model  # noqa: E402

EPS = 1e-7


def per_image_metrics(pred, gt):
    """pred, gt: (N, H, W) binary arrays -> dict of per-image arrays."""
    tp = (pred * gt).sum((1, 2)).astype(np.float64)
    fp = (pred * (1 - gt)).sum((1, 2)).astype(np.float64)
    fn = ((1 - pred) * gt).sum((1, 2)).astype(np.float64)
    return {
        "dice": (2 * tp + EPS) / (2 * tp + fp + fn + EPS),
        "iou": (tp + EPS) / (tp + fp + fn + EPS),
        "precision": (tp + EPS) / (tp + fp + EPS),
        "recall": (tp + EPS) / (tp + fn + EPS),
    }


@torch.no_grad()
def evaluate(dataset, seed, ckpt_dir, device, batch_size=8):
    ds = build_data(dataset, batch_size, 0, "phash")          # split random state 0 = the split used in the paper
    _, _, test_loader = ds.dataloaders()
    net = build_model("LiteMedSeg", base_channels=12).to(device)
    state = torch.load(Path(ckpt_dir) / f"LiteMedSeg_{dataset}_seed{seed}.pth", map_location="cpu")
    net.load_state_dict(state)
    net.eval()
    preds, gts = [], []
    for x, y in test_loader:
        out = net(x.to(device))
        out = out[0] if isinstance(out, (tuple, list)) else out
        preds.append((torch.sigmoid(out) > 0.5).cpu().numpy()[:, 0].astype(np.uint8))
        gts.append((y.numpy()[:, 0] > 0.5).astype(np.uint8))
    m = per_image_metrics(np.concatenate(preds), np.concatenate(gts))
    return {k: float(v.mean()) for k, v in m.items()}, len(np.concatenate(gts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["CVC-ClinicDB", "Kvasir-SEG", "PolypGen"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    print(f"{'dataset':14s} {'n_test':>6s}  " + "  ".join(f"{k:>16s}" for k in ("Dice", "IoU", "Precision", "Recall")) + "   (mean ± sd over seeds)")
    for d in a.datasets:
        rows = []
        for s in a.seeds:
            r, n = evaluate(d, s, a.ckpt_dir, a.device)
            rows.append(r)
            print(f"  {d} seed {s}: " + " ".join(f"{k}={v:.4f}" for k, v in r.items()))
        cells = []
        for k in ("dice", "iou", "precision", "recall"):
            v = [r[k] for r in rows]
            cells.append(f"{st.mean(v):.3f} ± {st.stdev(v) if len(v) > 1 else 0:.3f}")
        print(f"{d:14s} {n:6d}  " + "  ".join(f"{c:>16s}" for c in cells))


if __name__ == "__main__":
    main()
