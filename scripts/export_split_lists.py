"""Export explicit train/val/test sample lists (Reviewer 1.4 / 2.4) for the fixed split (random state 0).
Output: <out>/split_lists.csv with columns dataset, split, file, mask_empty, [centre, case_id, is_background for PolypGen]."""
import csv, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, ".")
from scripts.train import build_data
from segmentation_models.data.dataset import parse_polypgen_ids
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rows = []
for name in ("CVC-ClinicDB", "Kvasir-SEG", "PolypGen"):
    ds = build_data(name, 8, 0, "phash")
    for split, idx in (("train", ds.train_idx), ("val", ds.val_idx), ("test", ds.test_idx)):
        for i in idx:
            ip, mp = ds.image_paths[i], ds.mask_paths[i]
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            empty = int(m is None or m.max() == 0)
            r = dict(dataset=name, split=split, file=ip.name, mask_empty=empty, centre="", case_id="", is_background="")
            if name == "PolypGen":
                q = parse_polypgen_ids(ip.name)
                r.update(centre=q["center"], case_id=q.get("case_id", q.get("case", "")), is_background=int(q["is_background"]))
            rows.append(r)
with open(out / "split_lists.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
import collections
c = collections.Counter((r["dataset"], r["split"]) for r in rows); print(dict(c))
print("empty masks:", collections.Counter((r["dataset"], r["split"]) for r in rows if r["mask_empty"]))
