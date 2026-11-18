# Data setup

The datasets are publicly available and are not redistributed in this repository. Place them under `data/` (this folder is git-ignored) so that the layout is exactly:

```
data/
├── CVC-ClinicDB/
│   ├── Original/            # 612 frames (.tif)
│   └── Ground Truth/        # 612 masks (.tif)
├── Kvasir-SEG/
│   ├── images/              # 1000 images (.jpg)
│   └── masks/               # 1000 masks (.jpg)
└── PolypGen2021_segmentation/
    ├── train_segmentation/{images,masks}/   # official EndoCV2021 training partition (1,384 images)
    └── test_segmentation/{images,masks}/    # official EndoCV2021 test partition (153 images)
```

| Dataset | Source |
|---|---|
| CVC-ClinicDB | https://polyp.grand-challenge.org/CVCClinicDB/ |
| Kvasir-SEG | https://datasets.simula.no/kvasir-seg/ |
| PolypGen | https://www.synapse.org/Synapse:syn45200214 (free registration required; use the `*_segmentation` folders) |

All datasets are used under their respective research-use terms.

## Splits

All splits are generated deterministically by `segmentation_models/data/dataset.py` with random state 0 (`--split_seed 0`); the exact image lists are in
[`results/supplementary/S1_split_lists.csv`](../results/supplementary/S1_split_lists.csv) (columns: dataset, split, file, mask_empty, and for PolypGen centre, case_id, is_background),
and the split statistics in `results/final_protocol/split_reports/`.

| Dataset | Train / Val / Test | Grouping |
|---|---|---|
| CVC-ClinicDB | 502 / 60 / 50 (431 near-duplicate groups) | average hash (16×16), Hamming distance ≤ 6; every group lies in exactly one split |
| Kvasir-SEG | 799 / 101 / 100 | same |
| PolypGen | 1,326 / 58 / 153 | official train/test partition; validation subset drawn from the training partition grouped by case identifier (21 cases) |

## Known caveats of the data splits

* **No patient identifiers.** CVC-ClinicDB and Kvasir-SEG do not distribute patient/sequence IDs. Near-duplicate grouping removes near-identical consecutive frames but cannot replace a patient-level split.
* **PolypGen partition is not guaranteed case-disjoint.** The case identifier is parsed from the file names. 17 of the 39 test case identifiers (centres C1 and C4) also occur in the official training partition,
  and the file names of centres C2, C3, C5 and C6 do not identify individual cases. A near-duplicate check (same average hash) finds only 5 of the 153 test images with a near-duplicate in train/val; re-evaluating all models on the remaining
  148 test images changes the mean Dice by at most 0.011 (LiteMedSeg 0.706 vs 0.710; [`S3_polypgen_dedup_eval.csv`](../results/supplementary/S3_polypgen_dedup_eval.csv), script `scripts/polypgen_dedup_eval.py`).
  Frames of the same patient that are not near-identical cannot be excluded.
* **Empty masks.** 8 of the 153 PolypGen test images have empty ground-truth masks (listed in S1 with `mask_empty = 1`). They are scored with the same per-image formulas (smoothing 1e-7):
  an empty prediction on an empty mask scores 1, a false-positive pixel scores ≈ 0 for Dice and IoU.
* **Single fixed split.** The reported standard deviations come from training seeds only; variability due to the choice of split was not measured.
