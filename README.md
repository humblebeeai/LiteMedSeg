# LiteMedSeg: A Lightweight Framework for Efficient Polyp Segmentation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24347585.svg)](https://doi.org/10.5281/zenodo.24347585)

Official code, trained checkpoints and result files for the paper

> **LiteMedSeg: A Lightweight Framework for Efficient Polyp Segmentation**
> O. Eraliev, J. Alikhanov, F. Makhmudov, Y. Yoon. *Scientific Reports* (under review; this repository corresponds to the revised version).

LiteMedSeg is a 643K-parameter U-Net-style network for binary polyp segmentation. It combines a lightweight selective kernel (LSK),
efficient channel attention (ECA), mobile pyramid pooling (MPP) and Hadamard-gated attention (HGA).

> **Scope of the claims.** Under the common protocol below, LiteMedSeg reaches accuracy *comparable* to conventional networks with
> 12–42× more parameters and clearly higher accuracy than the three lightweight baselines. It is **not** significantly more accurate than the
> larger baselines (three seeds, low statistical power) and it is **not** faster than the other lightweight models. See [docs/PROTOCOL.md](docs/PROTOCOL.md) for all caveats.

## Main results

Test Dice (macro average over test images, mean ± SD over 3 training seeds; one fixed leakage-safe split per dataset; every model trained from scratch with the same pipeline).
Full table (precision, recall, IoU, FLOPs) is in [`results/final_protocol/table3_main.csv`](results/final_protocol/table3_main.csv); paired tests in [`results/supplementary/S2_paired_tests_main.csv`](results/supplementary/S2_paired_tests_main.csv).

| Model | Params (M) | GFLOPs | CVC-ClinicDB | Kvasir-SEG | PolypGen |
|---|---:|---:|---|---|---|
| U-Net | 7.763 | 13.75 | 0.830 ± 0.005 | 0.781 ± 0.007 | 0.667 ± 0.014 |
| UNet++ | 9.163 | 34.90 | 0.827 ± 0.012 | 0.783 ± 0.018 | 0.690 ± 0.006 |
| ResUNet | 16.159 | 26.78 | 0.802 ± 0.008 | 0.790 ± 0.015 | 0.605 ± 0.024 |
| ResUNet++ | 27.125 | 58.87 | 0.839 ± 0.035 | 0.810 ± 0.005 | 0.663 ± 0.007 |
| Attention U-Net | 7.855 | 14.03 | 0.825 ± 0.010 | 0.774 ± 0.006 | 0.647 ± 0.017 |
| DeepLabV3+ | 26.082 | 8.98 | 0.820 ± 0.007 | 0.814 ± 0.006 | 0.654 ± 0.012 |
| SCA-InceptionUNeXt | 10.319 | 24.90 | 0.787 ± 0.023 | 0.710 ± 0.016 | 0.591 ± 0.019 |
| UNeXt | 1.469 | 2.88 | 0.653 ± 0.048 | 0.689 ± 0.011 | 0.588 ± 0.011 |
| Fast-SCNN | 1.135 | 0.22 | 0.618 ± 0.025 | 0.550 ± 0.033 | 0.443 ± 0.029 |
| MobileViT-Seg | 1.047 | 0.32 | 0.608 ± 0.120 | 0.498 ± 0.033 | 0.339 ± 0.016 |
| **LiteMedSeg (ours)** | **0.643** | **1.32** | **0.851 ± 0.019** | **0.819 ± 0.008** | **0.710 ± 0.046** |

Additional experiments reported in the paper (all in [`results/`](results)): ablation study with module removal, replacement, HGA decomposition and
width-matched controls (`final_protocol_ablation/`); LiteMedSeg trained without auxiliary supervision (`final_protocol_aux/`); a PraNet re-implementation trained
from scratch (`final_protocol_polyp/`); CPU/GPU latency and module-level timing (`final_protocol/benchmark_*.csv`).

## Installation

```bash
git clone https://github.com/humblebeeai/LiteMedSeg.git
cd LiteMedSeg
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.14, PyTorch 2.11 (CUDA 12.8) and the package versions in [`requirements-tested.txt`](requirements-tested.txt); other versions have not been tested.
A GPU is recommended for training; evaluation of the released checkpoints runs on CPU.

## Data

The datasets are public and are **not** redistributed here. Download them and place them under `data/` as described in [docs/DATA.md](docs/DATA.md)
(CVC-ClinicDB, Kvasir-SEG, PolypGen). The exact train/validation/test image lists of every dataset are in
[`results/supplementary/S1_split_lists.csv`](results/supplementary/S1_split_lists.csv).

## Quick start

```bash
python tests/test_smoke.py            # sanity checks (or: python -m pytest -q tests), CPU, < 1 min (parameter count, shapes, HGA gradient regression test)
python scripts/evaluate.py           # re-evaluate the released LiteMedSeg checkpoints on the fixed test splits
```

`scripts/evaluate.py` should print Dice 0.851 ± 0.019 (CVC-ClinicDB, 50 test images), 0.819 ± 0.008 (Kvasir-SEG, 100) and 0.710 ± 0.046 (PolypGen, 153), as in the paper.

```python
import torch
from scripts.train import build_model

net = build_model("LiteMedSeg", base_channels=12)
net.load_state_dict(torch.load("checkpoints/LiteMedSeg_Kvasir-SEG_seed0.pth", map_location="cpu"))
net.eval()
with torch.no_grad():
    mask = torch.sigmoid(net(torch.rand(1, 3, 256, 256))) > 0.5     # input: RGB in [0, 1], 256×256
```

## Reproducing the experiments

All experiments use `scripts/train.py` / `scripts/run_campaign.py` (resumable; finished runs are skipped). Driver scripts for each result set:

| Paper element | Command | Output |
|---|---|---|
| Main comparison (Table 3), 11 models × 3 datasets × 3 seeds | `bash scripts/run_final_protocol.sh` then `bash scripts/run_step2.sh` | `results/final_protocol/` |
| Statistics and Table 3 | `python scripts/analyze_main.py` | `table3_main.csv`, `significance_main.csv` |
| Ablation (Tables 4–5) | `bash scripts/run_ablation.sh`, `bash scripts/run_ablation_matched.sh`, `python scripts/analyze_ablation.py` | `results/final_protocol_ablation/` |
| Auxiliary-supervision control | `bash scripts/run_aux_control.sh` | `results/final_protocol_aux/` |
| PraNet baseline | `bash scripts/run_polyp_baselines.sh` | `results/final_protocol_polyp/` |
| Latency / memory (Table 2) | `python scripts/benchmark_latency.py` | `benchmark_latency.csv`, `benchmark_modules.csv` |
| PolypGen near-duplicate check | `python scripts/polypgen_dedup_eval.py` | `polypgen_dedup_eval.csv` |
| Figures 8–10 | `python scripts/render_dice_vs_params_figure.py`, `python scripts/regen_qualitative_figures.py` | figures |
| Split lists (supplementary S1) | `python scripts/export_split_lists.py <out_dir>` | `split_lists.csv` |

The complete campaign (about 280 training runs; 100 epochs maximum, early stopping) was run on a single laptop GPU (Section 3.8 of the paper) and takes several days.
Training from scratch is not bit-wise deterministic across hardware, so re-running will give numbers close to, but not identical with, the reported ones (the released
checkpoints reproduce them exactly).

Single run example:

```bash
python scripts/train.py --model LiteMedSeg --dataset CVC-ClinicDB --seed 0 --split_seed 0 --epochs 100 --base_dir results/my_run
```

## Repository layout

```
segmentation_models/      model implementations (models/), data loading and leakage-safe splitting (data/), training engine (engine/), losses and metrics (utils/)
  models/lite_med_seg.py          LiteMedSeg (LSK, ECA, MPP, HGA)
  models/lite_med_seg_ablation.py configurable network for the ablation study (each module: lightweight / standard / removed)
scripts/                  training, campaign runners, analysis, benchmarks, figure generation
tests/                    sanity checks
checkpoints/              released LiteMedSeg weights (3 datasets × 3 seeds)
results/                  all result tables used in the paper (per-run CSVs, summaries, supplementary tables S1–S8, figures)
docs/                     DATA.md (dataset setup and split caveats), PROTOCOL.md (evaluation protocol, known limitations), CHANGES.md (differences from the earlier release)
```

## Evaluation protocol in brief

* Fixed split per dataset (random state 0): near-duplicate frames are grouped by an average hash and never cross splits (CVC-ClinicDB 502/60/50, Kvasir-SEG 799/101/100);
  PolypGen uses the official train/test partition with a case-grouped validation subset (1,326/58/153).
* Early stopping (patience 10) and checkpoint selection on **validation loss**; the selected checkpoint is evaluated once on the test split. No test data are used for any decision.
* Metrics are computed per image (threshold 0.5) and macro-averaged. Per-image F1 equals Dice for binary masks.
* Three training seeds (0, 1, 2) on the same split: the reported SD reflects training randomness, not split variability.
* All baselines are trained from scratch (no ImageNet weights) with BCE+Dice on the final output; LiteMedSeg additionally uses auxiliary deep supervision (its effect is quantified in the paper).

Please read [docs/PROTOCOL.md](docs/PROTOCOL.md) for known limitations, including that the official PolypGen partition is not guaranteed to be case-disjoint.

## Citation

If you use this code, please cite the paper (see [`CITATION.cff`](CITATION.cff)). An archived copy of release v1.0.0 with a permanent DOI is available on Zenodo: [10.5281/zenodo.24347585](https://doi.org/10.5281/zenodo.24347585).

## License

Released under the MIT License (see [`LICENSE`](LICENSE)). The datasets are subject to their own licences.
