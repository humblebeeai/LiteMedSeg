# Differences from the earlier release of this repository

The first public version of this repository (commits `4392481`, `daf8d8f`, `e035dbb`) accompanied the original submission. During the review we found problems in it; this release replaces it. Results obtained with the old code are **not valid** and must not be compared with the numbers in this release.

| Problem in the earlier release | What changed |
|---|---|
| The per-model notebooks passed the *test* loader as the validation loader, so early stopping and checkpoint selection used test data. | Notebooks removed. `scripts/train.py` / `scripts/run_campaign.py` always use three disjoint splits; the checkpoint is selected on validation loss and evaluated once on the test split. |
| The data module imported by the notebooks (`segmentation_models.data`) was not in the repository. | Re-implemented (`segmentation_models/data/`), with near-duplicate grouping and split reports. |
| Random image-level splits without grouping, and splits regenerated per seed. | Group-aware splits, one fixed split (random state 0) shared by all models and training seeds; image lists in `results/supplementary/S1_split_lists.csv`. |
| In HGA the last convolution of the gating branch was zero-initialised, so its gradient was exactly zero and the gate stayed at 0.5 (HGA behaved like plain multi-head self-attention). | Fixed (small random initialisation, bias 0.1, learnable gate scale/bias); a regression test is in `tests/test_smoke.py`. The branch is now called `descriptor_conv` (formerly `offset_conv`) because its output is never used as a sampling offset; old checkpoints still load. |
| The ablation code (`model_ablation_study/`) contained a stub that returned its input unchanged and instantiated unused parameters; "w/o module" meant "replaced by a heavier module". | Removed; replaced by `segmentation_models/models/lite_med_seg_ablation.py`, where each module can be lightweight, standard or removed, plus width-matched controls. |
| Mixed averaging (pixel-pooled precision/recall with per-image Dice/IoU); F1 overwritten by Dice. | All reported metrics are per-image macro averages. |
| Auxiliary deep supervision of LiteMedSeg was undocumented. | Documented in `docs/PROTOCOL.md`, with a control run without it (`results/final_protocol_aux/`). |
| No polyp-specific baseline, no lightweight baselines, no statistics. | Added UNeXt, Fast-SCNN, MobileViT-Seg, a PraNet re-implementation, three training seeds for every model, paired tests with Holm correction and confidence intervals. |
| Latency numbers could not be reproduced. | `scripts/benchmark_latency.py` with a fixed protocol; machine documented in `docs/PROTOCOL.md`. |

Removed because they are obsolete or invalid: the per-model notebooks, `ablation_study.ipynb`, `model_ablation_study/`, the `wandb/` logs and the old result folders (`model_comparisons*`, `multiseed_results*`).
Weights & Biases logging is still available through `--use_wandb` (set `WANDB_API_KEY` in `.env`, see `.env.example`).
