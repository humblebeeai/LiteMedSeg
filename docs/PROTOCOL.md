# Training and evaluation protocol

This document is the authoritative description of how every number in `results/` was produced. It mirrors Sections 3.7–3.10 of the paper.

## Training (all models)

| Item | Setting |
|---|---|
| Input | RGB, resized to 256×256 (bilinear for images, nearest for masks), scaled to [0, 1], no mean/std standardisation |
| Augmentation | none |
| Optimiser | AdamW, weight decay 1e-4, batch size 8, at most 100 epochs |
| Early stopping | patience 10 epochs on **validation loss**; the checkpoint with the lowest validation loss is kept |
| Initialisation | from scratch for all models (no ImageNet weights, including the DeepLabV3+ and PraNet backbones) |
| Seeds | training seeds 0, 1, 2 on **one fixed split** (`--split_seed 0`) |
| Per-model learning rate / scheduler | `MODEL_HPARAMS` in `scripts/train.py` (LiteMedSeg, UNeXt, Fast-SCNN, MobileViT-Seg: 1e-3, cosine annealing; U-Net, UNet++, Attention U-Net: 1e-4, ReduceLROnPlateau; ResUNet, ResUNet++, DeepLabV3+, SCA-InceptionUNeXt: 1e-3, ReduceLROnPlateau; PraNet: 1e-4, ReduceLROnPlateau). These values were taken from each model's earlier training configuration and **were not re-tuned** under the final protocol, so a baseline might perform better with other settings. |

### Loss and supervision (important for fair comparison)

* **LiteMedSeg** is trained with `LightweightLoss`: 0.6 × BCE+Dice(main output) + 0.4 × mean of BCE+Dice over two auxiliary decoder outputs. At inference only the main output is used.
* **All baselines** are trained with BCE+Dice on their final output only (deep supervision, where a model offers it, is not enabled). **PraNet** is the exception: it uses the same deep-supervision loss as LiteMedSeg.
* Training LiteMedSeg *without* the auxiliary term (`LiteMedSeg-noaux`, `results/final_protocol_aux/`) lowers the mean Dice from 0.793 to 0.774; the effect is not statistically significant with 3 seeds (see the paper, Section 4.1). Part of LiteMedSeg's margin over single-output baselines is therefore due to the training objective.

## Evaluation

* The selected checkpoint is evaluated **once** on the held-out test split.
* Predictions are binarised at 0.5 (sigmoid). All metrics are computed **per image** and averaged over the test images (macro average); a constant of 1e-7 is added to numerators and denominators.
  For binary masks the per-image F1 score equals the per-image Dice coefficient, hence the identical columns in `table3_main.csv`.
* `campaign_summary.csv` additionally stores pixel-pooled ("micro") precision/recall/F1 for completeness; they are not used in the paper's tables.
* Statistics (`scripts/analyze_main.py`, `scripts/analyze_ablation.py`): paired t-tests over the 3 seeds (df = 2) per dataset, 95% confidence intervals from the t-distribution, Holm correction over the comparisons.
  With n = 3 the tests have little power; most differences between LiteMedSeg and the larger baselines (0.005–0.020 Dice) are not significant.

## Efficiency measurements

`scripts/benchmark_latency.py`: batch size 1, fp32, eval mode, 256×256; median over 3 repeats of 100 (CPU) / 300 (GPU) timed passes after warm-up; GPU peak memory from `torch.cuda.max_memory_allocated`.
Measured on a single laptop (Intel Core Ultra 9 275HX, NVIDIA GeForce RTX 5080 Laptop GPU). CPU timings vary by up to ~25% between passes; no ARM/embedded measurements were made.
LiteMedSeg is **not** faster than the other lightweight models; its advantage is in parameter count, FLOPs and (relative to conventional networks) GPU memory.

## Known limitations

1. One fixed split per dataset: SDs reflect training randomness only.
2. Near-duplicate grouping (CVC-ClinicDB, Kvasir-SEG) is a heuristic substitute for patient-level splits; the official PolypGen partition is not guaranteed case-disjoint (see [DATA.md](DATA.md)).
3. Three seeds: low statistical power; ablation differences of individual modules are mostly not significant, and none survives Holm correction.
4. Baselines are trained from scratch, without augmentation and with per-model hyper-parameters that were not re-tuned; polyp-specific methods other than PraNet (Polyp-PVT, CaraNet, HarDNet-MSEG, UACANet) were not re-trained.
5. LiteMedSeg uses auxiliary supervision that the baselines do not receive.
6. Latency was measured on one machine only.
