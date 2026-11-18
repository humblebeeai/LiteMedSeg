#!/bin/bash
# Final protocol: leakage-safe split (phash; PolypGen case-grouped), ONE fixed split (split_seed=0),
# training seeds 0,1,2. Step 1 = LiteMedSeg + lightweight baselines, step 2 = 7 original baselines.
cd "$(dirname "$0")/.."
STEP1="LiteMedSeg UNeXt Fast-SCNN MobileViT-Seg"
STEP2="UNet UNet++ ResUNet ResUNet++ AttentionUNet DeepLabV3+ SCA-InceptionUNet"
for DS in CVC-ClinicDB Kvasir-SEG PolypGen; do
  echo "=== STEP1 $DS $(date) ==="
  python scripts/run_campaign.py --dataset $DS --models $STEP1 --seeds 0 1 2 --split_seed 0 --epochs 100 --base_dir results/final_protocol
done
for DS in CVC-ClinicDB Kvasir-SEG PolypGen; do
  echo "=== STEP2 $DS $(date) ==="
  python scripts/run_campaign.py --dataset $DS --models $STEP2 --seeds 0 1 2 --split_seed 0 --epochs 100 --base_dir results/final_protocol
done
echo "=== ALL DONE $(date) ==="
