#!/bin/bash
# Step 2: 7 original baselines under the final protocol (resumable: finished runs are skipped).
cd "$(dirname "$0")/.."
STEP2="UNet UNet++ ResUNet ResUNet++ AttentionUNet DeepLabV3+ SCA-InceptionUNet"
for DS in CVC-ClinicDB Kvasir-SEG PolypGen; do
  echo "=== STEP2 $DS $(date) ==="
  python scripts/run_campaign.py --dataset $DS --models $STEP2 --seeds 0 1 2 --split_seed 0 --epochs 100 --base_dir results/final_protocol
done
echo "=== STEP2 DONE $(date) ==="
