#!/bin/bash
# R1.7: polyp-specific baseline(s) under the final protocol (fixed split 0, seeds 0-2, no pretrained weights).
# Currently: PraNet. Resumable (finished runs are skipped).
cd "$(dirname "$0")/.."
MODELS=${MODELS:-"PraNet"}
for DS in CVC-ClinicDB Kvasir-SEG PolypGen; do
  python scripts/run_campaign.py --dataset $DS --models $MODELS --seeds 0 1 2 --split_seed 0 --epochs 100 --base_dir results/final_protocol_polyp
done
echo "=== POLYP BASELINES DONE $(date) ==="
