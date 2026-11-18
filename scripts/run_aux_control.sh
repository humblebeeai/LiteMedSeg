#!/bin/bash
# R1.3 control: LiteMedSeg trained with BCE+Dice on the main output only (no deep supervision),
# i.e. the same supervision the baselines receive. Compare with LiteMedSeg (aux 0.6/0.4) in results/final_protocol/.
cd "$(dirname "$0")/.."
for DS in CVC-ClinicDB Kvasir-SEG PolypGen; do
  python scripts/run_campaign.py --dataset $DS --models LiteMedSeg-noaux --seeds 0 1 2 --split_seed 0 --epochs 100 --base_dir results/final_protocol_aux
done
echo "=== AUX CONTROL DONE $(date) ==="
