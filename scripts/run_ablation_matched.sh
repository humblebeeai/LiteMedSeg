#!/bin/bash
# Parameter-matched ablation controls (R1.5): networks widened so their size is close to the full model (643K).
#   Abl-bare_b20 (708K)  ~ all four modules removed, widened
#   Abl-wo_hga_none_b16 (669K) ~ HGA removed, widened
# (wo_lsk_none / wo_mpp_none / wo_eca_none at B_Ch=12 are already within -12% / -10% / 0% of 643K.)
cd "$(dirname "$0")/.."
for DS in CVC-ClinicDB Kvasir-SEG PolypGen; do
  python scripts/run_campaign.py --dataset $DS --models Abl-bare_b20 Abl-wo_hga_none_b16 --seeds 0 1 2 --split_seed 0 --epochs 100 --base_dir results/final_protocol_ablation
done
echo "=== MATCHED DONE $(date) ==="
