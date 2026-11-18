#!/bin/bash
# Ablation under the final protocol (fixed split 0, phash/case-grouped, seeds 0-2, B_Ch=12).
# 'full' is NOT re-run here: it is identical to LiteMedSeg in results/final_protocol/ (643,268 params).
# TIER=1 (default): reference, removal, replacement, HGA decomposition  (11 variants)
# TIER=2          : single-module additions to the bare network + all-standard baseline
cd "$(dirname "$0")/.."
TIER=${TIER:-1}
if [ "$TIER" = "1" ]; then
  V="bare wo_lsk_none wo_eca_none wo_mpp_none wo_hga_none wo_lsk_std wo_eca_std wo_mpp_std hga_gate_only hga_attn_only hga_std_gate"
else
  V="bare_lsk bare_eca bare_mpp bare_hga baseline_std"
fi
MODELS=""; for v in $V; do MODELS="$MODELS Abl-${v}_b12"; done
for DS in CVC-ClinicDB Kvasir-SEG PolypGen; do
  echo "=== ABLATION tier$TIER $DS $(date) ==="
  python scripts/run_campaign.py --dataset $DS --models $MODELS --seeds 0 1 2 --split_seed 0 --epochs 100 --base_dir results/final_protocol_ablation
done
echo "=== ABLATION DONE $(date) ==="
