#!/usr/bin/env bash
# E6 calibration-component ablation (Issue 5, Table 4).
# alpha=0.3, eps=5, bounded_directional, seeds {42,123,456,789,2024}.
# Prereqs (run once, commit the outputs BEFORE this script):
#   experiments/configs/hetero_fit_coeffs.json      <- scripts/fit_hetero_variance.py
#   experiments/configs/E6_frozen_bounded_tau.json  <- scripts/freeze_bounded_tau.py
# Usage: scripts/run_e6.sh <KRUM_K> [seed ...]
set -euo pipefail
K="${1:?usage: run_e6.sh <KRUM_K> [seeds...]}"; shift || true
SEEDS=("${@:-42 123 456 789 2024}"); read -ra SEEDS <<< "${SEEDS[*]}"
FIT=experiments/configs/hetero_fit_coeffs.json
TAU=$(python3 -c "import json;print(json.load(open('experiments/configs/E6_frozen_bounded_tau.json'))['bounded_tau'])")
[ -f "$FIT" ] || { echo "missing $FIT"; exit 1; }
echo "frozen tau=$TAU  k=$K  seeds=${SEEDS[*]}"

for S in "${SEEDS[@]}"; do
  COMMON=(network --alpha 0.3 --epsilon 5 --attack-type bounded_directional
          --bounded-tau "$TAU" --bounded-margin 0.05
          --bounded-direction classifier_head_negate --krum-k "$K" --seed "$S")
  HET=(--hetero-fit-coeffs-json "$FIT")
  run() { tag=$1; shift
    rm -f "checkpoint_network_${tag}_seed${S}"*   # DP runs refuse stale checkpoints
    python3 main.py "${COMMON[@]}" --tag "$tag" "$@"
    grep -q '"use_dp": true' "experiment_config_network_${tag}_seed${S}.json" || { echo "DP not active: $tag"; exit 1; }
    test -s "results_network_${tag}_seed${S}_FINAL_TEST.csv" || { echo "empty FINAL_TEST: $tag"; exit 1; }
  }
  run e6_full       --ablation-mode calibrated_krum_dp_sweep "${HET[@]}"
  run e6_dponly     --ablation-mode calibrated_krum_dp_sweep "${HET[@]}" --no-hetero-calibration
  run e6_heteroonly --ablation-mode calibrated_krum_dp_sweep "${HET[@]}" --no-dp-calibration
  run e6_off        --ablation-mode krum_dp_sweep --aggregator adaptive_krum
done
