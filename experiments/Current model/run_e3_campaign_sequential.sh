#!/bin/bash
# E3 campaign -- sequential, ordered for an early go/no-go checkpoint.
# Run from your project root (where main.py lives).
#
# Frozen attack (stable across 25 rounds, confirmed this conversation):
#   --byzantine 1,2 --attack-type minmax --minmax-dev-type std
#   --minmax-search-iters 15   (gamma_init left at auto/None)
# Domain: network. ablation-mode: krum_baseline (USE_DP=False, matches
# E3's "no DP" condition) -- --aggregator switches Adaptive vs Calibrated.
# Calibration: experiments/configs/hetero_fit_coeffs_a0.3_a0.7_combined.json
# (2-alpha interim fit -- valid, but re-fit across all 5 alphas once this
# campaign's own data exists; don't leave this as the final calibration).

set -e
ALPHAS=(10 1 0.7 0.3 0.1)
BACKFILL_SEEDS=(123 456 789 2024)
HETERO_JSON="experiments/configs/hetero_fit_coeffs_a0.3_a0.7_combined.json"

run_adaptive () {
  local a="$1" s="$2"
  local tag="e3_adaptive_a${a}_seed${s}"
  echo "=== $tag ==="
  python main.py network --rounds 25 --alpha "$a" --byzantine 1,2 --seed "$s" \
    --aggregator adaptive_krum --ablation-mode krum_baseline \
    --attack-type minmax --minmax-dev-type std --minmax-search-iters 15 \
    --tag "$tag"
}

run_calibrated () {
  local a="$1" s="$2"
  local tag="e3_calibrated_a${a}_seed${s}"
  echo "=== $tag ==="
  python main.py network --rounds 25 --alpha "$a" --byzantine 1,2 --seed "$s" \
    --aggregator calibrated_krum --ablation-mode krum_baseline \
    --attack-type minmax --minmax-dev-type std --minmax-search-iters 15 \
    --hetero-fit-coeffs-json "$HETERO_JSON" \
    --tag "$tag"
}

# ---------------------------------------------------------------
# STAGE 1: seed 42, Adaptive Krum, all 5 alphas.  (~5 runs)
# ---------------------------------------------------------------
echo "########## STAGE 1: Adaptive Krum, seed 42, all alphas ##########"
for a in "${ALPHAS[@]}"; do
  run_adaptive "$a" 42
done

# ---------------------------------------------------------------
# STAGE 2: seed 42, Calibrated Krum, all 5 alphas.  (~5 runs)
# ---------------------------------------------------------------
echo "########## STAGE 2: Calibrated Krum, seed 42, all alphas ##########"
for a in "${ALPHAS[@]}"; do
  run_calibrated "$a" 42
done

# ---------------------------------------------------------------
# CHECKPOINT -- stop here manually if you want to inspect results
# before committing to the 40-run backfill below. To pause instead
# of auto-continuing, comment out STAGE 3/4 and re-run this script
# later with only those stages uncommented.
# ---------------------------------------------------------------
echo "########## CHECKPOINT: seed-42 read across both aggregators complete. ##########"
echo "########## Inspect per_client_krum_scores_network_e3_*_seed42.csv before continuing. ##########"
echo "########## Continuing to full seed backfill in 10s (Ctrl-C to stop here)... ##########"
sleep 10

# ---------------------------------------------------------------
# STAGE 3: remaining seeds, Adaptive Krum, all 5 alphas.  (~20 runs)
# ---------------------------------------------------------------
echo "########## STAGE 3: Adaptive Krum, seeds 123/456/789/2024, all alphas ##########"
for a in "${ALPHAS[@]}"; do
  for s in "${BACKFILL_SEEDS[@]}"; do
    run_adaptive "$a" "$s"
  done
done

# ---------------------------------------------------------------
# STAGE 4: remaining seeds, Calibrated Krum, all 5 alphas.  (~20 runs)
# ---------------------------------------------------------------
echo "########## STAGE 4: Calibrated Krum, seeds 123/456/789/2024, all alphas ##########"
for a in "${ALPHAS[@]}"; do
  for s in "${BACKFILL_SEEDS[@]}"; do
    run_calibrated "$a" "$s"
  done
done

echo "E3 campaign complete: 50 runs (25 Adaptive + 25 Calibrated, 5 alphas x 5 seeds each)."
