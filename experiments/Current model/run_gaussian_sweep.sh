#!/bin/bash
# ============================================================================
# Gaussian-noise attack sweep (Sweep 2)
#
# Mirrors Experiment 1's corrected/extended sign-flip sweep exactly, but with
# --attack-type gaussian instead of the sign_flip default. Requires main.py's
# ABLATION_MODE to be set to "krum_dp_sweep" (see the added elif block) —
# NOT one of pure_dp/pure_he/pure_zkp, none of which activate Adaptive Krum
# + DP + an active Byzantine attack together.
#
# Recommended: run this standalone, not concurrently with another sweep.
# The DGX Spark's unified CPU/GPU memory pool has already caused one
# documented contention incident (the vLLM episode) — no need to risk a
# repeat on a run whose completion status you're specifically trying to
# nail down this time.
#
# Estimated runtime: 2 models x 13 epsilon points x 25 rounds x ~140s/round
# (GPU, sequential in-process) ~= 25 hours. Run this inside tmux/screen with
# nohup so it survives an SSH disconnect:
#
#   tmux new -s gaussian_sweep
#   ./run_gaussian_sweep.sh 2>&1 | tee logs/gaussian_sweep_full.log
#   # detach: Ctrl-b d
#
# ============================================================================
set -e

cd "$(dirname "$0")"   # run from the directory main.py lives in
mkdir -p logs

MODEL_TYPES=("network" "application")
EPSILONS=(0.5 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)

echo "============================================================"
echo "  Gaussian sweep starting: $(date)"
echo "  Models: ${MODEL_TYPES[*]}"
echo "  Epsilons: ${EPSILONS[*]}"
echo "  Total runs: $(( ${#MODEL_TYPES[@]} * ${#EPSILONS[@]} ))"
echo "============================================================"

for MODEL in "${MODEL_TYPES[@]}"; do
  for EPS in "${EPSILONS[@]}"; do
    TAG="gaussian_dp${EPS}"

    # Guard against silently resuming into a stale/incompatible checkpoint
    # from an earlier partial/failed attempt at this exact tag.
    if [ -f "checkpoint_${MODEL}_${TAG}.npz" ]; then
      echo ">>> Existing checkpoint found for ${MODEL} ${TAG} -- resuming."
      echo ">>> If you changed any flag since that checkpoint was written,"
      echo ">>> delete checkpoint_${MODEL}_${TAG}.npz and _progress.json first."
    fi

    echo ""
    echo "=== [$(date +%T)] Running model=${MODEL} eps=${EPS} tag=${TAG} ==="
    python main.py "$MODEL" \
      --epsilon "$EPS" \
      --attack-type gaussian \
      --tag "$TAG" \
      2>&1 | tee "logs/gaussian_${MODEL}_eps${EPS}.log"

    echo "=== [$(date +%T)] Completed model=${MODEL} eps=${EPS} ==="
  done
done

echo ""
echo "============================================================"
echo "  Gaussian sweep finished: $(date)"
echo "  Results:  results_<model>_gaussian_dp<eps>.csv"
echo "  Configs:  experiment_config_<model>_gaussian_dp<eps>.json"
echo "  Checkpoints: checkpoint_<model>_gaussian_dp<eps>*.npz/.json"
echo "============================================================"
