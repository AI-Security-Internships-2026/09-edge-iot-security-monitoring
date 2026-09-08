#!/usr/bin/env bash
# scripts/run_task5_mad_k_sweep.sh
#
# Task 5, Table A: MAD-k sensitivity sweep.
# Fixed config: alpha=0.3, epsilon_target=5, f=2/10 (bounded-directional
# attack -- this codebase's closest match is sign_flip; confirm with
# whoever owns the paper's attack-model definition whether "bounded-
# directional" means something more specific than sign_flip before
# treating this as settled).
# Sweep: MAD-k in {2.0, 2.5, 3.0, 3.5, 4.0}, 5 seeds each, BOTH
# plain Adaptive Krum (krum_baseline mode) AND Calibrated Krum
# (calibrated_krum_dp_sweep mode -- the new mode that actually makes
# DP calibration reachable, see main.py's Issue-4 follow-up fix).
#
# HARD PREREQUISITE (per the order-note established earlier in this
# project): do not run this until
#   1. defences/krum.py's calibrated_adaptive_multi_krum is confirmed
#      working (tests/test_calibrated_krum.py passing -- confirmed).
#   2. main.py's --alpha flag is confirmed working (confirmed).
#   3. main.py's calibrated_krum_dp_sweep ablation mode exists (just
#      added -- confirmed via the isolated dispatch-logic check, but
#      NOT yet confirmed against a real end-to-end run, since this
#      environment has no GPU/dataset access. Run the timing_probe
#      stage below FIRST and visually confirm the printed startup
#      banner shows USE_DP=True and USE_CALIBRATED_KRUM's dispatch
#      actually firing before committing to the full sweep.)
#
# OPTIONAL (recommended, not required): pass HETERO_FIT_JSON=<path> to
# also activate hetero calibration using a real fit from
# scripts/fit_hetero_variance.py. If unset, hetero calibration stays a
# documented no-op (DP calibration still applies) -- this is a valid,
# honest partial-calibration run, not a broken one; just be clear in
# any resulting write-up that "Calibrated Krum" in this Table A run
# means "DP-calibrated only" unless HETERO_FIT_JSON was set.
#
# Usage:
#   bash scripts/run_task5_mad_k_sweep.sh timing_probe
#   bash scripts/run_task5_mad_k_sweep.sh run
#   HETERO_FIT_JSON=hetero_fit_coeffs.json bash scripts/run_task5_mad_k_sweep.sh run

set -euo pipefail

SEEDS=(42 123 456 789 2024)
K_VALUES=(2.0 2.5 3.0 3.5 4.0)
MODEL="network"          # Table A/B are single-model sensitivity tables;
                          # confirm with paper authors if "application"
                          # is also needed, or if network alone suffices
                          # for this inset.
ALPHA=0.3
EPSILON=5
BYZANTINE_CLIENTS="9,10"  # f=2/10, 1-indexed clients 9 and 10
RESULTS_DIR="experiments/results/task5_mad_k_sweep"
SESSION="task5_mad_k_sweep"

mkdir -p "$RESULTS_DIR"
HETERO_ARGS=()
if [[ -n "${HETERO_FIT_JSON:-}" ]]; then
    HETERO_ARGS=(--hetero-fit-coeffs-json "$HETERO_FIT_JSON")
fi

STAGE="${1:-}"

if [[ "$STAGE" == "timing_probe" ]]; then
    echo "Foreground probe: k=3.5, seed=42, plain THEN calibrated -- confirm"
    echo "both start cleanly and reach [ROUND 1/25] with USE_DP=True on"
    echo "the calibrated run before committing to the full sweep."
    echo ""
    echo "--- plain adaptive Krum ---"
    time python main.py "$MODEL" --ablation-mode krum_baseline \
        --alpha "$ALPHA" --krum-k 3.5 --byzantine "$BYZANTINE_CLIENTS" \
        --seed 42 --tag task5_madk_timing_probe_plain
    echo ""
    echo "--- calibrated Krum (DP-active) ---"
    time python main.py "$MODEL" --ablation-mode calibrated_krum_dp_sweep \
        --alpha "$ALPHA" --epsilon "$EPSILON" --krum-k 3.5 \
        --byzantine "$BYZANTINE_CLIENTS" "${HETERO_ARGS[@]}" \
        --seed 42 --tag task5_madk_timing_probe_calibrated
    exit 0
fi

if [[ "$STAGE" == "run" ]]; then
    echo "Launching MAD-k sweep as background tmux session: $SESSION"
    tmux new-session -d -s "$SESSION" -n runner
    tmux send-keys -t "$SESSION:runner" \
        "HETERO_FIT_JSON='${HETERO_FIT_JSON:-}' bash '$0' _run_inner" C-m
    echo "Attach with: tmux attach -t $SESSION"
    exit 0
fi

if [[ "$STAGE" == "_run_inner" ]]; then
    for k in "${K_VALUES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            tag_plain="task5_madk_plain_k${k}"
            echo "=== plain krum k=$k seed=$seed ==="
            python main.py "$MODEL" --ablation-mode krum_baseline \
                --alpha "$ALPHA" --krum-k "$k" --byzantine "$BYZANTINE_CLIENTS" \
                --seed "$seed" --tag "$tag_plain" \
                2>&1 | tee "$RESULTS_DIR/log_${tag_plain}_seed${seed}.txt"
            mv "per_client_krum_scores_${MODEL}_${tag_plain}_seed${seed}.csv" \
               "$RESULTS_DIR/" 2>/dev/null || echo "  WARNING: expected per_client_krum_scores CSV not found for $tag_plain seed $seed"

            tag_cal="task5_madk_calibrated_k${k}"
            echo "=== calibrated krum k=$k seed=$seed ==="
            python main.py "$MODEL" --ablation-mode calibrated_krum_dp_sweep \
                --alpha "$ALPHA" --epsilon "$EPSILON" --krum-k "$k" \
                --byzantine "$BYZANTINE_CLIENTS" "${HETERO_ARGS[@]}" \
                --seed "$seed" --tag "$tag_cal" \
                2>&1 | tee "$RESULTS_DIR/log_${tag_cal}_seed${seed}.txt"
            mv "per_client_krum_scores_${MODEL}_${tag_cal}_seed${seed}.csv" \
               "$RESULTS_DIR/" 2>/dev/null || echo "  WARNING: expected per_client_krum_scores CSV not found for $tag_cal seed $seed"
        done
    done
    echo "Sweep complete. Now run:"
    echo "  python scripts/aggregate_task5_results.py --table A --results-dir $RESULTS_DIR"
    exit 0
fi

echo "Usage: $0 {timing_probe|run}"
exit 1
