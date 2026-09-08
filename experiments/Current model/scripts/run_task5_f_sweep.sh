#!/usr/bin/env bash
# scripts/run_task5_f_sweep.sh
#
# Task 5, Table B: Byzantine-f sensitivity sweep.
# Fixed config: alpha=0.3, epsilon_target=5, MAD-k=3.5 (production
# default). Sweep: f in {1, 2, 3} out of 10, 5 seeds each, plain
# Adaptive Krum vs Calibrated Krum.
#
# f=1 -> --byzantine 9
# f=2 -> --byzantine 9,10
# f=3 -> --byzantine 8,9,10
#
# Same prerequisites and HETERO_FIT_JSON option as
# run_task5_mad_k_sweep.sh -- see that file's header comment.
#
# Usage:
#   bash scripts/run_task5_f_sweep.sh timing_probe
#   bash scripts/run_task5_f_sweep.sh run

set -euo pipefail

SEEDS=(42 123 456 789 2024)
declare -A F_TO_CLIENTS=( [1]="9" [2]="9,10" [3]="8,9,10" )
MODEL="network"
ALPHA=0.3
EPSILON=5
MAD_K=3.5
RESULTS_DIR="experiments/results/task5_f_sweep"
SESSION="task5_f_sweep"

mkdir -p "$RESULTS_DIR"
HETERO_ARGS=()
if [[ -n "${HETERO_FIT_JSON:-}" ]]; then
    HETERO_ARGS=(--hetero-fit-coeffs-json "$HETERO_FIT_JSON")
fi

STAGE="${1:-}"

if [[ "$STAGE" == "timing_probe" ]]; then
    echo "--- plain adaptive Krum, f=2 ---"
    time python main.py "$MODEL" --ablation-mode krum_baseline \
        --alpha "$ALPHA" --krum-k "$MAD_K" --byzantine "${F_TO_CLIENTS[2]}" \
        --seed 42 --tag task5_fsweep_timing_probe_plain
    echo ""
    echo "--- calibrated Krum (DP-active), f=2 ---"
    time python main.py "$MODEL" --ablation-mode calibrated_krum_dp_sweep \
        --alpha "$ALPHA" --epsilon "$EPSILON" --krum-k "$MAD_K" \
        --byzantine "${F_TO_CLIENTS[2]}" "${HETERO_ARGS[@]}" \
        --seed 42 --tag task5_fsweep_timing_probe_calibrated
    exit 0
fi

if [[ "$STAGE" == "run" ]]; then
    echo "Launching Byzantine-f sweep as background tmux session: $SESSION"
    tmux new-session -d -s "$SESSION" -n runner
    tmux send-keys -t "$SESSION:runner" \
        "HETERO_FIT_JSON='${HETERO_FIT_JSON:-}' bash '$0' _run_inner" C-m
    echo "Attach with: tmux attach -t $SESSION"
    exit 0
fi

if [[ "$STAGE" == "_run_inner" ]]; then
    for f in 1 2 3; do
        clients="${F_TO_CLIENTS[$f]}"
        for seed in "${SEEDS[@]}"; do
            tag_plain="task5_fsweep_plain_f${f}"
            echo "=== plain krum f=$f (clients=$clients) seed=$seed ==="
            python main.py "$MODEL" --ablation-mode krum_baseline \
                --alpha "$ALPHA" --krum-k "$MAD_K" --byzantine "$clients" \
                --seed "$seed" --tag "$tag_plain" \
                2>&1 | tee "$RESULTS_DIR/log_${tag_plain}_seed${seed}.txt"
            mv "per_client_krum_scores_${MODEL}_${tag_plain}_seed${seed}.csv" \
               "$RESULTS_DIR/" 2>/dev/null || echo "  WARNING: expected per_client_krum_scores CSV not found for $tag_plain seed $seed"

            tag_cal="task5_fsweep_calibrated_f${f}"
            echo "=== calibrated krum f=$f (clients=$clients) seed=$seed ==="
            python main.py "$MODEL" --ablation-mode calibrated_krum_dp_sweep \
                --alpha "$ALPHA" --epsilon "$EPSILON" --krum-k "$MAD_K" \
                --byzantine "$clients" "${HETERO_ARGS[@]}" \
                --seed "$seed" --tag "$tag_cal" \
                2>&1 | tee "$RESULTS_DIR/log_${tag_cal}_seed${seed}.txt"
            mv "per_client_krum_scores_${MODEL}_${tag_cal}_seed${seed}.csv" \
               "$RESULTS_DIR/" 2>/dev/null || echo "  WARNING: expected per_client_krum_scores CSV not found for $tag_cal seed $seed"
        done
    done
    echo "Sweep complete. Now run:"
    echo "  python scripts/aggregate_task5_results.py --table B --results-dir $RESULTS_DIR"
    exit 0
fi

echo "Usage: $0 {timing_probe|run}"
exit 1
