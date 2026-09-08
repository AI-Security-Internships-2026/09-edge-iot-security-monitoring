#!/usr/bin/env bash
# scripts/run_task4_federated_sweep.sh
#
# Task 4 (E1 baseline table) -- FEDERATED CONDITIONS ONLY.
#
# NOT COVERED by this script (real, open blocker -- see
# README_TASK4_TASK5.md): the CENTRALIZED conditions (CNN-LSTM, MLP,
# Random Forest, XGBoost). I have not been shown any centralized-training
# entrypoint in this codebase (no equivalent of main.py for a
# non-federated run), so I cannot honestly script that half without
# guessing at function names/conventions I have not verified. Send the
# real centralized-training script(s) (or confirm none exist yet) before
# that half of Table 1 can be scripted.
#
# Runs, per the ticket's exact schedule, for BOTH models (network,
# application) x 5 seeds {42,123,456,789,2024}:
#   1. FedAvg               -- PROX_MU=0, AGGREGATOR=fedavg
#   2. FedProx mu-sweep     -- PROX_MU in {0, 0.005, 0.02, 0.05, 0.1},
#                               AGGREGATOR=fedavg (mu is a LOCAL-training
#                               term, not a server-aggregation choice --
#                               see --prox-mu's help text in main.py)
#   3. Arch-swap variant    -- run ONLY after you've read off the
#                               argmax-Macro-F1 mu from step 2's
#                               aggregated results (see
#                               aggregate_task4_results.py) and filled it
#                               into BEST_MU below. --force-dp-safe-arch,
#                               PROX_MU=<BEST_MU>, USE_DP stays False.
#
# All runs are launched as background jobs inside a single tmux session
# (per the project's established "background/tmux, not interactive"
# convention for long compute jobs) so this script itself returns
# immediately -- attach to the session to watch progress.
#
# BEFORE RUNNING: get a fresh single-run timing estimate first (see the
# bottom of this file) -- do NOT launch the full 2-model x 5-seed x
# 6-condition x N-round schedule blind.
#
# Usage:
#   bash scripts/run_task4_federated_sweep.sh timing_probe   # 1 run, foreground, times it
#   bash scripts/run_task4_federated_sweep.sh fedavg_and_prox_sweep  # stage 1+2, background
#   BEST_MU=0.02 bash scripts/run_task4_federated_sweep.sh arch_swap # stage 3, after reading stage 2's argmax

set -euo pipefail

SEEDS=(42 123 456 789 2024)
MODELS=(network application)
MU_VALUES=(0 0.005 0.02 0.05 0.1)
RESULTS_DIR="experiments/results/task4_federated"
SESSION="task4_fl_sweep"

mkdir -p "$RESULTS_DIR"

STAGE="${1:-}"

if [[ "$STAGE" == "timing_probe" ]]; then
    echo "Running a single foreground timing probe (network, seed=42, FedAvg)..."
    echo "Get this number before committing to the full schedule."
    time python main.py network --ablation-mode baseline --aggregator fedavg \
        --prox-mu 0 --seed 42 --tag task4_timing_probe
    exit 0
fi

if [[ "$STAGE" == "fedavg_and_prox_sweep" ]]; then
    echo "Launching FedAvg + FedProx mu-sweep as a background tmux session: $SESSION"
    tmux new-session -d -s "$SESSION" -n runner
    tmux send-keys -t "$SESSION:runner" "bash '$0' _run_fedavg_and_prox_sweep_inner" C-m
    echo "Attach with: tmux attach -t $SESSION"
    exit 0
fi

if [[ "$STAGE" == "_run_fedavg_and_prox_sweep_inner" ]]; then
    for model in "${MODELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            # Condition 1: FedAvg (mu=0)
            tag="task4_fedavg"
            echo "=== $tag ==="
            python main.py "$model" --ablation-mode baseline --aggregator fedavg \
                --prox-mu 0 --seed "$seed" --tag "$tag" \
                2>&1 | tee "$RESULTS_DIR/log_${tag}_${model}_seed${seed}.txt"

            # Condition 2: FedProx mu-sweep
            for mu in "${MU_VALUES[@]}"; do
                tag="task4_fedprox_mu${mu}"
                echo "=== $tag ==="
                python main.py "$model" --ablation-mode baseline --aggregator fedavg \
                    --prox-mu "$mu" --seed "$seed" --tag "$tag" \
                    2>&1 | tee "$RESULTS_DIR/log_${tag}_${model}_seed${seed}.txt"
            done
        done
    done
    echo "Stage 1+2 complete. Now run:"
    echo "  python scripts/aggregate_task4_results.py --stage prox_sweep --results-dir $RESULTS_DIR"
    echo "to get the argmax-Macro-F1 mu, then re-run this script with STAGE=arch_swap and"
    echo "BEST_MU=<that value>."
    exit 0
fi

if [[ "$STAGE" == "arch_swap" ]]; then
    if [[ -z "${BEST_MU:-}" ]]; then
        echo "ERROR: set BEST_MU=<argmax mu from stage 2's aggregated results> first."
        echo "  e.g. BEST_MU=0.02 bash $0 arch_swap"
        exit 1
    fi
    echo "Launching arch-swap variant (PROX_MU=$BEST_MU) as background tmux session: ${SESSION}_archswap"
    tmux new-session -d -s "${SESSION}_archswap" -n runner
    tmux send-keys -t "${SESSION}_archswap:runner" \
        "bash '$0' _run_arch_swap_inner" C-m
    echo "Attach with: tmux attach -t ${SESSION}_archswap"
    exit 0
fi

if [[ "$STAGE" == "_run_arch_swap_inner" ]]; then
    for model in "${MODELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            tag="task4_fedprox_dpsafe_arch_no_dp"
            echo "=== $tag (PROX_MU=$BEST_MU, --force-dp-safe-arch, USE_DP stays False) ==="
            python main.py "$model" --ablation-mode baseline --aggregator fedavg \
                --prox-mu "$BEST_MU" --force-dp-safe-arch \
                --seed "$seed" --tag "$tag" \
                2>&1 | tee "$RESULTS_DIR/log_${tag}_${model}_seed${seed}.txt"
        done
    done
    exit 0
fi

echo "Usage: $0 {timing_probe|fedavg_and_prox_sweep|arch_swap}"
echo "  1. bash $0 timing_probe                          # run this FIRST, foreground"
echo "  2. bash $0 fedavg_and_prox_sweep                  # background tmux, FedAvg+FedProx sweep"
echo "  3. python scripts/aggregate_task4_results.py --stage prox_sweep --results-dir $RESULTS_DIR"
echo "  4. BEST_MU=<argmax from step 3> bash $0 arch_swap # background tmux, arch-swap condition"
echo "  5. python scripts/aggregate_task4_results.py --stage full --results-dir $RESULTS_DIR"
exit 1
