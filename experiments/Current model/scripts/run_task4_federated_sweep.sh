#!/usr/bin/env bash
# scripts/run_task4_federated_sweep.sh  (no-tmux / simple sequential version)
#
# Task 4 (E1 baseline table) -- FEDERATED CONDITIONS ONLY.
#
# NOT COVERED by this script (real, open blocker -- see
# README_TASK4_TASK5.md): the CENTRALIZED conditions (CNN-LSTM, MLP,
# Random Forest, XGBoost) -- those come from scripts/train_centralized.py.
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
# THIS VERSION RUNS EVERYTHING IN THE FOREGROUND, SEQUENTIALLY, NO TMUX.
# That means:
#   - The full schedule blocks your terminal until it's done. Use a
#     terminal multiplexer of your choice, `screen`, `&` + `disown`, or
#     just leave the terminal open -- your call, this script doesn't
#     manage that for you anymore.
#   - If the process dies partway (closed terminal, laptop sleep, etc.),
#     you'll need to re-run -- the script does NOT skip runs that already
#     have output files. Re-running is generally safe (files get
#     overwritten) but wastes compute on already-done seeds/conditions.
#     Check experiments/results/task4_federated/ for existing
#     results_*_FINAL_TEST.csv files before re-launching a stage.
#
# BEFORE RUNNING: get a fresh single-run timing estimate first (see the
# bottom of this file) -- do NOT launch the full 2-model x 5-seed x
# 6-condition x N-round schedule blind.
#
# Usage:
#   bash scripts/run_task4_federated_sweep.sh timing_probe
#   bash scripts/run_task4_federated_sweep.sh fedavg_and_prox_sweep
#   BEST_MU=0.02 bash scripts/run_task4_federated_sweep.sh arch_swap
#
#   # If network and application argmax to DIFFERENT mu (common -- check
#   # your prox_sweep aggregation output), run arch_swap twice instead:
#   BEST_MU=0.005 MODEL=network     bash scripts/run_task4_federated_sweep.sh arch_swap
#   BEST_MU=0     MODEL=application bash scripts/run_task4_federated_sweep.sh arch_swap

set -euo pipefail

SEEDS=(42 123 456 789 2024)
MODELS=(network application)
MU_VALUES=(0 0.005 0.02 0.05 0.1)
RESULTS_DIR="experiments/results/task4_federated"

mkdir -p "$RESULTS_DIR"

# Optional single-model override -- needed for arch_swap when network and
# application argmax to DIFFERENT mu values (they can't share one BEST_MU
# in that case). Leave MODEL unset to run both models in one invocation.
if [[ -n "${MODEL:-}" ]]; then
    if [[ "$MODEL" != "network" && "$MODEL" != "application" ]]; then
        echo "ERROR: MODEL must be 'network' or 'application', got '$MODEL'"
        exit 1
    fi
    MODELS=("$MODEL")
fi

STAGE="${1:-}"

if [[ "$STAGE" == "timing_probe" ]]; then
    echo "Running a single foreground timing probe (network, seed=42, FedAvg)..."
    echo "Get this number before committing to the full schedule."
    time python main.py network --ablation-mode baseline --aggregator fedavg \
        --prox-mu 0 --seed 42 --tag task4_timing_probe
    exit 0
fi

if [[ "$STAGE" == "fedavg_and_prox_sweep" ]]; then
    echo "Running FedAvg + FedProx mu-sweep sequentially in the foreground."
    echo "Models: ${MODELS[*]} | Seeds: ${SEEDS[*]} | Mu values: ${MU_VALUES[*]}"
    for model in "${MODELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            # Condition 1: FedAvg (mu=0)
            tag="task4_fedavg"
            echo "=== $tag ($model, seed=$seed) ==="
            python main.py "$model" --ablation-mode baseline --aggregator fedavg \
                --prox-mu 0 --seed "$seed" --tag "$tag" \
                2>&1 | tee "$RESULTS_DIR/log_${tag}_${model}_seed${seed}.txt"

            # Condition 2: FedProx mu-sweep
            for mu in "${MU_VALUES[@]}"; do
                tag="task4_fedprox_mu${mu}"
                echo "=== $tag ($model, seed=$seed) ==="
                python main.py "$model" --ablation-mode baseline --aggregator fedavg \
                    --prox-mu "$mu" --seed "$seed" --tag "$tag" \
                    2>&1 | tee "$RESULTS_DIR/log_${tag}_${model}_seed${seed}.txt"
            done
        done
    done
    echo "Stage 1+2 complete. Now run:"
    echo "  python scripts/aggregate_task4_results.py --stage prox_sweep --results-dir $RESULTS_DIR"
    echo "to get the argmax-Macro-F1 mu, then re-run this script with STAGE=arch_swap and"
    echo "BEST_MU=<that value> (per-model if they differ -- see header comment)."
    exit 0
fi

if [[ "$STAGE" == "arch_swap" ]]; then
    if [[ -z "${BEST_MU:-}" ]]; then
        echo "ERROR: set BEST_MU=<argmax mu from stage 2's aggregated results> first."
        echo "  e.g. BEST_MU=0.02 bash $0 arch_swap"
        echo "  or, for a per-model mu:"
        echo "  BEST_MU=0.005 MODEL=network     bash $0 arch_swap"
        echo "  BEST_MU=0     MODEL=application bash $0 arch_swap"
        exit 1
    fi
    echo "Running arch-swap variant (PROX_MU=$BEST_MU, MODELS=${MODELS[*]}) sequentially in the foreground."
    for model in "${MODELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            tag="task4_fedprox_dpsafe_arch_no_dp"
            echo "=== $tag ($model, seed=$seed, PROX_MU=$BEST_MU, --force-dp-safe-arch, USE_DP stays False) ==="
            python main.py "$model" --ablation-mode baseline --aggregator fedavg \
                --prox-mu "$BEST_MU" --force-dp-safe-arch \
                --seed "$seed" --tag "$tag" \
                2>&1 | tee "$RESULTS_DIR/log_${tag}_${model}_seed${seed}.txt"
        done
    done
    echo "arch_swap (MODELS=${MODELS[*]}) complete."
    exit 0
fi

echo "Usage: $0 {timing_probe|fedavg_and_prox_sweep|arch_swap}"
echo "  1. bash $0 timing_probe                          # run this FIRST"
echo "  2. bash $0 fedavg_and_prox_sweep                  # sequential, foreground, FedAvg+FedProx sweep"
echo "  3. python scripts/aggregate_task4_results.py --stage prox_sweep --results-dir $RESULTS_DIR"
echo "  4. BEST_MU=<argmax from step 3> bash $0 arch_swap # sequential, foreground, arch-swap condition"
echo "     If network and application argmax to DIFFERENT mu, run arch_swap twice instead:"
echo "       BEST_MU=<network argmax>     MODEL=network     bash $0 arch_swap"
echo "       BEST_MU=<application argmax> MODEL=application bash $0 arch_swap"
echo "  5. python scripts/aggregate_task4_results.py --stage full --results-dir $RESULTS_DIR"
exit 1
