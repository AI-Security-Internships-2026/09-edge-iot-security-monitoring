#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# FL-IDS Epsilon Sweep Driver — Experiment 1 continuation (1-increment grid)
# ---------------------------------------------------------------------------
# Interleaves network/application runs per epsilon value, so you get a
# progressively-complete curve for BOTH models as the sweep advances,
# rather than finishing all of one model before starting the other.
#
# Prerequisite: main.py must have USE_ADAPTIVE_KRUM=True, USE_HE=False,
# USE_HE_KRUM_HYBRID=False, USE_KRUM=False, USE_DP=True (see chat notes).
#
# ε=3, 9, 15 are intentionally excluded — already completed in Experiment 1.
#
# Usage:
#   chmod +x run_epsilon_sweep.sh
#   ./run_epsilon_sweep.sh                 # run the full new grid
#   ./run_epsilon_sweep.sh --dry-run        # print commands without running
# ---------------------------------------------------------------------------

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# epsilon:tag pairs — tag scheme fixes the dp9/dp09 inconsistency flagged
# in the master doc's Open Items (one decimal, dot->p, zero-padded int part)
EPS_TAGS=(
    "0.5:dp0p5"
    "1.0:dp01"
    "2.0:dp02"
    "4.0:dp04"
    "5.0:dp05"
    "6.0:dp06"
    "7.0:dp07"
    "8.0:dp08"
    "10.0:dp10"
    "11.0:dp11"
    "12.0:dp12"
    "13.0:dp13"
    "14.0:dp14"
)

MODELS=("network" "application")
KRUM_K=2.5   # pinned to Experiment 1's value — see chat notes on the
             # ADAPTIVE_KRUM_K default drift (2.5 -> 3.5) before this sweep

LOGDIR="sweep_logs"
mkdir -p "$LOGDIR"

TOTAL=$(( ${#EPS_TAGS[@]} * ${#MODELS[@]} ))
COUNT=0
START_TS=$(date +%s)

for pair in "${EPS_TAGS[@]}"; do
    eps="${pair%%:*}"
    tag="${pair##*:}"
    for model in "${MODELS[@]}"; do
        COUNT=$((COUNT + 1))
        LOGFILE="${LOGDIR}/${model}_${tag}.log"
        CKPT="checkpoint_${model}_${tag}.npz"
        PROG="checkpoint_${model}_${tag}_progress.json"

        echo "============================================================"
        echo " [$COUNT/$TOTAL] model=$model  epsilon=$eps  tag=$tag"
        echo "============================================================"

        if $DRY_RUN; then
            echo "python3 main.py $model --epsilon $eps --tag $tag --krum-k $KRUM_K"
            continue
        fi

        # Fresh run per condition — guard against accidentally resuming a
        # PARTIAL prior attempt at this exact tag under different flags
        # (main.py warns about this but doesn't block it automatically).
        if [[ -f "$CKPT" || -f "$PROG" ]]; then
            echo "  NOTE: existing checkpoint found for ${model}_${tag} —"
            echo "  resuming from it. Delete ${CKPT} / ${PROG} first if you"
            echo "  want a clean restart of this specific condition."
        fi

        RUN_START=$(date +%s)
        python3 main.py "$model" --epsilon "$eps" --tag "$tag" --krum-k "$KRUM_K" \
            2>&1 | tee "$LOGFILE"
        RUN_END=$(date +%s)
        echo "  [$model/$tag] finished in $(( (RUN_END - RUN_START) / 60 ))m"
        echo
    done
done

END_TS=$(date +%s)
echo "Sweep complete: $COUNT/$TOTAL conditions in $(( (END_TS - START_TS) / 3600 ))h"
