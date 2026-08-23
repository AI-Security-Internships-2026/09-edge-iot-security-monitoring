#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# FL-IDS Epsilon Sweep Driver v2 — FULL grid under the fixed sign-flip attack
# ---------------------------------------------------------------------------
# Supersedes the original run_epsilon_sweep.sh. That version deliberately
# EXCLUDED eps=3,9,15 because they were "already done" in Experiment 1 --
# but Experiment 1's anchors used the pre-fix sign_flip_attack (operates on
# untouched global_params, not a trained update -- see byzantine.py). Since
# main.py now trains-then-flips by default (sign_flip_attack_trained), those
# three anchors are no longer valid alongside the rest of this grid and MUST
# be re-run for a consistent, single-attacker-model 16-point curve.
#
# PREREQUISITE: run archive_pre_fix_results.sh FIRST. If old
# checkpoint_{tag}.npz files exist for any of these tags, main.py will
# silently RESUME from pre-fix weights instead of starting fresh --
# contaminating the corrected re-run with old attack behavior baked into
# the model state, with no visible seam in the output CSV.
#
# Usage:
#   ./archive_pre_fix_results.sh     # do this first, separately
#   ./run_epsilon_sweep_v2.sh --dry-run
#   ./run_epsilon_sweep_v2.sh
# ---------------------------------------------------------------------------

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# Full 16-point grid, consistent zero-padded naming throughout (fixes the
# old dp9/dp09 inconsistency AND extends the same scheme to 3/9/15, which
# never had a consistent tag in this script's naming before).
EPS_TAGS=(
    "0.5:dp0p5"
    "1.0:dp01"
    "2.0:dp02"
    "3.0:dp03"
    "4.0:dp04"
    "5.0:dp05"
    "6.0:dp06"
    "7.0:dp07"
    "8.0:dp08"
    "9.0:dp09"
    "10.0:dp10"
    "11.0:dp11"
    "12.0:dp12"
    "13.0:dp13"
    "14.0:dp14"
    "15.0:dp15"
)

MODELS=("network" "application")
KRUM_K=2.5   # pinned to match every condition -- see prior chat notes on
             # the ADAPTIVE_KRUM_K default drift; --krum-k still overrides
             # explicitly regardless of main.py's own current default

LOGDIR="sweep_logs_v2"
mkdir -p "$LOGDIR"

# Safety check -- refuse to proceed if any stale checkpoint for one of
# these exact tags is still sitting in the working directory (i.e. the
# archive script wasn't run, or was run before some of these tags existed).
STALE_FOUND=false
for pair in "${EPS_TAGS[@]}"; do
    tag="${pair##*:}"
    for model in "${MODELS[@]}"; do
        ckpt="checkpoint_${model}_${tag}.npz"
        if [[ -f "$ckpt" ]]; then
            echo "WARNING: stale checkpoint found: $ckpt"
            echo "  This WILL cause main.py to resume from pre-fix weights."
            STALE_FOUND=true
        fi
    done
done
if $STALE_FOUND && ! $DRY_RUN; then
    echo ""
    echo "Refusing to start -- run archive_pre_fix_results.sh first, or"
    echo "manually delete the checkpoint(s) listed above if you are certain"
    echo "they are already from a post-fix run."
    exit 1
fi

TOTAL=$(( ${#EPS_TAGS[@]} * ${#MODELS[@]} ))
COUNT=0
START_TS=$(date +%s)

for pair in "${EPS_TAGS[@]}"; do
    eps="${pair%%:*}"
    tag="${pair##*:}"
    for model in "${MODELS[@]}"; do
        COUNT=$((COUNT + 1))
        LOGFILE="${LOGDIR}/${model}_${tag}.log"

        echo "============================================================"
        echo " [$COUNT/$TOTAL] model=$model  epsilon=$eps  tag=$tag"
        echo "============================================================"

        if $DRY_RUN; then
            echo "python3 main.py $model --epsilon $eps --tag $tag --krum-k $KRUM_K"
            continue
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
echo ""
echo "Next: python3 build_sweep_table.py   (use build_sweep_table_fixed.py"
echo "if you have both old-attacker and new-attacker CSVs in the same dir --"
echo "it filters by adaptive_krum_k, not attack version, so also check"
echo "experiment_config_*.json's 'attack_function' field before trusting"
echo "any cross-run comparison.)"
