#!/usr/bin/env bash
# launch_e2_tmux.sh
#
# Splits the 90-cell E2 campaign across 5 tmux windows (18 cells each,
# exactly 3 per aggregator per window -- see run_e2_campaign.py's
# --num-shards sorted-index-mod split). Each window runs independently;
# a crash in one doesn't affect the others, and each is individually
# resumable.
#
# Prereqs (should already be done):
#   - experiments/configs/EXP1_campaign_E2.json exists (scripts/build_e2_campaign.py)
#   - experiments/configs/hetero_fit_coeffs_a0.7.json exists
#   - scripts/check_attack_difficulty.py has passed
#
# Usage:
#   scripts/launch_e2_tmux.sh                # start a fresh session
#   scripts/launch_e2_tmux.sh --resume       # reattach to existing session
#   tmux attach -t e2                        # watch any window live
#   tmux select-window -t e2:2               # jump to shard 2's window
#
# Each window's full stdout/stderr also goes to logs/e2_shard_N.log, so
# you can `tail -f` a shard without attaching to tmux at all.

set -euo pipefail

SESSION="e2"
NUM_SHARDS=5
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"

if [ "${1:-}" == "--resume" ]; then
    echo "Reattaching to existing session '${SESSION}'..."
    exec tmux attach -t "${SESSION}"
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "Session '${SESSION}' already exists. Use --resume to reattach," \
         "or 'tmux kill-session -t ${SESSION}' to tear it down first."
    exit 1
fi

mkdir -p "${LOG_DIR}"

echo "Starting tmux session '${SESSION}' with ${NUM_SHARDS} shard windows..."
echo "Repo root: ${REPO_ROOT}"
echo "Logs:      ${LOG_DIR}/e2_shard_*.log"

# Window 0: create the session itself.
tmux new-session -d -s "${SESSION}" -n "shard0" -c "${REPO_ROOT}"
tmux send-keys -t "${SESSION}:shard0" \
    "python scripts/run_e2_campaign.py --num-shards ${NUM_SHARDS} --shard-index 0 2>&1 | tee '${LOG_DIR}/e2_shard_0.log'" \
    C-m

# Windows 1..N-1.
for i in $(seq 1 $((NUM_SHARDS - 1))); do
    tmux new-window -t "${SESSION}" -n "shard${i}" -c "${REPO_ROOT}"
    tmux send-keys -t "${SESSION}:shard${i}" \
        "python scripts/run_e2_campaign.py --num-shards ${NUM_SHARDS} --shard-index ${i} 2>&1 | tee '${LOG_DIR}/e2_shard_${i}.log'" \
        C-m
done

tmux select-window -t "${SESSION}:shard0"

echo ""
echo "All ${NUM_SHARDS} shards launched (18 cells each)."
echo "  Attach:       tmux attach -t ${SESSION}"
echo "  Jump to N:    tmux select-window -t ${SESSION}:shardN"
echo "  Detach:       Ctrl-b d"
echo "  Tail a shard: tail -f ${LOG_DIR}/e2_shard_N.log"
echo "  Check all:    scripts/check_e2_shards.sh   (progress summary, see below)"
