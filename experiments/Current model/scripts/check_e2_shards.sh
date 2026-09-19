#!/usr/bin/env bash
# check_e2_shards.sh -- one-glance progress across all 5 E2 tmux shards.
# Reads logs/e2_shard_N.log (written by launch_e2_tmux.sh's `tee`), not
# tmux itself, so this works even if you've detached or closed the
# terminal that launched things.
#
# Usage: scripts/check_e2_shards.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
CELLS_PER_SHARD=18

total_ok=0
total_fail=0
any_running=0

for i in 0 1 2 3 4; do
    log="${LOG_DIR}/e2_shard_${i}.log"
    if [ ! -f "${log}" ]; then
        echo "shard ${i}: log not found (${log}) -- not started?"
        continue
    fi

    ok=$(grep -c '^  \[OK\]$' "${log}" || true)
    fail=$(grep -c '^  \[FAIL\]' "${log}" || true)
    done_count=$((ok + fail))
    last_line=$(tail -n 1 "${log}")
    running_marker=""

    if grep -q "^All .* completed and verified\.$" "${log}" 2>/dev/null; then
        status="COMPLETE"
    elif [ "${fail}" -gt 0 ] && grep -q "FAILED:" "${log}" 2>/dev/null; then
        status="FINISHED WITH FAILURES"
    else
        status="running"
        running_marker="(currently: ${last_line:0:70})"
    fi

    echo "shard ${i}: ${done_count}/${CELLS_PER_SHARD}  ok=${ok} fail=${fail}  [${status}] ${running_marker}"

    total_ok=$((total_ok + ok))
    total_fail=$((total_fail + fail))
done

echo ""
echo "TOTAL: ${total_ok} ok, ${total_fail} failed, out of 90."

if [ "${total_fail}" -gt 0 ]; then
    echo ""
    echo "Failed cell IDs (grep across all shard logs):"
    grep -B2 '^  \[FAIL\]' "${LOG_DIR}"/e2_shard_*.log | grep -oE 'E2-[0-9]{3}' | sort -u || true
fi
