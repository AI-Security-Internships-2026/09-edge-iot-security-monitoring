#!/usr/bin/env bash
# Start E5 in 3 detached tmux sessions (e5_0, e5_1, e5_2), each running a disjoint third of the cells.
#   bash scripts/launch_e5_tmux.sh                 # run from repo root
#   VENV=.venv bash scripts/launch_e5_tmux.sh      # activate a venv first
#   N=3 PARALLEL=1 bash scripts/launch_e5_tmux.sh
# Attach: tmux attach -t e5_0     Detach: Ctrl-b d     List: tmux ls
# Stop all: for i in 0 1 2; do tmux kill-session -t e5_$i; done
set -euo pipefail
N="${N:-3}"; PARALLEL="${PARALLEL:-1}"; CONFIG="${CONFIG:-experiments/configs/E5_campaign.json}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
command -v tmux >/dev/null || { echo "tmux not installed"; exit 1; }
cd "$ROOT"
python scripts/run_e5.py plan --config "$CONFIG" > /dev/null    # single audit BEFORE the shards start
for i in $(seq 0 $((N-1))); do
  S="e5_$i"
  if tmux has-session -t "$S" 2>/dev/null; then echo "session $S already exists - skipping"; continue; fi
  ACT=""; [ -n "${VENV:-}" ] && ACT="source $VENV/bin/activate && "
  tmux new-session -d -s "$S" -c "$ROOT" \
    "bash -lc '${ACT}python scripts/run_e5.py run --config $CONFIG --shard $i/$N --parallel $PARALLEL 2>&1 | tee e5_shard_$i.log; echo; echo SHARD $i DONE; exec bash'"
  echo "started $S  (log: e5_shard_$i.log)"
done
tmux ls
echo "When all three print SHARD DONE:  python scripts/run_e5.py plan && python scripts/run_e5.py summarize"
