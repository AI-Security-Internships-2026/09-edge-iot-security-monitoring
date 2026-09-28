#!/usr/bin/env bash
# launch_e7_tmux.sh -- runs the E7 campaign as two parallel tmux
# sessions (e7_shard0, e7_shard1), each handling half the 84 cells
# (round-robin by cell index, so both shards get a mix of FedAvg/
# AdaptiveKrum/CalibratedKrum and both alphas -- not e.g. all of
# FedAvg in one shard and nothing to overlap-check against).
#
# Usage (run from the repo root, next to main.py):
#   chmod +x launch_e7_tmux.sh run_e7_campaign.py
#   ./launch_e7_tmux.sh --smoke        # 1-cell dry-run style sanity check first
#   ./launch_e7_tmux.sh                # launch the real 84-cell campaign
#
# Requires: the "fedavg_attack" ablation-mode patch already applied to
# main.py (see main_py_patch.diff), and
# experiments/configs/frozen_minmax_params.json already built (Task 2).
set -euo pipefail

MAIN_PY="./main.py"
OUTPUT_ROOT="./experiments/results/E7"
FROZEN_JSON="./experiments/configs/frozen_minmax_params.json"
SUBSET_FRACTION="0.01"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_e7_campaign.py"

SMOKE=0
if [[ "${1:-}" == "--smoke" ]]; then
  SMOKE=1
fi

if [[ ! -f "$MAIN_PY" ]]; then
  echo "ERROR: $MAIN_PY not found -- run this from the repo root, next to main.py"
  exit 1
fi
if [[ ! -f "$FROZEN_JSON" ]]; then
  echo "ERROR: $FROZEN_JSON not found -- run scripts/check_attack_difficulty.py"
  echo "and build frozen_minmax_params.json first (see earlier E4 setup)."
  exit 1
fi

if [[ "$SMOKE" == "1" ]]; then
  echo "=== E7 dry-run: printing the full 84-cell plan, no execution ==="
  python "$RUNNER" \
    --main-py "$MAIN_PY" \
    --output-root "$OUTPUT_ROOT" \
    --frozen-attack-json "$FROZEN_JSON" \
    --ciciot-subset-fraction "$SUBSET_FRACTION" \
    --num-shards 1 --shard 0 \
    --dry-run
  echo
  echo "Dry-run above should show 84 cells total. If that looks right,"
  echo "re-run without --smoke to launch the real tmux sessions."
  exit 0
fi

mkdir -p "$OUTPUT_ROOT"

for shard in 0 1; do
  session="e7_shard${shard}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already exists -- not relaunching it."
    echo "  (attach with: tmux attach -t $session)"
    continue
  fi
  logfile="tmux_e7_shard${shard}.log"
  cmd="python '$RUNNER' \
    --main-py '$MAIN_PY' \
    --output-root '$OUTPUT_ROOT' \
    --frozen-attack-json '$FROZEN_JSON' \
    --ciciot-subset-fraction '$SUBSET_FRACTION' \
    --num-shards 2 --shard $shard \
    2>&1 | tee '$logfile'"
  tmux new-session -d -s "$session" "$cmd"
  echo "Launched $session -> log: $logfile"
done

sleep 3
echo
echo "=== tmux sessions ==="
tmux ls | grep e7_shard || echo "WARNING: no e7_shard sessions found -- check for a launch error above."
echo
echo "Attach:      tmux attach -t e7_shard0   (Ctrl-b d to detach)"
echo "Follow logs: tail -f tmux_e7_shard0.log tmux_e7_shard1.log"
echo "Progress:    find '$OUTPUT_ROOT' -name 'results_*_FINAL_TEST.csv' -size +0c | wc -l"
echo "             (target: 84)"
