#!/usr/bin/env bash
# ============================================================
# launch_multiseed_ablations.sh  (v3 -- 4 NEW seeds, 3 sessions, sequential)
#
# You already have 1 existing run per (mode, model) at seed=42 (the old
# hardcoded default, from before --seed existed). This script does NOT
# repeat that -- it only runs the 4 NEW seeds needed to reach 5 total
# per condition: 123, 7, 2024, 31337.
#
# NOTE ON YOUR EXISTING SEED=42 FILES: they almost certainly do NOT
# follow the new "{model}_{mode}_seed{N}" naming convention, since that
# convention didn't exist yet when they were produced (baseline/
# krum_baseline in particular didn't exist as named ABLATION_MODE
# values in main.py at all before this change -- those 2 modes' existing
# results came from some other/older script version). Before the later
# averaging step, you'll need to either rename/copy those 10 existing
# CSVs to match (results_{model}_{mode}_seed42.csv) or tell me their
# real filenames so the averaging script can find them as-is.
#
# 5 ablation modes total, but this script only runs 4 new seeds x 2
# models x 5 modes = 40 NEW runs (not 50) -- NOT re-running the epsilon
# sweeps or Experiment 2's exp2_unmitigated/exp2_mitigated, unchanged.
#
# Grouped into 3 tmux sessions. Within each session, jobs run ONE AT A
# TIME, in order -- the next job only starts once the previous one
# finishes. Sessions themselves run in parallel with each other.
#
#   Session 1  "s1_baseline_krumbaseline"  : baseline + krum_baseline
#              (2 modes x 2 models x 4 seeds = 16 sequential runs)
#   Session 2  "s2_zkp_he"                 : pure_zkp + pure_he
#              (2 modes x 2 models x 4 seeds = 16 sequential runs)
#   Session 3  "s3_dp"                     : pure_dp
#              (1 mode  x 2 models x 4 seeds =  8 sequential runs)
#
# Every job still gets a unique --tag via main.py's own _TAG logic
# ({model}_{ablation-mode}_seed{N}) -- no filename collisions, whether
# jobs run in the same session or different ones.
#
# Usage:
#   1. Edit REPO_DIR / PYTHON / VENV_ACTIVATE below.
#   2. chmod +x launch_multiseed_ablations.sh
#   3. ./launch_multiseed_ablations.sh
#   4. Monitor with: tmux ls
#                     tmux attach -t s1_baseline_krumbaseline
#                     tail -f logs/network_baseline_seed123.log
# ============================================================
set -euo pipefail

# ---- EDIT THESE THREE LINES FOR YOUR SETUP ----
REPO_DIR="/workspace/experiments/Current model"
PYTHON="/usr/bin/python3"
VENV_ACTIVATE=""                           # e.g. "source /path/to/venv/bin/activate" -- leave empty if not using a venv
# ------------------------------------------------

LOG_DIR="${REPO_DIR}/logs"
SCRIPT_DIR="${REPO_DIR}/_session_scripts"
mkdir -p "${LOG_DIR}" "${SCRIPT_DIR}"

MODELS=(network application)
SEEDS=(123 7 2024 31337)   # seed=42 already exists from your prior runs -- not repeated here

# session_name : space-separated ablation modes for that session
declare -A SESSIONS=(
  [s1_baseline_krumbaseline]="baseline krum_baseline"
  [s2_zkp_he]="pure_zkp pure_he"
  [s3_dp]="pure_dp"
)

# ---- Generate one runner script per session ----
for session in "${!SESSIONS[@]}"; do
  modes="${SESSIONS[$session]}"
  runner="${SCRIPT_DIR}/${session}.sh"

  {
    echo "#!/usr/bin/env bash"
    echo "set -uo pipefail   # NOT -e: one failed job shouldn't kill the rest of the queue"
    echo "cd '${REPO_DIR}'"
    if [[ -n "${VENV_ACTIVATE}" ]]; then
      echo "${VENV_ACTIVATE}"
    fi
    echo "echo \"Session ${session} starting: \$(date)\""
    job_num=0
    total_jobs=0
    for mode in ${modes}; do
      for model in "${MODELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          total_jobs=$((total_jobs + 1))
        done
      done
    done
    for mode in ${modes}; do
      for model in "${MODELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          job_num=$((job_num + 1))
          tag="${model}_${mode}_seed${seed}"
          log_file="${LOG_DIR}/${tag}.log"
          echo "echo \"[${session} ${job_num}/${total_jobs}] ${tag} starting: \$(date)\""
          echo "${PYTHON} main.py ${model} --ablation-mode ${mode} --seed ${seed} > '${log_file}' 2>&1"
          echo "echo \"[${session} ${job_num}/${total_jobs}] ${tag} finished: \$(date) (exit code \$?)\""
        done
      done
    done
    echo "echo \"Session ${session} ALL DONE: \$(date)\""
  } > "${runner}"

  chmod +x "${runner}"
  echo "generated ${runner}"
done

# ---- Launch each session's runner in its own tmux session ----
for session in "${!SESSIONS[@]}"; do
  runner="${SCRIPT_DIR}/${session}.sh"
  tmux new-session -d -s "${session}" "bash '${runner}'"
  echo "launched tmux session: ${session}  (jobs: ${SESSIONS[$session]})"
done

echo ""
echo "3 tmux sessions launched. Total NEW jobs across all three: 40."
echo "(Plus your existing 10 seed=42 runs, once renamed to match -- see"
echo " the note at the top of this file -- gives 5 total per condition.)"
echo "Check status any time with:  tmux ls"
echo "Watch a specific job's live log with:  tail -f logs/<model>_<mode>_seed<N>.log"
echo "Kill everything with:  tmux kill-server   (careful -- kills ALL tmux sessions on this machine)"
