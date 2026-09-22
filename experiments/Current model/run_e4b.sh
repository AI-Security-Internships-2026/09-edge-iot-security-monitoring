#!/usr/bin/env bash
# E4b runner -- DP architecture confound (network model only).
#   A: BatchNorm + LSTM, no DP           (--ablation-mode baseline)
#   B: GroupNorm + DPLSTM, no DP         (--ablation-mode baseline --force-dp-safe-arch)
#   C: GroupNorm + DPLSTM + DP-SGD       (--ablation-mode pure_dp --epsilon 5)
# Identical across arms: FedProx mu=0.005, alpha=0.7, no Byzantine attack, 25 rounds.
#
# Usage (run from the repo root, next to main.py):
#   ./run_e4b.sh --smoke                       # 1-round arm-C plumbing check (run this first)
#   ./run_e4b.sh --arms "A C" --seeds "42 123 456 789 2024"
#   ./run_e4b.sh --arms B --seeds 42           # clean rerun of the resumed arm-B seed
# Split seeds across machines by passing different --seeds lists.
# Set FORCE=1 to overwrite a cell that already has a FINAL_TEST file.
set -euo pipefail

MAIN="main.py"; MODEL="network"; MU="0.005"; ALPHA="0.7"; EPS="5"
OUT_ROOT="experiments/results/E4b"
ARMS="A C"; SEEDS="42 123 456 789 2024"; SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arms)  ARMS="$2";  shift 2;;
    --seeds) SEEDS="$2"; shift 2;;
    --main)  MAIN="$2";  shift 2;;
    --smoke) SMOKE=1;    shift;;
    *) echo "unknown arg $1"; exit 2;;
  esac
done

tag_for()   { case "$1" in A) echo e4b_A_bn_nodp;; B) echo e4b_B_gn_nodp;; C) echo "e4b_C_gn_dp_eps${EPS}";; esac; }
flags_for() {
  case "$1" in
    A) echo "--ablation-mode baseline";;
    B) echo "--ablation-mode baseline --force-dp-safe-arch";;
    C) echo "--ablation-mode pure_dp --epsilon ${EPS}";;
  esac
}

run_one() {
  local arm="$1" seed="$2" extra="${3:-}" tag_override="${4:-}"
  local tag; tag="${tag_override:-$(tag_for "$arm")}"
  local stem="${MODEL}_${tag}_seed${seed}"
  local dest="${OUT_ROOT}/${seed}/${tag}"
  if [[ -z "$tag_override" && -s "${dest}/results_${stem}_FINAL_TEST.csv" && "${FORCE:-0}" != "1" ]]; then
    echo "[skip] ${arm} seed ${seed}: ${dest} already has FINAL_TEST (FORCE=1 to overwrite)"; return 0
  fi
  # Always start from round 1: DP runs cannot resume, and A/B must be clean too.
  rm -f "checkpoint_${stem}.npz" "checkpoint_${stem}_progress.json" \
        "checkpoint_${stem}_best.npz" "checkpoint_${stem}_best.json" "results_${stem}.csv"
  echo "=== E4b arm ${arm}  seed ${seed}  tag ${tag} ==="
  # shellcheck disable=SC2086
  python "$MAIN" "$MODEL" $(flags_for "$arm") \
      --prox-mu "$MU" --alpha "$ALPHA" --seed "$seed" --tag "$tag" $extra \
      2>&1 | tee "log_${stem}.txt"
  mkdir -p "$dest"
  shopt -s nullglob
  for f in "results_${stem}"*.csv "per_client_krum_scores_${stem}.csv" \
           "experiment_config_${stem}.json" "dp_final_epsilon_${stem}".* "log_${stem}.txt"; do
    mv -f "$f" "$dest/"
  done
  rm -f "checkpoint_${stem}"*.npz "checkpoint_${stem}"*.json
  shopt -u nullglob
  echo "[done] ${arm} seed ${seed} -> ${dest}"
}

if [[ "$SMOKE" == "1" ]]; then
  # Plumbing check for arm C: are ALL 10 clients DP-active when the attack is off?
  # NOTE: this 1-round run still writes a FINAL_TEST file (main.py always does).
  # It is a throwaway; never read it or use it for any decision.
  run_one C 42 "--rounds 1" "e4b_C_smoke"
  d="${OUT_ROOT}/42/e4b_C_smoke"; stem="${MODEL}_e4b_C_smoke_seed42"
  n_states=$(grep -cE "Client +[0-9]+: sample_rate=" "${d}/log_${stem}.txt" || true)
  n_eps=$(( $(wc -l < "${d}/dp_final_epsilon_${stem}.csv") - 1 ))
  echo "DP client states printed: ${n_states}   final-epsilon rows: ${n_eps}"
  if [[ "$n_states" == "10" && "$n_eps" == "10" ]]; then
    echo "SMOKE PASS: all 10 clients are DP-active in pure_dp."
    rm -rf "$d"
  else
    echo "SMOKE FAIL: expected 10/10. Byzantine-labelled clients 1,2 are being excluded from DP."
    echo "Apply the one-line patch in main.py (pass [] instead of BYZANTINE_CLIENTS to"
    echo "build_dp_client_states when USE_BYZANTINE_ATTACK is False) and rerun --smoke."
    exit 1
  fi
  exit 0
fi

for seed in $SEEDS; do
  for arm in $ARMS; do
    run_one "$arm" "$seed"
  done
done
