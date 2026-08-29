#!/usr/bin/env bash
# Runs the full RAM/timing test matrix:
#   modes:    baseline, he_full, he_partial, he_partial_zkp, dp
#   profiles: unthrottled (700MB/1.0 vCPU), throttled (400MB/0.5 vCPU)
# for the given MODEL_TYPE (default: network, matching the old ablations'
# scope -- pass "application" as $1 to run that model too).
#
# Usage:
#   ./run_suite.sh [network|application]
#
# Prereqs:
#   1. Build partitions first (once per model type), on the HOST, not in
#      Docker (needs pandas/sklearn):
#        cd offline && pip install -r ../requirements-offline.txt
#        python build_docker_partitions.py --model network --rows 100000 --clients 2
#   2. Docker + docker compose installed and running.
set -euo pipefail

MODEL_TYPE="${1:-network}"
MODES=(baseline he_full he_partial he_partial_zkp dp)
PROFILES=(unthrottled throttled)

if [ ! -f "partitions/${MODEL_TYPE}/manifest.json" ]; then
  echo "ERROR: partitions/${MODEL_TYPE}/manifest.json not found."
  echo "Run the offline partition builder first -- see this script's header comment."
  exit 1
fi

mkdir -p results

for mode in "${MODES[@]}"; do
  for profile in "${PROFILES[@]}"; do
    RUN_TAG="${MODEL_TYPE}_${mode}_${profile}"
    echo ""
    echo "=============================================="
    echo " RUN: ${RUN_TAG}"
    echo "=============================================="

    export MODE="${mode}"
    export MODEL_TYPE="${MODEL_TYPE}"
    export RUN_TAG="${RUN_TAG}"

    if [ "${profile}" == "throttled" ]; then
      COMPOSE_FILES="-f docker-compose.yml -f docker-compose.throttled.yml"
    else
      COMPOSE_FILES="-f docker-compose.yml"
    fi

    mkdir -p "results/${RUN_TAG}"

    # Run both clients to completion.
    docker compose ${COMPOSE_FILES} up --build --abort-on-container-exit client0 client1

    # Run the server-side aggregation/timing step (Krum for baseline/dp,
    # HE-aggregate+decrypt [+ZKP verify/threshold] for he_* modes).
    docker compose ${COMPOSE_FILES} --profile server run --rm server

    docker compose ${COMPOSE_FILES} down -v

    echo " Done: ${RUN_TAG}  ->  results/${RUN_TAG}/"
  done
done

echo ""
echo "All runs complete. Results in ./results/<model>_<mode>_<profile>/"
