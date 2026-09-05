#!/usr/bin/env bash
# Runs the full RAM/timing test matrix:
#   modes:    baseline, he_full, he_partial, he_partial_zkp, dp
#   profiles: unthrottled (2GB/1.0 vCPU), throttled (2GB/0.5 vCPU)
# for the given MODEL_TYPE (default: network).
#
# ORCHESTRATION NOTE (fixed from an earlier version): client0 and
# client1 almost never finish at exactly the same time (different
# partition sizes -> different training time). This script brings the
# communication daemon (`server`) up first with --wait, then runs both
# clients as independent one-shot `run --rm` jobs backgrounded and
# waited on -- NOT `up --abort-on-container-exit`, which kills every
# other service the instant the FIRST one exits and was silently
# truncating the slower client mid-run.
#
# Usage:
#   ./run_suite.sh [network|application] [modes,comma,separated] [profiles,comma,separated]
#   ./run_suite.sh network                              # everything
#   ./run_suite.sh network dp                            # just dp, both profiles
#   ./run_suite.sh network dp throttled                  # just dp, throttled only
#   ./run_suite.sh network he_partial,he_partial_zkp      # both, both profiles
#
# Prereqs:
#   1. Build partitions first (once per model type), on the HOST, not
#      in Docker (needs pandas/sklearn) -- see offline/build_docker_partitions.py
#   2. Docker + docker compose installed and running.
set -euo pipefail

MODEL_TYPE="${1:-network}"
# Comma-separated lists, e.g.: ./run_suite.sh network dp throttled
MODES_ARG="${2:-baseline,he_full,he_partial,he_partial_zkp,dp}"
PROFILES_ARG="${3:-unthrottled,throttled}"
IFS=',' read -ra MODES <<< "${MODES_ARG}"
IFS=',' read -ra PROFILES <<< "${PROFILES_ARG}"

if [ ! -f "partitions/${MODEL_TYPE}/manifest.json" ]; then
  echo "ERROR: partitions/${MODEL_TYPE}/manifest.json not found."
  echo "Run the offline partition builder first -- see README.md."
  exit 1
fi

mkdir -p results

echo "Building images once up front..."
docker compose build

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

    # Phase 1a: bring the communication daemon up and wait for it to be
    # healthy BEFORE either client starts (so neither client's startup
    # depends_on check races against the other).
    docker compose ${COMPOSE_FILES} up -d --wait server

    # Phase 1b: run both clients as independent one-shot jobs,
    # concurrently, neither one able to kill the other on exit.
    docker compose ${COMPOSE_FILES} run --rm client0 &
    CLIENT0_PID=$!
    docker compose ${COMPOSE_FILES} run --rm client1 &
    CLIENT1_PID=$!

    CLIENT0_OK=0
    CLIENT1_OK=0
    wait "${CLIENT0_PID}" || CLIENT0_OK=$?
    wait "${CLIENT1_PID}" || CLIENT1_OK=$?

    if [ "${CLIENT0_OK}" -ne 0 ] || [ "${CLIENT1_OK}" -ne 0 ]; then
      echo "WARNING: client0 exit=${CLIENT0_OK} client1 exit=${CLIENT1_OK} -- check output above."
    fi

    # Phase 2: server-side aggregation timing (Krum / HE-aggregate+
    # decrypt / ZKP-verify+threshold), reading the artifacts both
    # clients just wrote to the shared volume. The daemon should have
    # already self-exited (it shuts down once it's received every
    # expected submission) -- `down -v` afterward cleans up regardless.
    docker compose ${COMPOSE_FILES} --profile aggregate run --rm aggregator

    docker compose ${COMPOSE_FILES} down -v

    echo " Done: ${RUN_TAG}  ->  results/${RUN_TAG}/"
  done
done

echo ""
echo "All runs complete. Results in ./results/<model>_<mode>_<profile>/"
echo "Run: python consolidate_results.py results"
