#!/usr/bin/env bash
# Runs the E8 Docker benchmark cells (2 client containers + 1 server container each; --cpus=<profile>, --memory=2048m).
# Run from anywhere:  bash e8_harness/run_e8.sh [mode ...]     modes: fedavg adaptive_krum calibrated_krum dp_calibrated_krum
# Environment (all optional):
#   DATA_DIR     host folder holding the Edge-IIoTset CSV; mounted READ-ONLY at the SAME path inside the container
#   TRAIN_ROWS   per-client train rows (set to the legacy client_0_results.json "train_rows")   TEST_ROWS likewise
#   DATA_SOURCE  auto (default) | real | synthetic       PROFILES="1.0 0.5"       RESULTS_ROOT=./e8_results
set -euo pipefail
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJECT_DIR="$(dirname "$HARNESS_DIR")"
IMAGE="${IMAGE:-flids-e8:latest}"; NET="e8bench-net"; RESULTS_ROOT="${RESULTS_ROOT:-$(pwd)/e8_results}"
read -r -a PROFILES <<< "${PROFILES:-1.0 0.5}"
MODES=("$@"); [ ${#MODES[@]} -eq 0 ] && MODES=(fedavg adaptive_krum calibrated_krum dp_calibrated_krum)
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "Building $IMAGE ..."; docker build -t "$IMAGE" "$HARNESS_DIR"; }
docker network inspect "$NET" >/dev/null 2>&1 || docker network create "$NET" >/dev/null
DATA_ARGS=(); [ -n "${DATA_DIR:-}" ] && DATA_ARGS=(-v "$DATA_DIR:$DATA_DIR:ro" -e "DATA_DIR=$DATA_DIR")
EXTRA=(--data-source "${DATA_SOURCE:-auto}"); [ -n "${TRAIN_ROWS:-}" ] && EXTRA+=(--train-rows "$TRAIN_ROWS"); [ -n "${TEST_ROWS:-}" ] && EXTRA+=(--test-rows "$TEST_ROWS")
COMMON=(--rm --network "$NET" -e PROJECT_ROOT=/app -v "$PROJECT_DIR:/app:ro")

for mode in "${MODES[@]}"; do for prof in "${PROFILES[@]}"; do
  tag="${mode}_${prof}vcpu"; out="$RESULTS_ROOT/$tag"; mkdir -p "$out"; cfg="/app/e8_harness/configs/$tag.json"
  echo "=== $tag ==="
  LIM=(--cpus="$prof" --memory=2048m --memory-swap=2048m)
  docker run "${COMMON[@]}" "${LIM[@]}" --name "e8-server-$tag" -v "$out:/results" "$IMAGE" \
      python /app/e8_harness/docker_bench_server.py --config "$cfg" --port 9000 --out-dir /results > "$out/server.log" 2>&1 &
  SPID=$!; sleep 3
  CPIDS=()
  for cid in 0 1; do
    docker run "${COMMON[@]}" "${LIM[@]}" "${DATA_ARGS[@]}" --name "e8-client$cid-$tag" -v "$out:/results" "$IMAGE" \
      python /app/e8_harness/docker_bench_client.py --config "$cfg" --client-id "$cid" --server-host "e8-server-$tag" \
      --server-port 9000 --out-dir /results "${EXTRA[@]}" > "$out/client$cid.log" 2>&1 &
    CPIDS+=($!)
  done
  for p in "${CPIDS[@]}"; do wait "$p" || echo "WARNING: a client container failed for $tag (see $out/client*.log)"; done
  wait "$SPID" || echo "WARNING: server container failed for $tag (see $out/server.log)"
  ls "$out"/*.json >/dev/null 2>&1 && echo "done: $out" || echo "NO RESULT FILES for $tag"
done; done
docker network rm "$NET" >/dev/null 2>&1 || true
echo "Finished. Send back the whole folder: $RESULTS_ROOT"
