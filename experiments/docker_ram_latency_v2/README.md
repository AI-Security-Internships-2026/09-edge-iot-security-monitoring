# Docker RAM/Latency Test Suite — Rerun (v2)

RAM and timing only. No accuracy claims are made or measured anywhere
in this suite — that's a deliberate scope decision, not an oversight.

## What this replaces

The old He-Full/He-Partial ablation measured 35 features because it ran
against a pre-text-feature-engineering / pre-label-fix snapshot of the
pipeline, not because of the row-subsample size. This rerun keeps the
**same data volume** (a 100k-row stratified subsample, same as before)
but runs it through the **current**, label-corrected `data_loader.py`
pipeline, so the resulting feature count and `total_params` are
measured fresh rather than assumed — see `partitions/<model>/manifest.json`
after building.

## What this measures

Every round, for every mode, captures **all** of: training time, the
mode-specific mechanism's time, communication time, and (in a separate
server-side pass) aggregation time. Nothing is measured in isolation.

| Mode | Client-side per round | Server-side (separate pass) |
|---|---|---|
| `baseline` | train | Krum timing (see scope note below) |
| `he_full` | train, full-model CKKS encrypt | HE aggregate + decrypt |
| `he_partial` | train, classifier-head-only CKKS encrypt | HE aggregate + decrypt |
| `he_partial_zkp` | train, partial encrypt, ciphertext-bound head-norm proof (isolated as its own stage) | HE aggregate + decrypt, ZKP verify + MAD threshold |
| `dp` | Opacus DP-SGD, split into `dp_setup` (PrivacyEngine calibration) and `dp_train` (the actual per-sample-gradient loop) as separate stages | Krum timing |

**Communication is measured on every round of every mode**, via a real
network round trip — not a shared-volume file write. A `server_daemon`
container runs concurrently with both clients for the whole run,
listening on the Docker compose network; each client serializes its
round's artifact (payload = the actual ciphertext dict for `he_*`
modes, or the plaintext param list for `baseline`/`dp`), times the
serialization, then POSTs it and times the full request/response round
trip. Recorded per round: `serialize_time_s`, `communication_send_time_s`,
`payload_bytes`. The daemon separately logs its own receive-side timing
(`server_communication_summary.json`) so you can compare client-perceived
vs. server-perceived communication cost if they diverge.

Every mode also runs at two resource profiles:
- **unthrottled**: 2GB RAM / 1.0 vCPU
- **throttled**: 2GB RAM / 0.5 vCPU (matches the old `pure_he` "constrained IoT gateway" CPU condition)

**RAM is capped at 2GB in every profile, not varied.** This is
deliberately a "don't crash, just measure peak/avg" ceiling rather than
a tight simulated constraint — the old ablations used inconsistent,
sometimes-too-tight limits (500m, a possibly-unenforced 256MB, 400MB)
that risked OOM kills mid-run. Only CPU is throttled between the two
profiles now; RAM is generous everywhere and `ram_peak_mb`/`ram_avg_mb`
are reported for every run so you can see actual usage against real
headroom instead of against an artificially tight ceiling.

**Krum timing** (`baseline`/`dp` modes' server step) and the **ZKP MAD
threshold** (`he_partial_zkp` mode's server step) both need more than 2
clients to be meaningful (Krum needs `n - f - 2 > 0`; a 2-point MAD is
degenerate). Since only 2 real Docker clients run — matching the old
ablations' own 2-client setup — this suite **synthesizes** additional
"clients" by jittering copies of the real, correctly-shaped/correctly-
sized trained parameters up to `SYNTHETIC_N_CLIENTS` (default 10,
matching the main pipeline's `NUM_CLIENTS`). This keeps model
dimensionality and client count realistic (the two things that actually
drive Krum's/the MAD-threshold's runtime) without claiming anything
about accuracy or detection for the synthetic copies. This is flagged
in the code, in every relevant output JSON's `note` field, and here —
flag it again in any write-up: **Krum/ZKP-threshold timing numbers
describe algorithm cost at a realistic (n, d), not a detection-rate
result.**

## Fixes included vs. the old suite

- **`mem_limit_mb` self-report bug (Contradiction #18) — fixed.** The
  old `client.py` hardcoded `"200"` into every result JSON regardless
  of the real limit. This suite reads the actual cgroup-enforced memory
  and CPU limits directly from `/sys/fs/cgroup` at runtime
  (`mem_profiler.read_cgroup_memory_limit_mb()` /
  `read_cgroup_cpu_limit()`) — real numbers, not a stale label. If a
  limit genuinely isn't enforced/readable, it reports `None` rather
  than guessing.
- **`total_params` mismatch (Contradiction #19)** — this rerun measures
  `total_params` directly from the model instantiated at whatever
  feature count the current pipeline actually produces on this data
  volume, recorded per-run in each client's result JSON. It won't
  silently inherit either of the old conflicting figures (129,352 /
  80,074).
- **CKKS parameters** use `he_aggregation.py`'s own historical
  Docker-tuned defaults (`poly_modulus_degree=4096`, depth-1 chain,
  64-bit security) — **not** `he_local.py`'s 8192/128-bit "local
  research" defaults, which would reject at n=4096 anyway. This matches
  what the old Docker ablations actually used.

## Setup

### 1. Offline: build partitions (host, needs pandas/sklearn)

```bash
cd offline
pip install -r ../requirements-offline.txt
python build_docker_partitions.py --model network --rows 100000 --clients 2
# and/or:
python build_docker_partitions.py --model application --rows 100000 --clients 2
```

This expects the raw CSV at the same relative path `data_loader.py`
already uses (`<repo_root>/datasets/Edge-IIoTset dataset/Selected dataset
for ML and DL/DNN-EdgeIIoT-dataset.csv`) — copy `data_loader.py` back
into its usual location relative to that dataset, or pass `--out` to
point the partition output wherever you like and adjust
`docker-compose.yml`'s `PARTITION_DIR` volume mount to match.

Check `partitions/<model>/manifest.json` afterward — it records the
**actual measured** feature count and per-client row counts for this
run. Don't assume 39/90 or any prior figure; read it from here.

### 2. Docker: run the full suite

```bash
./run_suite.sh network        # or: ./run_suite.sh application
```
(Windows: `powershell -ExecutionPolicy Bypass -File .\run_suite.ps1 network`)

Builds the images once, then runs all 5 modes × 2 resource profiles
(10 runs total) for the given model. Each run has two phases:
1. `server` (communication daemon) comes up first and waits until
   healthy. `client0` and `client1` then run as two independent
   one-shot jobs, concurrently. **They are deliberately NOT run via
   `docker compose up --abort-on-container-exit`** — client0 and
   client1 almost never finish at the same time (different partition
   sizes → different training time per round), and that flag kills
   every other service the instant the *first* one exits, silently
   truncating whichever client finishes second mid-run. Running them
   as independent `run --rm` jobs means neither can kill the other.
   The daemon self-exits once it's received every expected submission.
2. The `aggregator` service runs once, reading the artifacts both
   clients just wrote to the shared volume, timing HE aggregate/
   decrypt, ZKP verify/threshold, or Krum depending on mode.

Results land in `results/<model>_<mode>_<profile>/`, including
`server_communication_summary.json` (daemon's receive-side view) and
`server_<mode>_results.json` (aggregation timing).

If you ran an earlier version of this suite and hit
`FileNotFoundError: Missing artifact: .../client_0_round_N_artifact.json`
from the aggregator, that was this exact bug — delete `results/` and
rerun; the fix is in `run_suite.sh`/`run_suite.ps1` now.

### 3. Verify BEFORE exporting/zipping anything

```bash
python verify_results.py results
```

**Run this before you zip or upload any results folder, every time.**
It checks each `results/<tag>/` folder's own internal `mode`/CPU config
against what the folder name claims, flags any run that's missing
rounds (a client that crashed before writing its final summary),
flags any run missing its server-side aggregation file, and — this one
matters — hashes every folder's `client_0_results.json` and flags any
two folders that are byte-identical, since that means one of them is a
duplicate export, not a second real run. Fix or rerun anything flagged
here before it goes anywhere near a paper table.

You can target just the specific mode/profile combos that need
rerunning instead of the whole matrix:
```bash
./run_suite.sh network dp throttled           # just DP, throttled only
./run_suite.sh network he_partial,he_partial_zkp   # both profiles, both modes
```
(Windows: `.\run_suite.ps1 -ModelType network -Modes dp -Profiles throttled`)

### 4. Consolidate

```bash
python consolidate_results.py results
```

Writes `results/CONSOLIDATED_SUMMARY.json` and prints a summary table.
Only run this after `verify_results.py` reports everything clean —
consolidating over a mismatched or duplicated folder will silently
produce a wrong-but-plausible-looking summary.

## Files

```
offline/
  data_loader.py               <- unmodified copy, host-side only
  build_docker_partitions.py   <- subsamples + partitions (see above)
src/                            <- goes into the container image
  model_defs.py                <- unmodified copy
  krum.py                      <- unmodified copy (adaptive_multi_krum only — multi_krum() doesn't exist, per Contradiction #15, and isn't used here)
  zkp.py                       <- unmodified copy
  he_aggregation.py            <- unmodified copy
  he_local.py                  <- import paths patched for flat container layout
  mem_profiler.py              <- NEW: RAM sampler, stage timer, real cgroup-limit reader
  client_runner.py             <- NEW: per-client test driver (train, mechanism, communication)
  server_daemon.py             <- NEW: concurrent communication-timing daemon
  server_aggregate.py          <- NEW: one-shot aggregation timing (HE/ZKP/Krum)
Dockerfile
docker-compose.yml              <- unthrottled base (2GB/1.0 vCPU); services: server, client0, client1, aggregator
docker-compose.throttled.yml    <- CPU-only override (0.5 vCPU, RAM stays 2GB)
requirements-container.txt      <- torch(cpu)/tenseal/opacus/numpy only
requirements-offline.txt        <- pandas/sklearn/numpy, host-side only
run_suite.sh                    <- bash orchestrator (Linux/macOS/WSL/Git Bash)
run_suite.ps1                   <- PowerShell orchestrator, same logic (native Windows)
consolidate_results.py          <- collects results into one summary
verify_results.py               <- run BEFORE zipping/uploading: catches mislabeled/duplicate/incomplete folders
```

## What this suite deliberately does NOT do

- No accuracy/F1 measurement anywhere — RAM and timing only, per scope.
- No `multi_krum()` (fixed-m) — it doesn't exist in the uploaded
  codebase (Contradiction #15); only `adaptive_multi_krum()` is timed,
  per your explicit instruction to use adaptive Krum.
- No real multi-client FL loop — each Docker client trains
  independently for `ROUNDS` rounds against its own partition; there is
  no cross-client FedAvg/FedProx global-model exchange between rounds
  in this harness, since the goal is per-mechanism timing/RAM, not a
  reproduction of the main experiment pipeline's training dynamics.
