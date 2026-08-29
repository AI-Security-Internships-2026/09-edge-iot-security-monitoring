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

| Mode | What it times |
|---|---|
| `baseline` | Local training only (control condition) |
| `he_full` | Local training + full-model CKKS encryption + server aggregate/decrypt |
| `he_partial` | Local training + classifier-head-only CKKS encryption + server aggregate/decrypt |
| `he_partial_zkp` | Same as `he_partial`, **plus** the ciphertext-bound head-norm proof (client-side) and its server-side verify + MAD-threshold (isolated as separate stages, so you can see the ZKP guard's added cost on top of partial HE alone) |
| `dp` | Local training wrapped in Opacus DP-SGD (target ε=3.0, δ=1e-5, max_grad_norm=1.0, matching the old `pure_dp` Docker test's own config) |

Every mode also runs at two resource profiles:
- **unthrottled**: 700MB / 1.0 vCPU (matches the old `pure_he` unthrottled condition)
- **throttled**: 400MB / 0.5 vCPU (matches the old `pure_he` "constrained IoT gateway" condition)

This gives you a consistent grid across all five mechanisms at both
resource levels — the old runs used different, ad hoc limits per test
(500m for He-Full/Partial, a possibly-unenforced 256MB for `pure_dp`,
700MB/400MB for the throttle test). Standardizing here is a deliberate
improvement, not a hidden deviation — flagging it so you can decide if
you want it.

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
docker compose build
./run_suite.sh network        # or: ./run_suite.sh application
```

This runs all 5 modes × 2 resource profiles (10 runs total) for the
given model, writing to `results/<model>_<mode>_<profile>/`.

### 3. Consolidate

```bash
python consolidate_results.py results
```

Writes `results/CONSOLIDATED_SUMMARY.json` and prints a summary table.

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
  client_runner.py             <- NEW: per-client test driver
  server_aggregate.py          <- NEW: HE aggregate/decrypt, ZKP verify/threshold, Krum timing
Dockerfile
docker-compose.yml              <- unthrottled base (700MB/1.0 vCPU)
docker-compose.throttled.yml    <- throttled override (400MB/0.5 vCPU)
requirements-container.txt      <- torch(cpu)/tenseal/opacus/numpy only
requirements-offline.txt        <- pandas/sklearn/numpy, host-side only
run_suite.sh                    <- runs the full mode x profile matrix
consolidate_results.py          <- collects results into one summary
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
