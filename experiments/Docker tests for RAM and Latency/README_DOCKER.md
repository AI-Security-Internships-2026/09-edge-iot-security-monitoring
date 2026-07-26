# Running the FL-IDS 2-client IoT simulation on Docker

This folder is a complete, ready-to-build package: `Dockerfile`,
`docker-compose.yml`, and all the python files from the last round of
fixes, arranged exactly the way the code expects (`src/` → `/app` in
the container, everything else → `/workspace`).

## 1. Prerequisites

- Docker Desktop (or Docker Engine + Compose plugin) installed and running.
- Check with:
  ```
  docker --version
  docker compose version
  ```
  (If you only have the older standalone `docker-compose`, replace
  `docker compose` with `docker-compose` in everything below — same
  behavior.)

## 2. What's in `docker-compose.yml`

Three containers on one virtual network:
- `fl_server` — no memory limit (matches the "cloud, no RAM limit" design)
- `fl_client_0`, `fl_client_1` — **`mem_limit: 256m`**, `memswap_limit: 256m`

`memswap_limit` = `mem_limit` is deliberate: it disables swap for these
containers, so if a client actually exceeds 256MB it gets OOM-killed
immediately (exit code 137) instead of silently swapping to disk and
giving you misleadingly "fine" numbers. That's the honest way to test
whether it fits in a real constrained gateway.

It's currently set to 3 rounds, 2 local epochs, synthetic data
(no dataset volume mounted — see §5) so your first run is fast. Bump
`NUM_ROUNDS` / `LOCAL_EPOCHS` back up once you've confirmed it works.

## 3. Build and run

From this folder:

```bash
docker compose build
docker compose up
```

First build will take a few minutes (downloading the CPU-only torch
wheel, ~200MB, plus TenSEAL). Subsequent builds are cached and fast.

You'll see interleaved logs from all three containers — server
aggregation logs, and both clients' per-round training/DP/ZKP/HE logs
with a live RAM number in every line, e.g.:

```
fl_client_0  | [Client 0][INFO][187MB] Training done: 4.2s  RAM=187MB ...
fl_client_0  | [Client 0][INFO][201MB] HE encrypt done: 0.31s  chunks=3  RAM=201MB
```

## 4. Watching real container memory live (the actual thing you asked to check)

The RAM numbers printed by the app are `psutil` reading its own
process — useful, but the number that actually matters for "does this
fit in a 200-300MB IoT gateway" is what Docker's cgroup reports for
the **whole container**, including the `train_worker.py` subprocess
while it's alive. Open a second terminal and run:

```bash
docker stats fl_client_0 fl_client_1
```

This updates live and shows `MEM USAGE / LIMIT` (e.g. `238MiB / 256MiB`)
and `MEM %`. Watch it through one full round — you should see it climb
during the training phase, **drop back down** the moment
`train_worker.py` exits (this is the subprocess-isolation payoff:
memory is actually returned to the OS, not just `del`-eted and left in
Python's allocator arenas), then tick up again briefly during the HE
step.

If a client gets killed, `docker compose ps` will show it `Exited (137)`
— that's the OOM killer, not a bug. Lower `NUM_ROUNDS`/`LOCAL_EPOCHS`
won't help that (rounds don't stack memory, subprocess exit resets it);
what would help is dropping `mem_limit` back up towards 300m first to
confirm the pipeline works, then tightening down.

## 5. Where the numbers you asked about land

Every container writes to `./results` on your host (mounted via
`volumes:`), so after the run finishes:

```bash
cat results/client_0_results.json
cat results/client_1_results.json
cat results/server_timing.json
```

`client_X_results.json` has, per round: `train`, `dp`, `zkp`,
`he_encrypt` timings in seconds, and `peak` RAM in MB — this is your
direct answer to "how long does homomorphic encryption take." With the
partial-HE change (only the ~4,680-parameter classifier head is
encrypted, not the full 79,688-parameter model), expect `he_encrypt`
to be well under a second per round on a laptop CPU — the earlier
full-model HE pass was the expensive part this fix specifically cut.

`server_timing.json` has `aggregate_s` per round — how long the
server-side homomorphic decrypt + merge took.

## 6. Using the real dataset instead of synthetic data

Right now `USE_LOCAL_DP`/`USE_ZKP`/`USE_HE` are on but there's no
`/datasets` volume mounted, so `load_data()` falls back to synthetic
data automatically (see the log line `Using synthetic data (dataset
not mounted)`) — fine for timing/RAM testing, not for real accuracy
numbers. To use Edge-IIoTset, add this under each client in
`docker-compose.yml`:

```yaml
    volumes:
      - ./results:/results
      - /path/on/your/machine/to/Edge-IIoTset:/datasets
```

You'll also need `data_loader.py` (referenced by `client.py` but not
part of the files I've been given) present at `/app/data_loader.py`
— copy your existing one into `src/` before rebuilding.

## 7. Cleaning up

```bash
docker compose down
```

Add `-v` if you also want to remove the results volume mount's
contents (it won't — that's a host bind mount, not a docker volume,
so `./results` on your machine is untouched either way; `-v` here only
affects anonymous/named volumes, of which there are none in this
compose file).

## 8. Troubleshooting

- **TenSEAL fails to build during `docker compose build`**: the
  Dockerfile already installs `build-essential`/`cmake` as a source-build
  fallback, but on some ARM hosts (Apple Silicon) you may need
  `docker compose build --platform linux/amd64` to force the manylinux
  wheel path instead of a native arm64 build.
- **Client exits immediately with a traceback about `data_loader`**:
  that only happens if a `/datasets` volume is mounted but
  `data_loader.py` isn't in `src/` — either add the file or remove the
  volume mount (synthetic fallback needs no such file).
- **Both clients stuck at "Waiting for server..."**: `fl_server` is
  still building/starting — `depends_on` only waits for the container
  to *start*, not for Flask inside it to be ready; the client's own
  retry loop (60s timeout) handles this, just give it a few seconds.
