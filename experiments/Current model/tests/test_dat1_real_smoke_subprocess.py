"""
tests/test_dat1_real_smoke_subprocess.py

DAT1 (Mati86 review): a REAL 1-round, 10-client main.py run, with
runtime file-open instrumentation that covers WORKER PROCESSES too.

Why this replaces the old runpy + sys.addaudithook test:
  1. The old test passed `--model-type network`, but main.py takes the
     model type as a POSITIONAL argument (`main.py network`), so
     argparse would have rejected it before training started.
  2. The old hook appended to an in-memory list. Forked workers get
     their own COPY of that list, so the parent never saw anything a
     worker opened -- exactly the processes that run client training.
  3. It only worked under fork. Python 3.14 on Linux defaults to
     forkserver, and Windows/macOS use spawn.

Mechanism here:
  - main.py runs as a real subprocess, inside an ISOLATED COPY of the
    code (only .py/.json files are copied), so the run writes its own
    splits/ and results_*.csv into a temp dir and can never touch your
    real split/scaler artifacts or results.
  - A sitecustomize.py placed first on PYTHONPATH is imported by EVERY
    Python process the run creates (main, spawned, forkserver-ed). It
    installs sys.addaudithook and appends one line per file open to a
    shared log through a raw file descriptor (os.write; this does not
    itself trigger an "open" event). It also logs each process start
    and each os.fork() child, so the test can PROVE workers were
    instrumented rather than assume it.

What is asserted:
  A. Worker processes (every pid other than the main one) never open the
     dataset CSV, a TVT_global_*.npz split artifact, a scalers_*.pkl, or
     any FINAL_TEST / FINAL_VALIDATION result file. Client training runs
     in these workers on CPU; they receive their shard in memory.
  B. The FINAL_TEST csv is first opened only AFTER the last per-round
     log append, i.e. the test evaluation happens after all rounds.
  C. Non-vacuity: the run exited 0, the dataset CSV and the split
     artifact were opened by the main process, the FINAL_TEST csv was
     opened, at least one per-round log append happened, and (on CPU,
     where main.py uses a ProcessPoolExecutor) at least one worker
     process was recorded.

Scope (stated, not hidden): this is FILE-LEVEL evidence. In-memory use of
test arrays inside the main process is not observable through file
opens; it is covered by the static call-site test and the synthetic
partitioning test in test_dat1_smoke_run.py.

Run on a CPU runtime so the worker pool is exercised (a CUDA runtime
trains in-process and has no workers):

    python -m pytest tests/test_dat1_real_smoke_subprocess.py -v -s
"""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

import data_loader as dl

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.abspath(os.path.join(REPO_ROOT, "..", "configs"))

# Unique seed so nothing in this run can collide with a real seed's artifacts.
SMOKE_SEED = 987654

SITECUSTOMIZE = textwrap.dedent('''
    import os, sys
    _log = os.environ.get("DAT1_AUDIT_LOG")
    if _log:
        _fd = os.open(_log, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0))

        def _w(line):
            try:
                os.write(_fd, (line + "\\n").encode("utf-8", "replace"))
            except Exception:
                pass

        def _hook(event, args):
            if event == "open" and args:
                mode = args[1] if len(args) > 1 else ""
                _w("OPEN\\t%d\\t%s\\t%s" % (os.getpid(), mode, args[0]))

        sys.addaudithook(_hook)
        _w("START\\t%d\\t%d" % (os.getpid(), os.getppid()))
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(
                after_in_child=lambda: _w("FORK\\t%d\\t%d" % (os.getpid(), os.getppid())))
''')


def _write_synthetic_edge_iiot_csv(csv_path, rows_per_class=400,
                                   n_numeric_features=15, seed=999):
    """Schema-compatible stand-in for the real Edge-IIoTset CSV. 400 rows
    per class (not 50) so every client shard survives the Dirichlet
    partition and the stratified 90/10 local split."""
    rng = np.random.default_rng(seed)
    rows = []
    for cls in dl.ALL_CLASSES:
        for _ in range(rows_per_class):
            row = {f"feat_{i}": rng.normal() for i in range(n_numeric_features)}
            row["Attack_type"] = cls
            row["Attack_label"] = 0 if cls == "Normal" else 1
            rows.append(row)
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.to_csv(csv_path, index=False)


def _copy_code_isolated(dest_root):
    """Copy only .py/.json files (small) so the run is fully isolated and
    never writes into the real splits/ or results folders."""
    skip_dirs = {"tests", "splits", "__pycache__", "datasets", ".pytest_cache",
                 "_ciciot_cache", ".git"}
    code_dir = os.path.join(dest_root, "Current model")
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        rel = os.path.relpath(root, REPO_ROOT)
        out = code_dir if rel == "." else os.path.join(code_dir, rel)
        os.makedirs(out, exist_ok=True)
        for fn in files:
            if fn.endswith((".py", ".json")):
                with open(os.path.join(root, fn), "rb") as src, \
                        open(os.path.join(out, fn), "wb") as dst:
                    dst.write(src.read())
    cfg_out = os.path.join(dest_root, "configs")
    os.makedirs(cfg_out, exist_ok=True)
    for fn in os.listdir(CONFIGS_DIR):
        if fn.endswith(".json"):
            with open(os.path.join(CONFIGS_DIR, fn), "rb") as src, \
                    open(os.path.join(cfg_out, fn), "wb") as dst:
                dst.write(src.read())
    return code_dir


def _parse_log(log_path):
    opens, procs = [], {}
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f):
            parts = line.rstrip("\r\n").split("\t", 3)
            if parts[0] == "OPEN" and len(parts) == 4:
                opens.append({"i": idx, "pid": int(parts[1]),
                              "mode": parts[2], "path": parts[3]})
            elif parts[0] in ("START", "FORK") and len(parts) >= 3:
                procs[int(parts[1])] = (parts[0], int(parts[2]))
    return opens, procs


def test_real_smoke_run_test_holdout_never_opened_during_training(tmp_path):
    import torch  # fail loudly if torch is missing -- do NOT skip

    work = str(tmp_path / "isolated")
    code_dir = _copy_code_isolated(work)

    csv_path = str(tmp_path / "tiny_edge_iiot.csv")
    _write_synthetic_edge_iiot_csv(csv_path)

    hook_dir = tmp_path / "hook"
    hook_dir.mkdir()
    (hook_dir / "sitecustomize.py").write_text(SITECUSTOMIZE, encoding="utf-8")
    log_path = str(tmp_path / "opens.log")

    env = dict(os.environ)
    env["EDGE_IIOT_CSV_PATH"] = csv_path
    env["DAT1_AUDIT_LOG"] = log_path
    env["PYTHONPATH"] = str(hook_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    proc = subprocess.Popen(
        [sys.executable, "main.py", "network",
         "--seed", str(SMOKE_SEED), "--rounds", "1"],
        cwd=code_dir, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    try:
        out, _ = proc.communicate(timeout=1800)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
        pytest.fail("smoke run timed out after 1800s\n" + out[-3000:])
    main_pid = proc.pid

    assert proc.returncode == 0, (
        f"main.py exited with {proc.returncode}; last output:\n{out[-4000:]}"
    )

    opens, procs = _parse_log(log_path)
    csv_name = os.path.basename(csv_path)

    def is_protected(p):
        b = os.path.basename(p)
        return (b == csv_name
                or b.startswith("TVT_global_")
                or b.startswith("scalers_")
                or "FINAL_TEST" in b
                or "FINAL_VALIDATION" in b)

    # ---- C. non-vacuity -------------------------------------------------
    main_opens = [o for o in opens if o["pid"] == main_pid]
    assert any(os.path.basename(o["path"]) == csv_name for o in main_opens), (
        "dataset CSV never opened by the main process -- hook not working "
        "or EDGE_IIOT_CSV_PATH not honoured")
    assert any(os.path.basename(o["path"]).startswith("TVT_global_")
               for o in main_opens), "split artifact never opened by main process"

    final_test_opens = [o for o in opens if "FINAL_TEST" in os.path.basename(o["path"])]
    assert final_test_opens, "FINAL_TEST csv never opened -- run did not finish"

    round_log_appends = [
        o for o in opens
        if os.path.basename(o["path"]).startswith("results_")
        and os.path.basename(o["path"]).endswith(".csv")
        and "FINAL" not in os.path.basename(o["path"])
        and "a" in str(o["mode"])
    ]
    assert round_log_appends, "no per-round log append recorded"

    workers = {pid for pid in procs if pid != main_pid}
    if not torch.cuda.is_available():
        assert workers, (
            "no worker process was recorded, but on CPU main.py trains clients "
            "in a ProcessPoolExecutor -- the audit did not cover them")

    # ---- A. workers never touch protected files -------------------------
    worker_hits = [o for o in opens
                   if o["pid"] != main_pid and is_protected(o["path"])]
    assert not worker_hits, (
        "a WORKER process (client training) opened a protected data/split/"
        f"test file: {worker_hits}")

    # ---- B. test evaluation only after the last training round ----------
    assert min(o["i"] for o in final_test_opens) > max(o["i"] for o in round_log_appends), (
        "FINAL_TEST csv was opened before the last per-round log append -- "
        "test evaluation happened before training finished")

    print("\n=== DAT1 smoke-run audit ===")
    print(f"python            : {sys.version.split()[0]}  torch {torch.__version__}")
    print(f"platform          : {sys.platform}")
    print(f"main pid          : {main_pid}")
    print(f"worker pids seen  : {sorted(workers)} "
          f"({'CPU pool' if not torch.cuda.is_available() else 'CUDA: in-process'})")
    print(f"total open events : {len(opens)}")
    print(f"worker opens of protected files : 0")
    print(f"per-round log appends : {len(round_log_appends)}; "
          f"FINAL_TEST first opened at log line {min(o['i'] for o in final_test_opens)} "
          f"(last round append at {max(o['i'] for o in round_log_appends)})")
