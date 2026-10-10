#!/usr/bin/env python3
"""
dat1_grep_audit.py -- produces DAT1_grep_audit.txt, the saved evidence for
the acceptance item:

    grep -n for fit-then-transform calls in experiments/Current model/*.py.
    Every call must act on a variable whose data lineage traces to TRAIN
    rows only. Any such call on the pre-split full X_net_raw / X_app
    matrices = blocker.

(NOTE: this file deliberately never contains the literal searched-for text,
so the repo-wide audit test does not flag the audit script itself.)

Run from experiments/Current model:

    python dat1_grep_audit.py            # audit + the 3 audit unit tests
    python dat1_grep_audit.py --no-tests # audit only

Exit code 0 = PASS, 1 = FAIL. Output is written to DAT1_grep_audit.txt
(UTF-8) and echoed to the terminal.

The file records: timestamp, git commit, python version, SHA-256 of every
scanned file (so the evidence is tied to the exact code), the raw
grep -n style hits, each hit's enclosing function and argument, the
verdict, and (optionally) the audit tests' pytest output.
"""
import datetime
import hashlib
import os
import re
import subprocess
import sys

PATTERN = ".fit_" + "transform("
OUT_FILE = "DAT1_grep_audit.txt"

# file -> function that must ENCLOSE its (single) fit_transform call.
# Each is proven TRAIN-only by its own lineage test.
ALLOWED = {
    "data_loader.py": "_fit_or_load_scalers",
    "ciciot2023_loader.py": "load_and_preprocess_ciciot2023",
}

# Exact variable names that would mean fitting on the pre-split full matrix.
FORBIDDEN_ARGS = {"X_net_raw", "X_app", "X_app_raw", "X_all", "X_full",
                  "X_raw", "X", "X_rest", "X_val", "X_test"}

SKIP_DIRS = {"tests", "__pycache__", ".git", ".pytest_cache", "splits",
             "datasets", "_ciciot_cache"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    try:
        r = subprocess.run(["git", *args], capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else "(unavailable)"
    except Exception:
        return "(git not available)"


def enclosing_function(lines, idx):
    for j in range(idx, -1, -1):
        m = re.match(r"def\s+(\w+)\s*\(", lines[j])
        if m:
            return m.group(1)
    return "(module level)"


def call_argument(line):
    m = re.search(r"\.fit_transform\(\s*([^),]+)", line)
    return m.group(1).strip() if m else "?"


def collect_files(root):
    top = sorted(f for f in os.listdir(root)
                 if f.endswith(".py") and os.path.isfile(os.path.join(root, f)))
    extra = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        if os.path.abspath(dirpath) == os.path.abspath(root):
            continue
        for f in sorted(files):
            if f.endswith(".py"):
                extra.append(os.path.relpath(os.path.join(dirpath, f), root))
    return top, extra


def main():
    run_tests = "--no-tests" not in sys.argv
    root = os.getcwd()
    out = []
    p = out.append

    top, extra = collect_files(root)
    p("DAT1 GREP AUDIT -- fit-transform call lineage")
    p("=" * 70)
    p(f"timestamp (UTC) : {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}")
    p(f"directory       : {root}")
    p(f"git commit      : {git('rev-parse', 'HEAD')}")
    dirty = git("status", "--porcelain")
    p(f"git working tree: {'CLEAN' if dirty == '' else 'HAS UNCOMMITTED CHANGES -> commit before posting this as evidence'}")
    p(f"python          : {sys.version.split()[0]}")
    p("equivalent cmd  : grep -n \"" + PATTERN.replace(".", "\\.", 1).replace("(", "\\(") + "\" *.py   "
      f"({len(top)} top-level .py files"
      f"{', + ' + str(len(extra)) + ' in subfolders' if extra else ''})")
    p("")
    p("Scanned files (SHA-256):")
    for f in top + extra:
        p(f"  {sha256(os.path.join(root, f))}  {f}")
    p("")

    hits = []  # (file, lineno, text, func, arg, is_comment)
    for f in top + extra:
        with open(os.path.join(root, f), encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
        for i, line in enumerate(lines):
            if PATTERN in line:
                hits.append((f, i + 1, line.strip(), enclosing_function(lines, i),
                             call_argument(line), line.strip().startswith("#")))

    p("Raw hits (grep -n style):")
    if not hits:
        p("  (none)")
    for f, ln, text, *_ in hits:
        p(f"  {f}:{ln}:{text}")
    p("")

    p("Classification:")
    problems = []
    per_file_calls = {}
    for f, ln, text, func, arg, is_comment in hits:
        if is_comment:
            p(f"  {f}:{ln}  COMMENT (not a call) -- ignored")
            continue
        per_file_calls[f] = per_file_calls.get(f, 0) + 1
        expected_func = ALLOWED.get(f)
        reasons = []
        if expected_func is None:
            reasons.append("file is not on the audited allow-list")
        elif func != expected_func:
            reasons.append(f"enclosing function is {func!r}, expected {expected_func!r}")
        if arg in FORBIDDEN_ARGS:
            reasons.append(f"argument {arg!r} is a pre-split/full-data variable name")
        status = "OK  " if not reasons else "FAIL"
        p(f"  [{status}] {f}:{ln}  in {func}()  arg={arg}")
        p(f"           {text}")
        for r in reasons:
            p(f"           !! {r}")
            problems.append(f"{f}:{ln}: {r}")
    for f in ALLOWED:
        n = per_file_calls.get(f, 0)
        if n != 1 and os.path.exists(os.path.join(root, f)):
            msg = f"{f}: expected exactly 1 fit_transform call, found {n} -- re-audit"
            p(f"  [FAIL] {msg}")
            problems.append(msg)
    p("")

    verdict = "PASS" if not problems else "FAIL"
    p(f"AUDIT VERDICT: {verdict}")
    if problems:
        for pr in problems:
            p(f"  - {pr}")
    p("")
    p("Lineage proof for the allowed call sites (TRAIN-only) lives in:")
    p("  data_loader.py        -> tests/test_data_pipeline_dat1.py "
      "(test_grep_audit_fit_transform_only_inside_scaler_helper, "
      "test_scaler_unaffected_by_test_only_outlier)")
    p("  ciciot2023_loader.py  -> tests/test_dat1_ciciot_audit.py "
      "(static lineage + dynamic train-only scaler check)")

    tests_ok = True
    if run_tests:
        p("")
        p("=" * 70)
        p("Audit unit tests (pytest -v):")
        cmd = [sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider",
               "tests/test_data_pipeline_dat1.py::test_grep_audit_fit_transform_only_inside_scaler_helper",
               "tests/test_data_pipeline_dat1.py::test_grep_audit_fit_transform_repo_wide",
               "tests/test_data_pipeline_dat1.py::test_scaler_unaffected_by_test_only_outlier",
               "tests/test_dat1_ciciot_audit.py"]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        p("$ " + " ".join(os.path.basename(c) if i == 0 else c for i, c in enumerate(cmd)))
        p(r.stdout.rstrip())
        if r.stderr.strip():
            p(r.stderr.rstrip())
        tests_ok = (r.returncode == 0)
        p(f"audit tests exit code: {r.returncode} -> {'PASS' if tests_ok else 'FAIL'}")

    overall = verdict == "PASS" and tests_ok
    p("")
    p(f"OVERALL: {'PASS' if overall else 'FAIL'}")

    text = "\n".join(out) + "\n"
    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print(f"[written to {os.path.abspath(OUT_FILE)}]")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
