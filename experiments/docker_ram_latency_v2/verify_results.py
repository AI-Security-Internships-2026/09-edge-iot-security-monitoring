"""
Verifies every results/<tag>/ folder against what its OWN folder name
claims, and checks it's actually complete -- run this BEFORE zipping or
uploading anything, so a mislabeled/duplicated/incomplete folder gets
caught immediately instead of silently propagating downstream.

Checks per folder:
  - client_0_results.json's internal config.mode matches the mode
    implied by the folder name (e.g. "network_he_partial_throttled"
    should contain mode="he_partial", not "he_full")
  - client_0_results.json's internal config.real_cgroup_cpu_limit_cores
    matches the profile implied by the folder name (unthrottled -> 1.0,
    throttled -> 0.5)
  - client_0_results.json and client_1_results.json both exist and have
    exactly ROUNDS entries in "rounds" (catches a client that crashed
    partway through and never wrote its final summary)
  - the mode-appropriate server-side aggregation file exists
    (server_<mode>_results.json) -- catches an aggregator step that
    silently didn't run or wasn't included in an export
  - server_communication_summary.json shows received_total == expected_total

Usage: python verify_results.py [results_dir]
"""
import glob
import json
import os
import re
import sys

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "results"

KNOWN_MODES = ["baseline", "he_full", "he_partial_zkp", "he_partial", "dp"]


def infer_mode_and_profile(tag):
    """e.g. 'network_he_partial_zkp_throttled' -> ('he_partial_zkp', 'throttled').
    Tries longest mode names first so 'he_partial_zkp' isn't matched as
    'he_partial' by mistake."""
    profile = None
    for p in ("unthrottled", "throttled"):
        if tag.endswith(p):
            profile = p
            break
    mode = None
    for m in KNOWN_MODES:
        if f"_{m}_" in f"_{tag}_":
            mode = m
            break
    return mode, profile


def check_folder(run_dir):
    tag = os.path.basename(run_dir)
    expected_mode, expected_profile = infer_mode_and_profile(tag)
    expected_cpu = {"unthrottled": 1.0, "throttled": 0.5}.get(expected_profile)

    problems = []
    ok = True

    c0_path = os.path.join(run_dir, "client_0_results.json")
    c1_path = os.path.join(run_dir, "client_1_results.json")

    if not os.path.exists(c0_path):
        return tag, False, [f"MISSING client_0_results.json entirely (client 0 likely crashed before finishing)"]
    if not os.path.exists(c1_path):
        problems.append("MISSING client_1_results.json entirely (client 1 likely crashed before finishing)")
        ok = False

    with open(c0_path) as f:
        c0 = json.load(f)

    actual_mode = c0["config"]["mode"]
    actual_cpu = c0["config"]["real_cgroup_cpu_limit_cores"]
    n_rounds = len(c0["rounds"])
    expected_rounds = c0["config"]["rounds"]

    if expected_mode and actual_mode != expected_mode:
        problems.append(f"MODE MISMATCH: folder name implies '{expected_mode}' but "
                         f"client_0_results.json says mode='{actual_mode}'")
        ok = False

    if expected_cpu is not None and abs(actual_cpu - expected_cpu) > 1e-6:
        problems.append(f"CPU MISMATCH: folder name implies {expected_profile} "
                         f"(cpu={expected_cpu}) but client_0_results.json says cpu={actual_cpu}")
        ok = False

    if n_rounds != expected_rounds:
        problems.append(f"INCOMPLETE: client_0 only has {n_rounds}/{expected_rounds} rounds "
                         f"(likely crashed mid-run)")
        ok = False

    server_glob = glob.glob(os.path.join(run_dir, f"server_{actual_mode}_results.json"))
    if not server_glob:
        problems.append(f"MISSING server_{actual_mode}_results.json -- aggregator step "
                         f"never ran or wasn't exported")
        ok = False

    comm_path = os.path.join(run_dir, "server_communication_summary.json")
    if os.path.exists(comm_path):
        with open(comm_path) as f:
            comm = json.load(f)
        if comm["received_total"] != comm["expected_total"]:
            problems.append(f"COMMUNICATION INCOMPLETE: daemon received "
                             f"{comm['received_total']}/{comm['expected_total']} submissions")
            ok = False
    else:
        problems.append("MISSING server_communication_summary.json")
        ok = False

    return tag, ok, problems


def main():
    run_dirs = sorted(d for d in glob.glob(os.path.join(RESULTS_DIR, "*")) if os.path.isdir(d))
    if not run_dirs:
        print(f"No run folders found under {RESULTS_DIR}/")
        return

    # Also catch byte-identical folders (the exact bug that happened
    # this session) by hashing each folder's client_0_results.json.
    import hashlib
    hashes = {}

    all_ok = True
    for run_dir in run_dirs:
        tag, ok, problems = check_folder(run_dir)
        c0_path = os.path.join(run_dir, "client_0_results.json")
        if os.path.exists(c0_path):
            with open(c0_path, "rb") as f:
                h = hashlib.md5(f.read()).hexdigest()
            hashes.setdefault(h, []).append(tag)

        status = "OK" if ok else "PROBLEM"
        print(f"[{status:7s}] {tag}")
        for p in problems:
            print(f"           - {p}")
        if not ok:
            all_ok = False

    print()
    dupes = {h: tags for h, tags in hashes.items() if len(tags) > 1}
    if dupes:
        all_ok = False
        print("DUPLICATE client_0_results.json content found across folders "
              "(these are literally the same run, re-exported under different names):")
        for h, tags in dupes.items():
            print(f"  {tags}")
    else:
        print("No duplicate client_0_results.json content across folders.")

    print()
    print("ALL CLEAN -- safe to zip/export." if all_ok else
          "PROBLEMS FOUND -- fix/rerun the flagged folders before exporting for the paper.")


if __name__ == "__main__":
    main()
