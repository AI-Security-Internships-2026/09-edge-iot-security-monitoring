#!/usr/bin/env python3
"""
scripts/aggregate_task5_results.py

Computes Honest-FPR (Table A) and Honest-FPR + Byzantine-TPR (Table B)
from the REAL per_client_krum_scores.csv files Sub-task A's logger
(main.py, Issue 4 Task 1) writes -- one file per run, moved into the
sweep's results dir by run_task5_mad_k_sweep.sh / run_task5_f_sweep.sh.

METHODOLOGY CHOICE (flagged explicitly, per this project's reporting
discipline, not left implicit): "Honest-FPR per run" is computed by
POOLING the TP/FP/TN/FN classification column across ALL ROUNDS within
that run (not just the final round), then computing
FP / (FP + TN) from the pooled counts. This is a deliberate choice, not
the only possible one -- an alternative would be "final-round-only"
Honest-FPR. Pooling across rounds was chosen because:
  (a) it uses strictly more of the real logged data per run instead of
      discarding rounds 1-24's classifications,
  (b) a single-round FPR estimate from 8 honest clients is a very small
      sample (n=8) with high seed-to-seed variance; pooling across 25
      rounds gives ~200 honest-client-round observations per run before
      any cross-seed averaging.
If the paper's authors have an established different convention
(matching, e.g., Sub-task A's earlier 6-run sanity-sweep methodology),
reconcile against that BEFORE citing these numbers -- this is stated as
an assumption, not a confirmed-matching convention.

Each run's classification column already encodes TP/FP/TN/FN/N/A
directly from main.py's actual per-round Krum exclusion decision (see
Task 1's logger docstring) -- this script never re-derives
excluded/kept from the raw_krum_score column itself, exactly per Task
1's own stated discipline.

Usage:
    python scripts/aggregate_task5_results.py --table A \\
        --results-dir experiments/results/task5_mad_k_sweep \\
        --out experiments/results/param_sensitivity/table_a_mad_k.csv

    python scripts/aggregate_task5_results.py --table B \\
        --results-dir experiments/results/task5_f_sweep \\
        --out experiments/results/param_sensitivity/table_b_byzantine_f.csv
"""
import argparse
import csv
import glob
import math
import os
import re
from collections import defaultdict


def _mean_ci95(values):
    n = len(values)
    if n == 0:
        return None, None, 0
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0, n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(var)
    t_975_4 = 2.776  # n=5 -> df=4; see aggregate_task4_results.py's same note
    margin = t_975_4 * std / math.sqrt(n)
    return mean, margin, n


def _pooled_rates_from_csv(path):
    """
    Reads one real per_client_krum_scores.csv, pools TP/FP/TN/FN counts
    across ALL rounds in the file, returns (honest_fpr, byzantine_tpr).
    Rows with classification == "N/A" (no Krum-family exclusion decision
    that round -- e.g. a FedProx-fallback round) are excluded from both
    counts, exactly matching Task 1's own definition (N/A means no
    exclusion decision was made, not "excluded" or "kept").
    """
    counts = defaultdict(int)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            cls = row.get("classification", "N/A")
            if cls in ("TP", "FP", "TN", "FN"):
                counts[cls] += 1

    honest_total = counts["FP"] + counts["TN"]
    byz_total = counts["TP"] + counts["FN"]

    honest_fpr = counts["FP"] / honest_total if honest_total > 0 else None
    byz_tpr = counts["TP"] / byz_total if byz_total > 0 else None

    return honest_fpr, byz_tpr, dict(counts)


def collect_table_a(results_dir):
    """
    Table A: MAD-k sweep. Filenames from run_task5_mad_k_sweep.sh:
      per_client_krum_scores_network_task5_madk_plain_k<K>_seed<S>.csv
      per_client_krum_scores_network_task5_madk_calibrated_k<K>_seed<S>.csv
    Returns {k_value: {"plain": [fpr,...], "calibrated": [fpr,...]}}
    """
    pattern = os.path.join(results_dir, "per_client_krum_scores_*.csv")
    by_k = defaultdict(lambda: defaultdict(list))
    for path in sorted(glob.glob(pattern)):
        m = re.search(r"task5_madk_(plain|calibrated)_k([\d.]+)_seed(\d+)",
                       os.path.basename(path))
        if not m:
            print(f"  Skipping (doesn't match Table A naming): {path}")
            continue
        variant, k, seed = m.groups()
        fpr, tpr, counts = _pooled_rates_from_csv(path)
        if fpr is None:
            print(f"  WARNING: {path} has no honest-client classification "
                  f"rows (all N/A?) -- skipping this run for FPR purposes.")
            continue
        by_k[float(k)][variant].append(fpr)
    return by_k


def collect_table_b(results_dir):
    """
    Table B: Byzantine-f sweep. Filenames from run_task5_f_sweep.sh:
      per_client_krum_scores_network_task5_fsweep_plain_f<F>_seed<S>.csv
      per_client_krum_scores_network_task5_fsweep_calibrated_f<F>_seed<S>.csv
    Returns {f_value: {"plain": {"fpr":[...], "tpr":[...]},
                        "calibrated": {"fpr":[...], "tpr":[...]}}}
    """
    pattern = os.path.join(results_dir, "per_client_krum_scores_*.csv")
    by_f = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for path in sorted(glob.glob(pattern)):
        m = re.search(r"task5_fsweep_(plain|calibrated)_f(\d+)_seed(\d+)",
                       os.path.basename(path))
        if not m:
            print(f"  Skipping (doesn't match Table B naming): {path}")
            continue
        variant, f, seed = m.groups()
        fpr, tpr, counts = _pooled_rates_from_csv(path)
        if fpr is not None:
            by_f[int(f)][variant]["fpr"].append(fpr)
        if tpr is not None:
            by_f[int(f)][variant]["tpr"].append(tpr)
        else:
            print(f"  WARNING: {path} has no byzantine-client classification "
                  f"rows -- TPR unavailable for this run.")
    return by_f


def write_table_a(by_k, out_path):
    rows = []
    all_delta_ok = True
    for k in sorted(by_k.keys()):
        plain_vals = by_k[k].get("plain", [])
        cal_vals = by_k[k].get("calibrated", [])
        plain_mean, plain_ci, n_plain = _mean_ci95(plain_vals)
        cal_mean, cal_ci, n_cal = _mean_ci95(cal_vals)
        delta = (plain_mean - cal_mean) if (plain_mean is not None and cal_mean is not None) else None
        row = {
            "mad_k": k,
            "calibrated_honest_fpr_mean": cal_mean, "calibrated_honest_fpr_ci95": cal_ci,
            "calibrated_n_seeds": n_cal,
            "plain_adaptive_honest_fpr_mean": plain_mean, "plain_adaptive_honest_fpr_ci95": plain_ci,
            "plain_adaptive_n_seeds": n_plain,
            "delta_fpr_positive_is_calibrated_better": delta,
        }
        rows.append(row)
        if delta is not None and delta < 0:
            all_delta_ok = False
            print(f"  *** REQUIREMENT VIOLATION at k={k}: delta_FPR={delta:.4f} < 0 "
                  f"(Calibrated is WORSE than plain Adaptive Krum). Per the "
                  f"ticket's own instruction: do NOT force this to comply -- "
                  f"report it as a real limitation.")
        if n_plain < 5 or n_cal < 5:
            print(f"  WARNING: k={k} has fewer than 5 seeds "
                  f"(plain={n_plain}, calibrated={n_cal}) -- incomplete sweep.")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Table A written: {out_path}")
    print(f"  ALL delta_FPR >= 0 requirement: {'PASS' if all_delta_ok else 'FAIL -- see warnings above'}")
    return rows, all_delta_ok


def write_table_b(by_f, out_path):
    rows = []
    fpr_dominance_ok = True
    tpr_within_5pp_ok = True
    for f_val in sorted(by_f.keys()):
        plain = by_f[f_val].get("plain", {})
        cal = by_f[f_val].get("calibrated", {})

        plain_fpr_mean, plain_fpr_ci, n_plain_fpr = _mean_ci95(plain.get("fpr", []))
        cal_fpr_mean, cal_fpr_ci, n_cal_fpr = _mean_ci95(cal.get("fpr", []))
        plain_tpr_mean, plain_tpr_ci, n_plain_tpr = _mean_ci95(plain.get("tpr", []))
        cal_tpr_mean, cal_tpr_ci, n_cal_tpr = _mean_ci95(cal.get("tpr", []))

        row = {
            "f_byzantine": f_val,
            "calibrated_byzantine_tpr_mean": cal_tpr_mean, "calibrated_byzantine_tpr_ci95": cal_tpr_ci,
            "plain_adaptive_byzantine_tpr_mean": plain_tpr_mean, "plain_adaptive_byzantine_tpr_ci95": plain_tpr_ci,
            "calibrated_honest_fpr_mean": cal_fpr_mean, "calibrated_honest_fpr_ci95": cal_fpr_ci,
            "plain_adaptive_honest_fpr_mean": plain_fpr_mean, "plain_adaptive_honest_fpr_ci95": plain_fpr_ci,
            "n_seeds": min(n_plain_fpr, n_cal_fpr, n_plain_tpr, n_cal_tpr),
        }
        rows.append(row)

        if cal_fpr_mean is not None and plain_fpr_mean is not None:
            if not (cal_fpr_mean < plain_fpr_mean):
                fpr_dominance_ok = False
                print(f"  *** REQUIREMENT VIOLATION at f={f_val}: "
                      f"Calibrated HonestFPR ({cal_fpr_mean:.4f}) is NOT < "
                      f"PlainAdaptive HonestFPR ({plain_fpr_mean:.4f}).")

        if f_val in (1, 2) and cal_tpr_mean is not None and plain_tpr_mean is not None:
            gap_pp = abs(cal_tpr_mean - plain_tpr_mean) * 100
            if gap_pp > 5.0:
                tpr_within_5pp_ok = False
                print(f"  *** REQUIREMENT VIOLATION at f={f_val}: ByzantineTPR "
                      f"gap={gap_pp:.1f}pp exceeds the 5pp graceful-degradation "
                      f"requirement (required only for f=1,2, not f=3).")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Table B written: {out_path}")
    print(f"  HonestFPR dominance (Calibrated < Plain, all f): {'PASS' if fpr_dominance_ok else 'FAIL -- see warnings above'}")
    print(f"  ByzantineTPR within 5pp (f=1,2 only): {'PASS' if tpr_within_5pp_ok else 'FAIL -- see warnings above'}")
    return rows, fpr_dominance_ok, tpr_within_5pp_ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--table", choices=["A", "B"], required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.table == "A":
        by_k = collect_table_a(args.results_dir)
        if not by_k:
            print(f"  No matching per_client_krum_scores CSVs found under "
                  f"{args.results_dir!r}.")
            return
        out = args.out or "experiments/results/param_sensitivity/table_a_mad_k.csv"
        write_table_a(by_k, out)
    else:
        by_f = collect_table_b(args.results_dir)
        if not by_f:
            print(f"  No matching per_client_krum_scores CSVs found under "
                  f"{args.results_dir!r}.")
            return
        out = args.out or "experiments/results/param_sensitivity/table_b_byzantine_f.csv"
        write_table_b(by_f, out)


if __name__ == "__main__":
    main()
