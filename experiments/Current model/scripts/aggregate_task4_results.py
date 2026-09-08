#!/usr/bin/env python3
"""
scripts/aggregate_task4_results.py

Parses the REAL results_*_FINAL_TEST.csv and results_*_FINAL_VALIDATION.csv
files main.py and scripts/train_centralized.py both write (one pair per
run, identical schema/naming convention for federated AND centralized
conditions) and computes 5-seed mean +/- 95% CI per condition.

SPLIT_PROTOCOL.md: FedProx mu (and MAD-k, DP clip norm C) selection MUST
use the real global VALIDATION split, never TEST. Both main.py and
train_centralized.py write results_*_FINAL_VALIDATION.csv alongside
results_*_FINAL_TEST.csv -- this script computes the mu argmax from
VALIDATION specifically, and only ever uses TEST for the final reported
Table 1 numbers.

FALLBACK: if no VALIDATION files are found, falls back to a TEST-based
diagnostic-only argmax, clearly labeled as such.

CENTRALIZED CONDITIONS covered via scripts/train_centralized.py, which
writes the identical naming convention -- no special-casing needed here.

MINORITY RECALL / MINORITY AUC-PR: the 5 "rarest classes" span BOTH
models (network: Ransomware, MITM, Vulnerability_scanner; application:
XSS, Fingerprinting) -- no single row can hold all 5. Per-model
summaries are explicitly PARTIAL; the real 5-class average is computed
by combine_minority_across_models(), pairing network+application runs
by condition+seed, written to a separate table1_minority_combined.csv.

Usage:
    python scripts/aggregate_task4_results.py --stage prox_sweep \
        --results-dir experiments/results/task4_federated
    python scripts/aggregate_task4_results.py --stage full \
        --results-dir experiments/results/task4_federated \
        --out experiments/results/task4_federated/table1_federated.csv
"""
import argparse
import csv
import glob
import math
import os
import re
from collections import defaultdict

MINORITY_CLASSES = ["Ransomware", "MITM", "XSS", "Fingerprinting",
                     "Vulnerability_scanner"]


def _mean_ci95(values):
    n = len(values)
    if n == 0:
        return None, None, 0
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0, n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(var)
    t_975_4 = 2.776  # n=5 -> df=4
    margin = t_975_4 * std / math.sqrt(n)
    return mean, margin, n


def _mean_ci95_nan_aware(values):
    clean = [v for v in values if v == v]
    n_dropped = len(values) - len(clean)
    mean, ci, n = _mean_ci95(clean)
    return mean, ci, n, n_dropped


def _parse_final_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return rows[0]


def _condition_key_from_filename(fname, suffix):
    m = re.match(
        r"results_(network|application)_(task4_[a-zA-Z0-9_.]+)_seed(\d+)_FINAL_"
        + re.escape(suffix) + r"\.csv",
        os.path.basename(fname),
    )
    if not m:
        return None
    model, condition, seed = m.groups()
    return model, condition, int(seed)


def collect(results_dir, suffix="TEST"):
    by_condition = defaultdict(list)
    glob_suffix = f"FINAL_{suffix}.csv"
    pattern = os.path.join(results_dir, "**", f"results_*_{glob_suffix}")
    paths = glob.glob(pattern, recursive=True) + glob.glob(
        os.path.join(results_dir, f"results_*_{glob_suffix}")
    )
    paths = sorted(set(paths))
    if not paths:
        print(f"  WARNING: no results_*_{glob_suffix} files found under "
              f"{results_dir!r}.")
        return by_condition

    for path in paths:
        key = _condition_key_from_filename(path, suffix)
        if key is None:
            print(f"  Skipping (doesn't match expected naming): {path}")
            continue
        model, condition, seed = key
        row = _parse_final_csv(path)
        if row is None:
            print(f"  WARNING: {path} has no data row -- skipping.")
            continue
        row["_seed"] = seed
        row["_path"] = path
        by_condition[(model, condition)].append(row)

    return by_condition


def summarize(by_condition, minority_classes=MINORITY_CLASSES, prefix="test"):
    summaries = []
    for (model, condition), rows in sorted(by_condition.items()):
        n_seeds = len(rows)
        seeds_present = sorted(r["_seed"] for r in rows)

        acc_vals = [float(r[f"{prefix}_accuracy"]) for r in rows]
        f1_vals = [float(r[f"{prefix}_f1_macro"]) for r in rows]

        acc_mean, acc_ci, _ = _mean_ci95(acc_vals)
        f1_mean, f1_ci, _ = _mean_ci95(f1_vals)

        minority_f1_means = {}
        minority_recall_means = {}
        minority_aucpr_means = {}
        classes_in_this_model = []
        for cls in minority_classes:
            f1_col = f"{prefix}_f1_{cls}"
            if not all(f1_col in r for r in rows):
                minority_f1_means[cls] = (None, None)
                minority_recall_means[cls] = (None, None)
                minority_aucpr_means[cls] = (None, None, None)
                continue
            classes_in_this_model.append(cls)

            vals = [float(r[f1_col]) for r in rows]
            m, ci, _ = _mean_ci95(vals)
            minority_f1_means[cls] = (m, ci)

            recall_col = f"{prefix}_recall_{cls}"
            if all(recall_col in r for r in rows):
                rvals = [float(r[recall_col]) for r in rows]
                rm, rci, _ = _mean_ci95(rvals)
                minority_recall_means[cls] = (rm, rci)
            else:
                minority_recall_means[cls] = (None, None)

            aucpr_col = f"{prefix}_aucpr_{cls}"
            if all(aucpr_col in r for r in rows):
                avals = [float(r[aucpr_col]) for r in rows]
                am, aci, a_n, a_nan = _mean_ci95_nan_aware(avals)
                minority_aucpr_means[cls] = (am, aci, a_nan)
            else:
                minority_aucpr_means[cls] = (None, None, None)

        recall_present = [v[0] for v in minority_recall_means.values() if v[0] is not None]
        aucpr_present = [v[0] for v in minority_aucpr_means.values() if v[0] is not None]

        summaries.append({
            "model": model,
            "condition": condition,
            "n_seeds": n_seeds,
            "seeds_present": seeds_present,
            "seeds_missing": sorted(set([42, 123, 456, 789, 2024]) - set(seeds_present)),
            "accuracy_mean": acc_mean, "accuracy_ci95": acc_ci,
            "macro_f1_mean": f1_mean, "macro_f1_ci95": f1_ci,
            "minority_f1_by_class": minority_f1_means,
            "minority_recall_by_class": minority_recall_means,
            "minority_aucpr_by_class": minority_aucpr_means,
            "classes_in_this_model": classes_in_this_model,
            "minority_f1_avg_mean": (
                sum(v[0] for v in minority_f1_means.values() if v[0] is not None)
                / max(1, sum(1 for v in minority_f1_means.values() if v[0] is not None))
            ) if any(v[0] is not None for v in minority_f1_means.values()) else None,
            "minority_recall_avg_mean_PARTIAL": (
                sum(recall_present) / len(recall_present)
            ) if recall_present else None,
            "minority_aucpr_avg_mean_PARTIAL": (
                sum(aucpr_present) / len(aucpr_present)
            ) if aucpr_present else None,
        })
    return summaries


def combine_minority_across_models(by_condition_raw, prefix="test"):
    NETWORK_MINORITY = ["Ransomware", "MITM", "Vulnerability_scanner"]
    APP_MINORITY = ["XSS", "Fingerprinting"]

    net_by_condition = defaultdict(dict)
    app_by_condition = defaultdict(dict)
    for (model, condition), rows in by_condition_raw.items():
        target = net_by_condition if model == "network" else (
            app_by_condition if model == "application" else None
        )
        if target is None:
            continue
        for r in rows:
            target[condition][r["_seed"]] = r

    conditions = sorted(set(net_by_condition.keys()) & set(app_by_condition.keys()))
    if not conditions:
        print("  WARNING: no condition has BOTH a network and an "
              "application run -- cannot compute the real 5-class "
              "Minority Recall/AUC-PR for anything.")
        return []

    results = []
    for condition in conditions:
        net_rows = net_by_condition[condition]
        app_rows = app_by_condition[condition]
        paired_seeds = sorted(set(net_rows.keys()) & set(app_rows.keys()))
        missing = sorted((set(net_rows.keys()) | set(app_rows.keys())) - set(paired_seeds))
        if missing:
            print(f"  WARNING: condition={condition!r}: seeds {missing} "
                  f"present in only one of network/application -- "
                  f"excluded from the paired 5-class average.")

        per_seed_recall_avg = []
        per_seed_aucpr_avg = []
        total_aucpr_nan_dropped = 0
        for seed in paired_seeds:
            nrow, arow = net_rows[seed], app_rows[seed]

            recall_vals = (
                [float(nrow[f"{prefix}_recall_{c}"]) for c in NETWORK_MINORITY]
                + [float(arow[f"{prefix}_recall_{c}"]) for c in APP_MINORITY]
            )
            per_seed_recall_avg.append(sum(recall_vals) / len(recall_vals))

            aucpr_vals = (
                [float(nrow[f"{prefix}_aucpr_{c}"]) for c in NETWORK_MINORITY]
                + [float(arow[f"{prefix}_aucpr_{c}"]) for c in APP_MINORITY]
            )
            clean_aucpr = [v for v in aucpr_vals if v == v]
            total_aucpr_nan_dropped += (len(aucpr_vals) - len(clean_aucpr))
            if clean_aucpr:
                per_seed_aucpr_avg.append(sum(clean_aucpr) / len(clean_aucpr))

        recall_mean, recall_ci, n_recall = _mean_ci95(per_seed_recall_avg)
        aucpr_mean, aucpr_ci, n_aucpr = _mean_ci95(per_seed_aucpr_avg)

        results.append({
            "condition": condition,
            "n_seeds_paired": len(paired_seeds),
            "seeds_excluded_unpaired": missing,
            "minority_recall_avg_mean": recall_mean,
            "minority_recall_avg_ci95": recall_ci,
            "minority_aucpr_avg_mean": aucpr_mean,
            "minority_aucpr_avg_ci95": aucpr_ci,
            "n_aucpr_class_seed_points_dropped_as_nan": total_aucpr_nan_dropped,
        })
    return results


def write_minority_combined_csv(combined, out_path):
    fieldnames = [
        "condition", "n_seeds_paired", "seeds_excluded_unpaired",
        "minority_recall_avg_mean", "minority_recall_avg_ci95",
        "minority_aucpr_avg_mean", "minority_aucpr_avg_ci95",
        "n_aucpr_class_seed_points_dropped_as_nan",
    ]
    dirn = os.path.dirname(out_path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in combined:
            r = dict(row)
            r["seeds_excluded_unpaired"] = ";".join(map(str, r["seeds_excluded_unpaired"]))
            w.writerow(r)
    print(f"\n  Written (REAL 5-class Minority Recall/AUC-PR, network+application "
          f"paired by condition+seed): {out_path}")


def print_prox_sweep_argmax(val_summaries, test_summaries_fallback=None):
    if val_summaries:
        summaries = val_summaries
        print("\n  === FedProx mu argmax (VALIDATION-split Macro-F1, per")
        print("      SPLIT_PROTOCOL.md) ===")
    elif test_summaries_fallback:
        summaries = test_summaries_fallback
        print("\n  " + "!"*61)
        print("  [SPLIT_PROTOCOL WARNING] No results_*_FINAL_VALIDATION.csv")
        print("  files found -- falling back to TEST-holdout Macro-F1. This")
        print("  is a DIAGNOSTIC ESTIMATE ONLY, not the real selection")
        print("  mechanism ('tuning against TEST' is prohibited).")
        print("  " + "!"*61)
    else:
        print("\n  No FedProx mu-sweep data found.")
        return {}

    by_model = defaultdict(list)
    for s in summaries:
        m = re.match(r"task4_fedprox_mu([\d.]+)$", s["condition"])
        if m:
            mu = float(m.group(1))
            by_model[s["model"]].append((mu, s["macro_f1_mean"], s["n_seeds"]))

    best_mu_by_model = {}
    for model, entries in sorted(by_model.items()):
        entries.sort()
        print(f"\n  {model}:")
        best_mu, best_f1 = None, -1
        for mu, f1, n in entries:
            flag = "  <-- fewer than 5 seeds!" if n < 5 else ""
            f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
            print(f"    mu={mu:<7} macro_f1_mean={f1_str}  (n_seeds={n}){flag}")
            if f1 is not None and f1 > best_f1:
                best_f1, best_mu = f1, mu
        if best_mu is not None:
            print(f"    ARGMAX for {model}: mu={best_mu} (macro_f1={best_f1:.4f})")
            best_mu_by_model[model] = best_mu
        else:
            print(f"    No FedProx mu-sweep results found for {model}.")

    return best_mu_by_model


def write_table_csv(summaries, out_path):
    fieldnames = [
        "model", "condition", "n_seeds", "seeds_present", "seeds_missing",
        "accuracy_mean", "accuracy_ci95",
        "macro_f1_mean", "macro_f1_ci95",
        "minority_f1_avg_mean_PARTIAL",
        "minority_recall_avg_mean_PARTIAL",
        "minority_aucpr_avg_mean_PARTIAL",
        "classes_in_this_model",
    ] + [f"f1_{c}_mean" for c in MINORITY_CLASSES] + [f"f1_{c}_ci95" for c in MINORITY_CLASSES] \
      + [f"recall_{c}_mean" for c in MINORITY_CLASSES] + [f"recall_{c}_ci95" for c in MINORITY_CLASSES] \
      + [f"aucpr_{c}_mean" for c in MINORITY_CLASSES] + [f"aucpr_{c}_ci95" for c in MINORITY_CLASSES]

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in summaries:
            row = {
                "model": s["model"], "condition": s["condition"],
                "n_seeds": s["n_seeds"],
                "seeds_present": ";".join(map(str, s["seeds_present"])),
                "seeds_missing": ";".join(map(str, s["seeds_missing"])),
                "accuracy_mean": s["accuracy_mean"], "accuracy_ci95": s["accuracy_ci95"],
                "macro_f1_mean": s["macro_f1_mean"], "macro_f1_ci95": s["macro_f1_ci95"],
                "minority_f1_avg_mean_PARTIAL": s["minority_f1_avg_mean"],
                "minority_recall_avg_mean_PARTIAL": s["minority_recall_avg_mean_PARTIAL"],
                "minority_aucpr_avg_mean_PARTIAL": s["minority_aucpr_avg_mean_PARTIAL"],
                "classes_in_this_model": ";".join(s["classes_in_this_model"]),
            }
            for c in MINORITY_CLASSES:
                m, ci = s["minority_f1_by_class"][c]
                row[f"f1_{c}_mean"] = m
                row[f"f1_{c}_ci95"] = ci
                rm, rci = s["minority_recall_by_class"][c]
                row[f"recall_{c}_mean"] = rm
                row[f"recall_{c}_ci95"] = rci
                am_tuple = s["minority_aucpr_by_class"][c]
                am, aci = am_tuple[0], am_tuple[1]
                row[f"aucpr_{c}_mean"] = am
                row[f"aucpr_{c}_ci95"] = aci
            w.writerow(row)
    print(f"\n  Written: {out_path}")
    print(f"  NOTE: minority_*_avg_mean_PARTIAL columns cover only the "
          f"classes listed in classes_in_this_model (3-of-5 for network, "
          f"2-of-5 for application) -- the real 5-class Minority Recall/"
          f"AUC-PR is in the separate table1_minority_combined.csv.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--stage", choices=["prox_sweep", "full"], default="full")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    by_condition_test = collect(args.results_dir, suffix="TEST")
    by_condition_val = collect(args.results_dir, suffix="VALIDATION")

    if not by_condition_test:
        return

    test_summaries = summarize(by_condition_test, prefix="test")
    val_summaries = summarize(by_condition_val, prefix="val") if by_condition_val else []

    print(f"\n  Found {len(by_condition_test)} (model, condition) combinations "
          f"in FINAL_TEST files"
          + (f", {len(by_condition_val)} in FINAL_VALIDATION files."
             if by_condition_val else " (no FINAL_VALIDATION files found)."))
    for s in test_summaries:
        seed_flag = "" if s["n_seeds"] == 5 else f"  <-- only {s['n_seeds']}/5 seeds!"
        f1_str = f"{s['macro_f1_mean']:.4f}" if s['macro_f1_mean'] is not None else "N/A"
        print(f"    {s['model']:12s} {s['condition']:35s} n={s['n_seeds']} "
              f"macro_f1(TEST)={f1_str}{seed_flag}")

    print_prox_sweep_argmax(val_summaries, test_summaries_fallback=test_summaries)

    if args.stage == "full":
        out_path = args.out or os.path.join(args.results_dir, "table1_federated.csv")
        write_table_csv(test_summaries, out_path)

        minority_out_path = os.path.join(
            os.path.dirname(out_path) or ".", "table1_minority_combined.csv"
        )
        combined_minority = combine_minority_across_models(by_condition_test, prefix="test")
        if combined_minority:
            write_minority_combined_csv(combined_minority, minority_out_path)
            print("\n  Real 5-class Minority Recall / Minority AUC-PR (network+application")
            print("  paired by condition+seed):")
            for row in combined_minority:
                r_str = (f"{row['minority_recall_avg_mean']:.4f}"
                         if row['minority_recall_avg_mean'] is not None else "N/A")
                a_str = (f"{row['minority_aucpr_avg_mean']:.4f}"
                         if row['minority_aucpr_avg_mean'] is not None else "N/A")
                seed_flag = ("" if row["n_seeds_paired"] == 5
                             else f"  <-- only {row['n_seeds_paired']}/5 paired seeds!")
                print(f"    {row['condition']:35s} recall_avg(5-class)={r_str}  "
                      f"aucpr_avg(5-class)={a_str}{seed_flag}")


if __name__ == "__main__":
    main()
