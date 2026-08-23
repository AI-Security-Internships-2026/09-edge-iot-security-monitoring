#!/usr/bin/env python3
"""
build_sweep_table.py
=====================
Scans results_{model}_{tag}.csv + experiment_config_{model}_{tag}.json
pairs produced by main.py and builds two summary tables for the
epsilon sweep (Experiment 1 continuation):

  1. sweep_summary.csv / .md        — one row per (model, tag) condition:
                                       best round, best F1-Macro, best
                                       accuracy, Krum detection rate,
                                       mean krum_score_ratio, DP calibration
                                       (target vs achieved epsilon), noise
                                       multiplier.
  2. sweep_summary_per_class.csv    — same rows, but with per-class F1 at
                                       the best round broken out into
                                       separate columns.

FIX (this revision): the glob `results_*.csv` picks up EVERY results
file in the directory, including ones from unrelated experiments that
happen to share the models/naming pattern -- e.g. the Week 9 HE-hybrid
cross-configuration runs (network_byz2_7, network_byz4_10,
application_byz4_10), which ran at adaptive_krum_k=3.5 (that
experiment's own tuned value) while this epsilon sweep pins k=2.5 for
comparability with Experiment 1. Silently mixing those in means the
krum_vs_epsilon plot would show two different Krum configurations
plotted as if they were one continuous ε curve -- a real contamination
bug, not just clutter. This revision:
  - Reads adaptive_krum_k from each condition's experiment_config json
    (already available, just wasn't being checked before).
  - By default, KEEPS ONLY rows matching the MOST COMMON krum-k value
    found across all conditions (the epsilon sweep's own consistent
    value), and prints which tags got excluded and why.
  - Exposes --krum-k to override this filter explicitly instead of
    relying on majority-vote auto-detection.
  - Prints a loud warning if more than one krum-k value is present at
    all, even after filtering, so this can never again go unnoticed
    the way it did the first time.

Run this from the same directory as your results_*.csv / experiment_config_*.json
files (typically experiments/Current model/ on the DGX).

Usage:
    python3 build_sweep_table.py
    python3 build_sweep_table.py --krum-k 2.5      # explicit filter
    python3 build_sweep_table.py --no-filter        # include everything (debug only)
    python3 build_sweep_table.py --dir /path/to/results --out sweep_out/
"""

import argparse
import csv
import glob
import json
import os
import re
import statistics
import sys
from collections import Counter


def find_conditions(results_dir):
    conditions = []
    pattern = os.path.join(results_dir, "results_*.csv")
    for csv_path in sorted(glob.glob(pattern)):
        base = os.path.basename(csv_path)
        m = re.match(r"results_(.+)\.csv$", base)
        if not m:
            continue
        tag = m.group(1)
        config_path = os.path.join(results_dir, f"experiment_config_{tag}.json")
        if not os.path.exists(config_path):
            print(f"  [skip] {base}: no matching experiment_config_{tag}.json", file=sys.stderr)
            continue
        conditions.append((tag, csv_path, config_path))
    return conditions


def load_mean_rows(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = [row for row in reader if row.get("client") == "MEAN"]
    return header, rows


def per_class_columns(header):
    start = header.index("accuracy") + 1
    end = header.index("zkp_rejected")
    return header[start:end]


def safe_float(v):
    try:
        if v in (None, "", "N/A"):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def summarize_condition(tag, csv_path, config_path):
    with open(config_path) as f:
        config = json.load(f)

    header, mean_rows = load_mean_rows(csv_path)
    if not mean_rows:
        print(f"  [warn] {tag}: no MEAN rows found (run may not have started)", file=sys.stderr)
        return None

    f1_cols = per_class_columns(header)

    best_row = None
    best_f1_macro = -1.0
    for row in mean_rows:
        vals = [safe_float(row[c]) for c in f1_cols]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        f1_macro = sum(vals) / len(vals)
        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            best_row = row

    if best_row is None:
        print(f"  [warn] {tag}: could not compute F1-Macro for any round", file=sys.stderr)
        return None

    detection_rates = [safe_float(r.get("krum_detected_byzantine")) for r in mean_rows]
    detection_rates = [d for d in detection_rates if d is not None]
    mean_detection_rate = statistics.mean(detection_rates) if detection_rates else None

    score_ratios = [safe_float(r.get("krum_score_ratio")) for r in mean_rows]
    score_ratios = [s for s in score_ratios if s is not None]
    mean_score_ratio = statistics.mean(score_ratios) if score_ratios else None

    dp_spent = [safe_float(r.get("dp_epsilon_spent")) for r in mean_rows]
    dp_spent = [d for d in dp_spent if d is not None]
    mean_dp_epsilon_achieved = statistics.mean(dp_spent) if dp_spent else None

    noise_mults = [safe_float(r.get("dp_noise_multiplier")) for r in mean_rows]
    noise_mults = [n for n in noise_mults if n is not None]
    mean_noise_multiplier = statistics.mean(noise_mults) if noise_mults else None

    nan_rounds = sum(
        1 for r in mean_rows
        if str(r.get("nan_this_round")) == "1"
    )

    row_out = {
        "tag": tag,
        "model_type": config.get("model_type"),
        "dp_epsilon_target": config.get("dp_epsilon"),
        "dp_epsilon_achieved_mean": mean_dp_epsilon_achieved,
        "dp_noise_multiplier_mean": mean_noise_multiplier,
        "adaptive_krum_k": config.get("adaptive_krum_k"),
        "use_byzantine_attack": config.get("byzantine_attack"),
        "num_byzantine": config.get("num_byzantine"),
        "byzantine_clients": config.get("byzantine_clients"),
        "best_round": int(best_row["round"]),
        "best_f1_macro": round(best_f1_macro, 4),
        "best_accuracy": round(safe_float(best_row["accuracy"]), 4) if safe_float(best_row["accuracy"]) is not None else None,
        "final_round": int(mean_rows[-1]["round"]),
        "final_f1_macro": round(
            sum(v for v in (safe_float(mean_rows[-1][c]) for c in f1_cols) if v is not None) / len(f1_cols), 4
        ) if f1_cols else None,
        "krum_detection_rate_mean": round(mean_detection_rate, 4) if mean_detection_rate is not None else None,
        "krum_score_ratio_mean": round(mean_score_ratio, 4) if mean_score_ratio is not None else None,
        "nan_rounds": nan_rounds,
        "rounds_completed": len(mean_rows),
        "_best_row": best_row,
        "_f1_cols": f1_cols,
    }
    return row_out


SUMMARY_FIELDS = [
    "model_type", "tag", "dp_epsilon_target", "dp_epsilon_achieved_mean",
    "dp_noise_multiplier_mean", "adaptive_krum_k", "use_byzantine_attack",
    "byzantine_clients",
    "best_round", "best_f1_macro", "best_accuracy",
    "final_round", "final_f1_macro",
    "krum_detection_rate_mean", "krum_score_ratio_mean",
    "nan_rounds", "rounds_completed",
]


def write_csv(rows, out_path, fields):
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_markdown(rows, out_path, fields):
    with open(out_path, "w") as f:
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("|" + "|".join(["---"] * len(fields)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(c, "")) for c in fields) + " |\n")


def write_per_class_csv(rows, out_path):
    all_classes = []
    for r in rows:
        for c in r["_f1_cols"]:
            if c not in all_classes:
                all_classes.append(c)

    fields = ["model_type", "tag", "best_round", "best_f1_macro"] + all_classes
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            out = {
                "model_type": r["model_type"],
                "tag": r["tag"],
                "best_round": r["best_round"],
                "best_f1_macro": r["best_f1_macro"],
            }
            for c in r["_f1_cols"]:
                v = safe_float(r["_best_row"].get(c))
                out[c] = round(v, 4) if v is not None else None
            writer.writerow(out)


def apply_krum_k_filter(rows, explicit_k, no_filter):
    """
    Returns (kept_rows, excluded_rows, chosen_k). If no_filter is True,
    returns everything unfiltered (debug/inspection mode only -- the
    resulting sweep_summary.csv should NOT be fed into
    plot_epsilon_sweep.py in that mode without manual review).
    """
    if no_filter:
        return rows, [], None

    k_values = [r["adaptive_krum_k"] for r in rows if r["adaptive_krum_k"] is not None]
    if not k_values:
        print("  [warn] no adaptive_krum_k found in any config -- skipping filter", file=sys.stderr)
        return rows, [], None

    if explicit_k is not None:
        chosen_k = explicit_k
    else:
        counts = Counter(k_values)
        chosen_k = counts.most_common(1)[0][0]

    kept, excluded = [], []
    for r in rows:
        k = r["adaptive_krum_k"]
        if k is None or abs(k - chosen_k) < 1e-9:
            kept.append(r)
        else:
            excluded.append(r)

    return kept, excluded, chosen_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=".", help="Directory containing results_*.csv (default: current dir)")
    parser.add_argument("--out", default=".", help="Output directory for summary files (default: current dir)")
    parser.add_argument("--krum-k", type=float, default=None,
                       help="Keep only conditions run at this adaptive_krum_k. "
                            "Default: auto-detect the majority value and keep only that.")
    parser.add_argument("--no-filter", action="store_true",
                       help="Disable krum-k filtering entirely (debug only -- "
                            "do not feed the resulting summary into "
                            "plot_epsilon_sweep.py without manual review, it "
                            "will silently mix incompatible Krum configs)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Scanning {args.dir} for results_*.csv / experiment_config_*.json pairs...")
    conditions = find_conditions(args.dir)
    print(f"Found {len(conditions)} conditions.\n")

    rows = []
    for tag, csv_path, config_path in conditions:
        row = summarize_condition(tag, csv_path, config_path)
        if row is not None:
            rows.append(row)

    if not rows:
        print("No valid conditions found — nothing to summarize.", file=sys.stderr)
        sys.exit(1)

    kept, excluded, chosen_k = apply_krum_k_filter(rows, args.krum_k, args.no_filter)

    if excluded:
        print(f"\n[FILTER] {len(excluded)} condition(s) excluded — different "
              f"adaptive_krum_k than the sweep's own value ({chosen_k}):")
        for r in excluded:
            print(f"    {r['model_type']}/{r['tag']}: krum_k={r['adaptive_krum_k']} "
                  f"(likely a different experiment, e.g. a Krum cross-"
                  f"configuration run, not part of this ε sweep)")
        print(f"  These are NOT included in sweep_summary.csv. Use "
              f"--no-filter to include them anyway (not recommended for "
              f"plotting), or --krum-k <value> to pick a different value "
              f"explicitly.\n")
    elif chosen_k is not None:
        print(f"\n[FILTER] All {len(kept)} conditions consistently use "
              f"adaptive_krum_k={chosen_k} — no contamination detected.\n")

    rows = kept
    if not rows:
        print("All conditions were filtered out — check --krum-k or use --no-filter.",
              file=sys.stderr)
        sys.exit(1)

    rows.sort(key=lambda r: (
        r["model_type"] or "",
        r["dp_epsilon_target"] if r["dp_epsilon_target"] is not None else float("inf"),
    ))

    csv_out = os.path.join(args.out, "sweep_summary.csv")
    md_out = os.path.join(args.out, "sweep_summary.md")
    per_class_out = os.path.join(args.out, "sweep_summary_per_class.csv")

    write_csv(rows, csv_out, SUMMARY_FIELDS)
    write_markdown(rows, md_out, SUMMARY_FIELDS)
    write_per_class_csv(rows, per_class_out)

    print(f"Wrote {len(rows)} rows to:")
    print(f"  {csv_out}")
    print(f"  {md_out}")
    print(f"  {per_class_out}")

    for r in rows:
        if r["rounds_completed"] < 25:
            print(f"  [note] {r['model_type']}/{r['tag']}: only {r['rounds_completed']}/25 rounds logged")


if __name__ == "__main__":
    main()
