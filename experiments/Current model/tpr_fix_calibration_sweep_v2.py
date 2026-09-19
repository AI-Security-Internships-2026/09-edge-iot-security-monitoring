#!/usr/bin/env python3
"""
tpr_fix_calibration_sweep_v2.py

Replaces the coalition-SIZE sweep (tpr_fix_coalition_sweep.py), which we now
know is unreliable: growing the coalition changes the Min-Max attack's own
anchor point (w_avg = mean of the coalition's honestly-trained updates), and
that mean regresses toward the honest population's center as more members
are averaged in -- it is NOT guaranteed to push the crafted score further
from the honest cloud, and empirically did the opposite going from {1,2} to
{1,2,3}.

What this script sweeps instead -- both are things that can legitimately
change the result without falling into that regression-to-mean trap:

  1. dev_type: std (unsigned per-coordinate spread -- what you've been
     using) vs unit_vec / sign (both explicitly signed, -w_avg/||w_avg||-
     based directions -- push away from the coalition's own center rather
     than depending on coordinate-wise variance across members).
  2. WHICH pair of clients forms the coalition, holding size at 2 -- since
     the anchor is that pair's own mean, different pairs sit in different
     places relative to the honest population, independent of any gamma
     tuning. (--minmax-gamma-init is deliberately NOT swept here: at
     coalition size 2 the earlier diagnosis showed the ceiling isn't the
     binding constraint, so sweeping it is very unlikely to change
     anything and would waste a full 25-round run per value.)

For each combination it reports the same k=2.0..4.0 TPR/honest-FPR table
as before, so you can compare all candidates side by side and pick the
smallest, cleanest config that lands inside a workable band -- ideally
overlapping E5b's own k in {2.0, 2.5, 3.0, 3.5, 4.0} range, so the
"production" config used for E3/E4/E6 isn't sitting outside the range
E5b itself sweeps over.

Per Issue 5 Task 2: this is pre-freeze calibration against round-level
diagnostics from throwaway calibration runs -- NOT tuning against the
real campaign's final TEST results, which the ticket explicitly forbids.
Freeze whatever you pick here before launching the real E3/E4/E6 runs.

Usage:
    python tpr_fix_calibration_sweep_v2.py
    python tpr_fix_calibration_sweep_v2.py --coalitions "1,2" "3,4" "5,6" \
        --dev-types std unit_vec sign
    python tpr_fix_calibration_sweep_v2.py --skip-run   # analyze existing CSVs only
"""

import argparse
import csv
import glob
import subprocess
import sys
from collections import defaultdict

import numpy as np

K_VALUES = [2.0, 2.5, 3.0, 3.5, 4.0]

DEFAULT_ARGS = dict(
    model_type="network",
    rounds=25,
    alpha=0.7,
    seed=42,
    aggregator="adaptive_krum",
    ablation_mode="krum_baseline",
    attack_type="minmax",
    minmax_search_iters=15,
)


def tag_for(byz_str, dev_type):
    return f"tprcal_byz{byz_str.replace(',', '_')}_dev{dev_type}"


def run_one(byz_str, dev_type, main_py="main.py", python_bin=sys.executable):
    tag = tag_for(byz_str, dev_type)
    cmd = [
        python_bin, main_py, DEFAULT_ARGS["model_type"],
        "--rounds", str(DEFAULT_ARGS["rounds"]),
        "--alpha", str(DEFAULT_ARGS["alpha"]),
        "--byzantine", byz_str,
        "--seed", str(DEFAULT_ARGS["seed"]),
        "--aggregator", DEFAULT_ARGS["aggregator"],
        "--ablation-mode", DEFAULT_ARGS["ablation_mode"],
        "--attack-type", DEFAULT_ARGS["attack_type"],
        "--minmax-dev-type", dev_type,
        "--minmax-search-iters", str(DEFAULT_ARGS["minmax_search_iters"]),
        "--tag", tag,
    ]
    print("=" * 70)
    print("RUNNING:", " ".join(cmd))
    print("=" * 70)
    subprocess.run(cmd, check=True)
    return tag


def find_scores_csv(tag):
    candidates = glob.glob(f"per_client_krum_scores_network_{tag}_seed*.csv")
    if not candidates:
        candidates = glob.glob(f"**/per_client_krum_scores_network_{tag}_seed*.csv",
                                recursive=True)
    if not candidates:
        raise FileNotFoundError(
            f"No per_client_krum_scores_network_{tag}_seed*.csv found. "
            f"Did the run for tag={tag!r} actually complete?"
        )
    return candidates[0]


def load_rounds(csv_path):
    rounds = defaultdict(lambda: {"honest": [], "byzantine": []})
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                score = float(row["raw_krum_score"])
            except (KeyError, ValueError, TypeError):
                continue
            label = row.get("ground_truth_client_label", "")
            if label not in ("honest", "byzantine"):
                continue
            rounds[row["round_id"]][label].append(score)
    return rounds


def mad_threshold(all_scores, k):
    arr = np.asarray(all_scores, dtype=float)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    spread = 1.4826 * mad
    if spread <= 0:
        spread = 1e-9
    return median + k * spread


def summarize(csv_path):
    """Returns {k: (tpr_mean, tpr_min, fpr_mean, fpr_max)} plus raw
    byz/honest mean scores for a quick eyeball comparison."""
    rounds = load_rounds(csv_path)
    if not rounds:
        return None, None, None

    byz_means, honest_means = [], []
    per_k = {k: {"tpr": [], "fpr": []} for k in K_VALUES}

    for rid, groups in rounds.items():
        honest, byz = groups["honest"], groups["byzantine"]
        if not honest or not byz:
            continue
        byz_means.append(np.mean(byz))
        honest_means.append(np.mean(honest))
        all_scores = honest + byz
        for k in K_VALUES:
            thr = mad_threshold(all_scores, k)
            per_k[k]["tpr"].append(sum(1 for s in byz if s > thr) / len(byz))
            per_k[k]["fpr"].append(sum(1 for s in honest if s > thr) / len(honest))

    if not byz_means:
        return None, None, None

    table = {
        k: (
            np.mean(per_k[k]["tpr"]) * 100, np.min(per_k[k]["tpr"]) * 100,
            np.mean(per_k[k]["fpr"]) * 100, np.max(per_k[k]["fpr"]) * 100,
        )
        for k in K_VALUES
    }
    return table, float(np.mean(byz_means)), float(np.mean(honest_means))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coalitions", nargs="+", default=["1,2", "3,4", "5,6"],
                     help="Coalition PAIRS to try (size held at 2 unless you "
                          "pass a bigger group deliberately).")
    ap.add_argument("--dev-types", nargs="+", default=["std", "unit_vec", "sign"])
    ap.add_argument("--main-py", default="main.py")
    ap.add_argument("--skip-run", action="store_true")
    args = ap.parse_args()

    results = []
    for byz_str in args.coalitions:
        for dev_type in args.dev_types:
            tag = tag_for(byz_str, dev_type)
            if not args.skip_run:
                run_one(byz_str, dev_type, main_py=args.main_py)
            try:
                csv_path = find_scores_csv(tag)
            except FileNotFoundError as e:
                print(f"  [SKIP] {e}")
                continue
            table, byz_mean, honest_mean = summarize(csv_path)
            if table is None:
                print(f"  [WARN] no usable rows for {tag}")
                continue
            results.append((byz_str, dev_type, byz_mean, honest_mean, table))

    print("\n" + "=" * 100)
    print(f"{'coalition':>10} {'dev_type':>9} {'byz_mean':>12} {'honest_mean':>12} "
          f"{'k':>5} {'TPR_mean':>10} {'TPR_min':>9} {'honestFPR_mean':>16} {'honestFPR_max':>15}")
    print("-" * 100)
    for byz_str, dev_type, byz_mean, honest_mean, table in results:
        first = True
        for k in K_VALUES:
            tpr_mean, tpr_min, fpr_mean, fpr_max = table[k]
            prefix = (f"{byz_str:>10} {dev_type:>9} {byz_mean:>12,.0f} {honest_mean:>12,.0f}"
                      if first else " " * 45)
            print(f"{prefix} {k:>5.1f} {tpr_mean:>9.1f}% {tpr_min:>8.1f}% "
                  f"{fpr_mean:>15.1f}% {fpr_max:>14.1f}%")
            first = False
    print("=" * 100)
    print("\nLook for a row where TPR_mean is non-zero and non-saturated at a k "
          "inside {2.0-4.0} with honestFPR_max still near 0. Prefer configs "
          "where byz_mean clearly separates from honest_mean (not a razor-thin "
          "gap like the original 1,2/std case) so the result is robust across "
          "seeds, not just this one seed/round sample.")


if __name__ == "__main__":
    main()
