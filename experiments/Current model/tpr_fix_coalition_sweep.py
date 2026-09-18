#!/usr/bin/env python3
"""
tpr_fix_coalition_sweep.py

Fixes the "Min-Max vs Adaptive Krum always gives 0% TPR" problem by finding
the smallest Byzantine coalition size whose crafted score clears the MAD
threshold at k values inside E5b's own planned sweep {2.0, 2.5, 3.0, 3.5, 4.0}
-- instead of lowering k below that range, which would silently make E5b's
"MAD-k sensitivity" experiment vacuous (0% at every one of its k values too)
and is fragile against honest-client false positives in other rounds/seeds.

Run this from your project root (where main.py lives), with your real
environment/dataset/GPU -- it actually launches main.py, so it can't be run
in a sandbox with no project modules or data.

What it does:
  1. Runs main.py once per candidate coalition size (--byzantine "1,2",
     "1,2,3", "1,2,3,4"), full 25 rounds, alpha=0.7, seed=42, plain
     Adaptive Krum, frozen Min-Max attack (gamma_init=None/auto,
     dev_type=std, search_iters=15) -- everything held identical to your
     existing calibration except coalition size.
  2. Reads back the per-round per_client_krum_scores_*.csv each run writes.
  3. Reproduces krum.py's own MAD threshold math (median + k * 1.4826*MAD)
     round-by-round, for k in {2.0, 2.5, 3.0, 3.5, 4.0}, and computes what
     TPR and honest FPR *would have been* at each k.
  4. Prints a per-coalition-size, per-k summary table (mean +/- across
     rounds) so you can pick the smallest coalition that lands TPR in a
     non-zero, non-saturated band at a k that's still inside E5b's range.

Usage:
    python tpr_fix_coalition_sweep.py
    python tpr_fix_coalition_sweep.py --coalitions 1,2 1,2,3 1,2,3,4
    python tpr_fix_coalition_sweep.py --skip-run   # analyze existing CSVs only
"""

import argparse
import csv
import glob
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np

K_VALUES = [2.0, 2.5, 3.0, 3.5, 4.0]

DEFAULT_RUN_ARGS = dict(
    model_type="network",
    rounds=25,
    alpha=0.7,
    seed=42,
    aggregator="adaptive_krum",
    ablation_mode="krum_baseline",
    attack_type="minmax",
    minmax_dev_type="std",
    minmax_search_iters=15,
)


def tag_for(byz_str):
    return f"tpr_fix_byz{byz_str.replace(',', '_')}"


def run_one(byz_str, main_py="main.py", python_bin=sys.executable):
    tag = tag_for(byz_str)
    cmd = [
        python_bin, main_py, DEFAULT_RUN_ARGS["model_type"],
        "--rounds", str(DEFAULT_RUN_ARGS["rounds"]),
        "--alpha", str(DEFAULT_RUN_ARGS["alpha"]),
        "--byzantine", byz_str,
        "--seed", str(DEFAULT_RUN_ARGS["seed"]),
        "--aggregator", DEFAULT_RUN_ARGS["aggregator"],
        "--ablation-mode", DEFAULT_RUN_ARGS["ablation_mode"],
        "--attack-type", DEFAULT_RUN_ARGS["attack_type"],
        "--minmax-dev-type", DEFAULT_RUN_ARGS["minmax_dev_type"],
        "--minmax-search-iters", str(DEFAULT_RUN_ARGS["minmax_search_iters"]),
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
        # also check common output dirs
        candidates = glob.glob(f"**/per_client_krum_scores_network_{tag}_seed*.csv",
                                recursive=True)
    if not candidates:
        raise FileNotFoundError(
            f"No per_client_krum_scores_network_{tag}_seed*.csv found. "
            f"Did the run for tag={tag!r} actually complete?"
        )
    return candidates[0]


def load_rounds(csv_path):
    """Returns {round_id: {"honest": [scores], "byzantine": [scores]}}"""
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
    return median + k * spread, median, spread


def analyze(csv_path, byz_str):
    rounds = load_rounds(csv_path)
    if not rounds:
        print(f"  [WARN] no usable rows in {csv_path}")
        return

    print(f"\n--- coalition {byz_str}  ({csv_path}, {len(rounds)} rounds) ---")
    header = f"{'k':>5} {'mean_threshold':>16} {'TPR_mean':>10} {'TPR_min':>9} " \
             f"{'honestFPR_mean':>16} {'honestFPR_max':>15}"
    print(header)

    for k in K_VALUES:
        tprs, fprs = [], []
        for rid, groups in sorted(rounds.items()):
            honest = groups["honest"]
            byz = groups["byzantine"]
            if not honest or not byz:
                continue
            all_scores = honest + byz
            thr, _, _ = mad_threshold(all_scores, k)
            tp = sum(1 for s in byz if s > thr)
            fp = sum(1 for s in honest if s > thr)
            tprs.append(tp / len(byz))
            fprs.append(fp / len(honest))
        if not tprs:
            continue
        mean_thr = np.mean([
            mad_threshold(groups["honest"] + groups["byzantine"], k)[0]
            for groups in rounds.values()
            if groups["honest"] and groups["byzantine"]
        ])
        print(f"{k:>5.1f} {mean_thr:>16,.0f} {np.mean(tprs)*100:>9.1f}% "
              f"{np.min(tprs)*100:>8.1f}% {np.mean(fprs)*100:>15.1f}% "
              f"{np.max(fprs)*100:>14.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coalitions", nargs="+", default=["1,2", "1,2,3", "1,2,3,4"],
                     help='Byzantine coalitions to try, e.g. --coalitions "1,2" "1,2,3"')
    ap.add_argument("--main-py", default="main.py")
    ap.add_argument("--skip-run", action="store_true",
                     help="Skip launching main.py; just analyze existing CSVs "
                          "for the given --coalitions tags.")
    args = ap.parse_args()

    for byz_str in args.coalitions:
        tag = tag_for(byz_str)
        if not args.skip_run:
            run_one(byz_str, main_py=args.main_py)
        try:
            csv_path = find_scores_csv(tag)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue
        analyze(csv_path, byz_str)

    print("\nPick the smallest coalition size where TPR_mean is non-zero and "
          "non-saturated (not pinned at 0% or 100%) at a k in {2.0..4.0}, "
          "with honestFPR_max still near 0. That's your new frozen attack "
          "config for E3/E4/E6 -- keep --minmax-gamma-init/dev-type/"
          "search-iters exactly as used here.")


if __name__ == "__main__":
    main()
