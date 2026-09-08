#!/usr/bin/env python3
"""
scripts/fit_hetero_variance.py

Wraps defences/krum.py's fit_hetero_variance_regression() into a runnable
CLI. Produces the JSON file main.py's --hetero-fit-coeffs-json flag
consumes, which is what actually activates calibrated_adaptive_multi_krum's
use_hetero_calibration term for real (until this has been run once, that
term stays a documented no-op -- see krum.py's hetero_variance()).

PREREQUISITE (blocking, not yet satisfied in this environment): a REAL
per_client_krum_scores.csv, produced by Sub-task A's Task 1 sanity sweep
(main.py's Issue-4 logger, run for real against the actual Edge-IIoTset
pipeline / GPU). This script does not fabricate or simulate that data --
if you don't have a real CSV yet, run Sub-task A's 6-run sanity sweep
first (see README_TASK4_TASK5.md).

Per krum.py's fit_hetero_variance_regression() docstring: this fits ONLY
against rows where ground_truth_client_label == "honest", and the caller
is responsible for having already filtered the input CSV to rows from
DP-INACTIVE runs (so the fitted "hetero" term isn't itself contaminated
by an unremoved DP-noise contribution -- Task 1's schema doesn't carry a
"was DP active" column, so this filtering must happen upstream, e.g. by
only pointing this script at the epsilon=none runs' CSV rows).

Usage:
    python scripts/fit_hetero_variance.py \\
        --csv per_client_krum_scores_network_baseline_seed42.csv \\
        --out hetero_fit_coeffs.json

    # If your sanity sweep produced multiple CSVs (one per alpha/seed
    # combination) and you want to fit against all of them pooled:
    python scripts/fit_hetero_variance.py \\
        --csv results/*.csv \\
        --out hetero_fit_coeffs.json
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from defences.krum import fit_hetero_variance_regression


def _pool_csvs(csv_patterns, pooled_path):
    """
    fit_hetero_variance_regression() takes a single CSV path. If the
    caller passes multiple glob patterns (multiple sanity-sweep runs),
    pool them into one temp CSV first -- concatenating rows, keeping only
    the first header. Does NOT deduplicate rows; caller's responsibility
    to pass non-overlapping files.
    """
    import csv as _csv
    all_rows = []
    header = None
    n_files = 0
    for pattern in csv_patterns:
        for path in sorted(glob.glob(pattern)):
            n_files += 1
            with open(path, newline="") as f:
                reader = _csv.reader(f)
                file_header = next(reader)
                if header is None:
                    header = file_header
                elif file_header != header:
                    print(f"  WARNING: {path} has a different header than "
                          f"the first file -- skipping this file. Headers "
                          f"must match Task 1's exact schema.")
                    continue
                for row in reader:
                    all_rows.append(row)
    if n_files == 0:
        raise FileNotFoundError(
            f"No files matched any of: {csv_patterns}. Nothing to fit."
        )
    with open(pooled_path, "w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_rows)
    print(f"  Pooled {n_files} file(s), {len(all_rows)} total rows -> {pooled_path}")
    return pooled_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", nargs="+", required=True,
                    help="Path(s)/glob pattern(s) to real per_client_krum_scores.csv "
                         "file(s) from Sub-task A's sanity sweep. If more than one "
                         "path/pattern is given, they are pooled before fitting.")
    p.add_argument("--out", required=True,
                    help="Output path for the fitted coefficients JSON "
                         "(pass this to main.py's --hetero-fit-coeffs-json).")
    args = p.parse_args()

    if len(args.csv) == 1 and os.path.isfile(args.csv[0]):
        csv_path = args.csv[0]
    else:
        csv_path = _pool_csvs(args.csv, args.out + ".pooled_input.csv")

    print(f"  Fitting hetero_variance regression against: {csv_path}")
    coeffs = fit_hetero_variance_regression(csv_path)

    print(f"  Fit result:")
    print(f"    n_rows_fit          = {coeffs['n_rows_fit']}")
    print(f"    intercept           = {coeffs['intercept']:.6f}")
    print(f"    coef_n_samples_diff = {coeffs['coef_n_samples_diff']:.6e}")
    print(f"    coef_entropy_diff   = {coeffs['coef_entropy_diff']:.6f}")
    print(f"    r_squared           = {coeffs['r_squared']:.4f}")

    if coeffs["r_squared"] < 0.2:
        print(f"\n  WARNING: r_squared={coeffs['r_squared']:.4f} is low. Per "
              f"the reporting standard established throughout this project: "
              f"report this honestly rather than treating the fit as "
              f"validated. Consider whether hetero_variance's linear-in-"
              f"|diff| functional form is actually appropriate before "
              f"citing Task 5 numbers computed with these coefficients.")

    with open(args.out, "w") as f:
        json.dump(coeffs, f, indent=2)
    print(f"\n  Written to: {args.out}")
    print(f"  Use with: python main.py ... --hetero-fit-coeffs-json {args.out}")


if __name__ == "__main__":
    main()
