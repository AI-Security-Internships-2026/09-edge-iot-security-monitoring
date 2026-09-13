#!/usr/bin/env python3
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from defences.krum import fit_hetero_variance_regression


def _pool_csvs(csv_patterns, pooled_path):
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
                          f"the first file -- skipping this file.")
                    continue
                for row in reader:
                    all_rows.append(row)
    if n_files == 0:
        raise FileNotFoundError(f"No files matched: {csv_patterns}")
    with open(pooled_path, "w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_rows)
    print(f"  Pooled {n_files} file(s), {len(all_rows)} total rows -> {pooled_path}")
    return pooled_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if len(args.csv) == 1 and os.path.isfile(args.csv[0]):
        csv_path = args.csv[0]
    else:
        csv_path = _pool_csvs(args.csv, args.out + ".pooled_input.csv")

    print(f"  Fitting hetero_variance regression against: {csv_path}")
    coeffs = fit_hetero_variance_regression(csv_path)

    print(f"  Fit result:")
    for k, v in coeffs.items():
        print(f"    {k} = {v}")

    with open(args.out, "w") as f:
        json.dump(coeffs, f, indent=2)
    print(f"\n  Written to: {args.out}")


if __name__ == "__main__":
    main()
