#!/usr/bin/env python3
"""
Drop-in replacement for your fit_hetero_variance.py (same CLI: --csv ... --out ...).

Changes vs your version:
  1. UNIQUE COHORT PER RUN. fit_hetero_variance_regression() groups honest
     rows by (round_id, alpha, epsilon). Pooling several runs' CSVs with
     identical round_ids merges different seeds/runs into ONE fake cohort
     (n honest = 8*S instead of 8): the per-neighbour normaliser becomes
     8*S-2 instead of 6, and pairs are formed between clients that never
     coexisted. round_id is now prefixed with the source file's tag.
  2. INPUT VALIDATION via sibling experiment_config_<tag>.json: refuses
     DP-on runs (DP noise contaminates the 'hetero' term) and non-plain-
     adaptive-Krum runs (Calibrated Krum logs CALIBRATED scores in the
     raw_krum_score column).
  3. Prints how large the fitted variance term is at typical/extreme
     heterogeneity, so a dominating intercept is visible immediately.
"""
import argparse, csv, glob, json, os, re, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from defences.krum import fit_hetero_variance_regression  # noqa: E402


def _pool_csvs(patterns, pooled_path):
    header, rows, n_files = None, [], 0
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            base = os.path.basename(path)
            tag = re.sub(r"^per_client_krum_scores_|\.csv$", "", base)
            cfg_path = os.path.join(os.path.dirname(path) or ".",
                                    f"experiment_config_{tag}.json")
            if not os.path.exists(cfg_path):
                sys.exit(f"Missing {cfg_path}: cannot verify {base} is plain "
                         f"Adaptive Krum with DP off.")
            cfg = json.load(open(cfg_path))
            if cfg.get("use_dp"):
                sys.exit(f"{base}: DP ON -- contaminates the hetero fit.")
            if not cfg.get("use_adaptive_krum"):
                sys.exit(f"{base}: aggregator={cfg.get('aggregator')!r}, not plain "
                         f"adaptive_krum -- logged scores are not raw Krum scores.")
            with open(path, newline="") as f:
                r = csv.reader(f)
                h = next(r)
                if header is None:
                    header = h
                elif h != header:
                    print(f"  WARNING: {path} header differs -- skipped.")
                    continue
                ri = h.index("round_id")
                for row in r:
                    row[ri] = f"{tag}::{row[ri]}"      # unique cohort per run
                    rows.append(row)
            n_files += 1
    if n_files == 0:
        raise FileNotFoundError(f"No files matched: {patterns}")
    with open(pooled_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"  Pooled {n_files} file(s), {len(rows)} rows (cohorts kept separate).")
    return pooled_path, header, rows


def _scale_report(coeffs, header, rows):
    i_lab = header.index("ground_truth_client_label")
    i_n, i_e = header.index("client_n_samples"), header.index("client_class_entropy")
    hon = [(float(r[i_n]), float(r[i_e])) for r in rows if r[i_lab] == "honest"]
    ns = np.array([a for a, _ in hon]); es = np.array([b for _, b in hon])
    n90 = float(np.percentile(np.abs(ns[:, None] - ns[None, :]), 90))
    e90 = float(np.percentile(np.abs(es[:, None] - es[None, :]), 90))
    def var(nd, ed):
        p = coeffs["intercept"] + coeffs["coef_n_samples_diff"] * nd \
            + coeffs["coef_entropy_diff"] * ed
        return max(0.0, np.expm1(p)) ** 2
    v0, v90 = var(0, 0), var(n90, e90)
    print(f"  Fitted variance at zero heterogeneity      : {v0:.3e}")
    print(f"  Fitted variance at 90th-pct heterogeneity  : {v90:.3e} "
          f"(n_diff={n90:.0f}, ent_diff={e90:.2f})")
    print(f"  Max/min ratio across honest pairs           : {v90 / v0:.2f}x")
    if v90 / v0 < 1.5:
        print("  WARNING: term is nearly constant across pairs -- the intercept "
              "dominates, so hetero calibration ~ uniform rescaling (a no-op "
              "for selection). Full == DP-only in practice.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    pooled = a.out + ".pooled_input.csv"
    csv_path, header, rows = _pool_csvs(a.csv, pooled)
    print(f"  Fitting against: {csv_path}")
    coeffs = fit_hetero_variance_regression(csv_path)
    for k, v in coeffs.items():
        print(f"    {k} = {v}")
    _scale_report(coeffs, header, rows)
    json.dump(coeffs, open(a.out, "w"), indent=2)
    print(f"\n  Written to: {a.out}")


if __name__ == "__main__":
    main()
