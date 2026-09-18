#!/usr/bin/env python3
"""
Fit hetero_variance()'s regression (E6 fix #1) from PLAIN Adaptive Krum,
DP-OFF sanity runs and write the --hetero-fit-coeffs-json file.

Why this wrapper exists
-----------------------
1. Without a fit file, hetero calibration is a silent no-op, so E6's
   'Full' == 'DP-only' and 'Hetero-only' == 'Off'.
2. fit_hetero_variance_regression() groups honest rows by
   (round_id, alpha, epsilon). Concatenating CSVs from several seeds
   would therefore MERGE clients of different seeds into one fake
   "cohort". This wrapper makes round_id unique per source run.
3. Only clean input is accepted. Each CSV's sibling
   experiment_config_<tag>.json must show use_dp=False and
   use_adaptive_krum=True: calibrated runs log CALIBRATED scores under
   the column named raw_krum_score, and DP runs contaminate the
   'hetero' term with DP noise.

Recommended sanity sweep (plain Adaptive Krum, no DP, attack ON):
    for A in 10 0.7 0.3; do for S in 7 11 13; do
      python main.py network --ablation-mode krum_baseline \
        --aggregator adaptive_krum --alpha $A --seed $S \
        --attack-type sign_flip --tag fit_a${A}
    done; done
Then:
    python scripts/fit_hetero_variance.py \
      --csv-glob 'per_client_krum_scores_network_fit_a*_seed*.csv' \
      --out experiments/configs/hetero_fit_coeffs.json

Only training-time client statistics are used; no TEST/VALIDATION labels.
Seeds outside the evaluation set {42,123,456,789,2024} are recommended
(a warning is printed on overlap) so the fit is independent of the
campaign's partitions.
"""
import argparse, csv, glob, json, os, re, sys, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from defences.krum import fit_hetero_variance_regression  # noqa: E402

EVAL_SEEDS = {42, 123, 456, 789, 2024}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-r2-warn", type=float, default=0.02)
    a = ap.parse_args()

    files = sorted(glob.glob(a.csv_glob))
    if not files:
        sys.exit(f"No files match {a.csv_glob!r}")

    merged_rows, header, used = [], None, []
    for fpath in files:
        base = os.path.basename(fpath)
        tag = base[len("per_client_krum_scores_"):-len(".csv")]
        cfg_path = os.path.join(os.path.dirname(fpath) or ".",
                                f"experiment_config_{tag}.json")
        if not os.path.exists(cfg_path):
            sys.exit(f"Missing manifest {cfg_path} -- cannot verify {base} "
                     f"is plain Adaptive Krum with DP off.")
        cfg = json.load(open(cfg_path))
        if cfg.get("use_dp"):
            sys.exit(f"{base}: DP was ON -- would contaminate the hetero fit.")
        if not cfg.get("use_adaptive_krum"):
            sys.exit(f"{base}: not plain adaptive_krum "
                     f"(aggregator={cfg.get('aggregator')!r}); its logged scores "
                     f"are not raw Krum scores.")
        m = re.search(r"seed(\d+)", tag)
        seed = int(m.group(1)) if m else tag
        if isinstance(seed, int) and seed in EVAL_SEEDS:
            print(f"  WARNING: {base} uses evaluation seed {seed}; prefer "
                  f"pilot seeds outside {sorted(EVAL_SEEDS)}.")
        with open(fpath, newline="") as f:
            r = csv.reader(f)
            h = next(r)
            header = header or h
            ri = h.index("round_id")
            for row in r:
                row[ri] = f"{tag}::{row[ri]}"      # unique cohort per run
                merged_rows.append(row)
        used.append(base)

    with tempfile.NamedTemporaryFile("w", suffix=".csv", newline="",
                                     delete=False) as tmp:
        w = csv.writer(tmp)
        w.writerow(header)
        w.writerows(merged_rows)
        tmp_path = tmp.name
    try:
        coeffs = fit_hetero_variance_regression(tmp_path)
    finally:
        os.unlink(tmp_path)

    coeffs["source_files"] = used
    if coeffs["r_squared"] < a.min_r2_warn:
        print(f"  WARNING: r_squared={coeffs['r_squared']:.4f} is very low -- "
              f"report honestly; the hetero term will be weak.")
    if coeffs["coef_n_samples_diff"] < 0 or coeffs["coef_entropy_diff"] < 0:
        print("  WARNING: a coefficient is negative -- the variance term "
              "would shrink with more heterogeneity. Inspect before use.")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(coeffs, open(a.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in coeffs.items() if k != "source_files"},
                     indent=2))
    print(f"Wrote {a.out} from {len(used)} run(s).")


if __name__ == "__main__":
    main()
