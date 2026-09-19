#!/usr/bin/env python3
"""
refit_hetero_alpha0.7.py

Reproduces the existing alpha=0.3 hetero_fit_coeffs protocol
(documented in experiments/configs/hetero_fit_coeffs_a0.3.json's
write-up) at alpha=0.7, for E2's base condition -- the alpha=0.3 fit is
explicitly documented as not valid outside alpha=0.3, and E2 runs at
alpha=0.7, so calibrated_krum's E2 cell needs its own fit.

Protocol (identical to the alpha=0.3 fit, alpha changed only):
  1. Three data-collection runs: seeds 7, 11, 13; krum_baseline;
     --aggregator adaptive_krum; DP off; --attack-type sign_flip;
     --krum-k 3.5; --rounds 25; --alpha 0.7.
  2. Each run produces its own per_client_krum_scores_network_<tag>_
     seed<seed>.csv (main.py's Issue 4 Task 1 logger -- written
     automatically whenever USE_ADAPTIVE_KRUM/USE_CALIBRATED_KRUM is
     active, no extra flag needed).
  3. scripts/fit_hetero_variance_fixed.py consumes the three CSVs and
     calls defences/krum.py:fit_hetero_variance_regression() -- kept
     as a SEPARATE step here (not reimplemented) since that script
     already encodes the "keep each run's cohort separate" pooling fix
     described in the alpha=0.3 write-up.

ASSUMPTION FLAGGED: this script assumes scripts/fit_hetero_variance_fixed.py
accepts multiple --scores-csv arguments (one per seed) and a single
--out path, mirroring the alpha=0.3 artifact's stated inputs (three
per-seed CSVs -> one hetero_fit_coeffs_a0.7.json). fit_hetero_variance_fixed.py
itself was not available when this script was written -- if its real
CLI differs, only the FIT_CMD list below needs adjusting; the three
data-collection runs above are correct regardless of the fit script's
own interface.

Usage:
    python scripts/refit_hetero_alpha0.7.py
    python scripts/refit_hetero_alpha0.7.py --skip-run   # fit only, reuse existing CSVs
    python scripts/refit_hetero_alpha0.7.py --seeds 7,11,13,17  # add a 4th seed
"""

import argparse
import glob
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_MAIN_PY = os.path.join(_REPO_ROOT, "main.py")
_FIT_SCRIPT = os.path.join(_HERE, "fit_hetero_variance_fixed.py")

ALPHA = 0.7
DEFAULT_SEEDS = [7, 11, 13]
OUT_PATH = os.path.join(
    _REPO_ROOT, "experiments", "configs", "hetero_fit_coeffs_a0.7.json"
)


def tag_for(seed):
    return f"fit_a{str(ALPHA).replace('.', 'p')}_seed{seed}"


def run_one_seed(seed, python_bin=sys.executable):
    tag = tag_for(seed)
    cmd = [
        python_bin, _MAIN_PY, "network",
        "--ablation-mode", "krum_baseline",
        "--aggregator", "adaptive_krum",
        "--attack-type", "sign_flip",
        "--alpha", str(ALPHA),
        "--krum-k", "3.5",
        "--rounds", "25",
        "--seed", str(seed),
        "--tag", tag,
    ]
    print("=" * 70)
    print(f"[seed {seed}] RUNNING:", " ".join(cmd))
    print("=" * 70)
    subprocess.run(cmd, cwd=_REPO_ROOT, check=True)

    candidates = glob.glob(
        os.path.join(_REPO_ROOT, f"per_client_krum_scores_network_{tag}_seed{seed}.csv")
    )
    if not candidates:
        candidates = glob.glob(
            os.path.join(_REPO_ROOT, "**",
                         f"per_client_krum_scores_network_{tag}_seed{seed}.csv"),
            recursive=True,
        )
    if not candidates:
        raise FileNotFoundError(
            f"[seed {seed}] Expected per_client_krum_scores_network_{tag}_"
            f"seed{seed}.csv after a completed run -- not found. Did "
            f"main.py actually finish, and is USE_ADAPTIVE_KRUM active "
            f"(it should be, via --aggregator adaptive_krum + "
            f"--ablation-mode krum_baseline)?"
        )
    return candidates[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)),
                     help="Comma-separated seed list. Default matches the "
                          "alpha=0.3 fit's seeds (7,11,13) so the two fits "
                          "are directly comparable.")
    ap.add_argument("--skip-run", action="store_true",
                     help="Skip the 3 data-collection runs; assume their "
                          "per_client_krum_scores_*.csv files already "
                          "exist in the repo root and fit directly.")
    ap.add_argument("--out", type=str, default=OUT_PATH,
                     help="Output path for the fitted coefficients JSON.")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    csv_paths = []
    for seed in seeds:
        if args.skip_run:
            tag = tag_for(seed)
            matches = glob.glob(
                os.path.join(_REPO_ROOT,
                             f"per_client_krum_scores_network_{tag}_seed{seed}.csv")
            )
            if not matches:
                print(f"[seed {seed}] --skip-run given but no existing CSV "
                      f"found for tag={tag!r} -- cannot skip this seed.",
                      file=sys.stderr)
                sys.exit(1)
            csv_paths.append(matches[0])
        else:
            csv_paths.append(run_one_seed(seed))

    print("\nData-collection complete. CSVs:")
    for p in csv_paths:
        print(f"  {p}")

    if not os.path.exists(_FIT_SCRIPT):
        print(f"\n[MANUAL STEP REQUIRED] {_FIT_SCRIPT} was not found in "
              f"this environment, so the fit step was NOT run "
              f"automatically. Run it yourself against these {len(csv_paths)} "
              f"CSVs (check its actual --help for the real flag names -- "
              f"the ASSUMPTION note at the top of this script explains "
              f"why this couldn't be verified in advance):\n"
              f"    python {_FIT_SCRIPT} "
              f"{' '.join(csv_paths)} --out {args.out}\n"
              f"Then confirm the output JSON has \"schema_version\": 2 "
              f"(fit_hetero_variance_regression()'s current schema) "
              f"before pointing --hetero-fit-coeffs-json at it.")
        sys.exit(2)

    fit_cmd = [sys.executable, _FIT_SCRIPT, *csv_paths, "--out", args.out]
    print("\nRUNNING FIT:", " ".join(fit_cmd))
    subprocess.run(fit_cmd, cwd=_REPO_ROOT, check=True)

    print(f"\nDone. alpha=0.7 hetero fit written to: {args.out}")
    print("Sanity checks before trusting this for E2:")
    print("  - r_squared should be reported honestly, not suspiciously")
    print("    high (fit_hetero_variance_regression() already warns if")
    print("    r_squared > 0.95 -- read that warning if it fires).")
    print("  - schema_version must be 2.")
    print("  - Pass this file to E2's calibrated_krum cells via")
    print("    --hetero-fit-coeffs-json, NOT hetero_fit_coeffs_a0.3.json.")


if __name__ == "__main__":
    main()
