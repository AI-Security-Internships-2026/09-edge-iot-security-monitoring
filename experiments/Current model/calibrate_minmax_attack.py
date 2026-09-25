#!/usr/bin/env python3
"""
calibrate_minmax_attack.py -- Issue 5, Task 2 (attack difficulty check)

Sweeps candidate Min-Max attack parameters against a NO-DP Adaptive
Krum baseline (--ablation-mode krum_baseline) and reports, per
candidate:

  * Byzantine TPR / Honest FPR, derived ONLY from
    per_client_krum_scores_*.csv's per-round classification column
    -- a training-time diagnostic, never from results_*_FINAL_TEST.csv.
    This script does not read FINAL_TEST.csv at all, by design: Task 2
    explicitly prohibits tuning attack strength against TEST results,
    and the easiest way to guarantee that isn't violated is to never
    open that file here.

  * Attack-geometry diagnostics from attack_diag_*.jsonl (which your
    E4 campaign run also produces, but its CSV-only pull earlier
    didn't grab): mean constraint_ratio (how much of the
    Fang-et-al. "stay within the coalition's own max pairwise
    distance" budget the attack actually used) and whether gamma
    saturated at the search's own upper bound every round. This is
    the actionable part -- see WHY THIS MATTERS below.

WHY THIS MATTERS (read before picking a value)
------------------------------------------------
Your last E4 run used minmax_gamma_init=null (main.py's own "auto:
5x coalition max-pairwise-distance" default) and got 0/50 Byzantine
detections across all 7 epsilon values, both aggregators. Looking at
one round's raw scores directly: the two Byzantine clients scored
LOWER (more "central"/consistent) than every honest client -- the
opposite of what a detectable outlier should look like.

gamma_init is only the binary search's UPPER BOUND on how far the
crafted update is allowed to search for a feasible perturbation --
NOT the perturbation strength itself. Whether raising it actually
produces a stronger (more detectable) attack depends on whether the
search is SATURATING (gamma_star == gamma_search_upper_bound every
round, meaning the upper bound itself is comfortably feasible and a
larger one would very likely stay feasible and push farther) or
CONVERGING short of it (gamma_star meaningfully below gamma_hi,
meaning the coalition's own geometry -- how similar client 1's and
client 2's honestly-trained updates already are to each other -- is
the actual bottleneck, and raising gamma_init further won't help).

This script's "saturated" column tells you which regime you're in for
each candidate, so you're not guessing.

Usage
-----
    python calibrate_minmax_attack.py \\
        --main-py /path/to/experiments/Current\\ model/main.py \\
        --output-root /path/to/experiments/results/E4_calibration \\
        --rounds 10

Defaults to a small grid: gamma_init in {None, 1, 3, 10, 30} x
dev_type in {std, sign, unit_vec} = 15 short (no-DP) runs. Narrow with
--gammas / --dev-types if you already have a hunch (e.g. from the
diagnosis above -- if scores are landing well inside the honest range
at gamma_init=None, it's worth explicitly including a few LARGER
gamma_init candidates, which the default grid already does).

Target: pick the smallest candidate whose Byzantine TPR lands in
50-85% against this no-DP baseline (Task 2's stated range) -- not the
strongest attack available. A TPR near 100% here means the attack is
too easy for Adaptive Krum even before DP noise is added, which won't
usefully stress-test Calibrated Krum's added value in E4/E6.
"""
import argparse
import csv
import json
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

ALPHA = 0.7
DEFAULT_GAMMAS = [None, 1, 3, 10, 30]
DEFAULT_DEV_TYPES = ["std", "sign", "unit_vec"]


def cell_tag(gamma, dev_type):
    g = "auto" if gamma is None else str(gamma)
    return f"calib_minmax_g{g}_{dev_type}"


def run_one(main_py, run_dir, gamma, dev_type, rounds, seed, python_bin):
    tag = cell_tag(gamma, dev_type)
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_bin, str(main_py), "network",
        "--dataset", "edge_iiotset",
        "--ablation-mode", "krum_baseline",   # no DP -- isolates attack
        "--alpha", str(ALPHA),
        "--seed", str(seed),
        "--rounds", str(rounds),
        "--attack-type", "minmax",
        "--minmax-dev-type", dev_type,
        "--minmax-search-iters", "15",
        "--tag", tag,
    ]
    if gamma is not None:
        cmd += ["--minmax-gamma-init", str(gamma)]

    log_path = run_dir / "run.log"
    with open(log_path, "w") as logf:
        result = subprocess.run(cmd, cwd=run_dir, stdout=logf, stderr=subprocess.STDOUT)
    return tag, result.returncode, log_path


def read_tpr_fpr(run_dir):
    krum_csvs = list(run_dir.glob("per_client_krum_scores_*.csv"))
    if not krum_csvs:
        return None
    rows = list(csv.DictReader(open(krum_csvs[0])))
    c = Counter(r["classification"] for r in rows)
    tp, fp, tn, fn = c["TP"], c["FP"], c["TN"], c["FN"]
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "tpr": tpr, "fpr": fpr}


def read_attack_diag(run_dir):
    diag_files = list(run_dir.glob("attack_diag_*.jsonl"))
    if not diag_files:
        return None
    records = [json.loads(l) for l in open(diag_files[0]) if l.strip()]
    if not records:
        return None
    ratios = [r["constraint_ratio"] for r in records if "constraint_ratio" in r]
    saturated_flags = [
        abs(r["gamma"] - r["gamma_search_upper_bound"]) < 1e-9
        for r in records if "gamma" in r and "gamma_search_upper_bound" in r
    ]
    return {
        "mean_constraint_ratio": statistics.mean(ratios) if ratios else float("nan"),
        "pct_rounds_saturated": (100.0 * sum(saturated_flags) / len(saturated_flags)
                                  if saturated_flags else float("nan")),
        "n_rounds_logged": len(records),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--main-py", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--rounds", type=int, default=10,
                    help="Rounds per calibration run (default 10 -- enough "
                         "for Krum scores/attack geometry to reflect real "
                         "training dynamics without paying for a full "
                         "25-round DP run; this mode has no DP anyway).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gammas", type=str, nargs="+", default=None,
                    help="Override the default gamma_init grid. Pass 'none' "
                         "for the auto/5x-spread default, or a number, "
                         "e.g. --gammas none 1 3 10 30 100")
    p.add_argument("--dev-types", type=str, nargs="+",
                    default=DEFAULT_DEV_TYPES, choices=DEFAULT_DEV_TYPES)
    p.add_argument("--python-bin", default=sys.executable)
    args = p.parse_args()

    main_py = Path(args.main_py).resolve()
    if not main_py.is_file():
        p.error(f"--main-py not found: {main_py}")

    if args.gammas is None:
        gammas = DEFAULT_GAMMAS
    else:
        gammas = [None if g.lower() == "none" else float(g) for g in args.gammas]

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Calibrating minmax attack: {len(gammas)} gamma_init x "
          f"{len(args.dev_types)} dev_type = {len(gammas) * len(args.dev_types)} "
          f"runs, {args.rounds} rounds each, no-DP krum_baseline, alpha={ALPHA}.\n")

    results = []
    for dev_type in args.dev_types:
        for gamma in gammas:
            tag = cell_tag(gamma, dev_type)
            run_dir = output_root / tag
            tag, rc, log_path = run_one(main_py, run_dir, gamma, dev_type,
                                         args.rounds, args.seed, args.python_bin)
            if rc != 0:
                print(f"  [FAILED] {tag} -- see {log_path}")
                results.append({"tag": tag, "gamma_init": gamma,
                                 "dev_type": dev_type, "status": "FAILED"})
                continue

            tpr_fpr = read_tpr_fpr(run_dir)
            diag = read_attack_diag(run_dir)
            row = {"tag": tag, "gamma_init": gamma, "dev_type": dev_type,
                   "status": "OK"}
            if tpr_fpr:
                row.update(tpr_fpr)
            if diag:
                row.update(diag)
            results.append(row)

            g_str = "auto" if gamma is None else str(gamma)
            sat_str = (f"{row.get('pct_rounds_saturated', float('nan')):.0f}%"
                       if "pct_rounds_saturated" in row else "n/a")
            print(f"  gamma_init={g_str:>6s}  dev_type={dev_type:>9s}  "
                  f"TPR={row.get('tpr', float('nan')):.2f}  "
                  f"FPR={row.get('fpr', float('nan')):.2f}  "
                  f"mean_constraint_ratio={row.get('mean_constraint_ratio', float('nan')):.2f}  "
                  f"saturated={sat_str}")

    out_csv = output_root / "calibration_summary.csv"
    fieldnames = sorted({k for r in results for k in r.keys()},
                         key=lambda k: (k != "tag", k != "gamma_init",
                                        k != "dev_type", k != "tpr", k != "fpr"))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    print(f"\nWrote {out_csv}")
    print("\nPick the smallest gamma_init (least aggressive) whose TPR lands "
          "in 0.50-0.85 against this no-DP baseline. If EVERY candidate in "
          "the swept range saturates (saturated~=100%) AND still shows low "
          "TPR, the coalition's own honestly-trained updates are too "
          "similar to each other for Min-Max to ever look like an outlier "
          "at this alpha/dataset/client-count -- raising gamma_init further "
          "won't fix that; try a different --dev-types value, or treat "
          "'Min-Max is not a meaningfully stealthy attack under these "
          "conditions' as a legitimate, reportable finding for the paper "
          "(the issue explicitly allows mixed/negative results).")


if __name__ == "__main__":
    main()
