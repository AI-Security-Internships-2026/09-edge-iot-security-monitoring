#!/usr/bin/env python3
"""
run_e4_campaign.py -- Issue 5, E4 (Correct Cumulative epsilon Study)

Dispatches `main.py` subprocess runs across:
    epsilon grid (fixed at the issue's minimum required 7 points:
    0.2, 0.5, 1, 3, 5, 10, 15)  x  {Adaptive Krum, Calibrated Krum}
    x  1 seed (default 42)
holding fixed: Edge-IIoTset, network model, alpha=0.7, one frozen
stealth attack (Min-Max).

That's 7 x 2 x 1 = 14 cells total. NOTE: the issue's Task 2/E4
acceptance criteria call for "minimum 3 paired seeds at representative
epsilon values" for the final paired statistical comparison (Task 4)
-- a 1-seed pass gives you a first look at the curve shape and lets
you sanity-check runtime, but it is NOT sufficient on its own to
support Table 3 / Figure 4's paired seed-matched comparison. Rerun
with --seeds 42 123 456 (or run 2 more single-seed passes with
--seeds 123 and --seeds 456 later) before treating these as final.

--aggregators lets you shard the 14 cells for parallel execution
(e.g. across 2 tmux sessions, one per aggregator):
    session 1: --aggregators adaptive_krum    (7 cells)
    session 2: --aggregators calibrated_krum  (7 cells)

Design choices (stated explicitly so they're auditable, not implicit):

  * Attack parameters are read from --frozen-attack-json and passed
    through as-is, INCLUDING a null minmax_gamma_init: main.py's own
    default for --minmax-gamma-init is None ("auto: 5x coalition
    spread", recomputed each round from the actual client updates),
    so a null value here is a legitimate frozen *procedure*, not a
    missing literal. When the frozen JSON's minmax_gamma_init is
    null/None, the --minmax-gamma-init flag is OMITTED from the
    command entirely (passing the literal string "None" would crash
    argparse's type=float parser) -- main.py then falls through to
    its own None default, which is the same auto behavior.

  * Idempotent / resumable: before launching a cell, checks whether
    that cell's dp_final_epsilon_<tag>.json and
    results_<tag>_FINAL_TEST.csv already exist and are non-empty in
    the run's output directory. If so, the cell is SKIPPED. Use
    --force to ignore this and rerun anyway. This means the two tmux
    sessions can each be re-launched safely if one dies partway
    through -- already-done cells in its shard are just skipped.

  * Every subprocess call's cwd is a per-run directory, not the repo
    root -- main.py has no --outdir flag and writes tag-named files
    into its cwd, so isolating cwd per (aggregator, epsilon) keeps
    output files from colliding and makes each run traceable back to
    its own directory name.

Usage
-----
    python run_e4_campaign.py \\
        --main-py /path/to/experiments/Current\\ model/main.py \\
        --output-root /path/to/experiments/results/E4 \\
        --frozen-attack-json /path/to/experiments/configs/frozen_minmax_params.json \\
        --aggregators adaptive_krum \\
        [--seeds 42] [--dry-run] [--force]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

EPSILONS = [0.2, 0.5, 1, 3, 5, 10, 15]

AGGREGATORS = {
    # canonical_name -> (ablation_mode, extra_cli_args)
    "adaptive_krum":   ("krum_dp_sweep", ["--aggregator", "adaptive_krum"]),
    "calibrated_krum": ("calibrated_krum_dp_sweep", []),
}

ALPHA = 0.7
ATTACK_TYPE = "minmax"


def build_cells(aggregators, seeds):
    for agg in aggregators:
        for eps in EPSILONS:
            for seed in seeds:
                yield {"aggregator": agg, "epsilon": eps, "seed": seed}


def cell_tag(cell):
    return f"E4_{cell['aggregator']}_eps{cell['epsilon']}_a{ALPHA}_{ATTACK_TYPE}_seed{cell['seed']}"


def cell_already_done(run_dir):
    eps_files = list(run_dir.glob("dp_final_epsilon_*.json"))
    test_files = list(run_dir.glob("results_*_FINAL_TEST.csv"))
    return (
        any(f.stat().st_size > 0 for f in eps_files)
        and any(f.stat().st_size > 0 for f in test_files)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--main-py", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--frozen-attack-json", required=True)
    p.add_argument("--aggregators", nargs="+",
                    choices=list(AGGREGATORS.keys()),
                    default=list(AGGREGATORS.keys()),
                    help="Which aggregator(s) to run in THIS invocation. "
                         "Pass one per tmux session to shard the campaign, "
                         "e.g. 'adaptive_krum' in session 1 and "
                         "'calibrated_krum' in session 2.")
    p.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="Seed(s) to run. Default: single seed (42). "
                         "The issue requires >=3 for the final paired "
                         "comparison -- see module docstring.")
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    main_py = Path(args.main_py).resolve()
    if not main_py.is_file():
        p.error(f"--main-py not found: {main_py}")

    with open(args.frozen_attack_json) as f:
        frozen = json.load(f)
    required_keys = {"minmax_dev_type", "minmax_search_iters", "minmax_gamma_init"}
    missing = required_keys - frozen.keys()
    if missing:
        p.error(f"--frozen-attack-json is missing required key(s): {missing}")

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if len(args.seeds) < 3:
        print(f"*** WARNING: running with {len(args.seeds)} seed(s) "
              f"({args.seeds}). The issue's Task 2/E4 acceptance criteria "
              f"require >=3 paired seeds at these representative epsilon "
              f"points for the final Table 3/Figure 4 comparison. This "
              f"pass is fine for a first look / runtime check, but plan "
              f"to fill in the remaining seeds before treating it as "
              f"final. ***\n")

    cells = list(build_cells(args.aggregators, args.seeds))
    print(f"E4 campaign shard: {len(cells)} cells "
          f"(aggregators={args.aggregators}, epsilons={EPSILONS}, "
          f"seeds={args.seeds}, attack={ATTACK_TYPE}, alpha={ALPHA})")

    n_run = n_skip = n_fail = 0
    manifest_rows = []

    for cell in cells:
        tag = cell_tag(cell)
        run_dir = output_root / tag
        run_dir.mkdir(parents=True, exist_ok=True)

        if not args.force and cell_already_done(run_dir):
            print(f"  [SKIP] {tag} (output already present; use --force to rerun)")
            n_skip += 1
            manifest_rows.append({**cell, "tag": tag, "status": "REUSE_LOCAL"})
            continue

        ablation_mode, extra_args = AGGREGATORS[cell["aggregator"]]
        cmd = [
            args.python_bin, str(main_py), "network",
            "--dataset", "edge_iiotset",
            "--ablation-mode", ablation_mode,
            "--alpha", str(ALPHA),
            "--epsilon", str(cell["epsilon"]),
            "--seed", str(cell["seed"]),
            "--attack-type", ATTACK_TYPE,
            "--minmax-dev-type", str(frozen["minmax_dev_type"]),
            "--minmax-search-iters", str(frozen["minmax_search_iters"]),
            "--tag", tag,
        ] + extra_args

        # Only pass --minmax-gamma-init if the frozen value is an actual
        # number. A null/None here means main.py's own auto-init default
        # -- passing the literal string "None" would crash argparse's
        # type=float parser, so omit the flag instead in that case.
        gamma_init = frozen.get("minmax_gamma_init")
        if gamma_init is not None:
            cmd += ["--minmax-gamma-init", str(gamma_init)]

        print(f"  [NEW_RUN] {tag}")
        print(f"    cwd={run_dir}")
        print(f"    cmd={' '.join(cmd)}")

        if args.dry_run:
            manifest_rows.append({**cell, "tag": tag, "status": "DRY_RUN"})
            continue

        log_path = run_dir / "run.log"
        with open(log_path, "w") as logf:
            result = subprocess.run(cmd, cwd=run_dir, stdout=logf,
                                     stderr=subprocess.STDOUT)
        if result.returncode != 0:
            print(f"    *** FAILED (exit {result.returncode}) -- see {log_path}")
            n_fail += 1
            manifest_rows.append({**cell, "tag": tag, "status": "FAILED"})
        else:
            print(f"    OK -- log at {log_path}")
            n_run += 1
            manifest_rows.append({**cell, "tag": tag, "status": "NEW_RUN_OK"})

    manifest_path = output_root / f"e4_campaign_manifest_{'_'.join(args.aggregators)}.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "aggregators": args.aggregators, "epsilons": EPSILONS,
            "seeds": args.seeds, "alpha": ALPHA, "attack_type": ATTACK_TYPE,
            "frozen_attack_params": frozen,
            "cells": manifest_rows,
        }, f, indent=2)

    print(f"\nDone. run={n_run} skip={n_skip} fail={n_fail} "
          f"(dry_run={args.dry_run}). Manifest: {manifest_path}")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
