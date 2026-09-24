#!/usr/bin/env python3
"""
run_e4_campaign.py -- Issue 5, E4 (Correct Cumulative epsilon Study)

Dispatches `main.py` subprocess runs across:
    epsilon grid (fixed at the issue's minimum required 7 points:
    0.2, 0.5, 1, 3, 5, 10, 15)  x  {Adaptive Krum, Calibrated Krum}
    x  seeds (fixed at 3: 42, 123, 456)
holding fixed: Edge-IIoTset, network model, alpha=0.7, one frozen
stealth attack (Min-Max by default, per the issue spec unless
validation gives a documented reason to use another).

That's 7 x 2 x 3 = 42 cells total -- satisfies the issue's "minimum
3 paired seeds at representative epsilon values" requirement exactly,
since every one of these 7 points IS a representative point (they ARE
the minimum required fallback curve), so all 7 get full 3-seed
replication, uniformly.

Design choices (stated explicitly so they're auditable, not implicit):

  * Attack parameters (minmax gamma, dev-type, search-iters) are NOT
    invented here. Task 2 requires freezing them BEFORE final TEST
    runs, via your own calibration pass (e.g.
    scripts/check_attack_difficulty.py). This script reads them from
    a small JSON file (--frozen-attack-json) and refuses to run
    without one, rather than silently using main.py's un-frozen
    argparse defaults for a "final campaign" run.

  * Idempotent / resumable: before launching a cell, checks whether
    that cell's dp_final_epsilon_<tag>.json and
    results_<tag>_FINAL_TEST.csv already exist and are non-empty in
    the run's output directory. If so, the cell is SKIPPED (treated
    as already run) -- matches the reuse-audit spirit of not re-paying
    for a completed run. Use --force to ignore this and rerun anyway.

  * Every subprocess call's cwd is a per-run directory, not the repo
    root -- main.py has no --outdir flag and writes tag-named files
    into its cwd, so isolating cwd per (aggregator, epsilon, seed)
    keeps output files from colliding across parallel/sequential runs
    and makes the run trivially traceable back to its config directory
    name.

Usage
-----
    python run_e4_campaign.py \\
        --main-py /path/to/experiments/Current\\ model/main.py \\
        --output-root /path/to/experiments/results/E4 \\
        --frozen-attack-json /path/to/experiments/configs/frozen_minmax_params.json \\
        [--dense-grid] [--dry-run] [--force]

Default (no --dense-grid) uses the minimum required 7-point curve.
--dense-grid uses the full 19-point E4_dense_epsilon_sweep grid.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

# The issue's full dense grid (all cumulative full-run target epsilons).
DENSE_EPSILONS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5,
                   6, 7, 8, 9, 10, 15]

# The minimum required final curve if dense points are incomplete.
MINIMAL_EPSILONS = [0.2, 0.5, 1, 3, 5, 10, 15]

# Points that MUST get the full paired-seed replication regardless of
# which grid is selected (subset of MINIMAL_EPSILONS, by construction).
REPRESENTATIVE_EPSILONS = set(MINIMAL_EPSILONS)

AGGREGATORS = {
    # canonical_name -> (ablation_mode, extra_cli_args)
    "adaptive_krum":   ("krum_dp_sweep", ["--aggregator", "adaptive_krum"]),
    "calibrated_krum": ("calibrated_krum_dp_sweep", []),
}

ALPHA = 0.7
ATTACK_TYPE = "minmax"


def build_cells(dense_grid, dense_seeds, rep_seeds):
    """Yields one dict per experiment cell: aggregator, epsilon, seed."""
    epsilons = DENSE_EPSILONS if dense_grid else MINIMAL_EPSILONS
    for agg in AGGREGATORS:
        for eps in epsilons:
            seeds = rep_seeds if eps in REPRESENTATIVE_EPSILONS else dense_seeds
            for seed in seeds:
                yield {"aggregator": agg, "epsilon": eps, "seed": seed}


def cell_tag(cell):
    # Encodes every axis that distinguishes this cell -- readable AND
    # greppable, matches the --tag convention main.py already expects.
    return f"E4_{cell['aggregator']}_eps{cell['epsilon']}_a{ALPHA}_{ATTACK_TYPE}_seed{cell['seed']}"


def cell_already_done(run_dir, tag):
    eps_json = run_dir / f"dp_final_epsilon_network_{tag}_seed{tag.split('seed')[-1]}.json"
    # main.py's own _TAG is f"{MODEL_TYPE}_{tag_arg}_seed{seed}" -- the
    # seed is embedded in --tag here too (see cell_tag), so main.py's
    # constructed filename ends up seed-doubled. To avoid depending on
    # that exact string, just check any dp_final_epsilon_*.json and any
    # results_*_FINAL_TEST.csv exist and are non-empty in run_dir.
    eps_files = list(run_dir.glob("dp_final_epsilon_*.json"))
    test_files = list(run_dir.glob("results_*_FINAL_TEST.csv"))
    return (
        any(f.stat().st_size > 0 for f in eps_files)
        and any(f.stat().st_size > 0 for f in test_files)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--main-py", required=True,
                    help="Path to main.py (the FL-IDS unified training loop).")
    p.add_argument("--output-root", required=True,
                    help="Directory under which one subdirectory per cell is created.")
    p.add_argument("--frozen-attack-json", required=True,
                    help="JSON with frozen minmax attack params: "
                         "{'minmax_dev_type': 'std', 'minmax_search_iters': 15, "
                         "'minmax_gamma_init': <float>}. Must come from a real "
                         "calibration pass (Task 2), not invented here.")
    p.add_argument("--dense-grid", action="store_true",
                    help="Use the full 19-point epsilon grid instead of the "
                         "minimum required 7-point curve.")
    p.add_argument("--dense-seeds", type=int, nargs="+", default=[42],
                    help="Seed(s) used at non-representative epsilon points.")
    p.add_argument("--rep-seeds", type=int, nargs="+", default=[42, 123, 456],
                    help="Seed(s) used at REPRESENTATIVE_EPSILONS "
                         "(minimum 3 required by the issue).")
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument("--dry-run", action="store_true",
                    help="Print the commands that would run; execute nothing.")
    p.add_argument("--force", action="store_true",
                    help="Rerun a cell even if its output files already exist.")
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

    cells = list(build_cells(args.dense_grid, args.dense_seeds, args.rep_seeds))
    print(f"E4 campaign: {len(cells)} cells "
          f"({'dense 19-pt' if args.dense_grid else 'minimal 7-pt'} grid, "
          f"2 aggregators, attack={ATTACK_TYPE}, alpha={ALPHA})")

    n_run = n_skip = n_fail = 0
    manifest_rows = []

    for cell in cells:
        tag = cell_tag(cell)
        run_dir = output_root / tag
        run_dir.mkdir(parents=True, exist_ok=True)

        status = "NEW_RUN"
        if not args.force and cell_already_done(run_dir, tag):
            print(f"  [SKIP] {tag} (output already present; use --force to rerun)")
            n_skip += 1
            status = "REUSE_LOCAL"
            manifest_rows.append({**cell, "tag": tag, "status": status})
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
            "--minmax-gamma-init", str(frozen["minmax_gamma_init"]),
            "--tag", tag,
        ] + extra_args

        print(f"  [{status}] {tag}")
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
            status = "FAILED"
        else:
            print(f"    OK -- log at {log_path}")
            n_run += 1
            status = "NEW_RUN_OK"
        manifest_rows.append({**cell, "tag": tag, "status": status})

    manifest_path = output_root / "e4_campaign_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "grid": "dense_19pt" if args.dense_grid else "minimal_7pt",
            "alpha": ALPHA, "attack_type": ATTACK_TYPE,
            "frozen_attack_params": frozen,
            "cells": manifest_rows,
        }, f, indent=2)

    print(f"\nDone. run={n_run} skip={n_skip} fail={n_fail} "
          f"(dry_run={args.dry_run}). Manifest: {manifest_path}")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
