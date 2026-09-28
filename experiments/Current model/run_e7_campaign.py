#!/usr/bin/env python3
"""
run_e7_campaign.py -- Issue 5, E7 (Cross-Dataset External Validation, CICIoT2023)

Cell matrix (84 cells, NOT the spec's nominal 108 -- see note below):
    FedAvg:            2 alpha x 2 attacks x 3 seeds          = 12 cells (no epsilon axis)
    Adaptive Krum:     2 alpha x 3 epsilon x 2 attacks x 3 seeds = 36 cells
    Calibrated Krum:   2 alpha x 3 epsilon x 2 attacks x 3 seeds = 36 cells
    TOTAL = 84

WHY NOT 108: the spec's nominal "3 methods x 2a x 3eps x 2attacks x 3seeds
= 108" only makes sense if FedAvg also varies by epsilon -- but this
codebase has NO ablation mode that combines plain FedAvg aggregation
with DP-SGD (only krum_dp_sweep and calibrated_krum_dp_sweep apply
DP). Running FedAvg three times under three different epsilon LABELS
with no DP actually applied would silently fabricate a fake epsilon
axis. FedAvg is therefore run once per (alpha, attack, seed), with no
epsilon dimension -- 84 cells, not 108. If this reading is wrong,
fix AGGREGATOR_CONFIG below and rebuild the cell list.

PREREQUISITE (apply before running): main.py needs the "fedavg_attack"
ablation mode added -- see main_py_patch.diff shipped alongside this
script. Without it, there is no existing mode that gives plain FedAvg
aggregation with the Byzantine attack switched on and DP/Krum/HE all
off, which this campaign's FedAvg cells require.

Dataset: --dataset ciciot2023 --ciciot-subset-fraction <frac> is
passed on every cell (not just some), since E7 is cross-dataset
validation entirely on CICIoT2023, per the issue.

Attack params: Gaussian needs no frozen params. Min-Max reuses the
Task 2 frozen params (same JSON file run_e4_campaign.py already
requires) -- Edge-IIoTset-calibrated 'auto' gamma is passed through
unchanged; if CICIoT2023's honest-client score spread differs enough
that this needs re-validation, that is a separate, real risk flagged
in this campaign's manifest but NOT auto-corrected here.

Idempotent / resumable: same reuse-audit-style skip logic as
run_e4_campaign.py -- a cell is skipped if its
results_*_FINAL_TEST.csv already exists and is non-empty, unless
--force is passed.

Sharding for parallel tmux launch: pass --shard N --num-shards K to
run only the cells where (cell_index % K == N) -- see
launch_e7_tmux.sh, which starts K=2 tmux sessions this way so both
run concurrently and each cell still lands in one canonical run_dir
regardless of which shard executed it.

Usage
-----
    python run_e7_campaign.py \\
        --main-py /path/to/main.py \\
        --output-root /path/to/experiments/results/E7 \\
        --frozen-attack-json /path/to/experiments/configs/frozen_minmax_params.json \\
        --ciciot-subset-fraction 0.01 \\
        [--shard 0 --num-shards 2] [--dry-run] [--force]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ALPHAS = [1, 0.3]
EPSILONS = [1, 5, 10]
ATTACKS = {"gaussian": "gaussian", "minmax": "minmax"}  # spec label -> --attack-type value
SEEDS = [42, 123, 456]

# canonical_name -> (ablation_mode, needs_epsilon, extra_cli_args)
METHODS = {
    "fedavg":            ("fedavg_attack",             False, ["--aggregator", "fedavg"]),
    "adaptive_krum":     ("krum_dp_sweep",              True,  ["--aggregator", "adaptive_krum"]),
    "calibrated_krum":   ("calibrated_krum_dp_sweep",   True,  []),
}


def build_cells():
    cells = []
    for method in METHODS:
        _, needs_eps, _ = METHODS[method]
        eps_values = EPSILONS if needs_eps else [None]
        for alpha in ALPHAS:
            for eps in eps_values:
                for attack in ATTACKS:
                    for seed in SEEDS:
                        cells.append({
                            "method": method, "alpha": alpha,
                            "epsilon": eps, "attack": attack, "seed": seed,
                        })
    return cells


def cell_tag(cell):
    eps_part = f"_eps{cell['epsilon']}" if cell["epsilon"] is not None else ""
    return (f"E7_{cell['method']}_a{cell['alpha']}{eps_part}_"
            f"{cell['attack']}_seed{cell['seed']}")


def cell_already_done(run_dir):
    test_files = list(run_dir.glob("results_*_FINAL_TEST.csv"))
    return any(f.stat().st_size > 0 for f in test_files)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--main-py", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--frozen-attack-json", required=True,
                    help="Same frozen Min-Max params JSON used for E4 (Task 2). "
                         "Gaussian cells ignore this file's contents.")
    p.add_argument("--ciciot-subset-fraction", type=float, default=0.01)
    p.add_argument("--ciciot-five-feature-slice", action="store_true")
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument("--shard", type=int, default=0,
                    help="Run only cells where index %% num_shards == shard.")
    p.add_argument("--num-shards", type=int, default=1)
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
        p.error(f"--frozen-attack-json missing required key(s): {missing}")

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    all_cells = build_cells()
    shard_cells = [c for i, c in enumerate(all_cells) if i % args.num_shards == args.shard]

    print(f"E7 campaign: {len(all_cells)} total cells "
          f"(FedAvg x{sum(1 for c in all_cells if c['method']=='fedavg')}, "
          f"AdaptiveKrum x{sum(1 for c in all_cells if c['method']=='adaptive_krum')}, "
          f"CalibratedKrum x{sum(1 for c in all_cells if c['method']=='calibrated_krum')})")
    print(f"Shard {args.shard}/{args.num_shards}: {len(shard_cells)} cells assigned "
          f"to this process")

    n_run = n_skip = n_fail = 0
    manifest_rows = []

    for cell in shard_cells:
        tag = cell_tag(cell)
        run_dir = output_root / tag
        run_dir.mkdir(parents=True, exist_ok=True)

        status = "NEW_RUN"
        if not args.force and cell_already_done(run_dir):
            print(f"  [SKIP] {tag} (output already present)")
            n_skip += 1
            manifest_rows.append({**cell, "tag": tag, "status": "REUSE_LOCAL"})
            continue

        ablation_mode, needs_eps, extra_args = METHODS[cell["method"]]
        cmd = [
            args.python_bin, str(main_py), "network",
            "--dataset", "ciciot2023",
            "--ciciot-subset-fraction", str(args.ciciot_subset_fraction),
            "--ablation-mode", ablation_mode,
            "--alpha", str(cell["alpha"]),
            "--seed", str(cell["seed"]),
            "--attack-type", ATTACKS[cell["attack"]],
            "--tag", tag,
        ] + extra_args

        if args.ciciot_five_feature_slice:
            cmd += ["--ciciot-five-feature-slice"]

        if needs_eps:
            cmd += ["--epsilon", str(cell["epsilon"])]

        if cell["attack"] == "minmax":
            cmd += [
                "--minmax-dev-type", str(frozen["minmax_dev_type"]),
                "--minmax-search-iters", str(frozen["minmax_search_iters"]),
            ]
            if frozen["minmax_gamma_init"] is not None:
                cmd += ["--minmax-gamma-init", str(frozen["minmax_gamma_init"])]

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

    manifest_path = output_root / f"e7_campaign_manifest_shard{args.shard}.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "shard": args.shard, "num_shards": args.num_shards,
            "cell_count_total": len(all_cells), "cell_count_this_shard": len(shard_cells),
            "alphas": ALPHAS, "epsilons": EPSILONS, "attacks": list(ATTACKS),
            "seeds": SEEDS, "ciciot_subset_fraction": args.ciciot_subset_fraction,
            "frozen_attack_params": frozen, "cells": manifest_rows,
        }, f, indent=2)

    print(f"\nShard {args.shard} done. run={n_run} skip={n_skip} fail={n_fail} "
          f"(dry_run={args.dry_run}). Manifest: {manifest_path}")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
