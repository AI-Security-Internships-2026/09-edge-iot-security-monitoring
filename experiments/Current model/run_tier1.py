#!/usr/bin/env python3
"""
Tier 0 + Tier 1 run orchestrator for the FL-IDS Experiment 2 seed-priority
plan (see the paper's v2 submission plan, §3).

Run this FROM THE REPO ROOT (same directory as main.py), inside the
flids_dev container, on the DGX Spark. It shells out to main.py --
it does not reimplement anything.

WHAT THIS DOES
--------------
Phase 0 (once, informational only, no runs):
    Recomputes the classifier-head / total-parameter split directly from
    model_defs.py and prints the Tier-0 resolution (see chat for the
    derivation). Does not touch any file -- just confirms the numbers
    match what you expect before burning compute on a config that turns
    out to be based on a stale percentage.

Phase 1.1 -- k / assumed_f confound grid (single seed, 4 runs):
    Runs the HE+Krum hybrid, mitigated, network model, byzantine 1,2,
    at every (k, assumed_f) combination in K_GRID x ASSUMED_F_GRID.
    Prints each run's collateral-exclusion count so you can pick the
    canonical (k, assumed_f) pair before Phase 1.2.

Phase 1.2 -- multi-seed Experiment 2 (40 runs):
    5 seeds x 8 configs (2 unmitigated + 6 mitigated across both models
    and the three attacker identities from the paper: 1,2 / 2,7 / 4,10),
    all run at the CANONICAL_K / CANONICAL_ASSUMED_F you set below after
    reviewing Phase 1.1's output.

RESUMABILITY
------------
Every run is logged to run_manifest.csv (created next to this script,
under the --results-dir you pass) BEFORE it starts and updated after it
finishes. Re-running this script skips any (tag) already marked "done"
in the manifest -- safe to re-invoke after a crash, a Ctrl-C, or the
known "USE_DP=True can't resume" limitation (irrelevant here since
Experiment 2 never has USE_DP=True, but the skip-if-done logic is
general).

USAGE
-----
    # 1. Sanity-check Tier 0 first, no runs:
    python3 run_tier1.py --phase 0

    # 2. Run the k/assumed_f grid (4 single-seed runs):
    python3 run_tier1.py --phase 1.1 --results-dir results/tier1_run1

    # 3. Look at the printed collateral-exclusion counts, pick your
    #    canonical k / assumed_f, edit CANONICAL_K / CANONICAL_ASSUMED_F
    #    below, then:
    python3 run_tier1.py --phase 1.2 --results-dir results/tier1_run1

    # Add --dry-run to any phase to print the exact main.py commands
    # without executing them.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# EDIT THESE AFTER REVIEWING PHASE 1.1's OUTPUT
# ---------------------------------------------------------------------------
CANONICAL_K = 2.5
CANONICAL_ASSUMED_F = 1

# ---------------------------------------------------------------------------
# Fixed experiment grids -- match the paper's plan exactly
# ---------------------------------------------------------------------------
K_GRID = [2.5, 3.5]
ASSUMED_F_GRID = [1, 2]

SEEDS = [42, 43, 44, 45, 46]

# (model, ablation_mode, byzantine_string) -- the 8 Experiment 2 configs
EXPERIMENT2_CONFIGS = [
    ("network",     "exp2_unmitigated", "1,2"),
    ("application", "exp2_unmitigated", "1,2"),
    ("network",     "exp2_mitigated",   "1,2"),
    ("network",     "exp2_mitigated",   "2,7"),
    ("network",     "exp2_mitigated",   "4,10"),
    ("application", "exp2_mitigated",   "1,2"),
    ("application", "exp2_mitigated",   "2,7"),
    ("application", "exp2_mitigated",   "4,10"),
]

MANIFEST_FIELDS = [
    "tag", "phase", "model", "ablation_mode", "byzantine", "k",
    "assumed_f", "seed", "status", "started", "finished",
    "returncode", "log_path",
]


def phase0_tier0_check():
    """Recompute classifier-head / total-param split from model_defs.py."""
    try:
        from model_defs import get_model
    except ImportError:
        print("Could not import model_defs -- run this from the repo root.")
        sys.exit(1)

    import torch  # noqa -- required by model_defs

    for num_features in (39, 90):
        model = get_model(num_features=num_features, num_classes=8, dp_safe=False)
        state = model.state_dict()
        total = sum(v.numel() for v in state.values())
        head = sum(v.numel() for k, v in state.items() if k.startswith("classifier"))
        trainable = sum(p.numel() for p in model.parameters())
        print(f"num_features={num_features:3d}  "
              f"total(incl. buffers)={total:6d}  "
              f"trainable_only={trainable:6d}  "
              f"classifier_head={head:5d}  "
              f"pct_of_total={100*head/total:.3f}%  "
              f"pct_of_trainable={100*head/trainable:.3f}%")

    print("\nExpect: classifier_head=4680 both rows, total(incl. buffers)="
          "80074, pct_of_total=5.85% -- this is the number to cite in "
          "§III-F3, replacing the paper's current 3.6%/129,352 figure.")
    print("If your numbers differ, main.py's model architecture has "
          "changed since this script was written -- recompute the fix "
          "text before touching §III-F3.")


def load_manifest(path):
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {row["tag"]: row for row in csv.DictReader(f)}


def append_manifest(path, row):
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)


def run_one(model, ablation_mode, byzantine, k, assumed_f, seed, tag,
            results_dir, manifest_path, dry_run):
    existing = load_manifest(manifest_path)
    if tag in existing and existing[tag]["status"] == "done":
        print(f"[skip] {tag} already marked done in manifest")
        return

    cmd = [
        sys.executable, "main.py", model,
        "--ablation-mode", ablation_mode,
        "--byzantine", byzantine,
        "--krum-k", str(k),
        "--assumed-f", str(assumed_f),
        "--seed", str(seed),
        "--tag", tag,
    ]
    print(f"\n{'='*70}\n[{tag}]\n{' '.join(cmd)}\n{'='*70}")

    if dry_run:
        return

    log_path = results_dir / f"{tag}.log"
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    append_manifest(manifest_path, {
        "tag": tag, "phase": "", "model": model,
        "ablation_mode": ablation_mode, "byzantine": byzantine,
        "k": k, "assumed_f": assumed_f, "seed": seed,
        "status": "running", "started": started, "finished": "",
        "returncode": "", "log_path": str(log_path),
    })

    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)

    finished = time.strftime("%Y-%m-%d %H:%M:%S")
    append_manifest(manifest_path, {
        "tag": tag, "phase": "", "model": model,
        "ablation_mode": ablation_mode, "byzantine": byzantine,
        "k": k, "assumed_f": assumed_f, "seed": seed,
        "status": "done" if proc.returncode == 0 else "FAILED",
        "started": started, "finished": finished,
        "returncode": proc.returncode, "log_path": str(log_path),
    })

    if proc.returncode != 0:
        print(f"  !! {tag} exited {proc.returncode} -- see {log_path}")
    else:
        print(f"  ok  {tag} -- log at {log_path}")


def phase1_1(results_dir, manifest_path, dry_run):
    """k/assumed_f confound grid: 4 single-seed runs, network model,
    hybrid mitigated, byzantine 1,2 -- the exact condition the plan
    flags as confounded."""
    print("Phase 1.1 -- k/assumed_f grid (network, exp2_mitigated, "
          "byzantine 1,2, seed 42)")
    for k in K_GRID:
        for af in ASSUMED_F_GRID:
            tag = f"network_exp2_mitigated_byz1_2_k{k}_af{af}_seed42"
            run_one("network", "exp2_mitigated", "1,2", k, af, 42,
                     tag, results_dir, manifest_path, dry_run)

    print("\nPhase 1.1 done. Before running Phase 1.2:")
    print("  1. Open each run's per-client Krum log / LOG_CSV under "
          f"{results_dir} and compare collateral-exclusion rate "
          "(honest clients excluded every round) across the 4 configs.")
    print("  2. Pick the (k, assumed_f) that resolves the confound the "
          "way you want reported, and set CANONICAL_K / "
          "CANONICAL_ASSUMED_F at the top of this script.")
    print("  3. Then run: python3 run_tier1.py --phase 1.2 "
          f"--results-dir {results_dir}")


def phase1_2(results_dir, manifest_path, dry_run):
    """40-run multi-seed pass on the canonical (k, assumed_f), across all
    8 Experiment 2 configs x 5 seeds."""
    print(f"Phase 1.2 -- multi-seed Experiment 2 at "
          f"k={CANONICAL_K}, assumed_f={CANONICAL_ASSUMED_F} "
          f"({len(EXPERIMENT2_CONFIGS)} configs x {len(SEEDS)} seeds = "
          f"{len(EXPERIMENT2_CONFIGS) * len(SEEDS)} runs)")
    for model, mode, byz in EXPERIMENT2_CONFIGS:
        for seed in SEEDS:
            byz_tag = byz.replace(",", "_")
            tag = (f"{model}_{mode}_byz{byz_tag}_"
                   f"k{CANONICAL_K}_af{CANONICAL_ASSUMED_F}_seed{seed}")
            run_one(model, mode, byz, CANONICAL_K, CANONICAL_ASSUMED_F,
                     seed, tag, results_dir, manifest_path, dry_run)

    print("\nPhase 1.2 done (or dry-run printed). Aggregate results per "
          "(model, mode, byz) across the 5 seeds -- mean +/- std of "
          "best-round F1-Macro, detection rate, and kept/round -- before "
          "updating Table IX/X with confidence intervals.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True, choices=["0", "1.1", "1.2"])
    p.add_argument("--results-dir", type=Path, default=Path("results/tier1"))
    p.add_argument("--dry-run", action="store_true",
                    help="Print commands without running them or writing "
                         "to the manifest.")
    args = p.parse_args()

    if args.phase == "0":
        phase0_tier0_check()
        return

    args.results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.results_dir / "run_manifest.csv"

    if args.phase == "1.1":
        phase1_1(args.results_dir, manifest_path, args.dry_run)
    elif args.phase == "1.2":
        phase1_2(args.results_dir, manifest_path, args.dry_run)


if __name__ == "__main__":
    main()
