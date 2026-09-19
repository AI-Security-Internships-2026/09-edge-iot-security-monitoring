#!/usr/bin/env python3
"""
run_e2_campaign.py

Dispatches every cell in experiments/configs/EXP1_campaign_E2.json
against main.py, in order, and verifies each run actually produced a
complete, non-empty result before moving to the next cell (Issue 5's
"Result verifier detects missing/header-only/incomplete runs"
requirement, applied per-cell rather than as a separate later pass).

Does NOT run Task 2's pre-campaign attack-difficulty gate itself --
run scripts/check_attack_difficulty.py (with its Check 1/3 minmax/
minsum sweeps) separately FIRST, and only pass --skip-prereq-check
here once that has passed AND experiments/configs/hetero_fit_coeffs_a0.7.json
exists (via scripts/refit_hetero_alpha0.7.py). This script checks for
both by default and refuses to start otherwise -- see --skip-prereq-check
to override (e.g. for a --dry-run rehearsal of the command list without
actually training anything).

Usage:
    python scripts/run_e2_campaign.py --dry-run          # print commands only
    python scripts/run_e2_campaign.py                    # run everything
    python scripts/run_e2_campaign.py --only-aggregator calibrated_krum
    python scripts/run_e2_campaign.py --resume-from E2-042

    # 5-way parallel split (e.g. one per tmux window):
    python scripts/run_e2_campaign.py --num-shards 5 --shard-index 0
    python scripts/run_e2_campaign.py --num-shards 5 --shard-index 1
    ... etc through --shard-index 4
    # or just use scripts/launch_e2_tmux.sh, which wraps this.
"""

import argparse
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_MAIN_PY = os.path.join(_REPO_ROOT, "main.py")
_CAMPAIGN_JSON = os.path.join(_REPO_ROOT, "experiments", "configs", "EXP1_campaign_E2.json")
_HETERO_FIT_JSON = os.path.join(_REPO_ROOT, "experiments", "configs", "hetero_fit_coeffs_a0.7.json")


def cli_args_to_argv(cli_args):
    """Turn a cell's cli_args dict into an actual main.py argv list."""
    argv = [cli_args["positional_model_type"]]
    flag_map = {
        "ablation_mode": "--ablation-mode",
        "aggregator": "--aggregator",
        "attack_type": "--attack-type",
        "alpha": "--alpha",
        "seed": "--seed",
        "tag": "--tag",
        "dataset": "--dataset",
        "byzantine": "--byzantine",
        "gaussian_std": "--gaussian-std",
        "minmax_dev_type": "--minmax-dev-type",
        "minmax_search_iters": "--minmax-search-iters",
        "minmax_gamma_init": "--minmax-gamma-init",
        "bounded_tau": "--bounded-tau",
        "bounded_margin": "--bounded-margin",
        "bounded_direction": "--bounded-direction",
        "hetero_fit_coeffs_json": "--hetero-fit-coeffs-json",
    }
    for key, flag in flag_map.items():
        if key == "positional_model_type":
            continue
        if key not in cli_args:
            continue
        val = cli_args[key]
        if val is None:
            continue  # None means "omit -- let main.py use its own default/auto"
        argv += [flag, str(val)]
    return argv


def verify_result(cell, workdir):
    """Non-empty-file + minimally-parseable checks -- catches the
    'header-only/incomplete run' failure mode Task 1/6 call out
    explicitly, not just 'file exists'."""
    problems = []
    log_csv = os.path.join(workdir, cell["expected_log_csv"])
    manifest = os.path.join(workdir, cell["expected_manifest"])

    if not os.path.exists(log_csv):
        problems.append(f"missing {cell['expected_log_csv']}")
    else:
        with open(log_csv) as f:
            lines = f.readlines()
        if len(lines) <= 1:
            problems.append(
                f"{cell['expected_log_csv']} has only a header row "
                f"({len(lines)} lines) -- run did not produce any "
                f"round data"
            )

    if not os.path.exists(manifest):
        problems.append(f"missing {cell['expected_manifest']}")
    else:
        with open(manifest) as f:
            try:
                m = json.load(f)
            except json.JSONDecodeError as e:
                problems.append(f"{cell['expected_manifest']} is not valid JSON: {e}")
                m = {}
        if m.get("split_hash") is None:
            problems.append(
                f"{cell['expected_manifest']} has split_hash=None -- "
                f"the run likely crashed before the final-evaluation "
                f"backfill step (see main.py's manifest docstring)"
            )

    if cell["expected_krum_scores_csv"] is not None:
        scores_csv = os.path.join(workdir, cell["expected_krum_scores_csv"])
        if not os.path.exists(scores_csv):
            problems.append(f"missing {cell['expected_krum_scores_csv']}")

    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign-json", default=_CAMPAIGN_JSON)
    ap.add_argument("--workdir", default=_REPO_ROOT,
                     help="Where main.py's outputs land. Default: repo root.")
    ap.add_argument("--dry-run", action="store_true",
                     help="Print every command that would run; execute nothing.")
    ap.add_argument("--only-aggregator", default=None)
    ap.add_argument("--only-attack", default=None)
    ap.add_argument("--only-seed", type=int, default=None)
    ap.add_argument("--resume-from", default=None,
                     help="Cell id (e.g. E2-042) to resume from, skipping "
                          "everything before it.")
    ap.add_argument("--num-shards", type=int, default=None,
                     help="Split the campaign into N independent shards "
                          "(e.g. for N parallel tmux windows). Must be "
                          "used together with --shard-index.")
    ap.add_argument("--shard-index", type=int, default=None,
                     help="Which shard (0-indexed, < --num-shards) this "
                          "invocation runs. Assignment is by "
                          "crc32(cell_id) % num_shards -- stable "
                          "regardless of --only-* filters or list order, "
                          "so re-running the same --shard-index always "
                          "targets the same cells.")
    ap.add_argument("--skip-prereq-check", action="store_true",
                     help="Skip the hetero_fit_coeffs_a0.7.json existence "
                          "check. Does NOT skip per-cell result "
                          "verification. Use for --dry-run rehearsals.")
    args = ap.parse_args()

    with open(args.campaign_json) as f:
        campaign = json.load(f)

    cells = campaign["cells"]

    if (args.num_shards is None) != (args.shard_index is None):
        print("--num-shards and --shard-index must be given together "
              "(both or neither).", file=sys.stderr)
        sys.exit(1)
    if args.num_shards is not None:
        if not (0 <= args.shard_index < args.num_shards):
            print(f"--shard-index must be in [0, {args.num_shards}), "
                  f"got {args.shard_index}.", file=sys.stderr)
            sys.exit(1)
        # Sort by cell_id first so shard assignment is stable regardless
        # of the campaign JSON's on-disk order (E2-001..E2-090 sorts
        # lexicographically = numerically here), then take index modulo
        # num_shards -- gives an EXACT even split (90/5 = 18/18/18/18/18)
        # since cells are homogeneous cost (~25 rounds each regardless
        # of aggregator/attack), unlike a hash-based split which can
        # land anywhere from 15-21 per shard by chance.
        cells_sorted = sorted(cells, key=lambda c: c["cell_id"])
        cells = [c for i, c in enumerate(cells_sorted)
                 if i % args.num_shards == args.shard_index]

    if args.only_aggregator:
        cells = [c for c in cells if c["aggregator"] == args.only_aggregator]
    if args.only_attack:
        cells = [c for c in cells if c["attack_type"] == args.only_attack]
    if args.only_seed:
        cells = [c for c in cells if c["seed"] == args.only_seed]
    if args.resume_from:
        ids = [c["cell_id"] for c in cells]
        if args.resume_from not in ids:
            print(f"--resume-from {args.resume_from!r} not found in the "
                  f"filtered cell list.", file=sys.stderr)
            sys.exit(1)
        cells = cells[ids.index(args.resume_from):]

    needs_hetero = any(c["aggregator"] == "calibrated_krum" for c in cells)
    if needs_hetero and not args.skip_prereq_check and not args.dry_run:
        if not os.path.exists(_HETERO_FIT_JSON):
            print(f"REFUSING TO START: {len(cells)} cells include "
                  f"calibrated_krum, which needs "
                  f"{_HETERO_FIT_JSON!r}, which does not exist yet. Run "
                  f"scripts/refit_hetero_alpha0.7.py first, or pass "
                  f"--skip-prereq-check if you're intentionally running "
                  f"calibrated_krum with hetero calibration inactive.",
                  file=sys.stderr)
            sys.exit(1)

    print(f"Dispatching {len(cells)} E2 cell(s)"
          f"{' [DRY RUN]' if args.dry_run else ''}.")

    failures = []
    for i, cell in enumerate(cells, 1):
        argv = cli_args_to_argv(cell["cli_args"])
        cmd = [sys.executable, _MAIN_PY] + argv
        print(f"\n[{i}/{len(cells)}] {cell['cell_id']} "
              f"({cell['aggregator']} / {cell['attack_type']} / "
              f"seed={cell['seed']})")
        print("  " + " ".join(cmd))

        if args.dry_run:
            continue

        result = subprocess.run(cmd, cwd=args.workdir)
        if result.returncode != 0:
            print(f"  [FAIL] main.py exited {result.returncode}")
            failures.append((cell["cell_id"], "main.py nonzero exit"))
            continue

        problems = verify_result(cell, args.workdir)
        if problems:
            print(f"  [FAIL] result verification failed:")
            for p in problems:
                print(f"    - {p}")
            failures.append((cell["cell_id"], "; ".join(problems)))
        else:
            print(f"  [OK]")

    print(f"\n{'='*70}")
    if args.dry_run:
        print(f"Dry run complete -- {len(cells)} command(s) printed above, "
              f"none executed.")
    elif failures:
        print(f"{len(failures)}/{len(cells)} cell(s) FAILED:")
        for cell_id, reason in failures:
            print(f"  {cell_id}: {reason}")
        sys.exit(1)
    else:
        print(f"All {len(cells)} E2 cell(s) completed and verified.")
    print("=" * 70)


if __name__ == "__main__":
    main()
