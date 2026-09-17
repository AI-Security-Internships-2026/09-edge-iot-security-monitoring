"""
run_tier2.py -- Phase 1.3: bounded-magnitude/direction-only attack pilot.

Tests the norm-bound commitment guard's documented, previously-untested
limitation (paper Limitations / hmac_norm_guard.py Part 2: "a
bounded-magnitude, targeted misclassification attack... would not be
flagged by construction"). Uses bounded_directional_attack_trained()
with direction='classifier_head_negate' -- the classifier-head-only
variant, v3-patched so the crafted head-slice delta norm lands exactly
at tau-margin (the actual quantity the guard measures), backbone left
fully honest.

REQUIRES the byzantine.py fix applied alongside this script -- the
pre-fix version diluted the head-slice norm across the whole model
vector and would understate the attack.

Run matrix: 2 models x 2 attacker identities x 3 seeds = 12 runs,
matching the submission plan's Tier 2 spec ("2-3 seeds, ~8-12 runs").
Guard-enabled arm only (exp2_mitigated) -- this attack is meaningless
against the unmitigated arm, which has no norm check to evade.

Resume-safe: re-running this script after a crash skips any run
already marked "done" in the manifest CSV.

Usage:
    python3 run_tier2.py --dry-run          # print commands only
    python3 run_tier2.py                    # actually run everything
    python3 run_tier2.py --seeds 42,123     # override seed list
"""

import argparse
import csv
import itertools
import os
import subprocess

MANIFEST = "run_manifest_tier2.csv"

MODELS = ["network", "application"]
ATTACKER_IDENTITIES = ["1,2", "2,7"]  # matches Experiment 2's own spread
DEFAULT_SEEDS = [42, 123, 7]

MANIFEST_HEADER = ["tag", "model", "byzantine", "seed", "status"]


def _manifest_rows():
    if not os.path.exists(MANIFEST):
        return []
    with open(MANIFEST, newline="") as f:
        return list(csv.reader(f))


def already_done(tag):
    """A run counts as done only if its own row's status is 'done' --
    a row stuck at 'started' (crash mid-run) is re-run, not skipped."""
    for row in _manifest_rows():
        if row and row[0] == tag and len(row) >= 5 and row[4] == "done":
            return True
    return False


def log_manifest(row):
    write_header = not os.path.exists(MANIFEST)
    with open(MANIFEST, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(MANIFEST_HEADER)
        w.writerow(row)


def build_command(model, byz, seed, margin):
    tag = f"tier2_{model}_byz{byz.replace(',', '')}_seed{seed}"
    cmd = [
        "python3", "main.py", model,
        "--ablation-mode", "exp2_mitigated",
        # REQUIRED -- without this, exp2_mitigated's dispatch precedence
        # silently replaces --attack-type with classifier_head_flip_attack
        # regardless of what's passed here. See main.py's own comment at
        # the --byzantine-full-model flag definition.
        "--byzantine-full-model",
        "--attack-type", "bounded_directional",
        "--bounded-direction", "classifier_head_negate",
        "--bounded-margin", str(margin),
        # --bounded-tau deliberately omitted: leaving it None makes the
        # attack use the live estimate_norm_guard_tau() path (honest
        # training on round 1, since no prior-round honest norms exist
        # yet; attacking from round 2 on) -- the causally-realistic
        # attacker, rather than handing it the guard's exact threshold.
        "--byzantine", byz,
        "--seed", str(seed),
        "--tag", tag,
    ]
    return tag, cmd


def run_one(model, byz, seed, margin, dry_run):
    tag, cmd = build_command(model, byz, seed, margin)
    if already_done(tag):
        print(f"[skip] {tag} already completed")
        return

    log_manifest([tag, model, byz, seed, "started"])
    print(" ".join(cmd))
    if not dry_run:
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            log_manifest([tag, model, byz, seed, f"FAILED (exit {e.returncode})"])
            raise
    log_manifest([tag, model, byz, seed, "done"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true",
                    help="Print commands without executing them.")
    p.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated seed override, e.g. 42,123,7. "
                         f"Default: {DEFAULT_SEEDS}")
    p.add_argument("--margin", type=float, default=0.05,
                    help="Safety margin subtracted from tau (passed to "
                         "--bounded-margin). Default 0.05, matching the "
                         "function's own default.")
    args = p.parse_args()

    seeds = ([int(s.strip()) for s in args.seeds.split(",")]
              if args.seeds else DEFAULT_SEEDS)

    total = len(MODELS) * len(ATTACKER_IDENTITIES) * len(seeds)
    print(f"Tier 2 pilot: {len(MODELS)} models x {len(ATTACKER_IDENTITIES)} "
          f"attacker identities x {len(seeds)} seeds = {total} runs")
    print(f"Manifest: {MANIFEST}\n")

    for model, byz, seed in itertools.product(MODELS, ATTACKER_IDENTITIES, seeds):
        run_one(model, byz, seed, args.margin, args.dry_run)


if __name__ == "__main__":
    main()
