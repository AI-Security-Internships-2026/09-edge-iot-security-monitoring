"""
Issue 5 Task 2 -- Attack difficulty check, run BEFORE the full E1-E8
campaign (Task 3). Not a new aggregator/attack implementation -- this is
a thin runner around the existing main.py CLI (script entry, per the
issue spec: "Task 2 -- Attack difficulty check before full campaign
(script entry, no new file)") plus a parser for the LOG_CSV it already
writes. If the checks below fail, DO NOT proceed to the full 5-seed
campaign -- tune the attack hyperparameters flagged in each failure
message and re-run this script until both checks pass.

Two checks, both from the issue's acceptance criteria:

  1. Plain Adaptive Krum (uncalibrated baseline) vs Min-Max attack,
     alpha=0.7, epsilon=none. Requirement: Byzantine TPR strictly in
     (50%, 85%) -- not 100% (attack isn't evasive enough to be a
     meaningful stress test) and not 0% (attack is so aggressive it's
     trivially caught, i.e. functionally identical to the easy-control
     sign-flip/Gaussian attacks this whole issue exists to move past).

  2. HMAC norm guard vs bounded_directional attack, tau matched to the
     guard's own threshold. Requirement: TPR == 0% EXACTLY -- this is
     the attack's whole point (defeat a magnitude-only check by
     truthfully reporting a norm under threshold) and is meant to be
     documented as a known, honest limitation, not something to "fix".

Usage
-----
    python scripts/check_attack_difficulty.py

    # to sweep --minmax-gamma-init while tuning, e.g.:
    python scripts/check_attack_difficulty.py --gamma-candidates 0.5,1,2,4,8

Exit code is 0 iff both checks pass; non-zero (with a specific message
about which check failed and which knob to adjust) otherwise, so this
is safe to wire into a pre-campaign CI/Make target that gates Task 3.
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_MAIN_PY = os.path.join(_REPO_ROOT, "main.py")

MINMAX_TPR_LOW, MINMAX_TPR_HIGH = 0.50, 0.85
CALIBRATION_ROUNDS = 5          # short run: fast enough to iterate gamma
DEFAULT_BYZANTINE = "1,2"       # matches main.py's own default clients


def _run_main(extra_args, tag, workdir, byzantine_clients=None, rounds=None):
    """
    Invoke main.py as a subprocess (not imported directly -- main.py
    builds its whole config at MODULE IMPORT time from argparse, so it
    cannot be safely called twice in-process with different --attack-
    type / --aggregator values; subprocess is the only clean way to run
    it repeatedly here). Runs from `workdir` so LOG_CSV/etc. land in an
    isolated scratch directory instead of the repo root.

    byzantine_clients : str or None
        Comma-separated 1-indexed client list, e.g. "1,2,3,4" -- same
        format as main.py's own --byzantine flag. Defaults to
        DEFAULT_BYZANTINE when omitted. Exposed as a parameter so
        Check 1 can sweep coalition SIZE, not just --minmax-gamma-init
        -- a larger coalition is a genuinely different lever than
        gamma, since a bigger coalition is harder to make
        simultaneously look honest with one shared crafted vector.
    rounds : int or None
        Overrides CALIBRATION_ROUNDS for this run. If models haven't
        diverged much from the early-training regime yet, natural
        honest-client score spread can be artificially small, which
        makes evasion look artificially easy -- bump this if a
        coalition-size/gamma sweep alone doesn't move TPR off 0%.

    Returns the path to the run's real results_{model}_{tag}_seed{seed}.csv.
    """
    byz = byzantine_clients if byzantine_clients is not None else DEFAULT_BYZANTINE
    n_rounds = rounds if rounds is not None else CALIBRATION_ROUNDS
    cmd = [
        sys.executable, _MAIN_PY, "network",
        "--tag", tag,
        "--rounds", str(n_rounds),
        "--alpha", "0.7",
        "--byzantine", byz,
        "--seed", "42",
    ] + extra_args

    print(f"\n{'='*70}\n  RUNNING: {' '.join(cmd)}\n  (cwd={workdir})\n{'='*70}")
    result = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    print(result.stdout[-4000:])  # tail -- full run log is long
    if result.returncode != 0:
        print(result.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(
            f"main.py exited with code {result.returncode} for tag={tag!r} "
            f"-- see stderr above. Cannot compute TPR from a run that "
            f"didn't complete."
        )
    # main.py's real LOG_CSV naming (see main.py's _TAG/LOG_CSV construction):
    # results_{MODEL_TYPE}_{tag}_seed{seed}.csv -- NOT results_{tag}.csv.
    # _run_main() always invokes main.py with the "network" positional
    # model-type arg and a fixed seed=42 (see `cmd` above), so both pieces
    # are always known here. (Confirmed via direct execution: the naive
    # results_{tag}.csv guess never matches, so every run of this script
    # crashed with FileNotFoundError immediately after main.py finished
    # successfully -- this file never actually completed a check before.)
    return os.path.join(workdir, f"results_network_{tag}_seed42.csv")


def _byzantine_tpr_from_log(log_csv_path, byzantine_clients_1indexed,
                             rounds_to_average="last"):
    """
    Parse a results_*.csv (LOG_CSV) and compute Byzantine detection TPR.

    LOG_CSV has one row per client per round (client column = 1-indexed
    client number as an int; a separate "mean" row per round is written
    with client == "mean" and is skipped here) plus a
    'krum_detected_byzantine' column (1/0 per client row). TPR for a
    round = (# Byzantine clients with krum_detected_byzantine==1) /
    (# Byzantine clients that round).

    rounds_to_average: 'last' (default, most representative of a
    converged run) or 'all' (mean over every logged round).
    """
    per_round_hits = {}   # round_num -> [detected_bool, ...]
    with open(log_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            client_raw = row["client"]
            if client_raw.strip().upper() == "MEAN":
                continue
            client_id = int(client_raw)
            if client_id not in byzantine_clients_1indexed:
                continue
            round_num = int(row["round"])
            detected = int(row["krum_detected_byzantine"])
            per_round_hits.setdefault(round_num, []).append(detected)

    if not per_round_hits:
        raise ValueError(
            f"No Byzantine-client rows found in {log_csv_path} for "
            f"clients {byzantine_clients_1indexed} -- check "
            f"--byzantine matches what was actually passed to main.py."
        )

    rounds_sorted = sorted(per_round_hits)
    if rounds_to_average == "last":
        selected_rounds = [rounds_sorted[-1]]
    else:
        selected_rounds = rounds_sorted

    hits, total = 0, 0
    for r in selected_rounds:
        hits += sum(per_round_hits[r])
        total += len(per_round_hits[r])
    return hits / total if total > 0 else float("nan")


def check_minmax_vs_plain_adaptive_krum(workdir, gamma_init=None,
                                         minmax_dev_type="std",
                                         minmax_search_iters=15,
                                         byzantine_clients=None,
                                         rounds=None):
    """Acceptance item 1: plain Adaptive Krum TPR on Min-Max must land
    strictly in (50%, 85%).

    byzantine_clients : str or None
        Coalition to attack with, e.g. "1,2,3,4" -- defaults to
        DEFAULT_BYZANTINE ("1,2"). A bigger coalition is harder to
        make simultaneously look honest with one shared crafted
        vector, so if an exhaustive gamma/dev_type sweep alone can't
        move TPR off 0%, growing the coalition is the next thing to
        try before concluding the attack is just robustly stealthy at
        this size.
    rounds : int or None
        Overrides CALIBRATION_ROUNDS -- see _run_main()'s docstring.
    """
    byz = byzantine_clients if byzantine_clients is not None else DEFAULT_BYZANTINE
    extra = [
        "--aggregator", "adaptive_krum",
        # BUG FOUND (reviewing main.py's ablation-mode logic more
        # closely after building run_campaign.py): "baseline" mode
        # hardcodes USE_BYZANTINE_ATTACK=False UNCONDITIONALLY -- with
        # this mode, no client is ever actually attacked regardless of
        # --attack-type/--byzantine, so the "byzantine"-labeled ground-
        # truth clients (1,2) train perfectly honestly, Krum correctly
        # never flags them (nothing wrong to catch), and TPR sits at
        # ~0% no matter how gamma_init is tuned -- Check 1 as
        # originally written could never pass. "krum_baseline" is the
        # mode that actually forces USE_BYZANTINE_ATTACK=True while
        # still honoring --aggregator (see main.py's ablation-mode
        # block) -- this is the same fix already applied throughout
        # scripts/run_campaign.py's condition builders; this script
        # was built earlier and missed it until this review.
        "--ablation-mode", "krum_baseline",
        "--attack-type", "minmax",
        "--minmax-dev-type", minmax_dev_type,
        "--minmax-search-iters", str(minmax_search_iters),
    ]
    if gamma_init is not None:
        extra += ["--minmax-gamma-init", str(gamma_init)]

    log_path = _run_main(extra, tag="check_minmax_adaptive_krum", workdir=workdir,
                          byzantine_clients=byz, rounds=rounds)
    byz_clients = [int(c) for c in byz.split(",")]
    tpr = _byzantine_tpr_from_log(log_path, byz_clients)

    print(f"\n[CHECK 1] Plain Adaptive Krum vs Min-Max (gamma_init="
          f"{gamma_init!r}, byzantine={byz!r}, rounds={rounds!r}): "
          f"TPR = {tpr:.2%}")

    if tpr >= 1.0 - 1e-9:
        raise SystemExit(
            f"FAIL: TPR={tpr:.2%} == 100%. Min-Max attack is NOT evasive "
            f"enough -- plain Adaptive Krum still catches every instance, "
            f"making this comparison meaningless (identical to the easy-"
            f"control sign-flip/Gaussian attacks). Fix: INCREASE "
            f"--minmax-gamma-init (try 2x-4x current value: "
            f"{'unset (auto = 5x coalition spread)' if gamma_init is None else gamma_init}), "
            f"and/or increase --minmax-search-iters, and/or try "
            f"--minmax-dev-type sign or unit_vec."
        )
    if tpr <= 0.0 + 1e-9:
        raise SystemExit(
            f"FAIL: TPR={tpr:.2%} == 0%. Min-Max attack is TOO aggressive "
            f"-- plain Adaptive Krum catches nothing, which means the "
            f"crafted update likely violates its own Min-Max feasibility "
            f"bound in practice (e.g. gamma_init set far too high, "
            f"overshooting the coalition's own pairwise-distance spread "
            f"after the binary search saturates at the upper bound). Fix: "
            f"DECREASE --minmax-gamma-init; if an exhaustive gamma AND "
            f"dev_type sweep both stay at 0%, try a LARGER --byzantine "
            f"coalition (harder to blend a bigger group in with one "
            f"shared crafted vector) and/or more --rounds before "
            f"crafting (more time for honest per-client scores to "
            f"diverge from the early-training regime)."
        )
    if not (MINMAX_TPR_LOW < tpr < MINMAX_TPR_HIGH):
        raise SystemExit(
            f"FAIL: TPR={tpr:.2%} is outside the required "
            f"({MINMAX_TPR_LOW:.0%}, {MINMAX_TPR_HIGH:.0%}) range. "
            f"{'Increase' if tpr >= MINMAX_TPR_HIGH else 'Decrease'} "
            f"--minmax-gamma-init and re-run."
        )

    print(f"[CHECK 1] PASS -- TPR={tpr:.2%} is within "
          f"({MINMAX_TPR_LOW:.0%}, {MINMAX_TPR_HIGH:.0%}).")
    return tpr


def check_bounded_directional_vs_norm_guard(workdir, bounded_tau=None,
                                             bounded_margin=0.05,
                                             byzantine_clients=None,
                                             rounds=None):
    """Acceptance item 2: HMAC norm guard TPR on bounded_directional
    must be EXACTLY 0% -- the documented, expected limitation."""
    byz = byzantine_clients if byzantine_clients is not None else DEFAULT_BYZANTINE
    extra = [
        "--ablation-mode", "pure_norm_guard",
        "--attack-type", "bounded_directional",
        "--bounded-margin", str(bounded_margin),
        # BUG FOUND (same review pass as Check 1's fix above):
        # pure_norm_guard mode hardcodes BYZANTINE_HEAD_ONLY=True, and
        # main.py's attack dispatch checks that flag BEFORE --attack-
        # type (see _train_one_client(): "if (USE_HE or
        # USE_HE_KRUM_HYBRID or USE_NORM_GUARD) and
        # BYZANTINE_HEAD_ONLY:") -- meaning --attack-type
        # bounded_directional was being SILENTLY IGNORED and this check
        # was actually running classifier_head_flip_attack instead (an
        # unbounded, full-scale attack the norm guard SHOULD and would
        # catch, making the "TPR=0% by design" assertion fail for a
        # completely different reason than the one it's meant to
        # verify). --byzantine-full-model is a main.py flag that forces
        # BYZANTINE_HEAD_ONLY=False regardless of ablation mode, so
        # --attack-type is actually honored here.
        "--byzantine-full-model",
    ]
    if bounded_tau is not None:
        extra += ["--bounded-tau", str(bounded_tau)]

    log_path = _run_main(extra, tag="check_bounded_directional_norm_guard",
                          workdir=workdir, byzantine_clients=byz, rounds=rounds)
    byz_clients = [int(c) for c in byz.split(",")]
    # Use 'all' rounds here, not just 'last': round 1 has no prior-round
    # honest norms to estimate tau from (if --bounded-tau wasn't given),
    # so the attack is skipped that round by design (see
    # bounded_directional's client_cfg["bounded_tau"] handling in
    # main.py) -- averaging over all rounds still correctly yields 0%
    # as long as every round WITH an active attack is caught 0% of the
    # time, and surfaces a non-zero result clearly if the guard ever
    # catches it even once.
    tpr = _byzantine_tpr_from_log(log_path, byz_clients, rounds_to_average="all")

    print(f"\n[CHECK 2] HMAC norm guard vs bounded_directional "
          f"(tau={bounded_tau!r}): TPR = {tpr:.2%}")

    if tpr > 0.0 + 1e-9:
        raise SystemExit(
            f"FAIL: TPR={tpr:.2%} != 0%. The norm guard caught at least "
            f"one bounded_directional attack instance -- this means the "
            f"crafted update's norm was NOT actually under the guard's "
            f"real threshold (tau estimate too loose, or --bounded-tau "
            f"doesn't match what the guard is really enforcing this "
            f"round). Fix: pass --bounded-tau explicitly, matched to the "
            f"guard's own printed '[Norm guard] ... threshold=...' value "
            f"from this run's stdout, and/or increase --bounded-margin "
            f"for more safety headroom."
        )

    print("[CHECK 2] PASS -- TPR=0.00% exactly, as expected by design "
          "(documented HMAC norm-guard limitation).")
    return tpr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma-candidates", type=str, default=None,
                         help="Comma-separated --minmax-gamma-init values "
                              "to try in order until Check 1 passes, e.g. "
                              "'0.5,1,2,4,8'. If omitted, runs once with "
                              "the auto default (5x coalition spread).")
    parser.add_argument("--byzantine-candidates", type=str, default=None,
                         help="Semicolon-separated coalition candidates "
                              "to try for Check 1, e.g. "
                              "'1,2;1,2,3;1,2,3,4'. Tried as the OUTER "
                              "loop, gamma-candidates as the INNER loop. "
                              "If omitted, uses DEFAULT_BYZANTINE ('1,2') "
                              "only.")
    parser.add_argument("--calibration-rounds", type=int, default=None,
                         help="Overrides CALIBRATION_ROUNDS (default 5) "
                              "for both checks.")
    parser.add_argument("--bounded-tau", type=float, default=None,
                         help="Explicit tau for Check 2. If omitted, "
                              "Check 2 uses main.py's own online estimate "
                              "(may be None on round 1 -- see that "
                              "check's docstring).")
    parser.add_argument("--keep-workdir", action="store_true",
                         help="Don't delete the scratch run directory "
                              "after the checks complete (for inspecting "
                              "results_*.csv / stdout logs by hand).")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="fl_ids_attack_check_") as workdir:
        gamma_candidates = (
            [float(g) for g in args.gamma_candidates.split(",")]
            if args.gamma_candidates else [None]
        )
        byzantine_candidates = (
            args.byzantine_candidates.split(";")
            if args.byzantine_candidates else [DEFAULT_BYZANTINE]
        )

        passed = False
        last_error = None
        for byz in byzantine_candidates:
            for gamma in gamma_candidates:
                try:
                    check_minmax_vs_plain_adaptive_krum(
                        workdir, gamma_init=gamma, byzantine_clients=byz,
                        rounds=args.calibration_rounds,
                    )
                    passed = True
                    break
                except SystemExit as e:
                    last_error = e
                    print(f"  (byzantine={byz!r}, gamma_init={gamma!r} "
                          f"failed, trying next candidate if any remain)")
            if passed:
                break
        if not passed:
            print(f"\nAll byzantine/gamma candidates exhausted without "
                  f"passing Check 1.\n{last_error}")
            if not args.keep_workdir:
                pass  # TemporaryDirectory cleans up on exit
            sys.exit(1)

        check_bounded_directional_vs_norm_guard(
            workdir, bounded_tau=args.bounded_tau,
            rounds=args.calibration_rounds,
        )

        print(f"\n{'='*70}\n"
              f"  ALL TASK 2 CHECKS PASSED -- safe to proceed to the full\n"
              f"  E1-E8 5-seed campaign (Task 3).\n"
              f"{'='*70}")

        if args.keep_workdir:
            # Copy scratch dir contents somewhere durable before the
            # TemporaryDirectory context manager deletes it.
            import shutil
            dest = os.path.join(_REPO_ROOT, "check_attack_difficulty_output")
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.copytree(workdir, dest)
            print(f"  Run artifacts copied to: {dest}")


if __name__ == "__main__":
    main()
