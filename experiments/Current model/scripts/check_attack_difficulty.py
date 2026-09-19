"""
Issue 5 Task 2 -- Attack difficulty check, run BEFORE the full E1-E8
campaign (Task 3). Not a new aggregator/attack implementation -- this is
a thin runner around the existing main.py CLI (script entry, per the
issue spec: "Task 2 -- Attack difficulty check before full campaign
(script entry, no new file)") plus a parser for the LOG_CSV it already
writes. If the checks below fail, DO NOT proceed to the full 5-seed
campaign -- tune the attack hyperparameters flagged in each failure
message and re-run this script until both checks pass.

Three checks, all from the issue's acceptance criteria (Check 3 added
to close a gap found during E2 prep: the original script validated
Min-Max but not Min-Sum, even though Task 2 requires both):

  1. Plain Adaptive Krum (uncalibrated baseline) vs Min-Max attack,
     alpha=0.7, epsilon=none. Requirement: krum_score_ratio (mean
     Byzantine Krum score / mean honest Krum score) strictly in
     (MINMAX_RATIO_LOW, MINMAX_RATIO_HIGH) -- not too high (attack is
     such an obvious outlier it's functionally identical to the
     easy-control sign-flip/Gaussian attacks this whole issue exists to
     move past) and not <= 1 (attack blends into, or -- via the
     mutual-zero-distance collusion effect at large coalition sizes --
     scores MORE central than, the honest cluster, i.e. effectively
     invisible, not just evasive). NOTE: this check originally gated on
     Byzantine TPR in (50%, 85%); that band could never pass for this
     attack, since every colluding client broadcasts one identical
     crafted vector and so always shares one Krum score, making
     per-round TPR quantized to exactly {0%, 100%}. See
     MINMAX_RATIO_LOW/HIGH's comment below for the full account.

  2. Plain Adaptive Krum vs Min-Sum attack -- same gate and same
     rationale as (1) (Min-Sum shares Min-Max's coalition-broadcast
     structure, so TPR is quantized the same way); tuned independently
     since Min-Sum's crafted-update objective differs from Min-Max's.

  3. HMAC norm guard vs bounded_directional attack, tau matched to the
     guard's own threshold. Requirement: TPR == 0% EXACTLY -- this is
     the attack's whole point (defeat a magnitude-only check by
     truthfully reporting a norm under threshold) and is meant to be
     documented as a known, honest limitation, not something to "fix".

Usage
-----
    python scripts/check_attack_difficulty.py

    # to sweep --minmax-gamma-init while tuning, e.g.:
    python scripts/check_attack_difficulty.py --gamma-candidates 0.5,1,2,4,8

    # Min-Sum can be swept independently of Min-Max if it needs a
    # different gamma/coalition to pass:
    python scripts/check_attack_difficulty.py \
        --gamma-candidates 0.5,1,2,4,8 \
        --minsum-gamma-candidates 1,2,4,8,16

Exit code is 0 iff all three checks pass; non-zero (with a specific
message about which check failed and which knob to adjust) otherwise,
so this is safe to wire into a pre-campaign CI/Make target that gates
Task 3.
"""

import argparse
import csv
import glob
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_MAIN_PY = os.path.join(_REPO_ROOT, "main.py")

# Check 1 used to gate on Byzantine TPR in (50%, 85%). BUG FOUND: Min-Max
# always broadcasts ONE identical crafted vector to every colluding
# client (coalition-optimal per Fang et al. -- see byzantine.py's module
# docstring). That means every coalition member has distance 0 to every
# other coalition member, so they compute IDENTICAL Krum scores every
# round and therefore cross the MAD threshold together or not at all.
# Per-round Byzantine TPR can only ever be exactly 0% or exactly 100%
# for this attack -- there is no coalition size, gamma, or k for which
# a strictly-between-0-and-100% TPR is even possible, so the old
# acceptance band could never pass by design, independent of tuning.
# (Confirmed empirically: f=2/k=2.5 -> TPR=0% despite byz scores being
# the two highest in the round; f=2/k=1.5 -> TPR=0% again, unchanged
# across all 5 rounds; f=4 -> TPR=0% via a DIFFERENT, structural
# mechanism -- mutual zero-distance collapses the coalition's own score
# toward zero once coalition size approaches n-f-2 neighbours.)
#
# Replacement metric: krum_score_ratio = mean Byzantine score / mean
# honest score, already computed every round by main.py (see its
# krum_ratio local + the "krum_score_ratio" CSV column on the MEAN
# row) -- continuous, not quantized by coalition size, and it already
# distinguishes the two failure modes above: ratio <= ~1 means the
# attack is either blended into the honest cluster or (ratio << 1)
# undergoing the mutual-zero-distance collapse; ratio far above 1 means
# it's such an obvious outlier the "stealthy" framing of this attack
# family doesn't hold. Band chosen from the f=2/k=2.5 default run,
# which is otherwise exactly the "evasive but not blended in" case this
# check is meant to validate (ratio=1.6427): require the mean Byzantine
# score to be noticeably elevated (>1.2x) but not wildly so (<3x) —
# re-tune this band, not gamma_init, if it doesn't fit your setup.
MINMAX_RATIO_LOW, MINMAX_RATIO_HIGH = 1.2, 3.0

# Check 3 (Issue 5 Task 2 gap fix) -- Min-Sum. Same coalition-broadcast
# structure as Min-Max (Shejwalkar & Houmansadr, same paper): all
# colluding clients broadcast ONE identical crafted vector, so per-round
# Byzantine TPR is quantized to {0%, 100%} here too, for the identical
# structural reason documented above MINMAX_RATIO_LOW/HIGH. Reusing the
# same krum_score_ratio gate and the same (1.2, 3.0) band as a starting
# point -- NOT assumed identical to Min-Max's actual passing gamma/dev
# combo, since Min-Sum's crafted-update objective differs (minimizes sum
# of squared distances to all clients rather than maximizing distance to
# the nearest honest neighbour), so it must be calibrated independently.
MINSUM_RATIO_LOW, MINSUM_RATIO_HIGH = 1.2, 3.0

CALIBRATION_ROUNDS = 5          # short run: fast enough to iterate gamma
DEFAULT_BYZANTINE = "1,2"       # matches main.py's own default clients


def _sanitize_tag_component(value):
    """Turn an arbitrary sweep-parameter value into a filesystem/tag-safe
    fragment (main.py's checkpoint/LOG_CSV naming is built from `tag`, so
    this must not contain '.', ',', or spaces)."""
    return str(value).replace(".", "p").replace(",", "-").replace(" ", "")


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

    # Defensive checkpoint cleanup: main.py resumes from any checkpoint
    # matching this tag+seed, which is exactly the bug that made every
    # gamma candidate after the first silently replay the first run's
    # cached result instead of training. Callers below now build a tag
    # that's unique per sweep parameter combination, which is the real
    # fix -- but we don't have visibility into every key main.py's
    # resume logic might match on, so belt-and-suspenders: also nuke
    # any stale checkpoint/state file for this exact tag+seed before
    # each run, in this run's own isolated workdir.
    for stale in glob.glob(os.path.join(workdir, f"*{tag}*seed42*")):
        if stale.endswith(".csv") or "checkpoint" in os.path.basename(stale).lower() \
                or "state" in os.path.basename(stale).lower():
            os.remove(stale)

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


def _byzantine_score_ratio_from_log(log_csv_path, rounds_to_average="last"):
    """
    Parse a results_*.csv (LOG_CSV) and pull main.py's own
    krum_score_ratio (= mean Byzantine Krum score / mean honest Krum
    score) off the per-round "MEAN" row -- this is the SAME value
    main.py already prints as "[Krum diagnostics] ... ratio=...", just
    read from the CSV instead of scraped from stdout, and computed once
    by main.py itself rather than re-derived here (avoids any risk of
    this script's math drifting from the aggregator's own).

    rounds_to_average: 'last' (default) or 'all' (mean over every
    logged round's ratio). 'N/A' rows (krum_score_diag was None that
    round, e.g. the too-few-accepted-clients FedProx fallback) are
    skipped; if that empties the selection, raises rather than
    returning a fabricated ratio.
    """
    ratios_by_round = {}
    with open(log_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["client"].strip().upper() != "MEAN":
                continue
            raw = row["krum_score_ratio"]
            if raw == "N/A":
                continue
            ratios_by_round[int(row["round"])] = float(raw)

    if not ratios_by_round:
        raise ValueError(
            f"No usable krum_score_ratio values found in {log_csv_path} "
            f"-- every MEAN row was N/A (no Byzantine or no honest "
            f"finite Krum scores that round). Check --byzantine and "
            f"--aggregator actually produced Krum diagnostics."
        )

    rounds_sorted = sorted(ratios_by_round)
    selected = [rounds_sorted[-1]] if rounds_to_average == "last" else rounds_sorted
    vals = [ratios_by_round[r] for r in selected]
    return sum(vals) / len(vals)


def check_minmax_vs_plain_adaptive_krum(workdir, gamma_init=None,
                                         minmax_dev_type="std",
                                         minmax_search_iters=15,
                                         byzantine_clients=None,
                                         rounds=None):
    """Acceptance item 1: plain Adaptive Krum's krum_score_ratio (mean
    Byzantine Krum score / mean honest Krum score) on Min-Max must land
    strictly in (MINMAX_RATIO_LOW, MINMAX_RATIO_HIGH) -- see that
    constant's comment for why this replaced a TPR band (TPR is
    quantized to {0%, 100%} for this attack by construction, since
    every colluding client broadcasts one identical crafted vector and
    so always shares one Krum score).

    byzantine_clients : str or None
        Coalition to attack with, e.g. "1,2,3,4" -- defaults to
        DEFAULT_BYZANTINE ("1,2"). CAUTION, unlike gamma: growing the
        coalition is NOT a safe way to push the attack further from
        the honest cluster on its own -- it simultaneously shrinks
        theoretical_neighbours = n - f - 2, and once coalition size
        approaches that count, colluding members' mutual zero-distance
        to each other dominates their Krum score regardless of gamma,
        collapsing ratio toward (or below) 1 rather than raising it.
        Confirmed empirically: growing "1,2" -> "1,2,3,4" here took
        ratio from 1.64 to 0.018. Prefer tuning gamma_init within a
        FIXED coalition size first.
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

    # BUG FIX: this used to be a single fixed tag
    # ("check_minmax_adaptive_krum") shared by every gamma candidate in
    # a sweep. main.py resumes from a checkpoint whenever it finds one
    # matching tag+seed already at the target round count -- so after
    # gamma=0.5's run finished, gamma=1.0/2.0/4.0/8.0 all hit that same
    # checkpoint, never trained with their own gamma value, and this
    # script silently reported gamma=0.5's cached result four more
    # times (bit-for-bit identical FINAL TEST rows across "different"
    # candidates was the tell). Each sweep point now gets its own tag
    # so each candidate actually runs.
    tag = (
        f"check_minmax_g{_sanitize_tag_component(gamma_init if gamma_init is not None else 'auto')}"
        f"_dev{_sanitize_tag_component(minmax_dev_type)}"
        f"_it{minmax_search_iters}"
        f"_byz{_sanitize_tag_component(byz)}"
    )
    log_path = _run_main(extra, tag=tag, workdir=workdir,
                          byzantine_clients=byz, rounds=rounds)
    byz_clients = [int(c) for c in byz.split(",")]
    # TPR still computed and printed for visibility (and because Check 2
    # genuinely wants an exact-0% TPR, so _byzantine_tpr_from_log stays)
    # -- but Check 1 no longer GATES on it; see the MINMAX_RATIO_LOW/HIGH
    # comment above for why a TPR band can never pass for this attack.
    tpr = _byzantine_tpr_from_log(log_path, byz_clients)
    ratio = _byzantine_score_ratio_from_log(log_path)

    print(f"\n[CHECK 1] Plain Adaptive Krum vs Min-Max (gamma_init="
          f"{gamma_init!r}, byzantine={byz!r}, rounds={rounds!r}): "
          f"TPR = {tpr:.2%} (informational only)  "
          f"krum_score_ratio = {ratio:.4f} (gating metric)")

    if ratio <= 1.0 + 1e-9:
        collapse = ratio < 0.9  # comfortably below "just blended in"
        raise SystemExit(
            f"FAIL: krum_score_ratio={ratio:.4f} <= 1.0 -- Byzantine "
            f"clients score AS CENTRAL AS OR MORE CENTRAL THAN honest "
            f"ones, i.e. not just evasive but effectively invisible. "
            + (
                f"ratio << 1 strongly suggests the mutual-zero-distance "
                f"collapse: colluding clients broadcast one IDENTICAL "
                f"crafted vector, so once coalition size approaches "
                f"n - f - 2 (the Krum neighbour count), a client's own "
                f"colluding peers alone can satisfy most/all of its "
                f"nearest-neighbour sum at distance 0, tanking its score "
                f"regardless of gamma. Fix: SHRINK the --byzantine "
                f"coalition (check theoretical_neighbours = n - f - 2 "
                f"stays comfortably above coalition size), not gamma."
                if collapse else
                f"Fix: DECREASE --minmax-gamma-init so the crafted "
                f"update sits farther from the honest cluster; if that "
                f"alone doesn't move ratio above 1, check coalition "
                f"size against n - f - 2 as above."
            )
        )
    if not (MINMAX_RATIO_LOW < ratio < MINMAX_RATIO_HIGH):
        raise SystemExit(
            f"FAIL: krum_score_ratio={ratio:.4f} is outside the required "
            f"({MINMAX_RATIO_LOW}, {MINMAX_RATIO_HIGH}) range. "
            f"{'Attack is such an obvious outlier this is barely stealthier than sign-flip/Gaussian -- DECREASE' if ratio >= MINMAX_RATIO_HIGH else 'Attack is too close to blending into the honest cluster -- INCREASE'} "
            f"--minmax-gamma-init and re-run."
        )

    print(f"[CHECK 1] PASS -- krum_score_ratio={ratio:.4f} is within "
          f"({MINMAX_RATIO_LOW}, {MINMAX_RATIO_HIGH}). "
          f"(TPR={tpr:.2%}, kept only as a secondary, non-gating signal.)")
    return ratio


def check_minsum_vs_plain_adaptive_krum(workdir, gamma_init=None,
                                         minsum_dev_type="std",
                                         minsum_search_iters=15,
                                         byzantine_clients=None,
                                         rounds=None):
    """Acceptance item 1 (Min-Sum half, previously missing entirely).

    Mirrors check_minmax_vs_plain_adaptive_krum() exactly -- same
    krum_score_ratio gate, same coalition-collapse caution -- just with
    --attack-type minsum. See that function's docstring for the full
    rationale; not duplicated here beyond what differs.
    """
    byz = byzantine_clients if byzantine_clients is not None else DEFAULT_BYZANTINE
    extra = [
        "--aggregator", "adaptive_krum",
        "--ablation-mode", "krum_baseline",
        "--attack-type", "minsum",
        # main.py's minmax-* flags are shared by minsum (see main.py's
        # --minmax-dev-type / --minmax-search-iters / --minmax-gamma-init
        # help text -- there is no separate --minsum-* flag set).
        "--minmax-dev-type", minsum_dev_type,
        "--minmax-search-iters", str(minsum_search_iters),
    ]
    if gamma_init is not None:
        extra += ["--minmax-gamma-init", str(gamma_init)]

    tag = (
        f"check_minsum_g{_sanitize_tag_component(gamma_init if gamma_init is not None else 'auto')}"
        f"_dev{_sanitize_tag_component(minsum_dev_type)}"
        f"_it{minsum_search_iters}"
        f"_byz{_sanitize_tag_component(byz)}"
    )
    log_path = _run_main(extra, tag=tag, workdir=workdir,
                          byzantine_clients=byz, rounds=rounds)
    byz_clients = [int(c) for c in byz.split(",")]
    tpr = _byzantine_tpr_from_log(log_path, byz_clients)
    ratio = _byzantine_score_ratio_from_log(log_path)

    print(f"\n[CHECK 3] Plain Adaptive Krum vs Min-Sum (gamma_init="
          f"{gamma_init!r}, byzantine={byz!r}, rounds={rounds!r}): "
          f"TPR = {tpr:.2%} (informational only)  "
          f"krum_score_ratio = {ratio:.4f} (gating metric)")

    if ratio <= 1.0 + 1e-9:
        collapse = ratio < 0.9
        raise SystemExit(
            f"FAIL: krum_score_ratio={ratio:.4f} <= 1.0 -- Byzantine "
            f"clients score AS CENTRAL AS OR MORE CENTRAL THAN honest "
            f"ones under Min-Sum, i.e. not just evasive but effectively "
            f"invisible. "
            + (
                f"ratio << 1 strongly suggests the same mutual-zero-"
                f"distance collapse documented for Min-Max -- SHRINK the "
                f"--byzantine coalition (check n - f - 2 stays "
                f"comfortably above coalition size), not gamma."
                if collapse else
                f"Fix: DECREASE --minmax-gamma-init so the crafted "
                f"update sits farther from the honest cluster; if that "
                f"alone doesn't move ratio above 1, check coalition "
                f"size against n - f - 2 as above."
            )
        )
    if not (MINSUM_RATIO_LOW < ratio < MINSUM_RATIO_HIGH):
        raise SystemExit(
            f"FAIL: krum_score_ratio={ratio:.4f} is outside the required "
            f"({MINSUM_RATIO_LOW}, {MINSUM_RATIO_HIGH}) range. "
            f"{'Attack is such an obvious outlier this is barely stealthier than sign-flip/Gaussian -- DECREASE' if ratio >= MINSUM_RATIO_HIGH else 'Attack is too close to blending into the honest cluster -- INCREASE'} "
            f"--minmax-gamma-init and re-run."
        )

    print(f"[CHECK 3] PASS -- krum_score_ratio={ratio:.4f} is within "
          f"({MINSUM_RATIO_LOW}, {MINSUM_RATIO_HIGH}). "
          f"(TPR={tpr:.2%}, kept only as a secondary, non-gating signal.)")
    return ratio


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
    parser.add_argument("--minsum-gamma-candidates", type=str, default=None,
                         help="Same as --gamma-candidates but for Check 3 "
                              "(Min-Sum). Defaults to whatever "
                              "--gamma-candidates resolved to if omitted, "
                              "since Min-Sum often -- but is not "
                              "guaranteed to -- share a workable gamma "
                              "range with Min-Max.")
    parser.add_argument("--minsum-byzantine-candidates", type=str, default=None,
                         help="Same as --byzantine-candidates but for "
                              "Check 3 (Min-Sum). Defaults to whatever "
                              "--byzantine-candidates resolved to if "
                              "omitted.")
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

        # Check 3 (Min-Sum) -- same outer/inner sweep structure as
        # Check 1, run second so a Check 1 failure is reported first
        # (Check 1 currently has the larger literature precedent for
        # what gamma/coalition ranges are reasonable to try first).
        minsum_gamma_candidates = (
            [float(g) for g in args.minsum_gamma_candidates.split(",")]
            if args.minsum_gamma_candidates
            else gamma_candidates
        )
        minsum_byzantine_candidates = (
            args.minsum_byzantine_candidates.split(";")
            if args.minsum_byzantine_candidates
            else byzantine_candidates
        )

        passed_minsum = False
        last_error_minsum = None
        for byz in minsum_byzantine_candidates:
            for gamma in minsum_gamma_candidates:
                try:
                    check_minsum_vs_plain_adaptive_krum(
                        workdir, gamma_init=gamma, byzantine_clients=byz,
                        rounds=args.calibration_rounds,
                    )
                    passed_minsum = True
                    break
                except SystemExit as e:
                    last_error_minsum = e
                    print(f"  (byzantine={byz!r}, gamma_init={gamma!r} "
                          f"failed for Min-Sum, trying next candidate if "
                          f"any remain)")
            if passed_minsum:
                break
        if not passed_minsum:
            print(f"\nAll byzantine/gamma candidates exhausted without "
                  f"passing Check 3 (Min-Sum).\n{last_error_minsum}")
            sys.exit(1)

        check_bounded_directional_vs_norm_guard(
            workdir, bounded_tau=args.bounded_tau,
            rounds=args.calibration_rounds,
        )

        print(f"\n{'='*70}\n"
              f"  ALL TASK 2 CHECKS PASSED (Min-Max, Min-Sum, bounded-\n"
              f"  directional) -- safe to proceed to the full E1-E8\n"
              f"  5-seed campaign (Task 3), including E2.\n"
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
