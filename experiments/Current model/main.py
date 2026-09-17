"""
Unified FL-IDS Main Loop
========================
PRV1 FIX -- Cumulative DP accounting, implemented LITERALLY per the
issue spec:

  - One opacus.PrivacyEngine per DP-active client, created EXACTLY ONCE
    (before Round 1) and reused, unmodified, for that client's entire
    NUM_ROUNDS-round lifespan. Enforced via
    dp_persistent_client_state.py: a plain in-process dict holding
    each DP-active client's (model, optimizer, engine), built once
    before Round 1 and reused every round -- no multiprocessing, no
    threads.
    (This codebase's execution target is CUDA-only. GPU client
    training/eval already runs sequentially, in-process, with no
    ProcessPoolExecutor at all -- see the "fork+CUDA hang" fix
    elsewhere in this file's history. A dedicated-OS-process-per-client
    design for DP would reintroduce exactly that hazard, for no
    parallelism benefit, since GPU training is sequential regardless of
    client count. Task 1's "same object, never recreated" guarantee is
    satisfied trivially by keeping a plain dict alive in the one
    process that's already running everything sequentially.)
  - _train_one_client() (still used for Byzantine clients, and for
    every client on any USE_DP=False run) contains NO
    `PrivacyEngine(...)` constructor call -- verified both by grep and
    by an AST-based test (tests/test_dp_accounting_composition.py's
    test_ast_audit_no_privacy_engine_in_train_one_client). There is no
    DP branch in that function at all; DP-active honest clients never
    call it -- they go through
    dp_persistent_client_state.run_dp_client_round() instead, within
    the same sequential per-round loop.
  - final_total_epsilon is read directly off each client's own
    never-reset engine, once, after the last round (via
    get_final_epsilons()) -- not recomputed from a separately
    maintained accountant.

KNOWN, ACCEPTED TRADE-OFF: mid-run checkpoint resume is NOT supported
for USE_DP=True runs. The cumulative-epsilon accounting state lives
entirely inside live PrivacyEngine objects in this process's memory --
there is no way to serialize "the same DPOptimizer/GradSampleModule
hook state" to disk and reconstruct it faithfully after a crash. A
DP-active run that crashes must be restarted from Round 1. This is
flagged loudly at run time (see main()) rather than silently producing
a plausible-looking but incorrect resumed epsilon. Confirmed
acceptable -- not treated as an open item.

Merges:
  - DP/Norm-Guard/HE main.py  (privacy stack structure)
  - Krum main.py        (working Multi-Krum aggregation)
--------------------------------------------------------------------------
"""

import os
import sys
import csv
import copy
import json
import time
import warnings
import contextlib
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

from config_loader import load_hyperparams_config, get_value

# ---------------------------------------------------------------------------
# Path setup -- allow running from project root OR from src/
# ---------------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

import argparse

_parser = argparse.ArgumentParser(
    description="FL-IDS unified training loop."
)
_parser.add_argument("model_type", choices=["network", "application"],
                      nargs="?", default="network")
_parser.add_argument("--dataset", type=str, default="edge_iiotset",
                      choices=["edge_iiotset", "ciciot2023"],
                      help="Issue 5 Task 3/6 (E7): which dataset to load. "
                           "'edge_iiotset' (default) is this codebase's "
                           "original DNN-EdgeIIoT-dataset.csv, unaffected "
                           "by this flag entirely. 'ciciot2023' swaps in "
                           "ciciot2023_loader.py as a drop-in replacement "
                           "for data_loader.py's network-model surface "
                           "(see that module's docstring for exactly how) "
                           "-- ONLY valid with model_type='network' "
                           "(positional arg above); ciciot2023_loader has "
                           "no application-model equivalent and will "
                           "raise NotImplementedError if you combine "
                           "--dataset ciciot2023 with the 'application' "
                           "positional arg. Set CICIOT2023_DATASET_DIR "
                           "env var to point at the real dataset "
                           "directory before using this.")
_parser.add_argument("--ciciot-subset-fraction", type=float, default=0.10,
                      help="Issue 5 Task 3 (E7): stratified subset "
                           "fraction of the full ~46M-row CICIoT2023 "
                           "dataset to load. Only meaningful with "
                           "--dataset ciciot2023.")
_parser.add_argument("--ciciot-five-feature-slice", action="store_true",
                      help="Issue 5 Task 3 (E7): use the 5-feature "
                           "Edge-IIoTset-semantic-overlap slice instead "
                           "of CICIoT2023's full 46-feature schema (issue "
                           "spec's compute-lightweight fallback, "
                           "'if full subset is too heavy'). Only "
                           "meaningful with --dataset ciciot2023.")
_parser.add_argument("--epsilon", type=float, default=None,
                      help="PRV1: Override DP_EPSILON -- this is the "
                           "FULL-RUN target epsilon (composed over all "
                           "NUM_ROUNDS x LOCAL_EPOCHS optimizer steps for "
                           "the client's whole lifespan), NOT a per-round "
                           "value. e.g. --epsilon 9.0 means 'this client's "
                           "cumulative privacy loss after all 25 rounds "
                           "should be ~9', matching E4_dense_epsilon_sweep.json's "
                           "full_run_target_epsilon grid.")
_parser.add_argument("--tag", type=str, default=None,
                      help="Suffix on every output filename.")
_parser.add_argument("--byzantine", type=str, default=None,
                      help="Comma-separated 1-indexed client numbers to make "
                           "Byzantine, e.g. --byzantine 4,10. Default is "
                           "clients 1,2.")
_parser.add_argument("--krum-k", type=float, default=None,
                      help="Override ADAPTIVE_KRUM_K / HEAD_NORM_GUARD_K.")
_parser.add_argument("--assumed-f", type=int, default=None,
                      help="Override ADAPTIVE_KRUM_HYBRID_ASSUMED_F (the "
                           "assumed attacker count the HE+Krum hybrid "
                           "pipeline's plaintext-slice Krum step uses), "
                           "before clamping to NUM_BYZANTINE. Falls back "
                           "to hyperparams.json's "
                           "adaptive_krum_hybrid_assumed_f if omitted.")
_parser.add_argument("--aggregator", type=str, default=None,
                      choices=["fedavg", "krum", "multi_krum", "median",
                               "trimmed_mean", "adaptive_krum",
                               "calibrated_krum"],
                      help="BAS1 (Issue 3) Task 1: which plaintext "
                           "aggregator to run. Only meaningful when "
                           "--ablation-mode is 'baseline' or "
                           "'krum_baseline' -- ignored (with a warning) "
                           "for any other mode, since those modes' "
                           "aggregation is governed by their own "
                           "USE_HE/USE_HE_KRUM_HYBRID/USE_NORM_GUARD flags, "
                           "not by this dispatch table. "
                           "'calibrated_krum' is reserved for Issue 4 "
                           "and will raise NotImplementedError if "
                           "actually selected.")
_parser.add_argument("--attack-type", type=str, default="sign_flip",
                      choices=["sign_flip", "gaussian", "zero_gradient",
                               "minmax", "minsum", "bounded_directional"],
                      help="Which Byzantine attack the malicious clients use. "
                           "Issue 5 Task 1 adds minmax/minsum (Fang et al. "
                           "USENIX'20 stealthy, distance-aware, AGR-agnostic "
                           "poisoners -- coalition-broadcast, see "
                           "defences/byzantine.py) and bounded_directional "
                           "(targeted direction, magnitude capped just under "
                           "the norm-guard threshold -- defeats the HMAC "
                           "norm guard by design; documented limitation).")
_parser.add_argument("--gaussian-std", type=float, default=None,
                      help="Std dev for --attack-type gaussian.")
_parser.add_argument("--minmax-dev-type", type=str, default="std",
                      choices=["std", "sign", "unit_vec"],
                      help="Issue 5 Task 1: perturbation-direction estimator "
                           "for --attack-type minmax/minsum. 'std' = per-"
                           "coordinate std of the Byzantine coalition's own "
                           "honestly-trained updates (paper default).")
_parser.add_argument("--minmax-search-iters", type=int, default=15,
                      help="Issue 5 Task 1: binary-search iterations for the "
                           "Min-Max/Min-Sum perturbation scale gamma.")
_parser.add_argument("--minmax-gamma-init", type=float, default=None,
                      help="Issue 5 Task 1: upper-bound search scale gamma "
                           "for Min-Max/Min-Sum. Default: 5x the coalition's "
                           "own pairwise-distance spread. Task 2 requires "
                           "tuning this so plain Adaptive Krum TPR on this "
                           "attack lands strictly in 50-85%% -- raise it if "
                           "TPR is stuck at 100%% (not evasive enough), "
                           "lower it if TPR collapses to 0%% (too aggressive).")
_parser.add_argument("--bounded-tau", type=float, default=None,
                      help="Issue 5 Task 1: assumed norm-guard threshold tau "
                           "for --attack-type bounded_directional. If "
                           "omitted, estimated each round from the PRIOR "
                           "round's verified honest head-norms via "
                           "defences/byzantine.py:estimate_norm_guard_tau() "
                           "(same MAD-k rule as the guard itself). On round "
                           "1 (no prior data), the attack is skipped and the "
                           "client trains honestly that round -- logged, not "
                           "silent. Prefer passing this explicitly (matched "
                           "to the guard's own computed threshold) for "
                           "Task 2's exact 'TPR=0%% by design' check.")
_parser.add_argument("--bounded-margin", type=float, default=0.05,
                      help="Issue 5 Task 1: safety margin subtracted from "
                           "tau for --attack-type bounded_directional -- "
                           "crafted update norm = tau - bounded-margin.")
_parser.add_argument("--bounded-direction", type=str, default="negate",
                      choices=["negate", "classifier_head_negate"],
                      help="Issue 5 Task 1: adversarial direction for "
                           "--attack-type bounded_directional. 'negate' "
                           "flips the client's whole trained update; "
                           "'classifier_head_negate' flips only the "
                           "classifier-head slice (targeted rare-class "
                           "attack) before the whole vector is rescaled to "
                           "the shared norm budget.")
_parser.add_argument("--no-dp-calibration", action="store_true",
                      help="Issue 5 Task 3 (E6 ablation): force "
                           "use_dp_calibration=False in "
                           "calibrated_adaptive_multi_krum(), regardless of "
                           "AGGREGATOR. Previously use_dp_calibration=True "
                           "was hardcoded at the call site with no CLI path "
                           "to turn it off -- E6's 4-variant ablation "
                           "('Full proposed' / '-DP calibration only' / "
                           "'-Heterogeneity calibration only' / 'Calibration "
                           "off == plain Adaptive Krum') needs exactly this "
                           "toggle, independently of --no-hetero-calibration "
                           "below. Only meaningful when AGGREGATOR== "
                           "'calibrated_krum'; ignored otherwise.")
_parser.add_argument("--no-hetero-calibration", action="store_true",
                      help="Issue 5 Task 3 (E6 ablation): force "
                           "use_hetero_calibration=False -- same rationale "
                           "as --no-dp-calibration above, independent toggle "
                           "for the heterogeneity-variance term. Passing "
                           "BOTH --no-dp-calibration and "
                           "--no-hetero-calibration reproduces plain "
                           "Adaptive Krum's client selection exactly (E6 "
                           "variant 4), since with every calibration term "
                           "off, calibrated_adaptive_multi_krum's distance "
                           "scoring reduces to plain Adaptive Krum's -- see "
                           "the 'calibrated_krum_dp_sweep' ablation-mode "
                           "help text above for the same equivalence "
                           "argument applied to noise_multiplier=None.")
_parser.add_argument("--rounds", type=int, default=None,
                      help="Issue 5 Task 2: override NUM_ROUNDS (default 25, "
                           "or 2 under SANITY_CHECK). Exists so "
                           "scripts/check_attack_difficulty.py can run a "
                           "short (e.g. 5-round) calibration pass to tune "
                           "--minmax-gamma-init without paying for a full "
                           "25-round run on every tuning iteration. NOT "
                           "used by the full E1-E8 campaign runners, which "
                           "rely on the real default.")
_parser.add_argument("--byzantine-full-model", action="store_true",
                      help="Issue 5 (bug fix): force BYZANTINE_HEAD_ONLY="
                           "False regardless of --ablation-mode. "
                           "pure_norm_guard and exp2_unmitigated/"
                           "exp2_mitigated all hardcode BYZANTINE_HEAD_ONLY"
                           "=True, and the attack dispatch checks that flag "
                           "BEFORE --attack-type -- meaning --attack-type "
                           "was silently ignored (replaced with "
                           "classifier_head_flip_attack) under those modes "
                           "with no way to actually test e.g. "
                           "bounded_directional's real purpose (evading a "
                           "LIVE norm guard) until this flag existed. "
                           "Required by scripts/check_attack_difficulty.py's "
                           "Check 2.")
_parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for torch/numpy/python-random and "
                           "the client Dirichlet partition.")
_parser.add_argument("--alpha", type=float, default=None,
                      help="Issue 4 Task 0/1: Dirichlet concentration "
                           "parameter controlling client-level label "
                           "heterogeneity, passed through to "
                           "load_partition_network()/load_partition_"
                           "application()'s existing alpha= kwarg in "
                           "data_loader.py (_dirichlet_partition()). "
                           "Lower alpha = more heterogeneous (label-"
                           "skewed) client partitions. If omitted, uses "
                           "data_loader.py's own default (0.7). NOTE: "
                           "this flag did NOT exist before Issue 4 -- "
                           "the underlying alpha-Dirichlet partitioner "
                           "itself already existed in data_loader.py, "
                           "it was simply never exposed on the CLI, so "
                           "every prior run silently used alpha=0.7 "
                           "regardless of intent.")
_parser.add_argument("--ablation-mode", type=str, default=None,
                      choices=["pure_dp", "pure_he", "pure_norm_guard",
                               "krum_dp_sweep", "exp2_unmitigated",
                               "exp2_mitigated", "baseline", "krum_baseline",
                               "calibrated_krum_dp_sweep"],
                      help="Override the hardcoded ABLATION_MODE below via CLI. "
                           "calibrated_krum_dp_sweep (Issue 4 follow-up fix): "
                           "mirrors krum_dp_sweep's shape (USE_DP=True, "
                           "Byzantine attack ON) but routes aggregation "
                           "through calibrated_adaptive_multi_krum instead of "
                           "plain adaptive_multi_krum. Exists because neither "
                           "'baseline' nor 'krum_baseline' (the only two "
                           "modes that honour --aggregator=calibrated_krum) "
                           "ever set USE_DP=True -- without this mode, "
                           "calibrated_krum's DP-noise-variance term was "
                           "mathematically unreachable through any CLI-driven "
                           "run (verified: with every client's "
                           "noise_multiplier=None, calibrated_adaptive_multi_"
                           "krum produces IDENTICAL client selection to plain "
                           "adaptive_multi_krum, since dividing every pairwise "
                           "distance by the same round-level constant "
                           "preserves score ranking exactly).")
_parser.add_argument("--force-dp-safe-arch", action="store_true",
                      help="BAS1 (Issue 3) Task 4: force dp_safe=True (the "
                           "GroupNorm+DPLSTM architecture swap) even when "
                           "the selected --ablation-mode would otherwise "
                           "leave USE_DP=False (and therefore DP_SAFE=False, "
                           "since DP_SAFE is normally derived directly from "
                           "USE_DP). Exists specifically for the "
                           "'FedProx GroupNorm+DPLSTM no-DP' E1 baseline row "
                           "(fl_fedprox_dpsafe_arch_no_dp), which isolates "
                           "the architecture swap's effect on performance "
                           "from the DP-SGD noise mechanism's effect -- "
                           "USE_DP must stay False (no noise injected) while "
                           "DP_SAFE must be True (DP-safe layers used "
                           "anyway). Has no effect when USE_DP is already "
                           "True for the selected ablation mode (DP_SAFE is "
                           "already True there).")
_parser.add_argument("--prox-mu", type=float, default=None,
                      help="Task 4 fix: override PROX_MU (the FedProx "
                           "proximal-term coefficient applied during LOCAL "
                           "client training, NOT a server-aggregation "
                           "choice -- 'FedAvg' in this codebase's convention "
                           "is simply PROX_MU=0, 'FedProx' is any nonzero "
                           "value). Previously PROX_MU was only settable by "
                           "editing hyperparams.json's fedprox_mu field "
                           "between runs, which made Task 4's required "
                           "5-point sweep (mu in {0, 0.005, 0.02, 0.05, "
                           "0.1}) impractical to script across 5 seeds x 2 "
                           "models without either hand-editing config 50 "
                           "times or writing a config-mutating wrapper. "
                           "Omitting this flag reproduces prior behavior "
                           "exactly (falls back to hyperparams.json's "
                           "fedprox_mu).")
_parser.add_argument("--hetero-fit-coeffs-json", type=str, default=None,
                      help="Issue 4 Task 2: path to a JSON file containing "
                           "{'intercept', 'coef_n_samples_diff', "
                           "'coef_entropy_diff', 'r_squared', 'n_rows_fit'} "
                           "-- the output of defences/krum.py's "
                           "fit_hetero_variance_regression(), itself fit "
                           "against a REAL per_client_krum_scores.csv from "
                           "Sub-task A's sanity sweep (scripts/"
                           "fit_hetero_variance.py wraps this). Only "
                           "consumed when USE_CALIBRATED_KRUM is True. If "
                           "omitted, hetero_fit_coeffs stays None and "
                           "use_hetero_calibration remains a documented "
                           "no-op for the hetero term specifically (DP "
                           "calibration, if reachable via this run's "
                           "ablation mode, is unaffected) -- this is the "
                           "correct, honest default until a real fit "
                           "exists; never fabricate a placeholder value "
                           "here to make this flag 'do something.'")
_args = _parser.parse_args()

MODEL_TYPE = _args.model_type

if _args.hetero_fit_coeffs_json is not None:
    import json as _json_hetero
    with open(_args.hetero_fit_coeffs_json) as _f_hetero:
        HETERO_FIT_COEFFS = _json_hetero.load(_f_hetero)
    print(f"  [Issue 4] Loaded real hetero_fit_coeffs from "
          f"{_args.hetero_fit_coeffs_json}: r_squared="
          f"{HETERO_FIT_COEFFS.get('r_squared', 'N/A')}, "
          f"n_rows_fit={HETERO_FIT_COEFFS.get('n_rows_fit', 'N/A')}")
else:
    HETERO_FIT_COEFFS = None

# Issue 5 Task 3 (E6 ablation): derive the two calibration-term toggles
# from the new --no-dp-calibration / --no-hetero-calibration flags.
# Defaults (neither flag passed) => both True, IDENTICAL to the prior
# hardcoded calibrated_adaptive_multi_krum(use_dp_calibration=True,
# use_hetero_calibration=True) call -- purely additive.
USE_DP_CALIBRATION = not _args.no_dp_calibration
USE_HETERO_CALIBRATION = not _args.no_hetero_calibration

import random
random.seed(_args.seed)
np.random.seed(_args.seed)
torch.manual_seed(_args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_args.seed)

SANITY_CHECK = False

_HP_CONFIG = load_hyperparams_config()

# FL hyperparameters
NUM_ROUNDS    = _args.rounds if _args.rounds is not None else (2 if SANITY_CHECK else 25)
NUM_CLIENTS   = 10
LOCAL_EPOCHS  = 5
LEARNING_RATE = 0.001

# PRV1 Task 1.3 -- the real optimizer.step() horizon a client's DP
# accountant must be calibrated against. Computed from the actual
# NUM_ROUNDS/LOCAL_EPOCHS in effect this run -- never hardcoded.
TOTAL_EPOCHS_PER_CLIENT = NUM_ROUNDS * LOCAL_EPOCHS
PROX_MU = _args.prox_mu if _args.prox_mu is not None else get_value(_HP_CONFIG, "fedprox_mu")

if _args.byzantine is not None:
    _byzantine_1indexed = sorted(int(c.strip()) for c in _args.byzantine.split(","))
    BYZANTINE_CLIENTS = [c - 1 for c in _byzantine_1indexed]
    NUM_BYZANTINE = len(BYZANTINE_CLIENTS)
    assert len(set(BYZANTINE_CLIENTS)) == NUM_BYZANTINE, \
        f"--byzantine has duplicate client numbers: {_args.byzantine}"
    assert all(1 <= c <= NUM_CLIENTS for c in _byzantine_1indexed), \
        (f"--byzantine client numbers must be in [1, {NUM_CLIENTS}], "
         f"got {_byzantine_1indexed}")
else:
    NUM_BYZANTINE = 2
    BYZANTINE_CLIENTS = list(range(NUM_BYZANTINE))

# Issue 4 Task 0/1: real, CLI-controllable Dirichlet alpha. Defaults to
# data_loader.py's own default (0.7) when not passed, so any existing
# script that doesn't pass --alpha gets IDENTICAL behavior to before
# this change.
ALPHA_DIRICHLET = _args.alpha if _args.alpha is not None else 0.7

ATTACK_SCALE = 5.0 if MODEL_TYPE == "network" else 2.0
ATTACK_TYPE = _args.attack_type

_GAUSSIAN_STD_DEFAULT = 50.0 if MODEL_TYPE == "network" else 30.0
GAUSSIAN_STD = _args.gaussian_std if _args.gaussian_std is not None else _GAUSSIAN_STD_DEFAULT

# Issue 5 Task 1 -- stealthy attack hyperparameters.
MINMAX_DEV_TYPE      = _args.minmax_dev_type
MINMAX_SEARCH_ITERS  = _args.minmax_search_iters
MINMAX_GAMMA_INIT    = _args.minmax_gamma_init      # None => auto (5x coalition spread)
BOUNDED_TAU_OVERRIDE = _args.bounded_tau            # None => estimate from prior round
BOUNDED_MARGIN       = _args.bounded_margin
BOUNDED_DIRECTION    = _args.bounded_direction

# ---------------------------------------------------------------------------
# ABLATION MODE SELECTOR
# ---------------------------------------------------------------------------
ABLATION_MODE = (_args.ablation_mode if _args.ablation_mode is not None
                  else "exp2_unmitigated")

if ABLATION_MODE == "pure_dp":
    USE_KRUM = USE_ADAPTIVE_KRUM = USE_HE = USE_HE_KRUM_HYBRID = USE_NORM_GUARD = False
    USE_DP = True
    USE_BYZANTINE_ATTACK = False
    BYZANTINE_HEAD_ONLY = False

elif ABLATION_MODE == "pure_he":
    USE_HE = True
    USE_KRUM = USE_ADAPTIVE_KRUM = USE_HE_KRUM_HYBRID = USE_NORM_GUARD = USE_DP = False
    USE_BYZANTINE_ATTACK = False
    BYZANTINE_HEAD_ONLY = False

elif ABLATION_MODE == "pure_norm_guard":
    USE_NORM_GUARD = True
    USE_HE = USE_KRUM = USE_ADAPTIVE_KRUM = USE_HE_KRUM_HYBRID = USE_DP = False
    USE_BYZANTINE_ATTACK = True
    BYZANTINE_HEAD_ONLY = True

elif ABLATION_MODE == "krum_dp_sweep":
    # Issue 5 Task 3 fix: previously hardcoded USE_ADAPTIVE_KRUM=True
    # with no --aggregator override, matching only "baseline" and
    # "krum_baseline"'s ORIGINAL (pre-dispatch-table) hardcoded shape.
    # That left every DP+Byzantine-attack run stuck on a single
    # aggregator (plain Adaptive Krum) -- there was no CLI-reachable
    # way to run e.g. 'median + DP + Byzantine attack' or 'fedavg + DP
    # + Byzantine attack', which Issue 5's E4 (dense epsilon sweep,
    # "3 aggregators x alpha x epsilon") needs for its third,
    # non-calibrated reference aggregator alongside adaptive_krum and
    # calibrated_krum (the latter already reachable via the separate
    # calibrated_krum_dp_sweep mode below). Mirrors krum_baseline's
    # exact override pattern (see that block + the "AGGREGATOR not in
    # globals()" dispatch-derivation block right after this whole
    # if/elif chain, which re-derives USE_KRUM/USE_ADAPTIVE_KRUM/
    # USE_CALIBRATED_KRUM from whatever AGGREGATOR ends up being here)
    # -- purely additive, existing scripts that omit --aggregator get
    # IDENTICAL behavior to before (falls back to "adaptive_krum").
    USE_KRUM = USE_HE = USE_HE_KRUM_HYBRID = USE_NORM_GUARD = False
    USE_DP = True
    USE_BYZANTINE_ATTACK = True
    BYZANTINE_HEAD_ONLY = False
    AGGREGATOR = _args.aggregator if _args.aggregator is not None else "adaptive_krum"

elif ABLATION_MODE == "exp2_unmitigated":
    USE_HE_KRUM_HYBRID = True
    USE_KRUM = USE_ADAPTIVE_KRUM = USE_HE = USE_NORM_GUARD = False
    USE_DP = False
    USE_BYZANTINE_ATTACK = True
    BYZANTINE_HEAD_ONLY = True
    USE_HEAD_NORM_GUARD = False

elif ABLATION_MODE == "exp2_mitigated":
    USE_HE_KRUM_HYBRID = True
    USE_KRUM = USE_ADAPTIVE_KRUM = USE_HE = USE_NORM_GUARD = False
    USE_DP = False
    USE_BYZANTINE_ATTACK = True
    BYZANTINE_HEAD_ONLY = True
    USE_HEAD_NORM_GUARD = True

elif ABLATION_MODE == "baseline":
    USE_KRUM = USE_ADAPTIVE_KRUM = USE_HE = USE_HE_KRUM_HYBRID = USE_NORM_GUARD = False
    USE_DP = False
    USE_BYZANTINE_ATTACK = False
    BYZANTINE_HEAD_ONLY = False
    AGGREGATOR = _args.aggregator if _args.aggregator is not None else "fedavg"

elif ABLATION_MODE == "krum_baseline":
    USE_ADAPTIVE_KRUM = True
    USE_KRUM = USE_HE = USE_HE_KRUM_HYBRID = USE_NORM_GUARD = False
    USE_DP = False
    USE_BYZANTINE_ATTACK = True
    BYZANTINE_HEAD_ONLY = False
    AGGREGATOR = _args.aggregator if _args.aggregator is not None else "adaptive_krum"

elif ABLATION_MODE == "calibrated_krum_dp_sweep":
    # Issue 4 follow-up fix -- see this mode's --ablation-mode help text
    # above for the full "why this mode exists" explanation. Shape
    # mirrors krum_dp_sweep (USE_DP=True, Byzantine attack ON,
    # BYZANTINE_HEAD_ONLY=False) but forces AGGREGATOR="calibrated_krum"
    # so the dispatch-derivation block below sets USE_CALIBRATED_KRUM=True
    # -- this is the ONLY ablation mode where USE_DP and
    # USE_CALIBRATED_KRUM are both True, i.e. the only mode where
    # dp_variance() ever receives a real (non-None) noise_multiplier.
    AGGREGATOR = "calibrated_krum"
    USE_KRUM = USE_ADAPTIVE_KRUM = USE_HE = USE_HE_KRUM_HYBRID = USE_NORM_GUARD = False
    USE_DP = True
    USE_BYZANTINE_ATTACK = True
    BYZANTINE_HEAD_ONLY = False

else:
    raise ValueError(f"Unknown ABLATION_MODE={ABLATION_MODE!r}")

# Issue 5 (found via check_attack_difficulty.py review): pure_norm_guard
# and exp2_unmitigated/exp2_mitigated all hardcode BYZANTINE_HEAD_ONLY=
# True, and the attack-dispatch precedence (see _train_one_client():
# "if (USE_HE or USE_HE_KRUM_HYBRID or USE_NORM_GUARD) and
# BYZANTINE_HEAD_ONLY:") checks that flag BEFORE checking ATTACK_TYPE --
# meaning --attack-type was being SILENTLY IGNORED (replaced with
# classifier_head_flip_attack) for any of those three modes, with no
# CLI-reachable way to test e.g. bounded_directional's actual intended
# purpose (evading a LIVE norm guard) under pure_norm_guard. This flag
# closes that gap without touching any mode's hardcoded default --
# purely opt-in, existing scripts unaffected.
if _args.byzantine_full_model:
    BYZANTINE_HEAD_ONLY = False

# BAS1 (Issue 3) Task 1 -- dispatch table. Additive: ABLATION_MODE still
# governs everything it always has (this was an explicit constraint --
# DP+Krum, HE+Krum hybrid, and norm-guard-only combinations aren't
# expressible as a single aggregator-name string). --aggregator only
# ever applies to the two purely-plaintext modes above, replacing what
# used to be a hardcoded USE_KRUM/USE_ADAPTIVE_KRUM/else-fedprox choice
# baked into the mode itself. Existing scripts that only pass
# --ablation-mode (no --aggregator) get IDENTICAL behavior to before
# this change: baseline -> fedavg, krum_baseline -> adaptive_krum.
USE_CALIBRATED_KRUM = False   # default; only True when AGGREGATOR=="calibrated_krum"
                              # -- reachable via --aggregator=calibrated_krum under
                              # ABLATION_MODE in {baseline, krum_baseline} (USE_DP
                              # always False there), OR via
                              # --ablation-mode calibrated_krum_dp_sweep (USE_DP
                              # always True there, AGGREGATOR forced to
                              # "calibrated_krum" directly, no --aggregator needed).
if "AGGREGATOR" not in globals():
    AGGREGATOR = None
    if _args.aggregator is not None:
        warnings.warn(
            f"--aggregator={_args.aggregator!r} was passed but "
            f"ABLATION_MODE={ABLATION_MODE!r} is not 'baseline', "
            f"'krum_baseline', or 'calibrated_krum_dp_sweep' -- ignored. "
            f"This mode's aggregation is governed by its own "
            f"USE_HE/USE_HE_KRUM_HYBRID/USE_NORM_GUARD flags, not the "
            f"aggregator dispatch table."
        )
else:
    # Derive the existing booleans FROM the dispatch selection so every
    # existing downstream consumer (imports, mutual-exclusion assert,
    # _krum_active, manifest dict, startup print) keeps working
    # unmodified -- this is what makes the dispatch table additive
    # rather than a competing selection mechanism. "krum" and
    # "multi_krum" are treated as aliases for the same fixed-M function
    # (krum.py only defines one fixed-M Multi-Krum implementation).
    USE_KRUM = AGGREGATOR in ("krum", "multi_krum")
    USE_ADAPTIVE_KRUM = (AGGREGATOR == "adaptive_krum")
    # Issue 4 Task 2 FOLLOW-UP FIX (previously an open item, now closed):
    # a prior revision of this file left calibrated_krum's DP-calibration
    # term mathematically unreachable, because --aggregator was only ever
    # honoured under ABLATION_MODE in {"baseline", "krum_baseline"}, and
    # BOTH of those modes hardcode USE_DP=False -- every
    # per_client_metadata[i]["noise_multiplier"] was always None, so
    # dp_variance() always returned 0 regardless of use_dp_calibration's
    # value. Verified at the time: with every noise_multiplier=None (and
    # hetero_fit_coeffs=None, see below), calibrated_adaptive_multi_krum
    # produced IDENTICAL client selection to plain adaptive_multi_krum on
    # every tested seed -- dividing every pairwise distance by the same
    # round-level constant preserves score ranking exactly, so the whole
    # calibration mechanism was a silent no-op through any CLI-reachable
    # path. FIXED by adding ABLATION_MODE="calibrated_krum_dp_sweep"
    # above, which sets AGGREGATOR="calibrated_krum" directly (bypassing
    # --aggregator entirely) alongside USE_DP=True -- this is now the
    # mode Task 5's MAD-k/f sensitivity sweeps should use whenever DP
    # calibration needs to be exercised for real.
    #
    # REMAINING OPEN ITEM (NOT fixed by the above, still real): the
    # hetero-calibration term is separately unreachable, for a different
    # reason -- hetero_variance() requires fit_coeffs regressed against a
    # REAL per_client_krum_scores.csv from Sub-task A's sanity sweep,
    # which does not exist yet in this environment. See
    # --hetero-fit-coeffs-json below: if not passed, hetero_fit_coeffs
    # stays None and use_hetero_calibration=True remains a documented
    # no-op (a one-time warning fires from krum.py's hetero_variance()).
    # Task 5 can still produce a real, meaningful result with DP
    # calibration alone (use_hetero_calibration can be left True with no
    # ill effect -- it just contributes 0 until a real fit is supplied)
    # -- but anyone reading Task 5's numbers before a real hetero fit
    # exists should know the hetero half of "DP- and
    # Heterogeneity-Calibrated" is not yet doing anything.
    USE_CALIBRATED_KRUM = (AGGREGATOR == "calibrated_krum")
    TRIMMED_MEAN_BETA = get_value(_HP_CONFIG, "trimmed_mean_beta")

assert sum([USE_KRUM, USE_ADAPTIVE_KRUM, USE_HE, USE_HE_KRUM_HYBRID,
            USE_CALIBRATED_KRUM]) <= 1, \
    "USE_KRUM, USE_ADAPTIVE_KRUM, USE_HE, USE_HE_KRUM_HYBRID, and " \
    "USE_CALIBRATED_KRUM are mutually exclusive aggregation branches " \
    "-- pick at most one."

DP_SAFE = USE_DP or _args.force_dp_safe_arch

HE_POLY_DEGREE = 8192

if "USE_HEAD_NORM_GUARD" not in globals():
    USE_HEAD_NORM_GUARD = True
HEAD_NORM_GUARD_K = (_args.krum_k if _args.krum_k is not None
                      else get_value(_HP_CONFIG, "adaptive_krum_k"))
HEAD_NORM_GUARD_MIN_KEEP_FRACTION = 0.5


def _resolve_aggregator_canonical():
    """
    Issue 5 Task 4: a single canonical aggregator slug for this run,
    used by scripts/analysis_paper.py to group results (Table 2's
    7-aggregator comparison, E3/E4/E5/E6's 3-aggregator comparisons,
    etc.). Previously there was NO field in the run manifest recording
    which aggregator actually ran -- only the boolean flags below,
    which the round loop's aggregation elif chain reads directly.

    Mirrors that elif chain's EXACT branch-selection precedence (see
    main()'s round loop: 'if USE_HE ... elif USE_HE_KRUM_HYBRID ...
    elif USE_NORM_GUARD ... elif USE_KRUM ... elif USE_ADAPTIVE_KRUM
    ... elif USE_CALIBRATED_KRUM ... elif AGGREGATOR == ...') so this
    always matches what actually ran, without duplicating any branch
    BODY -- only the branch SELECTION order, which is static per-run
    config (these flags are never reassigned inside the round loop).
    """
    if USE_HE:
        return "he_only"
    if USE_HE_KRUM_HYBRID:
        return ("he_krum_hybrid_head_norm_guard" if USE_HEAD_NORM_GUARD
                 else "he_krum_hybrid")
    if USE_NORM_GUARD:
        return "norm_guard_only"
    if USE_KRUM:
        return "multi_krum"
    if USE_ADAPTIVE_KRUM:
        return "adaptive_krum"
    if USE_CALIBRATED_KRUM:
        return AGGREGATOR  # "calibrated_krum"
    if AGGREGATOR in ("median", "trimmed_mean", "fedavg"):
        return AGGREGATOR
    return "fedavg_implicit_default"


AGGREGATOR_CANONICAL = _resolve_aggregator_canonical()

# PRV1 Task 2 -- DP_EPSILON is now unambiguously the FULL-RUN composed
# target epsilon (see E4_dense_epsilon_sweep.json's full_run_target_epsilon
# grid), never a per-round value.
DP_EPSILON       = _args.epsilon if _args.epsilon is not None else 15.0
DP_DELTA         = 1e-5
DP_MAX_GRAD_NORM = get_value(_HP_CONFIG, "dp_max_grad_norm")
DP_BATCH_SIZE    = 512

KRUM_M = NUM_CLIENTS - NUM_BYZANTINE - 1

ADAPTIVE_KRUM_K = (_args.krum_k if _args.krum_k is not None
                    else get_value(_HP_CONFIG, "adaptive_krum_k"))
ADAPTIVE_KRUM_METHOD = "mad"
ADAPTIVE_KRUM_MIN_KEEP_FRACTION = 0.5

ADAPTIVE_KRUM_HYBRID_ASSUMED_F = min(
    (_args.assumed_f if _args.assumed_f is not None
     else get_value(_HP_CONFIG, "adaptive_krum_hybrid_assumed_f")),
    NUM_BYZANTINE
)

# ---------------------------------------------------------------------------
# Device / parallelization settings
# ---------------------------------------------------------------------------
_CPU_COUNT = os.cpu_count() or 4
_CUDA_AVAILABLE = torch.cuda.is_available()
_DEVICE = torch.device("cuda" if _CUDA_AVAILABLE else "cpu")

# NOTE (PRV1): when USE_DP=True, non-Byzantine client TRAINING goes
# through dp_states (dp_persistent_client_state.py) instead of this pool
# -- see main(). CLIENT_POOL_WORKERS below is still reported/used for the
# EVAL wave (all clients, any run); Byzantine-client training under
# USE_DP=True runs sequentially alongside dp_states, not through this pool.
CLIENT_POOL_WORKERS = 1 if _CUDA_AVAILABLE else min(4, NUM_CLIENTS)
_THREADS_PER_WORKER = max(1, _CPU_COUNT // CLIENT_POOL_WORKERS)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
_TAG = (f"{MODEL_TYPE}_{ABLATION_MODE}_seed{_args.seed}"
        if _args.tag is None
        else f"{MODEL_TYPE}_{_args.tag}_seed{_args.seed}")
CHECKPOINT_PARAMS        = f"checkpoint_{_TAG}.npz"
CHECKPOINT_PROGRESS      = f"checkpoint_{_TAG}_progress.json"
CHECKPOINT_BEST_PARAMS   = f"checkpoint_{_TAG}_best.npz"
CHECKPOINT_BEST_PROGRESS = f"checkpoint_{_TAG}_best.json"
LOG_CSV                  = f"results_{_TAG}.csv"
FINAL_TEST_CSV           = f"results_{_TAG}_FINAL_TEST.csv"
FINAL_VALIDATION_CSV     = f"results_{_TAG}_FINAL_VALIDATION.csv"

# PRV1 Task 1.5 -- the paper-ready, once-per-client final composed
# epsilon, written exactly once after the last round.
DP_FINAL_EPSILON_JSON = f"dp_final_epsilon_{_TAG}.json"
DP_FINAL_EPSILON_CSV  = f"dp_final_epsilon_{_TAG}.csv"

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
# Issue 5 Task 3/6 (E7): --dataset ciciot2023 swap. MUST happen here,
# before the `from data_loader import ...` / `from task import ...`
# lines immediately below -- task.py does its OWN
# `from data_loader import (NETWORK_NAMES, ...)` at ITS module-import
# time (see task.py's top-level imports), so this sys.modules
# substitution has to be in place before task.py is first imported
# anywhere in the process, not just before main.py's own data_loader
# import. See ciciot2023_loader.py's module docstring ("HOW THE SWAP
# WORKS") for the full rationale and why this narrow substitution was
# chosen over threading a --dataset parameter through every call site.
if _args.dataset == "ciciot2023":
    if MODEL_TYPE != "network":
        raise ValueError(
            "--dataset ciciot2023 is only valid with model_type="
            "'network' (positional arg) -- ciciot2023_loader has no "
            "application-model equivalent. See ciciot2023_loader.py's "
            "module docstring."
        )
    import ciciot2023_loader
    sys.modules["data_loader"] = ciciot2023_loader
    print(f"  [Issue 5 E7] --dataset ciciot2023 active: data_loader has "
          f"been swapped for ciciot2023_loader (subset_fraction="
          f"{_args.ciciot_subset_fraction}, five_feature_slice="
          f"{_args.ciciot_five_feature_slice}). CICIOT2023_DATASET_DIR="
          f"{os.environ.get('CICIOT2023_DATASET_DIR', '(not set, using default path)')}")
    # Pre-warm the cache with this run's actual subset_fraction /
    # five_feature_slice choice -- load_and_preprocess_ciciot2023() is
    # a no-op on every subsequent call for this seed (see its own
    # early-return-if-cached logic), so calling it once here with the
    # REAL CLI values, before any code path calls the swapped
    # get_global_test_holdout()/load_partition_network() with only
    # (model_type, seed) and no way to pass subset_fraction through,
    # guarantees those downstream calls hit an already-correctly-
    # configured cache rather than silently falling back to
    # ciciot2023_loader's DEFAULT_SUBSET_FRACTION.
    ciciot2023_loader.load_and_preprocess_ciciot2023(
        seed=_args.seed, subset_fraction=_args.ciciot_subset_fraction,
        use_five_feature_slice=_args.ciciot_five_feature_slice,
    )

if MODEL_TYPE == "network":
    from data_loader import (load_partition_network as load_partition,
                              NETWORK_NAMES as ATTACK_NAMES,
                              NUM_NETWORK_CLASSES as NUM_CLASSES)
    from task import (get_model, get_model_parameters, set_model_parameters,
                       train, test, build_criterion_network as build_criterion)
else:
    from data_loader import (load_partition_application as load_partition,
                              APP_NAMES as ATTACK_NAMES,
                              NUM_APP_CLASSES as NUM_CLASSES)
    from task import (get_model, get_model_parameters, set_model_parameters,
                       train, test, build_criterion_application as build_criterion)

from data_loader import get_global_test_holdout, get_global_validation_holdout

from defences.byzantine import (sign_flip_attack, sign_flip_attack_trained,
                                 classifier_head_flip_attack, gaussian_attack,
                                 gaussian_attack_trained, zero_gradient_attack,
                                 minmax_attack_trained, minsum_attack_trained,
                                 bounded_directional_attack_trained,
                                 estimate_norm_guard_tau)

if USE_KRUM:
    from defences.krum import multi_krum

if USE_ADAPTIVE_KRUM or USE_HE_KRUM_HYBRID:
    from defences.krum import adaptive_multi_krum

if USE_CALIBRATED_KRUM:
    from defences.krum import calibrated_adaptive_multi_krum

# BAS1 (Issue 3) Task 1 -- only imported when the dispatch table
# actually selects one of these (AGGREGATOR is None on every
# non-baseline/krum_baseline ablation mode, per the block above).
if AGGREGATOR in ("median", "trimmed_mean", "fedavg"):
    from defences.krum import coordinate_median, trimmed_mean, fedavg as _fedavg_agg

if USE_HE or USE_HE_KRUM_HYBRID or USE_NORM_GUARD:
    from defences import he_local

if USE_NORM_GUARD or (USE_HE_KRUM_HYBRID and USE_HEAD_NORM_GUARD):
    from defences import hmac_norm_guard as norm_guard

# PRV1: Opacus is only ever imported/used inside
# dp_persistent_client_state.py now, for DP-active honest clients. main.py
# itself only needs to know whether it's installed, to fail loudly up
# front instead of partway through Round 1.
if USE_DP:
    try:
        import opacus  # noqa: F401
        _OPACUS_AVAILABLE = True
    except ImportError:
        warnings.warn("Opacus not installed -- USE_DP will be skipped. "
                       "Install with: pip install opacus")
        _OPACUS_AVAILABLE = False
    from dp_persistent_client_state import (
        build_dp_client_states, run_dp_client_round, get_final_epsilons,
    )
else:
    _OPACUS_AVAILABLE = False

if USE_HE or USE_HE_KRUM_HYBRID or USE_NORM_GUARD:
    try:
        import tenseal as ts
        _TENSEAL_AVAILABLE = True
    except ImportError:
        raise ImportError("TenSEAL required for USE_HE/USE_HE_KRUM_HYBRID/"
                           "USE_NORM_GUARD=True. Install with Python 3.11: "
                           "pip install tenseal")


# ---------------------------------------------------------------------------
# Issue 4 Task 1 -- per-round, per-client structured Krum-score logger.
# ---------------------------------------------------------------------------
# Consolidated output: one row per (round, client), written to
# per_client_krum_scores.csv at the end of the run. Source of truth for
# computing Honest-FPR / Byzantine-TPR in Issue 4 Tasks 4-5.
PER_CLIENT_KRUM_LOG_CSV = f"per_client_krum_scores_{_TAG}.csv"
PER_CLIENT_KRUM_LOG_HEADER = [
    "round_id", "client_id", "raw_krum_score", "ground_truth_client_label",
    "classification", "client_n_samples", "client_class_entropy",
    "active_epsilon_config", "alpha_dirichlet_config",
]
# Buffer, flushed to disk once at the end of the run (matches this
# codebase's existing convention of writing FINAL_TEST_CSV/DP_FINAL_
# EPSILON_* once, after the round loop, rather than appending per-row
# like the main per-round LOG_CSV does).
_per_client_krum_log_rows = []


def _shannon_entropy_from_counts(counts):
    """
    Shannon entropy (natural log / nats, NOT base-2 bits -- documented
    choice per Issue 4 Task 1's requirement to pick one and state it)
    of a client's local label distribution for one round's training
    partition. `counts` is a per-class count array/list (e.g. from
    np.bincount(y_tr, minlength=NUM_CLASSES), exactly what
    print_data_split() already computes for its own printout).

    Classes with zero count contribute 0 (standard convention,
    0*log(0) treated as 0, not NaN).
    """
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


def _classify_krum_outcome(is_ground_truth_byzantine, was_excluded_this_round):
    """
    Issue 4 Task 1's classification field, per the handoff doc's
    explicit definition (stated here in the docstring so it is not
    re-derived differently later):
        TP = ground-truth Byzantine client, excluded this round
        FP = ground-truth honest client,    excluded this round
        TN = ground-truth honest client,    kept this round
        FN = ground-truth Byzantine client, kept this round
    """
    if is_ground_truth_byzantine and was_excluded_this_round:
        return "TP"
    if (not is_ground_truth_byzantine) and was_excluded_this_round:
        return "FP"
    if (not is_ground_truth_byzantine) and (not was_excluded_this_round):
        return "TN"
    return "FN"


def get_round_lr(base_lr, round_num, num_rounds, min_lr_frac=0.15):
    progress = round_num / num_rounds
    decay = 0.5 * (1 + np.cos(np.pi * progress))
    return base_lr * (min_lr_frac + (1 - min_lr_frac) * decay)


def _apply_dp_safe_prox_step(real_model, global_dict, mu, lr):
    if global_dict is None or mu == 0:
        return
    with torch.no_grad():
        for name, param in real_model.named_parameters():
            if name not in global_dict:
                continue
            g = torch.as_tensor(global_dict[name], dtype=param.dtype,
                                 device=param.device)
            param -= lr * mu * (param - g)

# ---------------------------------------------------------------------------
# PARALLEL / SEQUENTIAL CLIENT TRAINING (USE_DP=False path ONLY)
# ---------------------------------------------------------------------------
# PRV1: when USE_DP=True, non-Byzantine clients never call
# _train_one_client() for training -- see main()'s round loop, which
# calls dp_persistent_client_state.run_dp_client_round() for them instead.
# Byzantine clients (any run) and all clients on USE_DP=False runs still
# use this function, unchanged.


def _pool_worker_init():
    import torch as _torch
    _torch.set_num_threads(_THREADS_PER_WORKER)


def _train_one_client(client_idx, X_tr, y_tr, global_params, client_cfg):
    """
    Called either via ProcessPoolExecutor (CPU) or directly in-process
    (GPU). USE_DP=True runs never call this function for a non-Byzantine
    client's training -- see dp_persistent_client_state.py instead (this
    function is still used for Byzantine clients even on DP-active runs,
    and for every client on USE_DP=False runs). Grep/AST audit (PRV1
    acceptance item 1): this function constructs no Opacus privacy
    engine anywhere in its body -- there is no DP code path in this
    function at all, so the acceptance requirement ("no per-round engine
    construction when USE_DP=True") holds trivially here.

    Returns (client_idx, params).
    """
    device = client_cfg.get("device", "cpu")

    model = get_model(num_features=client_cfg["sample_features"],
                       num_classes=client_cfg["num_classes"],
                       dp_safe=client_cfg["dp_safe"])
    set_model_parameters(model, global_params)
    model = model.to(device)

    if client_cfg["use_byzantine_attack"] and client_idx in client_cfg["byzantine_clients"]:
        if (client_cfg["use_he"] or client_cfg["use_he_hybrid"] or client_cfg["use_norm_guard"]) \
                and client_cfg["byzantine_head_only"]:
            criterion = client_cfg["criterion"]
            train(model, X_tr, y_tr, criterion,
                  epochs=client_cfg["local_epochs"],
                  lr=client_cfg["learning_rate"],
                  global_params=global_params,
                  mu=client_cfg["prox_mu"],
                  device=device)
            trained_params   = get_model_parameters(model)
            model_state_keys = list(model.state_dict().keys())
            params = classifier_head_flip_attack(
                trained_params, model_state_keys, scale=client_cfg["attack_scale"]
            )
        else:
            attack_type = client_cfg["attack_type"]
            if attack_type == "zero_gradient":
                params = zero_gradient_attack(global_params)
            else:
                criterion = client_cfg["criterion"]
                train(model, X_tr, y_tr, criterion,
                      epochs=client_cfg["local_epochs"],
                      lr=client_cfg["learning_rate"],
                      global_params=global_params,
                      mu=client_cfg["prox_mu"],
                      device=device)
                trained_params = get_model_parameters(model)
                if attack_type == "gaussian":
                    params = gaussian_attack_trained(trained_params,
                                                      std=client_cfg["gaussian_std"])
                elif attack_type in ("minmax", "minsum"):
                    # Issue 5 Task 1: coalition-aware -- crafting needs
                    # every Byzantine coalition member's trained params,
                    # which don't all exist yet at this point in the
                    # (possibly parallel) per-client training wave.
                    # Return the client's own HONEST trained params here
                    # unmodified; main()'s round loop replaces every
                    # Byzantine client's entry with the coalition-crafted
                    # vector in a post-training-wave pass -- see the
                    # "Issue 5 Task 1: Min-Max/Min-Sum coalition crafting"
                    # block right after the training wave completes.
                    params = trained_params
                elif attack_type == "bounded_directional":
                    tau = client_cfg["bounded_tau"]
                    if tau is None:
                        # No usable tau this round (typically round 1,
                        # before any prior-round honest norms exist) --
                        # train honestly rather than crash or guess.
                        # Logged at the call site, not silently.
                        params = trained_params
                    else:
                        model_state_keys = (
                            list(model.state_dict().keys())
                            if client_cfg["bounded_direction"] == "classifier_head_negate"
                            else None
                        )
                        params = bounded_directional_attack_trained(
                            trained_params, global_params, tau=tau,
                            margin=client_cfg["bounded_margin"],
                            direction=client_cfg["bounded_direction"],
                            model_state_keys=model_state_keys,
                        )
                else:
                    params = sign_flip_attack_trained(trained_params,
                                                       scale=client_cfg["attack_scale"])
    else:
        criterion = client_cfg["criterion"]
        train(model, X_tr, y_tr, criterion,
              epochs=client_cfg["local_epochs"],
              lr=client_cfg["learning_rate"],
              global_params=global_params,
              mu=client_cfg["prox_mu"],
              device=device)
        params = get_model_parameters(model)

    return client_idx, params


def _eval_one_client(client_idx, global_params, X_te, y_te, eval_cfg):
    device = eval_cfg.get("device", "cpu")
    model = get_model(num_features=eval_cfg["sample_features"],
                       num_classes=eval_cfg["num_classes"],
                       dp_safe=eval_cfg["dp_safe"])
    set_model_parameters(model, global_params)
    model = model.to(device)
    loss_v, acc_v, f1_per_class = test(model, X_te, y_te,
                                        eval_cfg["num_classes"],
                                        device=device)
    return client_idx, loss_v, acc_v, f1_per_class


def _run_training_wave(executor, clients_data, global_params, round_client_cfg):
    """
    USE_DP=False path only. Returns {client_idx: params}.
    """
    if executor is None:
        results_by_client = {}
        for i, (X_tr, y_tr, X_te, y_te) in enumerate(clients_data):
            client_idx, params = _train_one_client(
                i, X_tr, y_tr, global_params, round_client_cfg
            )
            results_by_client[client_idx] = params
        return results_by_client

    futures = {
        executor.submit(
            _train_one_client, i, X_tr, y_tr, global_params, round_client_cfg
        ): i
        for i, (X_tr, y_tr, X_te, y_te) in enumerate(clients_data)
    }
    results_by_client = {}
    for future in as_completed(futures):
        client_idx, params = future.result()
        results_by_client[client_idx] = params
    return results_by_client


def _run_eval_wave(executor, clients_data, global_params, eval_cfg):
    if executor is None:
        results = {}
        for i, (X_tr, y_tr, X_te, y_te) in enumerate(clients_data):
            client_idx, loss_v, acc_v, f1_per_class = _eval_one_client(
                i, global_params, X_te, y_te, eval_cfg
            )
            results[client_idx] = (loss_v, acc_v, f1_per_class)
        return results

    eval_futures = {
        executor.submit(
            _eval_one_client, i, global_params, X_te, y_te, eval_cfg
        ): i
        for i, (X_tr, y_tr, X_te, y_te) in enumerate(clients_data)
    }
    results = {}
    for future in as_completed(eval_futures):
        client_idx, loss_v, acc_v, f1_per_class = future.result()
        results[client_idx] = (loss_v, acc_v, f1_per_class)
    return results


# ---------------------------------------------------------------------------
# AGGREGATION HELPERS
# ---------------------------------------------------------------------------

def fedprox_aggregate(all_params: list, weights: list) -> list:
    total = sum(weights)
    result = []
    for layer_idx in range(len(all_params[0])):
        layer_avg = sum(
            p[layer_idx] * (w / total)
            for p, w in zip(all_params, weights)
        )
        result.append(layer_avg)
    return result


# ---------------------------------------------------------------------------
# CHECKPOINT HELPERS
# ---------------------------------------------------------------------------

def save_checkpoint(global_params: list, round_num: int):
    np.savez(CHECKPOINT_PARAMS, *global_params)
    with open(CHECKPOINT_PROGRESS, "w") as f:
        json.dump({"last_completed_round": round_num}, f)


def load_checkpoint():
    if not (os.path.exists(CHECKPOINT_PARAMS) and
            os.path.exists(CHECKPOINT_PROGRESS)):
        return None, 0
    data = np.load(CHECKPOINT_PARAMS)
    params = [data[f"arr_{i}"] for i in range(len(data.files))]
    with open(CHECKPOINT_PROGRESS) as f:
        progress = json.load(f)
    return params, progress["last_completed_round"]


def _checkpoint_round(global_params, round_num):
    """
    PRV1: no DP accountant state is ever written -- USE_DP=True runs
    cannot resume from a checkpoint (confirmed acceptable; see module
    docstring). This function only ever checkpoints model parameters.
    """
    save_checkpoint(global_params, round_num)


def save_best_checkpoint(global_params: list, round_num: int, f1_macro: float):
    np.savez(CHECKPOINT_BEST_PARAMS, *global_params)
    with open(CHECKPOINT_BEST_PROGRESS, "w") as f:
        json.dump({"best_round": round_num, "best_f1_macro": float(f1_macro)}, f)


# ---------------------------------------------------------------------------
# CSV LOGGING
# ---------------------------------------------------------------------------

_CSV_HEADER = (
    ["round", "client", "loss", "accuracy"]
    + ATTACK_NAMES
    + ["norm_guard_rejected", "krum_selected", "krum_detected_byzantine",
       "dp_epsilon_spent_cumulative", "round_time_s",
       "dp_full_run_target_epsilon", "dp_per_round_target_epsilon_deprecated",
       "dp_noise_multiplier",
       "krum_scores_byzantine_mean", "krum_scores_honest_mean",
       "krum_score_ratio", "nan_this_round"]
)


def init_log_csv(resume: bool = False):
    if not resume and os.path.exists(LOG_CSV):
        os.remove(LOG_CSV)
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline="") as f:
            csv.writer(f).writerow(_CSV_HEADER)


def append_log_row(round_num, client_label, loss, accuracy,
                    per_class_f1, norm_guard_rejected, krum_selected,
                    krum_detected, dp_eps, round_time, is_mean: bool = False,
                    dp_full_run_target_epsilon=None,
                    dp_per_round_target_epsilon_deprecated=None,
                    dp_noise_multiplier=None,
                    krum_scores_byzantine_mean=None, krum_scores_honest_mean=None,
                    krum_score_ratio=None, nan_this_round=None):
    """
    NOTE on dp_per_round_target_epsilon_deprecated: this codebase no
    longer computes this value at all (the single-process
    dp_persistent_client_state module has no per-round-reset accountant
    to derive it from, and never did -- see PRV1's module docstring).
    The CSV column is kept, always written as "N/A", purely so old and
    new results_*.csv files have the same column layout and can be
    concatenated/diffed without a schema break. `_fmt()` below handles
    a None value safely; the caller (main()) never has to guard this
    itself.
    """
    if is_mean:
        krum_selected_field = krum_selected
        krum_detected_field = (
            f"{krum_detected:.4f}" if krum_detected is not None else "N/A"
        )
    else:
        krum_selected_field = 1 if krum_selected else 0
        krum_detected_field = 1 if krum_detected else 0

    def _fmt(v, spec=".6f"):
        return format(v, spec) if v is not None else "N/A"

    row = (
        [round_num, client_label,
         f"{loss:.6f}", f"{accuracy:.6f}"]
        + [f"{v:.6f}" for v in per_class_f1]
        + [int(norm_guard_rejected),
           krum_selected_field,
           krum_detected_field,
           f"{dp_eps:.4f}" if dp_eps is not None else "N/A",
           f"{round_time:.2f}",
           _fmt(dp_full_run_target_epsilon, ".2f"),
           _fmt(dp_per_round_target_epsilon_deprecated, ".4f"),
           _fmt(dp_noise_multiplier, ".4f"),
           _fmt(krum_scores_byzantine_mean, ".4e"),
           _fmt(krum_scores_honest_mean, ".4e"),
           _fmt(krum_score_ratio, ".4f"),
           ("N/A" if nan_this_round is None else int(bool(nan_this_round)))]
    )
    with open(LOG_CSV, "a", newline="") as f:
        csv.writer(f).writerow(row)


# ---------------------------------------------------------------------------
# MAIN TRAINING LOOP
# ---------------------------------------------------------------------------

def main():
    if (USE_HE or USE_HE_KRUM_HYBRID or USE_NORM_GUARD) and BYZANTINE_HEAD_ONLY:
        _attack_function_label = "classifier_head_flip_attack"
    else:
        _attack_function_label = {
            "sign_flip":           "sign_flip_attack_trained",
            "gaussian":            "gaussian_attack_trained",
            "zero_gradient":       "zero_gradient_attack",
            "minmax":              "minmax_attack_trained",
            "minsum":              "minsum_attack_trained",
            "bounded_directional": "bounded_directional_attack_trained",
        }[ATTACK_TYPE]

    print(f"\n{'='*65}")
    print(f"  FL-IDS Unified Loop -- MODEL: {MODEL_TYPE.upper()}")
    print(f"  Ablation mode: {ABLATION_MODE}")
    if SANITY_CHECK:
        print(f"  *** SANITY_CHECK MODE -- {NUM_ROUNDS} rounds only ***")
    print(f"  Rounds={NUM_ROUNDS}  Clients={NUM_CLIENTS}  Epochs={LOCAL_EPOCHS}")
    print(f"  Device={_DEVICE}  (CUDA available: {_CUDA_AVAILABLE})")
    print(f"  Byzantine={NUM_BYZANTINE} (clients "
          f"{[c+1 for c in BYZANTINE_CLIENTS]})  "
          f"Attack={'ON' if USE_BYZANTINE_ATTACK else 'OFF'}")
    print(f"  Attack function: {_attack_function_label}")
    print(f"  USE_KRUM={USE_KRUM}  USE_ADAPTIVE_KRUM={USE_ADAPTIVE_KRUM}  "
          f"USE_HE={USE_HE}  USE_HE_KRUM_HYBRID={USE_HE_KRUM_HYBRID}  "
          f"USE_DP={USE_DP}  USE_NORM_GUARD={USE_NORM_GUARD}")
    if USE_DP:
        print(f"  DP: eps(full-run target)={DP_EPSILON}  delta={DP_DELTA}  "
              f"max_grad_norm={DP_MAX_GRAD_NORM}  batch_size={DP_BATCH_SIZE}  "
              f"accountant=rdp")
        print(f"  [PRV1] DP-active clients each get a persistent, "
              f"in-process (model, optimizer, engine) state "
              f"(dp_persistent_client_state.py) -- one PrivacyEngine per "
              f"client, created once before Round 1, reused unmodified "
              f"for all {NUM_ROUNDS} rounds. NOT using the "
              f"ProcessPoolExecutor pool for training this run.")
    print(f"{'='*65}\n")

    torch.set_num_threads(_CPU_COUNT)

    print("Loading data partitions...")
    clients_data = []
    for i in range(NUM_CLIENTS):
        print(f"  Partition {i+1}/{NUM_CLIENTS}...", end="\r")
        clients_data.append(load_partition(i, NUM_CLIENTS, seed=_args.seed,
                                            alpha=ALPHA_DIRICHLET))
    sample_features = clients_data[0][0].shape[1]
    print(f"\nFeature count (measured, not assumed): {sample_features}")
    print(f"All {NUM_CLIENTS} clients loaded.\n")

    # Issue 4 Task 1: client_class_entropy is a property of each
    # client's TRAIN partition, which is fixed for the whole run
    # (Dirichlet partitioning happens once, above -- not per round) --
    # computed once here, reused every round in the logger below rather
    # than recomputed NUM_ROUNDS times for an unchanging quantity.
    CLIENT_CLASS_ENTROPY = {}
    for i, (X_tr, y_tr, X_te, y_te) in enumerate(clients_data):
        counts = np.bincount(y_tr.astype(int), minlength=NUM_CLASSES)
        CLIENT_CLASS_ENTROPY[i] = _shannon_entropy_from_counts(counts)

    def print_data_split():
        print("  -- Data split (train partition, per client) --")
        name_w = 8
        header = "    Client  Total   " + "  ".join(
            f"{n[:name_w]:>{name_w}}" for n in ATTACK_NAMES
        )
        print(header)
        for i, (X_tr, y_tr, X_te, y_te) in enumerate(clients_data):
            counts = np.bincount(y_tr.astype(int), minlength=NUM_CLASSES)
            tag = " [BYZANTINE]" if i in BYZANTINE_CLIENTS else ""
            counts_str = "  ".join(f"{c:>{name_w}}" for c in counts)
            print(f"    {i+1:>2}      {len(y_tr):>5}  {counts_str}{tag}")
        print()

    print("Building criterion once (class weights, FocalLoss)...")
    precomputed_criterion = build_criterion(seed=_args.seed).to(_DEVICE)
    print("Criterion built.\n")

    client_cfg = {
        "sample_features":      sample_features,
        "num_classes":          NUM_CLASSES,
        "dp_safe":              DP_SAFE,
        "use_byzantine_attack": USE_BYZANTINE_ATTACK,
        "criterion":            precomputed_criterion,
        "byzantine_clients":    BYZANTINE_CLIENTS,
        "attack_scale":         ATTACK_SCALE,
        "attack_type":          ATTACK_TYPE,
        "gaussian_std":         GAUSSIAN_STD,
        # Issue 5 Task 1: bounded_directional applies inside
        # _train_one_client (needs only a scalar tau, known before the
        # round starts); minmax/minsum are crafted AFTER the training
        # wave (need every coalition member's trained params -- see the
        # post-wave crafting block in main()'s round loop). bounded_tau
        # is mutated in-place each round (estimated from the prior
        # round's verified honest head-norms) since it can only be
        # known causally, round by round.
        "bounded_tau":          BOUNDED_TAU_OVERRIDE,
        "bounded_margin":       BOUNDED_MARGIN,
        "bounded_direction":    BOUNDED_DIRECTION,
        "use_he":               USE_HE,
        "use_he_hybrid":        USE_HE_KRUM_HYBRID,
        "use_norm_guard":              USE_NORM_GUARD,
        "byzantine_head_only":  BYZANTINE_HEAD_ONLY,
        "use_dp":                USE_DP,
        "local_epochs":         LOCAL_EPOCHS,
        "learning_rate":        LEARNING_RATE,
        "prox_mu":              PROX_MU,
        "device":               _DEVICE,
    }

    eval_cfg = {
        "sample_features": sample_features,
        "num_classes":     NUM_CLASSES,
        "dp_safe":         DP_SAFE,
        "device":          _DEVICE,
    }

    he_context = None
    if (USE_HE or USE_HE_KRUM_HYBRID or USE_NORM_GUARD) and _TENSEAL_AVAILABLE:
        he_context = he_local.create_ckks_context(HE_POLY_DEGREE)
        print(f"CKKS context initialised (poly_degree={HE_POLY_DEGREE}).\n")

    MODEL_STATE_KEYS = None
    if USE_HE or USE_HE_KRUM_HYBRID or USE_NORM_GUARD:
        _keys_model = get_model(num_features=sample_features,
                                 num_classes=NUM_CLASSES, dp_safe=DP_SAFE)
        MODEL_STATE_KEYS = list(_keys_model.state_dict().keys())
        del _keys_model

    global_params, start_round = load_checkpoint()

    # ------------------------------------------------------------------
    # PRV1 -- DP-active runs cannot resume from a mid-run checkpoint
    # (confirmed acceptable). A resumed run under the OLD design
    # silently reset per-client accounting to zero steps already-spent,
    # which is exactly the bug this issue exists to fix -- rather than
    # repeat that mistake in a new shape, DP-active resumes are refused
    # outright, loudly, here.
    # ------------------------------------------------------------------
    if USE_DP and global_params is not None and start_round > 0:
        print(f"  [PRV1] REFUSING TO RESUME: found a checkpoint at round "
              f"{start_round} for a USE_DP=True run. This implementation's "
              f"cumulative epsilon accounting lives entirely inside live "
              f"PrivacyEngine objects in this process's memory -- that "
              f"state cannot be serialized and faithfully reconstructed "
              f"after a crash. Resuming would either silently under-report "
              f"epsilon (if accounting restarted from zero) or require "
              f"re-deriving state this code has no way to verify. Delete "
              f"{CHECKPOINT_PARAMS} and {CHECKPOINT_PROGRESS} and restart "
              f"this run from Round 1.")
        raise RuntimeError(
            "USE_DP=True runs cannot resume from a checkpoint in this "
            "implementation -- delete the checkpoint files and restart "
            "from Round 1 (see printed message above)."
        )

    if global_params is None:
        global_params = get_model_parameters(
            get_model(num_features=sample_features,
                      num_classes=NUM_CLASSES,
                      dp_safe=DP_SAFE)
        )
        start_round = 0
        print("Starting fresh run.\n")
    else:
        print(f"Resuming from round {start_round}.\n")
        print("  NOTE: if you changed ABLATION_MODE or any other experiment "
              f"flag since the last run, delete {CHECKPOINT_PARAMS} and "
              f"{CHECKPOINT_PROGRESS} before continuing.\n")

    resume = start_round > 0
    init_log_csv(resume=resume)
    best_f1_macro = -1.0
    if resume and os.path.exists(CHECKPOINT_BEST_PROGRESS):
        with open(CHECKPOINT_BEST_PROGRESS) as f:
            best_f1_macro = json.load(f).get("best_f1_macro", -1.0)

    # -----------------------------------------------------------------
    # PRV1 -- build the persistent, in-process, per-client DP state
    # BEFORE Round 1 (regression guard, Task 1.6: if USE_DP is False,
    # nothing in this block runs -- no engine of any kind is created).
    # -----------------------------------------------------------------
    dp_states = {}
    dp_noise_multiplier_by_client = {}
    if USE_DP and _OPACUS_AVAILABLE:
        print(f"  [PRV1] Building {NUM_CLIENTS - NUM_BYZANTINE} persistent "
              f"DP client state(s), in-process -- each DP-active client's "
              f"PrivacyEngine created exactly once, calibrated against the "
              f"FULL {TOTAL_EPOCHS_PER_CLIENT}-optimizer-epoch horizon "
              f"({NUM_ROUNDS} rounds x {LOCAL_EPOCHS} local epochs), "
              f"target full_run_target_epsilon={DP_EPSILON}...")
        dp_states = build_dp_client_states(
            clients_data, BYZANTINE_CLIENTS, USE_BYZANTINE_ATTACK,
            sample_features, NUM_CLASSES, DP_SAFE, LEARNING_RATE,
            DP_BATCH_SIZE, DP_EPSILON, DP_DELTA, DP_MAX_GRAD_NORM,
            TOTAL_EPOCHS_PER_CLIENT, _DEVICE,
        )
        for i, state in sorted(dp_states.items()):
            dp_noise_multiplier_by_client[i] = state["sigma"]
            print(f"    Client {i+1:2d}: sample_rate={state['sample_rate']:.5f}  "
                  f"sigma={state['sigma']:.4f}")
        print()

    meta_path = f"experiment_config_{_TAG}.json"
    with open(meta_path, "w") as f:
        json.dump({
            "ablation_mode": ABLATION_MODE,
            # Issue 5 Task 4: single canonical aggregator slug -- see
            # _resolve_aggregator_canonical()'s docstring above for why
            # this was missing before and what precedence it mirrors.
            # This is the field scripts/analysis_paper.py groups on.
            "aggregator": AGGREGATOR_CANONICAL,
            "aggregator_cli_arg": AGGREGATOR,
            "model_type": MODEL_TYPE,
            # Issue 5 Task 3/6 (E7): which dataset this run actually
            # used -- analysis_paper.py's Table 6 builder should filter
            # on this rather than assume every run is Edge-IIoTset.
            "dataset": _args.dataset,
            "ciciot_subset_fraction": _args.ciciot_subset_fraction if _args.dataset == "ciciot2023" else None,
            "ciciot_five_feature_slice": _args.ciciot_five_feature_slice if _args.dataset == "ciciot2023" else None,
            "sanity_check": SANITY_CHECK,
            "num_rounds": NUM_ROUNDS,
            "num_clients": NUM_CLIENTS,
            "num_features_measured": sample_features,
            "alpha_dirichlet": ALPHA_DIRICHLET,
            "alpha_dirichlet_cli_override": _args.alpha,
            "local_epochs": LOCAL_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "prox_mu": PROX_MU,
            "byzantine_attack": USE_BYZANTINE_ATTACK,
            "num_byzantine": NUM_BYZANTINE,
            "byzantine_clients": BYZANTINE_CLIENTS,
            "attack_scale": ATTACK_SCALE,
            "attack_type": ATTACK_TYPE,
            "gaussian_std": GAUSSIAN_STD if ATTACK_TYPE == "gaussian" else None,
            # Issue 5 Task 1 -- stealthy attack hyperparameters, logged
            # regardless of whether the currently-selected ATTACK_TYPE
            # uses them, so a run's manifest fully documents what CLI
            # knobs were available/overridable for reproduction.
            "minmax_dev_type": MINMAX_DEV_TYPE if ATTACK_TYPE in ("minmax", "minsum") else None,
            "minmax_search_iters": MINMAX_SEARCH_ITERS if ATTACK_TYPE in ("minmax", "minsum") else None,
            "minmax_gamma_init": MINMAX_GAMMA_INIT if ATTACK_TYPE in ("minmax", "minsum") else None,
            "bounded_tau_override": BOUNDED_TAU_OVERRIDE if ATTACK_TYPE == "bounded_directional" else None,
            "bounded_margin": BOUNDED_MARGIN if ATTACK_TYPE == "bounded_directional" else None,
            "bounded_direction": BOUNDED_DIRECTION if ATTACK_TYPE == "bounded_directional" else None,
            "attack_function": _attack_function_label,
            "use_krum": USE_KRUM,
            "krum_m": KRUM_M,
            "krum_discards": NUM_CLIENTS - KRUM_M,
            "use_adaptive_krum": USE_ADAPTIVE_KRUM,
            "adaptive_krum_k": ADAPTIVE_KRUM_K,
            "adaptive_krum_hybrid_assumed_f": ADAPTIVE_KRUM_HYBRID_ASSUMED_F,
            "byzantine_clients_cli_override": _args.byzantine,
            "adaptive_krum_method": ADAPTIVE_KRUM_METHOD,
            "adaptive_krum_min_keep_fraction": ADAPTIVE_KRUM_MIN_KEEP_FRACTION,
            "use_calibrated_krum": USE_CALIBRATED_KRUM,
            # Issue 5 Task 3 (E6 ablation) -- logged unconditionally
            # (not just when USE_CALIBRATED_KRUM) so a manifest always
            # documents what these flags WERE for this run, matching
            # this file's existing convention for attack hyperparams.
            "use_dp_calibration": USE_DP_CALIBRATION if USE_CALIBRATED_KRUM else None,
            "use_hetero_calibration": USE_HETERO_CALIBRATION if USE_CALIBRATED_KRUM else None,
            "use_he": USE_HE,
            "use_he_krum_hybrid": USE_HE_KRUM_HYBRID,
            "use_norm_guard": USE_NORM_GUARD,
            "use_head_norm_guard": USE_HEAD_NORM_GUARD,
            "head_norm_guard_k": HEAD_NORM_GUARD_K if (USE_HEAD_NORM_GUARD or USE_NORM_GUARD) else None,
            "head_norm_guard_min_keep_fraction": HEAD_NORM_GUARD_MIN_KEEP_FRACTION if (USE_HEAD_NORM_GUARD or USE_NORM_GUARD) else None,
            "he_poly_degree": HE_POLY_DEGREE if (USE_HE or USE_HE_KRUM_HYBRID or USE_NORM_GUARD) else None,
            "use_dp": USE_DP,
            "dp_full_run_target_epsilon": DP_EPSILON,
            "dp_total_epochs_per_client": TOTAL_EPOCHS_PER_CLIENT,
            "dp_delta": DP_DELTA,
            "dp_max_grad_norm": DP_MAX_GRAD_NORM,
            "dp_batch_size": DP_BATCH_SIZE,
            "dp_accountant": "rdp",
            "dp_accounting_note": (
                "Each DP-active client's noise_multiplier is calibrated "
                "ONCE, in dp_persistent_client_state.py, before Round 1, "
                "against dp_total_epochs_per_client (= num_rounds * "
                "local_epochs) and that client's REAL Dirichlet-partition "
                "size. That SAME PrivacyEngine object is reused, "
                "unmodified, for the client's entire lifespan; its own "
                "get_epsilon() IS the cumulative, composed, paper-citable "
                "epsilon reported every round -- there is no separate "
                "main-process accountant. USE_DP=True runs cannot resume "
                "from a checkpoint (see main.py's module docstring)."
            ) if USE_DP else None,
            "byzantine_head_only": BYZANTINE_HEAD_ONLY,
            "byzantine_full_model_cli_override": _args.byzantine_full_model,
            "dp_safe": DP_SAFE,
            "force_dp_safe_arch_cli_flag": _args.force_dp_safe_arch,
            "prox_mu_cli_override": _args.prox_mu,
            "hetero_fit_coeffs_json_path": _args.hetero_fit_coeffs_json,
            "hetero_fit_coeffs_active": HETERO_FIT_COEFFS is not None,
            "device": str(_DEVICE),
            "cuda_available": _CUDA_AVAILABLE,
            "client_pool_workers": CLIENT_POOL_WORKERS,
            "threads_per_worker": _THREADS_PER_WORKER,
            "framework": "custom Python simulation (direct, parallel client training)",
        }, f, indent=2)

    # ========================================================================
    # ROUND LOOP
    # PRV1: when USE_DP=True, non-Byzantine client TRAINING goes through
    # dp_states (dp_persistent_client_state.run_dp_client_round()) instead
    # of _train_one_client() -- Byzantine clients still use the latter,
    # unchanged. Both run sequentially, in this process; the
    # ProcessPoolExecutor/executor below is used ONLY for USE_DP=False runs.
    # ========================================================================
    # Issue 4 Task 2: round-to-round calibration state for Calibrated
    # Krum. None on round 1 triggers calibrated_adaptive_multi_krum's
    # own internal bootstrap (see that function's docstring); every
    # subsequent round threads through whatever it returned as
    # "new_baseline_honest_std" -- this is the ONLY piece of Krum-
    # family state that persists across rounds in this file (mirrors
    # how dp_states persists PrivacyEngine objects across rounds for
    # the same reason: some things are genuinely round-to-round state,
    # not per-round-recomputed).
    _calibrated_krum_baseline_std = None

    # Issue 5 Task 1: causal, round-to-round estimate of the norm-guard
    # threshold for --attack-type bounded_directional, updated at the
    # end of each round's norm-guard verification block below (honest
    # clients' verified head-norms only). None until a round with
    # USE_NORM_GUARD/USE_HEAD_NORM_GUARD actually populates it -- see
    # client_cfg["bounded_tau"] update at the top of the round loop.
    _prior_round_honest_head_norms = None

    pool_cm = (
        contextlib.nullcontext()
        if _CUDA_AVAILABLE
        else ProcessPoolExecutor(max_workers=CLIENT_POOL_WORKERS,
                                  initializer=_pool_worker_init)
    )

    with pool_cm as executor:
        for round_num in range(start_round + 1, NUM_ROUNDS + 1):
            round_start = time.time()
            print(f"[ROUND {round_num}/{NUM_ROUNDS}]")
            print_data_split()

            accepted_params         = []
            accepted_weights        = []
            accepted_client_indices = []

            norm_guard_rejected_this_round      = []
            dp_cumulative_eps_this_round = []
            dp_noise_mult_this_round     = []

            # Issue 5 Task 1: resolve this round's bounded_directional
            # tau -- explicit CLI override wins outright; otherwise
            # estimate from the prior round's verified honest head-norms
            # (None on round 1 / any run where the guard hasn't produced
            # data yet -- client_cfg["bounded_tau"] stays None and
            # _train_one_client trains that client honestly that round,
            # logged there, not silently skipped here).
            if ATTACK_TYPE == "bounded_directional" and BOUNDED_TAU_OVERRIDE is None:
                if _prior_round_honest_head_norms:
                    client_cfg["bounded_tau"] = estimate_norm_guard_tau(
                        _prior_round_honest_head_norms, k=HEAD_NORM_GUARD_K
                    )
                else:
                    client_cfg["bounded_tau"] = None

            _train_wave_start = time.time()

            if USE_DP and _OPACUS_AVAILABLE:
                # PRV1: for DP-active runs, each non-Byzantine client
                # trains through its own persistent (model, optimizer,
                # engine) in dp_states -- built once, before Round 1,
                # reused unmodified every round (run_dp_client_round()
                # reads cumulative_epsilon directly off that SAME
                # never-reset engine; no central accountant bookkeeping
                # needed). Byzantine clients (excluded from dp_states
                # entirely -- they have no real DP accounting to do)
                # still go through the unchanged _train_one_client()
                # path. Both run sequentially, in this one process --
                # matches how this codebase already runs everything on
                # GPU (no ProcessPoolExecutor here regardless of
                # USE_DP, per the "fork+CUDA hang" fix elsewhere in
                # this file).
                trained_params_by_client = {}
                for i, (X_tr, y_tr, X_te, y_te) in enumerate(clients_data):
                    if i in dp_states:
                        params, cum_eps = run_dp_client_round(
                            i, dp_states, global_params,
                            client_cfg["criterion"], LOCAL_EPOCHS,
                            LEARNING_RATE, PROX_MU, DP_DELTA,
                        )
                        trained_params_by_client[i] = params
                        dp_cumulative_eps_this_round.append(cum_eps)
                        dp_noise_mult_this_round.append(dp_noise_multiplier_by_client[i])
                    else:
                        _, params = _train_one_client(
                            i, X_tr, y_tr, global_params, client_cfg
                        )
                        trained_params_by_client[i] = params
            else:
                trained_params_by_client = _run_training_wave(
                    executor, clients_data, global_params, client_cfg
                )

            _train_wave_elapsed = time.time() - _train_wave_start
            print(f"  [Timing] Training wave (all {NUM_CLIENTS} clients): "
                  f"{_train_wave_elapsed:.1f}s")

            # ----------------------------------------------------------
            # Issue 5 Task 1: Min-Max/Min-Sum coalition crafting.
            # _train_one_client() returned each Byzantine coalition
            # member's OWN honestly-trained params unmodified (see that
            # function's minmax/minsum branch) because crafting these
            # attacks needs every coalition member's trained update at
            # once -- only available now that the full training wave has
            # completed (true whether this round went through the DP
            # branch or the plain _run_training_wave() branch above --
            # both populate the same trained_params_by_client dict).
            # Every Byzantine client is overwritten with the SAME
            # crafted vector (coalition-optimal broadcast, per Fang et
            # al. -- see defences/byzantine.py's module docstring).
            # ----------------------------------------------------------
            if USE_BYZANTINE_ATTACK and ATTACK_TYPE in ("minmax", "minsum"):
                _coalition_honest_params = [
                    trained_params_by_client[i] for i in BYZANTINE_CLIENTS
                    if i in trained_params_by_client
                ]
                if len(_coalition_honest_params) > 0:
                    _craft_fn = (minmax_attack_trained if ATTACK_TYPE == "minmax"
                                 else minsum_attack_trained)
                    _crafted_params, _minmax_diag = _craft_fn(
                        _coalition_honest_params,
                        dev_type=MINMAX_DEV_TYPE,
                        gamma_init=MINMAX_GAMMA_INIT,
                        search_iters=MINMAX_SEARCH_ITERS,
                        return_diagnostics=True,
                    )
                    for i in BYZANTINE_CLIENTS:
                        if i in trained_params_by_client:
                            trained_params_by_client[i] = _crafted_params
                    print(f"  [{ATTACK_TYPE.upper()} attack] "
                          f"gamma={_minmax_diag['gamma']:.4f}  "
                          f"dev_type={_minmax_diag['dev_type_used']}  "
                          f"coalition_size={_minmax_diag['coalition_size']}")

            for i, (X_tr, y_tr, X_te, y_te) in enumerate(clients_data):
                params = trained_params_by_client[i]

                if USE_BYZANTINE_ATTACK and i in BYZANTINE_CLIENTS:
                    if (USE_HE or USE_HE_KRUM_HYBRID or USE_NORM_GUARD) and BYZANTINE_HEAD_ONLY:
                        tag = "head-only"
                    elif ATTACK_TYPE == "minmax":
                        tag = "minmax"
                    elif ATTACK_TYPE == "minsum":
                        tag = "minsum"
                    elif ATTACK_TYPE == "bounded_directional":
                        tag = ("bounded-directional"
                               if client_cfg["bounded_tau"] is not None
                               else "bounded-directional [SKIPPED, no tau yet -- trained honestly]")
                    elif ATTACK_TYPE == "gaussian":
                        tag = "gaussian (trained)"
                    elif ATTACK_TYPE == "zero_gradient":
                        tag = "zero-gradient"
                    else:
                        tag = "sign-flip (trained)"
                    if ATTACK_TYPE in ("minmax", "minsum", "bounded_directional"):
                        # These attacks don't use the fixed ATTACK_SCALE
                        # multiplier -- their magnitude is set by the
                        # gamma search / tau-margin rescale instead.
                        print(f"  Client {i+1:2d}  [BYZANTINE -- {tag}]")
                    else:
                        print(f"  Client {i+1:2d}  [BYZANTINE -- {tag} x{ATTACK_SCALE}]")

                if (USE_HE or USE_HE_KRUM_HYBRID or USE_NORM_GUARD) and _TENSEAL_AVAILABLE and he_context is not None:
                    if (USE_HE_KRUM_HYBRID or USE_NORM_GUARD) and USE_HEAD_NORM_GUARD:
                        client_enc = he_local.encrypt_params_with_norm_guard(
                            params, MODEL_STATE_KEYS, he_context, HE_POLY_DEGREE,
                            global_params
                        )
                    else:
                        client_enc = he_local.encrypt_params(
                            params, MODEL_STATE_KEYS, he_context, HE_POLY_DEGREE
                        )
                    if round_num == start_round + 1 and len(accepted_params) == 0:
                        print(f"  [HE] {client_enc['pct_encrypted']:.1f}% of "
                              f"params encrypted (classifier head), rest "
                              f"plaintext (bulk).")
                    accepted_params.append(client_enc)
                else:
                    accepted_params.append(params)

                accepted_weights.append(len(X_tr))
                accepted_client_indices.append(i)

            krum_selected_ids  = set()
            krum_discarded_ids = set()
            krum_detected_byz  = set()
            krum_score_diag    = None
            krum_scored_client_indices = None

            if len(accepted_params) == 0:
                print("  WARNING: All clients rejected -- skipping round.")
                _checkpoint_round(global_params, round_num)
                continue

            if USE_HE and _TENSEAL_AVAILABLE:
                enc_aggregate = he_local.aggregate_encrypted(
                    accepted_params, accepted_weights, he_context
                )
                global_params = he_local.decrypt_params(enc_aggregate)
                agg_label = "HE (partial, classifier-head-only -- full-client average, no Krum)"

            elif USE_HE_KRUM_HYBRID and _TENSEAL_AVAILABLE:
                if USE_HEAD_NORM_GUARD:
                    verified_positions = []
                    verified_norms = []
                    norm_guard_rejected_ids = set()
                    for pos, c in enumerate(accepted_params):
                        proof = c.get("head_norm_proof")
                        chunks = c["sensitive_enc"]["chunks"]
                        is_valid, reason = (
                            norm_guard.verify_head_norm_proof(proof, chunks)
                            if proof is not None else (False, "PROOF_MISSING")
                        )
                        if is_valid:
                            verified_positions.append(pos)
                            verified_norms.append(proof["norm"])
                        else:
                            norm_guard_rejected_ids.add(accepted_client_indices[pos])
                            print(f"  [Head-norm guard] Client "
                                  f"{accepted_client_indices[pos]+1} REJECTED "
                                  f"at verification: {reason}")

                    guard_kept_rel, guard_dropped_rel, norm_guard_diag = \
                        norm_guard.mad_threshold_head_norms(
                            verified_norms, k=HEAD_NORM_GUARD_K,
                            min_keep_fraction=HEAD_NORM_GUARD_MIN_KEEP_FRACTION
                        )
                    norm_guard_survivor_positions = [
                        verified_positions[i] for i in guard_kept_rel
                    ]
                    for i in guard_dropped_rel:
                        norm_guard_rejected_ids.add(
                            accepted_client_indices[verified_positions[i]]
                        )

                    print(f"  [Head-norm guard] {norm_guard_diag} "
                          f"kept={len(norm_guard_survivor_positions)}/"
                          f"{len(accepted_params)}  "
                          f"rejected_ids={sorted(norm_guard_rejected_ids)}")

                    # Issue 5 Task 1: record this round's verified HONEST
                    # (ground-truth non-Byzantine) head-norms for next
                    # round's estimate_norm_guard_tau() call, if
                    # --attack-type bounded_directional is running
                    # without an explicit --bounded-tau override.
                    # Deliberately excludes Byzantine clients' norms
                    # even when they passed verification (e.g. a
                    # bounded_directional attacker truthfully reporting
                    # norm < tau) -- the estimate should track the
                    # honest population only.
                    _prior_round_honest_head_norms = [
                        verified_norms[k] for k, pos in enumerate(verified_positions)
                        if accepted_client_indices[pos] not in BYZANTINE_CLIENTS
                    ]

                    hybrid_accepted_params = [
                        accepted_params[pos] for pos in norm_guard_survivor_positions
                    ]
                    hybrid_accepted_weights = [
                        accepted_weights[pos] for pos in norm_guard_survivor_positions
                    ]
                    hybrid_accepted_client_indices = [
                        accepted_client_indices[pos] for pos in norm_guard_survivor_positions
                    ]
                else:
                    norm_guard_rejected_ids = set()
                    norm_guard_diag = None
                    hybrid_accepted_params = accepted_params
                    hybrid_accepted_weights = accepted_weights
                    hybrid_accepted_client_indices = accepted_client_indices

                if len(hybrid_accepted_params) - ADAPTIVE_KRUM_HYBRID_ASSUMED_F - 2 < 1:
                    selected_positions = list(range(len(hybrid_accepted_params)))
                    krum_score_diag = None
                    agg_label = ("HE+Krum hybrid (fallback -- too few "
                                 "norm-guard-surviving clients)")
                else:
                    bulk_param_lists = [c["bulk"] for c in hybrid_accepted_params]
                    _, selected_positions, krum_score_diag = adaptive_multi_krum(
                        bulk_param_lists,
                        hybrid_accepted_weights,
                        num_byzantine=ADAPTIVE_KRUM_HYBRID_ASSUMED_F,
                        k=ADAPTIVE_KRUM_K,
                        method=ADAPTIVE_KRUM_METHOD,
                        min_keep_fraction=ADAPTIVE_KRUM_MIN_KEEP_FRACTION,
                        return_diagnostics=True,
                    )
                    krum_scored_client_indices = hybrid_accepted_client_indices
                    agg_label = None

                krum_selected_ids = {
                    hybrid_accepted_client_indices[pos] for pos in selected_positions
                }
                krum_discarded_ids = (
                    {idx for idx in hybrid_accepted_client_indices
                     if idx not in krum_selected_ids}
                    | norm_guard_rejected_ids
                )
                krum_detected_byz = krum_discarded_ids & set(BYZANTINE_CLIENTS)

                selected_enc_clients = [hybrid_accepted_params[pos] for pos in selected_positions]
                selected_weights = [hybrid_accepted_weights[pos] for pos in selected_positions]
                enc_aggregate = he_local.aggregate_encrypted(
                    selected_enc_clients, selected_weights, he_context
                )
                global_params = he_local.decrypt_params(enc_aggregate)

                if agg_label is None:
                    agg_label = (
                        f"HE+Krum hybrid (adaptive, {ADAPTIVE_KRUM_METHOD}, "
                        f"k={ADAPTIVE_KRUM_K})  selected={sorted(krum_selected_ids)}  "
                        f"discarded={sorted(krum_discarded_ids)}  "
                        f"detected_byz={sorted(krum_detected_byz)}"
                    )

            elif USE_NORM_GUARD and _TENSEAL_AVAILABLE:
                verified_positions = []
                verified_norms = []
                norm_guard_rejected_ids = set()
                for pos, c in enumerate(accepted_params):
                    proof = c.get("head_norm_proof")
                    chunks = c["sensitive_enc"]["chunks"]
                    is_valid, reason = (
                        norm_guard.verify_head_norm_proof(proof, chunks)
                        if proof is not None else (False, "PROOF_MISSING")
                    )
                    if is_valid:
                        verified_positions.append(pos)
                        verified_norms.append(proof["norm"])
                    else:
                        norm_guard_rejected_ids.add(accepted_client_indices[pos])
                        print(f"  [Norm guard] Client "
                              f"{accepted_client_indices[pos]+1} REJECTED "
                              f"at verification: {reason}")

                guard_kept_rel, guard_dropped_rel, norm_guard_diag = \
                    norm_guard.mad_threshold_head_norms(
                        verified_norms, k=HEAD_NORM_GUARD_K,
                        min_keep_fraction=HEAD_NORM_GUARD_MIN_KEEP_FRACTION,
                    )
                survivor_positions = [verified_positions[i] for i in guard_kept_rel]
                for i in guard_dropped_rel:
                    norm_guard_rejected_ids.add(accepted_client_indices[verified_positions[i]])

                krum_selected_ids = {accepted_client_indices[pos] for pos in survivor_positions}
                krum_discarded_ids = norm_guard_rejected_ids
                krum_detected_byz = norm_guard_rejected_ids & set(BYZANTINE_CLIENTS)
                norm_guard_rejected_this_round = sorted(norm_guard_rejected_ids)

                print(f"  [Norm guard] {norm_guard_diag} "
                      f"kept={len(survivor_positions)}/{len(accepted_params)}  "
                      f"rejected_ids={sorted(norm_guard_rejected_ids)}  "
                      f"detected_byz={sorted(krum_detected_byz)}")

                # Issue 5 Task 1: same honest-norm tracking as the
                # HE+Krum hybrid head-norm-guard branch above, for
                # estimate_norm_guard_tau() -- see that branch's
                # identical comment for the rationale.
                _prior_round_honest_head_norms = [
                    verified_norms[k] for k, pos in enumerate(verified_positions)
                    if accepted_client_indices[pos] not in BYZANTINE_CLIENTS
                ]

                if len(survivor_positions) == 0:
                    print("  WARNING: Norm guard rejected ALL clients "
                          "this round -- skipping round.")
                    _checkpoint_round(global_params, round_num)
                    continue

                survivor_enc = [accepted_params[pos] for pos in survivor_positions]
                survivor_weights = [accepted_weights[pos] for pos in survivor_positions]
                enc_aggregate = he_local.aggregate_encrypted(
                    survivor_enc, survivor_weights, he_context
                )
                global_params = he_local.decrypt_params(enc_aggregate)
                agg_label = (f"Norm guard only (no Krum)  "
                             f"selected={sorted(krum_selected_ids)}  "
                             f"rejected={sorted(norm_guard_rejected_ids)}  "
                             f"detected_byz={sorted(krum_detected_byz)}")

            elif USE_KRUM:
                effective_m = min(KRUM_M, len(accepted_params) - 1)
                if effective_m < 1:
                    global_params = fedprox_aggregate(accepted_params, accepted_weights)
                    agg_label = "FedProx (Krum fallback)"
                else:
                    global_params, selected_positions = multi_krum(
                        accepted_params, accepted_weights,
                        num_byzantine=NUM_BYZANTINE, m=effective_m,
                    )
                    krum_selected_ids = {accepted_client_indices[pos] for pos in selected_positions}
                    krum_discarded_ids = {idx for idx in accepted_client_indices if idx not in krum_selected_ids}
                    krum_detected_byz = krum_discarded_ids & set(BYZANTINE_CLIENTS)
                    agg_label = (f"Multi-Krum (m={effective_m})  "
                                 f"selected={sorted(krum_selected_ids)}  "
                                 f"discarded={sorted(krum_discarded_ids)}  "
                                 f"detected_byz={sorted(krum_detected_byz)}")

            elif USE_ADAPTIVE_KRUM:
                if len(accepted_params) - NUM_BYZANTINE - 2 < 1:
                    global_params = fedprox_aggregate(accepted_params, accepted_weights)
                    agg_label = "FedProx (Adaptive-Krum fallback -- too few accepted clients)"
                else:
                    global_params, selected_positions, krum_score_diag = adaptive_multi_krum(
                        accepted_params, accepted_weights,
                        num_byzantine=NUM_BYZANTINE, k=ADAPTIVE_KRUM_K,
                        method=ADAPTIVE_KRUM_METHOD,
                        min_keep_fraction=ADAPTIVE_KRUM_MIN_KEEP_FRACTION,
                        return_diagnostics=True,
                    )
                    krum_selected_ids = {accepted_client_indices[pos] for pos in selected_positions}
                    krum_discarded_ids = {idx for idx in accepted_client_indices if idx not in krum_selected_ids}
                    krum_detected_byz = krum_discarded_ids & set(BYZANTINE_CLIENTS)
                    krum_scored_client_indices = accepted_client_indices
                    agg_label = (f"Adaptive Multi-Krum ({ADAPTIVE_KRUM_METHOD}, "
                                 f"k={ADAPTIVE_KRUM_K})  "
                                 f"selected={sorted(krum_selected_ids)}  "
                                 f"discarded={sorted(krum_discarded_ids)}  "
                                 f"detected_byz={sorted(krum_detected_byz)}")

            elif USE_CALIBRATED_KRUM:
                if len(accepted_params) - NUM_BYZANTINE - 2 < 1:
                    global_params = fedprox_aggregate(accepted_params, accepted_weights)
                    agg_label = "FedProx (Calibrated-Krum fallback -- too few accepted clients)"
                else:
                    # Issue 4 Task 1/2: per_client_metadata keyed by the
                    # SAME 0-indexed position used in accepted_params/
                    # accepted_weights (i.e. position pos <-> original
                    # client accepted_client_indices[pos]). noise_
                    # multiplier is real sigma whenever USE_DP is active
                    # for this run -- reachable for real now via
                    # --ablation-mode calibrated_krum_dp_sweep (see that
                    # mode's definition above; USE_DP is False and this
                    # is always None under 'baseline'/'krum_baseline')
                    # -- dp_variance() treats None as "no DP noise for
                    # this client", not missing data.
                    _calibrated_metadata = {
                        pos: {
                            "n_samples": accepted_weights[pos],
                            "class_entropy": CLIENT_CLASS_ENTROPY[orig_id],
                            "noise_multiplier": dp_noise_multiplier_by_client.get(orig_id)
                                                 if USE_DP else None,
                            "epsilon": DP_EPSILON if USE_DP else None,
                        }
                        for pos, orig_id in enumerate(accepted_client_indices)
                    }
                    global_params, selected_positions, krum_score_diag = calibrated_adaptive_multi_krum(
                        accepted_params, accepted_weights, _calibrated_metadata,
                        num_byzantine=NUM_BYZANTINE, k=ADAPTIVE_KRUM_K,
                        method=ADAPTIVE_KRUM_METHOD,
                        min_keep_fraction=ADAPTIVE_KRUM_MIN_KEEP_FRACTION,
                        baseline_honest_std_from_prior_round=_calibrated_krum_baseline_std,
                        dp_max_grad_norm=DP_MAX_GRAD_NORM,
                        alpha_dirichlet=ALPHA_DIRICHLET,
                        # Real fit if --hetero-fit-coeffs-json was passed
                        # (see top of file); otherwise None, and
                        # krum.py's hetero_variance() fires its one-time
                        # "no-op" warning -- honest default, never
                        # fabricated here.
                        hetero_fit_coeffs=HETERO_FIT_COEFFS,
                        # Issue 5 Task 3 (E6 ablation): was hardcoded
                        # True/True -- now CLI-toggleable via
                        # --no-dp-calibration / --no-hetero-calibration
                        # (both default off, i.e. both calibration terms
                        # on, IDENTICAL to prior hardcoded behavior when
                        # neither flag is passed).
                        use_dp_calibration=USE_DP_CALIBRATION,
                        use_hetero_calibration=USE_HETERO_CALIBRATION,
                        return_diagnostics=True,
                    )
                    _calibrated_krum_baseline_std = krum_score_diag["new_baseline_honest_std"]
                    krum_selected_ids = {accepted_client_indices[pos] for pos in selected_positions}
                    krum_discarded_ids = {idx for idx in accepted_client_indices if idx not in krum_selected_ids}
                    krum_detected_byz = krum_discarded_ids & set(BYZANTINE_CLIENTS)
                    krum_scored_client_indices = accepted_client_indices
                    agg_label = (f"Calibrated Adaptive Multi-Krum "
                                 f"({ADAPTIVE_KRUM_METHOD}, k={ADAPTIVE_KRUM_K})  "
                                 f"selected={sorted(krum_selected_ids)}  "
                                 f"discarded={sorted(krum_discarded_ids)}  "
                                 f"detected_byz={sorted(krum_detected_byz)}")

            elif AGGREGATOR == "median":
                global_params = coordinate_median(accepted_params, accepted_weights)
                agg_label = "Coordinate Median"

            elif AGGREGATOR == "trimmed_mean":
                global_params = trimmed_mean(accepted_params, accepted_weights,
                                              beta=TRIMMED_MEAN_BETA)
                agg_label = f"Trimmed Mean (beta={TRIMMED_MEAN_BETA})"

            elif AGGREGATOR == "fedavg":
                global_params = _fedavg_agg(accepted_params, accepted_weights)
                agg_label = "FedAvg (explicit, no defence)"

            else:
                # fedprox_aggregate is left in place, untouched, as the
                # fallback for every non-baseline ABLATION_MODE (pure_dp,
                # krum_dp_sweep, exp2_* when they reach a plain-average
                # path, etc.) -- it is mathematically identical to
                # krum.fedavg, but is not deleted/redirected here since
                # AGGREGATOR is None outside the two baseline modes.
                global_params = fedprox_aggregate(accepted_params, accepted_weights)
                agg_label = "FedProx"

            print(f"  Aggregation: {agg_label}")
            if norm_guard_rejected_this_round:
                print(f"  Norm guard rejected: {norm_guard_rejected_this_round}")

            _krum_active = (USE_KRUM or USE_ADAPTIVE_KRUM or USE_HE_KRUM_HYBRID
                             or USE_NORM_GUARD or USE_CALIBRATED_KRUM)

            # Issue 4 Task 1: map this round's raw per-position Krum
            # scores back to original client IDs, BEFORE the per-client
            # loop below (the existing byz/honest-mean diagnostic block
            # further down does the same mapping again, later, for a
            # different purpose -- this is intentionally a separate,
            # minimal computation so that existing block is left
            # untouched).
            _raw_krum_score_by_client_id = {}
            if krum_score_diag is not None:
                _scored_indices_for_log = (
                    krum_scored_client_indices
                    if krum_scored_client_indices is not None
                    else accepted_client_indices
                )
                for _pos, _orig_id in enumerate(_scored_indices_for_log):
                    _s = krum_score_diag["scores"][_pos]
                    _raw_krum_score_by_client_id[_orig_id] = (
                        float(_s) if np.isfinite(_s) else float("nan")
                    )

            _eval_wave_start = time.time()
            eval_results_by_client = _run_eval_wave(
                executor, clients_data, global_params, eval_cfg
            )
            _eval_wave_elapsed = time.time() - _eval_wave_start
            print(f"  [Timing] Eval wave (all {NUM_CLIENTS} clients): "
                  f"{_eval_wave_elapsed:.1f}s")

            round_losses, round_accs, round_f1s = [], [], []
            for i, (X_tr, y_tr, X_te, y_te) in enumerate(clients_data):
                loss_v, acc_v, f1_per_class = eval_results_by_client[i]
                round_losses.append(loss_v)
                round_accs.append(acc_v)
                round_f1s.append(f1_per_class)

                is_norm_guard_rejected = i in norm_guard_rejected_this_round
                is_krum_selected = (i in krum_selected_ids) if _krum_active else False
                is_krum_detected = (i in krum_detected_byz) if _krum_active else False

                append_log_row(
                    round_num=round_num, client_label=i + 1,
                    loss=loss_v, accuracy=acc_v, per_class_f1=f1_per_class,
                    norm_guard_rejected=is_norm_guard_rejected, krum_selected=is_krum_selected,
                    krum_detected=is_krum_detected, dp_eps=None, round_time=0.0,
                    is_mean=False,
                )

                # Issue 4 Task 1 -- consolidated per_client_krum_scores.csv
                # row. "excluded this round" is defined the same way
                # regardless of WHICH aggregation branch produced it
                # (Krum/Adaptive-Krum/HE-Krum-hybrid/Calibrated-Krum/norm
                # guard): a client counts as excluded if Krum-family
                # logic was active this round AND it was not selected
                # (mirrors is_krum_selected's own definition above, so
                # this can never silently disagree with the printed/
                # logged krum_selected field for the same row).
                _is_byzantine_gt = i in BYZANTINE_CLIENTS
                _was_excluded = _krum_active and not is_krum_selected
                _classification = (
                    _classify_krum_outcome(_is_byzantine_gt, _was_excluded)
                    if _krum_active else "N/A"
                )
                _per_client_krum_log_rows.append([
                    round_num,
                    i + 1,
                    _raw_krum_score_by_client_id.get(i, float("nan")),
                    "byzantine" if _is_byzantine_gt else "honest",
                    _classification,
                    len(X_tr),
                    CLIENT_CLASS_ENTROPY[i],
                    (DP_EPSILON if USE_DP else None),
                    ALPHA_DIRICHLET,
                ])

            mean_loss = float(np.mean(round_losses))
            mean_acc  = float(np.mean(round_accs))
            mean_f1   = np.mean(round_f1s, axis=0)
            round_f1_macro = float(mean_f1.mean())
            if round_f1_macro > best_f1_macro:
                best_f1_macro = round_f1_macro
                save_best_checkpoint(global_params, round_num, best_f1_macro)
                print(f"  [Best checkpoint] New best F1-Macro: {best_f1_macro:.4f} "
                      f"(round {round_num}) -> {CHECKPOINT_BEST_PARAMS}")
            round_time = time.time() - round_start

            print(f"  Loss: {mean_loss:.4f}  Acc: {mean_acc:.4f}  "
                  f"F1-Macro: {mean_f1.mean():.4f}  [{round_time:.1f}s]")
            print("  Per-class F1:")
            for name, f1 in zip(ATTACK_NAMES, mean_f1):
                bar = "#" * int(f1 * 20)
                print(f"    {name:<28} {f1:.4f}  {bar}")
            print()

            krum_detection_rate = (
                len(krum_detected_byz) / NUM_BYZANTINE
                if (_krum_active and NUM_BYZANTINE > 0) else None
            )
            if _krum_active and krum_detection_rate is not None:
                krum_label = ("Krum" if USE_KRUM
                              else "HE+Krum Hybrid (plaintext-slice)" if USE_HE_KRUM_HYBRID
                              else "Norm guard (classifier-head only, no Krum)" if USE_NORM_GUARD
                              else "Calibrated Adaptive Krum" if USE_CALIBRATED_KRUM
                              else "Adaptive Krum")
                print(f"  [{krum_label}] Detection rate this round: "
                      f"{krum_detection_rate:.2%}  "
                      f"({len(krum_detected_byz)}/{NUM_BYZANTINE} Byzantine detected, "
                      f"{len(krum_selected_ids)}/{NUM_CLIENTS - len(norm_guard_rejected_this_round)} "
                      f"legitimate-eligible clients selected)")

            # PRV1: mean, across this round's DP-active clients, of
            # each client's OWN never-reset engine's cumulative-so-far
            # epsilon -- read directly, no re-derivation. There is no
            # deprecated per-round-only value to compute anymore (see
            # append_log_row's docstring) -- always logged as "N/A".
            mean_dp_eps_cumulative = (
                float(np.mean(dp_cumulative_eps_this_round))
                if dp_cumulative_eps_this_round else None
            )
            mean_dp_noise_mult = (
                float(np.mean(dp_noise_mult_this_round))
                if dp_noise_mult_this_round else None
            )

            krum_byz_mean = krum_honest_mean = krum_ratio = None
            nan_this_round = False
            if krum_score_diag is not None:
                nan_this_round = krum_score_diag["num_nan"] > 0
                pos_scores = krum_score_diag["scores"]
                scored_indices = (krum_scored_client_indices
                                  if krum_scored_client_indices is not None
                                  else accepted_client_indices)
                byz_scores, honest_scores = [], []
                for pos, orig_id in enumerate(scored_indices):
                    s = pos_scores[pos]
                    if not np.isfinite(s):
                        continue
                    (byz_scores if orig_id in BYZANTINE_CLIENTS else honest_scores).append(s)
                if byz_scores:
                    krum_byz_mean = float(np.mean(byz_scores))
                if honest_scores:
                    krum_honest_mean = float(np.mean(honest_scores))
                if krum_byz_mean is not None and krum_honest_mean not in (None, 0):
                    krum_ratio = krum_byz_mean / krum_honest_mean
                print(f"  [Krum diagnostics] byz_mean_score={krum_byz_mean!r}  "
                      f"honest_mean_score={krum_honest_mean!r}  "
                      f"ratio={krum_ratio!r}  nan_this_round={nan_this_round}")

            if USE_DP and mean_dp_eps_cumulative is not None:
                print(f"  [PRV1] DP epsilon (cumulative, composed thru "
                      f"round {round_num}): mean={mean_dp_eps_cumulative:.4f}  "
                      f"target(full-run)={DP_EPSILON}")

            append_log_row(
                round_num=round_num, client_label="MEAN",
                loss=mean_loss, accuracy=mean_acc, per_class_f1=mean_f1,
                norm_guard_rejected=len(norm_guard_rejected_this_round),
                krum_selected=len(krum_selected_ids) if _krum_active else None,
                krum_detected=krum_detection_rate,
                dp_eps=mean_dp_eps_cumulative, round_time=round_time,
                is_mean=True,
                dp_full_run_target_epsilon=(DP_EPSILON if USE_DP else None),
                dp_per_round_target_epsilon_deprecated=None,
                dp_noise_multiplier=mean_dp_noise_mult,
                krum_scores_byzantine_mean=krum_byz_mean,
                krum_scores_honest_mean=krum_honest_mean,
                krum_score_ratio=krum_ratio,
                nan_this_round=nan_this_round,
            )

            _checkpoint_round(global_params, round_num)

    # -------------------------------------------------------------
    # PRV1 Task 1.5 -- final_total_epsilon, computed EXACTLY ONCE per
    # client, here, after the last round, directly off each client's
    # own never-reset engine (still alive in dp_states -- no process
    # to stop/join, no handshake needed; get_final_epsilons() just
    # calls .get_epsilon() on each engine object that's been sitting
    # in this same process the whole time).
    # -------------------------------------------------------------
    final_total_epsilon = {}
    if USE_DP and _OPACUS_AVAILABLE and dp_states:
        final_total_epsilon = get_final_epsilons(dp_states, DP_DELTA)
        print("\n" + "-"*65)
        print("  [PRV1] FINAL COMPOSED EPSILON (paper-citable -- read "
              "once, after the last round, directly off each client's "
              "own never-reset PrivacyEngine)")
        for i in sorted(final_total_epsilon.keys()):
            print(f"    Client {i+1:2d}: final_total_epsilon="
                  f"{final_total_epsilon[i]:.4f}  "
                  f"(target={DP_EPSILON}, "
                  f"sigma={dp_noise_multiplier_by_client.get(i, float('nan')):.4f})")

        with open(DP_FINAL_EPSILON_JSON, "w") as f:
            json.dump({
                "model_type": MODEL_TYPE,
                "ablation_mode": ABLATION_MODE,
                "seed": _args.seed,
                "num_rounds": NUM_ROUNDS,
                "dp_full_run_target_epsilon": DP_EPSILON,
                "dp_delta": DP_DELTA,
                "dp_total_epochs_per_client": TOTAL_EPOCHS_PER_CLIENT,
                "final_total_epsilon_by_client": {
                    str(i): eps for i, eps in final_total_epsilon.items()
                },
                "noise_multiplier_by_client": {
                    str(i): sigma for i, sigma in dp_noise_multiplier_by_client.items()
                },
            }, f, indent=2)

        with open(DP_FINAL_EPSILON_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["client", "final_total_epsilon",
                        "dp_full_run_target_epsilon", "noise_multiplier",
                        "dp_delta", "dp_total_epochs_per_client"])
            for i in sorted(final_total_epsilon.keys()):
                w.writerow([i + 1, final_total_epsilon[i], DP_EPSILON,
                            dp_noise_multiplier_by_client[i], DP_DELTA,
                            TOTAL_EPOCHS_PER_CLIENT])

        print(f"  Written to: {DP_FINAL_EPSILON_JSON}, {DP_FINAL_EPSILON_CSV}")
        print("-"*65)

    # -----------------------------------------------------------------
    # DAT1 Task 1.10 -- FINAL TEST-HOLDOUT EVALUATION.
    # Per SPLIT_PROTOCOL.md: TEST is used for EXACTLY one final
    # evaluation, never for any selection decision (e.g. FedProx mu
    # argmax -- that now uses FINAL_VALIDATION_CSV below instead).
    # -----------------------------------------------------------------
    X_test_holdout, y_test_holdout = get_global_test_holdout(
        MODEL_TYPE, seed=_args.seed
    )

    _final_model = get_model(num_features=sample_features,
                              num_classes=NUM_CLASSES, dp_safe=DP_SAFE)
    set_model_parameters(_final_model, global_params)
    _final_model = _final_model.to(_DEVICE)

    final_test_loss, final_test_acc, final_test_f1_per_class, \
        final_test_recall_per_class, final_test_aucpr_per_class = test(
        _final_model, X_test_holdout, y_test_holdout, NUM_CLASSES,
        device=_DEVICE, return_extended=True
    )
    final_test_f1_macro = float(np.mean(final_test_f1_per_class))

    with open(FINAL_TEST_CSV, "w", newline="") as _f:
        _writer = csv.writer(_f)
        _writer.writerow(
            ["model_type", "seed", "ablation_mode", "num_rounds",
             "test_loss", "test_accuracy", "test_f1_macro"]
            + [f"test_f1_{name}" for name in ATTACK_NAMES]
            + [f"test_recall_{name}" for name in ATTACK_NAMES]
            + [f"test_aucpr_{name}" for name in ATTACK_NAMES]
        )
        _writer.writerow(
            [MODEL_TYPE, _args.seed, ABLATION_MODE, NUM_ROUNDS,
             final_test_loss, final_test_acc, final_test_f1_macro]
            + [float(v) for v in final_test_f1_per_class]
            + [float(v) for v in final_test_recall_per_class]
            # AUC-PR entries may be NaN (class absent from this seed's
            # TEST holdout -- see task.py's test() docstring for the
            # return_extended=True path); written as real NaN, not
            # silently zeroed, so aggregate_task4_results.py can
            # distinguish "no positive examples this seed" from "model
            # detected nothing."
            + [float(v) for v in final_test_aucpr_per_class]
        )

    print("\n" + "-"*65)
    print(f"  [FINAL TEST-HOLDOUT] loss={final_test_loss:.4f}  "
          f"acc={final_test_acc:.4f}  F1-Macro={final_test_f1_macro:.4f}")
    print(f"    Written to: {FINAL_TEST_CSV}")
    print("-"*65)

    # -----------------------------------------------------------------
    # Issue 4 Task 4 fix -- FINAL VALIDATION-HOLDOUT EVALUATION.
    # Per SPLIT_PROTOCOL.md's provenance table: any selection decision
    # (FedProx mu argmax, MAD-k, DP clip norm C) must be made against
    # VALIDATION, never TEST. This block exists so that decision has a
    # real number to use -- computed identically to the TEST block
    # above, just against get_global_validation_holdout() instead.
    # Same model, same round's global_params -- this is NOT a second,
    # independently-trained model; it's the SAME final model evaluated
    # against a second, disjoint holdout split.
    # -----------------------------------------------------------------
    X_val_holdout, y_val_holdout = get_global_validation_holdout(
        MODEL_TYPE, seed=_args.seed
    )

    final_val_loss, final_val_acc, final_val_f1_per_class, \
        final_val_recall_per_class, final_val_aucpr_per_class = test(
        _final_model, X_val_holdout, y_val_holdout, NUM_CLASSES,
        device=_DEVICE, return_extended=True
    )
    final_val_f1_macro = float(np.mean(final_val_f1_per_class))

    with open(FINAL_VALIDATION_CSV, "w", newline="") as _f:
        _writer = csv.writer(_f)
        _writer.writerow(
            ["model_type", "seed", "ablation_mode", "num_rounds",
             "val_loss", "val_accuracy", "val_f1_macro"]
            + [f"val_f1_{name}" for name in ATTACK_NAMES]
            + [f"val_recall_{name}" for name in ATTACK_NAMES]
            + [f"val_aucpr_{name}" for name in ATTACK_NAMES]
        )
        _writer.writerow(
            [MODEL_TYPE, _args.seed, ABLATION_MODE, NUM_ROUNDS,
             final_val_loss, final_val_acc, final_val_f1_macro]
            + [float(v) for v in final_val_f1_per_class]
            + [float(v) for v in final_val_recall_per_class]
            + [float(v) for v in final_val_aucpr_per_class]
        )

    print(f"  [FINAL VALIDATION-HOLDOUT] loss={final_val_loss:.4f}  "
          f"acc={final_val_acc:.4f}  F1-Macro={final_val_f1_macro:.4f}")
    print(f"    Written to: {FINAL_VALIDATION_CSV}")
    print("-"*65)

    # -----------------------------------------------------------------
    # Issue 4 Task 1 -- write the consolidated per-client Krum-score CSV
    # exactly once, here, after the round loop (same "write once, after
    # training" convention as FINAL_TEST_CSV/DP_FINAL_EPSILON_* above).
    # -----------------------------------------------------------------
    with open(PER_CLIENT_KRUM_LOG_CSV, "w", newline="") as _f:
        _writer = csv.writer(_f)
        _writer.writerow(PER_CLIENT_KRUM_LOG_HEADER)
        for _row in _per_client_krum_log_rows:
            _writer.writerow(_row)
    print(f"  [Issue 4 Task 1] Per-client Krum-score log written to: "
          f"{PER_CLIENT_KRUM_LOG_CSV} ({len(_per_client_krum_log_rows)} rows)")

    print("\n" + "="*65)
    print(f"  Training complete -- {NUM_ROUNDS} rounds  [{MODEL_TYPE.upper()}]  "
          f"[ABLATION_MODE={ABLATION_MODE}]")
    print(f"  Results logged to:     {LOG_CSV}")
    print(f"  Checkpoint:            {CHECKPOINT_PARAMS} (round {NUM_ROUNDS})")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()