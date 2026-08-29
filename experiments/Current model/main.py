"""
Unified FL-IDS Main Loop
========================
Merges:
  - DP/ZKP/HE main.py  (privacy stack structure)
  - Krum main.py        (working Multi-Krum aggregation)

Aggregation / defence branches, selected by flags (normally set for you by
ABLATION_MODE below -- see that section):
  1. USE_HE=True                   -> CKKS homomorphic aggregation (partial,
                                       classifier-head-only, via he_local.py),
                                       ALL accepted clients averaged, no Krum.
  2. USE_KRUM=True, USE_HE=False   -> Multi-Krum, fixed m (plaintext, Byzantine-robust)
  3. USE_ADAPTIVE_KRUM=True        -> Adaptive Multi-Krum, dynamic MAD/Z-score
                                       threshold instead of a fixed m (plaintext)
  4. USE_HE_KRUM_HYBRID=True       -> Experiment 2: plaintext-slice Adaptive
                                       Krum + encrypted-slice HE, gated by an
                                       optional HMAC head-norm guard pre-filter.
  5. USE_ZKP=True                  -> NEW: defences/zkp.py Part 2's
                                       ciphertext-bound HMAC head-norm guard,
                                       in ISOLATION -- no Krum call at all.
                                       Decouples the guard stage from branch 4's
                                       hybrid pipeline so it can be tested as
                                       its own standalone defence.
  6. All of the above False        -> plain FedAvg / FedProx

Bug fixed (kept from earlier revisions): ZKP-rejected clients are removed
from accepted_params before Krum is called, so accepted_params is a
COMPACTED list. Multi-Krum returns positions within that compacted list.
We track accepted_client_indices in parallel so we can map positions back
to original 0-indexed client IDs before comparing against BYZANTINE_CLIENTS
for detection-rate logging. Adaptive Multi-Krum and the new standalone ZKP
guard branch use the exact same compaction/translation logic.

Run:
    python src/main.py network      # network-layer model
    python src/main.py application  # application-layer model

--------------------------------------------------------------------------
CHANGELOG (this revision)
--------------------------------------------------------------------------
1-16. (see previous revisions -- krum_detected truthy fix, KRUM_M=7,
      measured feature count logging, DP_MAX_GRAD_NORM=1.5, params
      extraction UnboundLocalError fix, parallel client training,
      adaptive Multi-Krum / Condition 5, criterion built once,
      eval parallelized, EMA removed, noise_multiplier caching)

17. GPU DEVICE SUPPORT -- added (see prior revision).

18. SANITY_CHECK toggle added (see prior revision).

19. FIX -- fork+CUDA hang (see prior revision -- sequential in-process
    training/eval on GPU runs instead of a forked ProcessPoolExecutor).

20. FIX -- sign_flip_attack was non-standard versus the literature (see
    prior revision -- sign_flip_attack_trained() now used, trains the
    attacking client first, then negates the result).

21. ADDED -- Gaussian noise attack, trains-first version
    (gaussian_attack_trained), wired in as a second selectable attack
    type alongside sign-flip via the new --attack-type CLI flag.
    --attack-type zero_gradient also added as a third option.

22. FIX -- "attack_function" in experiment_config_*.json is now computed
    dynamically from the actual attack path taken (including head-only
    routing), not hardcoded.

23. FIX -- stale comment on USE_DP removed.

24. FIX -- GAUSSIAN_STD now model-aware (network=50.0, application=30.0)
    instead of a flat default, matching how ATTACK_SCALE already varies
    per model. Still fully overridable via --gaussian-std.

25. NEW -- Three standalone-mechanism ABLATION runs added,
    selected via the new ABLATION_MODE switch below ("pure_dp", "pure_he",
    "pure_zkp"). Specifically:
      a. FIX -- the old standalone USE_HE branch built its own local
         he_aggregate() function, which never decrypted before returning
         and averaged unweighted -- global_params ended up as a list of
         still-encrypted CKKS vectors, which would crash the very next
         set_model_parameters() call in eval. USE_HE now routes through
         the SAME he_local.encrypt_params()/aggregate_encrypted()/
         decrypt_params() pipeline USE_HE_KRUM_HYBRID already validated
         (Experiment 2), just without any Krum call -- all accepted
         clients are encrypted (classifier-head-only, partial CKKS) and
         averaged unconditionally. The old he_aggregate() function and
         the old ts.ckks_vector()-based per-client encryption loop are
         DELETED -- nothing should ever call the old broken path again.
      b. FIX -- the old standalone USE_ZKP branch used a bare, uncalibrated
         zkp_verify_norm(params, max_norm=ZKP_MAX_NORM=10.0) check on the
         FULL trained parameter vector -- not defences/zkp.py's actual
         HMAC commitment/proof machinery at all (that module was never
         imported under USE_ZKP). USE_ZKP is now redefined as: run
         defences/zkp.py Part 2's ciphertext-bound HMAC head-norm guard
         (the same mechanism USE_HE_KRUM_HYBRID's USE_HEAD_NORM_GUARD
         uses) on the classifier-head slice, IN ISOLATION -- no Krum call
         at all in this branch. This directly tests whether the guard
         alone (decoupled from Experiment 2's hybrid pipeline's second,
         redundant Krum stage) detects a classifier-head-only Byzantine
         attacker. zkp_verify_norm() and ZKP_MAX_NORM are DELETED.
      c. USE_HE and USE_ZKP now both trigger the classifier_head_flip_attack
         path when BYZANTINE_HEAD_ONLY=True, exactly like USE_HE_KRUM_HYBRID
         already did -- both new branches operate on the classifier-head
         CKKS slice, so an attacker corrupting that slice (not the whole
         model) is the relevant threat model for both.
      d. USE_HE + USE_ADAPTIVE_KRUM together is UNCHANGED and still
         forbidden by the mutual-exclusion assert below -- that combination
         is intentionally NOT implemented; USE_HE_KRUM_HYBRID is the correct
         (already-implemented, already-validated) way to combine partial HE
         with plaintext-slice Krum. Bypassing the assert would let Krum
         score a mismatched/incomplete parameter structure (encrypt_params()
         dicts, not flat param lists) and produce meaningless results.

26. NEW (this revision) -- Fourth ABLATION_MODE, "krum_dp_sweep", added.
    Reproduces Experiment 1's exact recipe (USE_ADAPTIVE_KRUM=True,
    USE_DP=True, USE_BYZANTINE_ATTACK=True, BYZANTINE_HEAD_ONLY=False,
    k=2.5 default) so that Sweep 2 (Gaussian-noise attack, via
    --attack-type gaussian) can reuse the exact same aggregation/DP
    recipe as the original/corrected sign-flip sweep, differing only in
    which --attack-type is passed on the CLI. None of pure_dp/pure_he/
    pure_zkp activate Adaptive Krum + DP + an active attack together, so
    this was previously only reachable by hand-editing flags outside the
    ABLATION_MODE block entirely (as the block's own docstring describes)
    -- that approach is error-prone across a multi-run sweep since it's
    easy to forget the edit is even flag-driven at all. This mode makes
    that recipe a named, reproducible option instead.
    NOTE: ABLATION_MODE is still not CLI-controllable (open item) --
    switch it back to "pure_dp"/"pure_he"/"pure_zkp" by hand for other
    ablation work after this sweep completes.
--------------------------------------------------------------------------
KNOWN OPEN ITEMS -- NOT YET RESOLVED, FLAGGED FOR NEXT REVISION
--------------------------------------------------------------------------
- PROX_MU is 0.02 here (user-confirmed intended value).
- LR decay disabled (user-confirmed decision) -- get_round_lr() kept but unused.
- task.py has been patched (separately) to register FocalLoss's weight via
  register_buffer() and accept a `device` kwarg on train()/test() -- confirm
  the version on disk matches before running; this file's calls assume it.
- DP_BATCH_SIZE=512 was tuned for CPU. DGX Spark's unified CPU/GPU memory
  means an Opacus per-sample-gradient OOM here can degrade the WHOLE
  system rather than cleanly killing the job -- watch `free -h` on the
  first real (non-sanity-check) DP round; drop DP_BATCH_SIZE if memory
  pressure shows up.
- Gaussian noise draws are UNSEEDED (np.random.normal, no explicit seed).
  This affects exact reproducibility of a given run's specific noise
  realization only -- it is independent of GAUSSIAN_STD's calibrated
  value (network=50.0, application=30.0, set via measure_param_scale.py),
  which is a fixed constant, not a random draw. Two runs at the same
  epsilon/std will differ in exact numbers but not in statistical
  behavior. Still an open, undecided item as of this revision.
- The pure_zkp ablation's detection rate is NOT directly comparable to
  Experiment 2's mitigated hybrid runs (which had Krum as a second,
  redundant layer behind the guard) -- a discrepancy here (guard alone
  missing an attacker Krum would've caught, or vice versa) is exactly the
  kind of result that tells you whether the two stages do independent
  work or the guard alone was already carrying the whole defence. State
  this explicitly in any write-up using this ablation's numbers.
--------------------------------------------------------------------------
"""

import os
import sys
import csv
import json
import time
import warnings
import contextlib
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

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
_parser.add_argument("--epsilon", type=float, default=None,
                     help="Override DP_EPSILON, e.g. --epsilon 9.0")
_parser.add_argument("--tag", type=str, default=None,
                     help="Suffix on every output filename -- "
                          "e.g. --tag dp15 -> results_network_dp15.csv. "
                          "If omitted, defaults to '<model_type>_<ABLATION_MODE>' "
                          "so ablation runs are auto-labelled.")
_parser.add_argument("--byzantine", type=str, default=None,
                     help="Comma-separated client numbers to make Byzantine, "
                          "using the SAME 1-indexed numbering the console "
                          "output prints everywhere ('Client 4', 'Client 10', "
                          "etc.) -- e.g. --byzantine 4,10. Default (no flag) "
                          "is clients 1,2. Overrides both BYZANTINE_CLIENTS "
                          "and NUM_BYZANTINE below -- NUM_BYZANTINE becomes "
                          "len(this list). Numbers must be in [1, NUM_CLIENTS].")
_parser.add_argument("--krum-k", type=float, default=None,
                     help="Override ADAPTIVE_KRUM_K / HEAD_NORM_GUARD_K "
                          "(default 2.5) -- the MAD sensitivity multiplier. "
                          "Larger k -> more permissive (fewer clients dropped).")
_parser.add_argument("--attack-type", type=str, default="sign_flip",
                     choices=["sign_flip", "gaussian", "zero_gradient"],
                     help="Which Byzantine attack the malicious clients use "
                          "(ignored when BYZANTINE_HEAD_ONLY routes to "
                          "classifier_head_flip_attack instead). Default sign_flip.")
_parser.add_argument("--gaussian-std", type=float, default=None,
                     help="Standard deviation for --attack-type gaussian. "
                          "Ignored for other attack types. Default is "
                          "model-aware (network=50.0, application=30.0).")
_args = _parser.parse_args()

MODEL_TYPE = _args.model_type

# -- Sanity-check toggle --------------------------------------------------
SANITY_CHECK = False

# FL hyperparameters
NUM_ROUNDS    = 2 if SANITY_CHECK else 25
NUM_CLIENTS   = 10
LOCAL_EPOCHS  = 10
LEARNING_RATE = 0.001
PROX_MU       = 0.02       # FedProx proximal coefficient (0 = plain FedAvg)

# Byzantine client selection (WHICH clients are attackers, if any attack is
# active this run -- whether the attack is actually active is decided below,
# per ABLATION_MODE, via USE_BYZANTINE_ATTACK).
if _args.byzantine is not None:
    # --byzantine "4,10" -> the clients printed as "Client 4" and "Client 10"
    # everywhere in the console output are attacked. Internally
    # BYZANTINE_CLIENTS/client_idx are 0-indexed (array positions); this is
    # where that translation happens, one place, so the mismatch between
    # "what the logs show" and "what the code stores" can't leak out.
    _byzantine_1indexed = sorted(int(c.strip()) for c in _args.byzantine.split(","))
    BYZANTINE_CLIENTS   = [c - 1 for c in _byzantine_1indexed]
    NUM_BYZANTINE       = len(BYZANTINE_CLIENTS)
    assert len(set(BYZANTINE_CLIENTS)) == NUM_BYZANTINE, \
        f"--byzantine has duplicate client numbers: {_args.byzantine}"
    assert all(1 <= c <= NUM_CLIENTS for c in _byzantine_1indexed), \
        (f"--byzantine client numbers must be in [1, {NUM_CLIENTS}] "
        f"(1-indexed, matching the console output's 'Client N' labels), "
        f"got {_byzantine_1indexed}")
else:
    # Default -- clients 0 and 1 (0-indexed) are malicious, i.e. "Client 1"
    # and "Client 2" in the console output.
    NUM_BYZANTINE     = 2
    BYZANTINE_CLIENTS = list(range(NUM_BYZANTINE))

ATTACK_SCALE  = 5.0 if MODEL_TYPE == "network" else 2.0
ATTACK_TYPE   = _args.attack_type

# Model-aware default, same philosophy as ATTACK_SCALE -- measured via
# measure_param_scale.py against this codebase's actual trained-delta
# magnitude (network: delta_std~4.17, application: delta_std~2.64).
_GAUSSIAN_STD_DEFAULT = 50.0 if MODEL_TYPE == "network" else 30.0
GAUSSIAN_STD  = _args.gaussian_std if _args.gaussian_std is not None else _GAUSSIAN_STD_DEFAULT

# ---------------------------------------------------------------------------
# ABLATION MODE SELECTOR (revision 25, extended revision 26)
# ---------------------------------------------------------------------------
# Picks one of the standalone-mechanism ablation runs, or the DP+Krum sweep
# recipe. Each mode sets EVERY defence/privacy flag explicitly -- nothing is
# left at a stale default from a previous experiment's config.
#
#   "pure_dp"        -> DP-SGD+FedProx only (fedprox_aggregate, no Krum/HE/
#                        ZKP). No Byzantine attack -- clean utility-cost
#                        ablation, matches Experiment 3's gap-table request
#                        for a standalone +DP row with no attack in the
#                        picture.
#   "pure_he"         -> Partial (classifier-head-only) CKKS HE only, via
#                        he_local.py. No Krum, no attack -- clean cost/
#                        behaviour ablation. (Fixes the old broken
#                        he_aggregate() path -- see changelog #25a above.)
#   "pure_zkp"        -> defences/zkp.py Part 2's ciphertext-bound HMAC
#                        head-norm guard, in ISOLATION -- NO Krum call at
#                        all. WITH the classifier-head-only Byzantine
#                        attack active, so this directly tests whether the
#                        guard alone (decoupled from Experiment 2's hybrid
#                        pipeline's second, redundant Krum stage) detects a
#                        head-only attacker.
#   "krum_dp_sweep"    -> Experiment 1's DP+Adaptive-Krum recipe, WITH a
#                        full-model (not head-only) Byzantine attack active.
#                        Attack type is chosen via --attack-type on the CLI
#                        (sign_flip for the original/corrected Experiment 1
#                        sweep, gaussian for Sweep 2). This is the mode to
#                        use for any epsilon-sweep-style run that needs
#                        Adaptive Krum, DP, and an active attack together --
#                        none of the three modes above activate all three
#                        at once.
#
# To run a config outside these four, set ABLATION_MODE to one of these as
# a base and hand-edit the derived flags below -- or just set the flags
# directly and remove/bypass this block.
# ---------------------------------------------------------------------------
ABLATION_MODE = "krum_dp_sweep"   # <-- set for the Gaussian-noise sweep
                                   # (Sweep 2). Switch back to "pure_dp" /
                                   # "pure_he" / "pure_zkp" for other
                                   # single-mechanism ablation runs.

if ABLATION_MODE == "pure_dp":
    USE_KRUM = USE_ADAPTIVE_KRUM = USE_HE = USE_HE_KRUM_HYBRID = USE_ZKP = False
    USE_DP = True
    USE_BYZANTINE_ATTACK = False
    BYZANTINE_HEAD_ONLY = False

elif ABLATION_MODE == "pure_he":
    USE_HE = True
    USE_KRUM = USE_ADAPTIVE_KRUM = USE_HE_KRUM_HYBRID = USE_ZKP = USE_DP = False
    USE_BYZANTINE_ATTACK = False
    BYZANTINE_HEAD_ONLY = False

elif ABLATION_MODE == "pure_zkp":
    USE_ZKP = True
    USE_HE = USE_KRUM = USE_ADAPTIVE_KRUM = USE_HE_KRUM_HYBRID = USE_DP = False
    USE_BYZANTINE_ATTACK = True
    BYZANTINE_HEAD_ONLY = True   # the whole point of this ablation -- attack
                                 # exactly the slice the HMAC guard covers

elif ABLATION_MODE == "krum_dp_sweep":
    # Experiment 1's recipe: Adaptive Krum (k=2.5 default, NOT the
    # HE-hybrid-specific k=3.5) scoring plaintext params directly, plus
    # DP-SGD+FedProx on honest clients, plus a full-model attack on the
    # Byzantine clients (train-then-corrupt, per Week 10's fix -- routed
    # via --attack-type, not the classifier-head-only stealthy path).
    USE_ADAPTIVE_KRUM = True
    USE_KRUM = USE_HE = USE_HE_KRUM_HYBRID = USE_ZKP = False
    USE_DP = True
    USE_BYZANTINE_ATTACK = True
    BYZANTINE_HEAD_ONLY = False   # full-model attack (sign_flip/gaussian/
                                  # zero_gradient), not the classifier-head
                                  # CKKS slice -- there is no CKKS slice in
                                  # this mode at all (USE_HE/HYBRID/ZKP are
                                  # all False here).

else:
    raise ValueError(f"Unknown ABLATION_MODE={ABLATION_MODE!r} -- must be "
                     f"'pure_dp', 'pure_he', 'pure_zkp', or 'krum_dp_sweep'.")

# UNCHANGED, deliberately -- USE_HE + USE_ADAPTIVE_KRUM (or USE_KRUM)
# together is still forbidden. USE_HE_KRUM_HYBRID is the correct,
# already-implemented, already-validated way to combine partial HE with
# plaintext-slice Krum (Experiment 2). USE_ZKP's standalone head-norm guard
# is a separate, non-Krum defence and is intentionally NOT part of this
# mutual-exclusion group. krum_dp_sweep sets USE_ADAPTIVE_KRUM=True with
# USE_HE=False, so it satisfies this assert trivially (sum=1).
assert sum([USE_KRUM, USE_ADAPTIVE_KRUM, USE_HE, USE_HE_KRUM_HYBRID]) <= 1, \
    "USE_KRUM, USE_ADAPTIVE_KRUM, USE_HE, and USE_HE_KRUM_HYBRID are mutually " \
    "exclusive aggregation branches -- pick at most one."

DP_SAFE = USE_DP

# CKKS parameters for the partial (classifier-head-only) HE path, used by
# USE_HE, USE_HE_KRUM_HYBRID, and USE_ZKP (all three need real ciphertext to
# either aggregate over or bind a proof to). Matches the "standard,
# non-RAM-constrained" config he_local.py already defines (n=8192,
# [60,40,40,60], scale=2**40). Unused (harmless) under krum_dp_sweep, since
# none of USE_HE/USE_HE_KRUM_HYBRID/USE_ZKP are active in that mode.
HE_POLY_DEGREE = 8192

# Head-norm guard config -- used by USE_HE_KRUM_HYBRID (as a pre-filter
# before Krum) and by USE_ZKP (as the ENTIRE defence, no Krum). See
# defences/zkp.py Part 2. Unused (harmless) under krum_dp_swee
USE_HEAD_NORM_GUARD = True
HEAD_NORM_GUARD_K = _args.krum_k if _args.krum_k is not None else 2.5
HEAD_NORM_GUARD_MIN_KEEP_FRACTION = 0.5

DP_EPSILON       = _args.epsilon if _args.epsilon is not None else 15.0
DP_DELTA         = 1e-5
DP_MAX_GRAD_NORM = 1.5
DP_BATCH_SIZE    = 16

KRUM_M = NUM_CLIENTS - NUM_BYZANTINE - 1

ADAPTIVE_KRUM_K                 = _args.krum_k if _args.krum_k is not None else 2.5
ADAPTIVE_KRUM_METHOD             = "mad"
ADAPTIVE_KRUM_MIN_KEEP_FRACTION  = 0.5

# Only meaningful for USE_HE_KRUM_HYBRID -- see that branch's comments.
ADAPTIVE_KRUM_HYBRID_ASSUMED_F  = min(1, NUM_BYZANTINE)

# ---------------------------------------------------------------------------
# Device / parallelization settings
# ---------------------------------------------------------------------------
_CPU_COUNT      = os.cpu_count() or 4
_CUDA_AVAILABLE = torch.cuda.is_available()
_DEVICE         = torch.device("cuda" if _CUDA_AVAILABLE else "cpu")

# GPU note: when CUDA is available, no ProcessPoolExecutor is created at
# all -- client training/eval runs sequentially in-process (see
# _run_training_wave/_run_eval_wave). CLIENT_POOL_WORKERS is kept as a
# reported/logged value (still 1 on GPU) even though no pool actually
# exists in that case.
CLIENT_POOL_WORKERS = 1 if _CUDA_AVAILABLE else min(4, NUM_CLIENTS)
_THREADS_PER_WORKER = max(1, _CPU_COUNT // CLIENT_POOL_WORKERS)

# ---------------------------------------------------------------------------
# Output paths -- one set per model type/ablation so runs don't collide
# ---------------------------------------------------------------------------
_TAG               = (f"{MODEL_TYPE}_{ABLATION_MODE}" if _args.tag is None
                      else f"{MODEL_TYPE}_{_args.tag}")
CHECKPOINT_PARAMS       = f"checkpoint_{_TAG}.npz"
CHECKPOINT_PROGRESS     = f"checkpoint_{_TAG}_progress.json"
CHECKPOINT_BEST_PARAMS   = f"checkpoint_{_TAG}_best.npz"
CHECKPOINT_BEST_PROGRESS = f"checkpoint_{_TAG}_best.json"
LOG_CSV                 = f"results_{_TAG}.csv"

# ---------------------------------------------------------------------------
# Imports (deferred so errors are clear)
# ---------------------------------------------------------------------------
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

from defences.byzantine import (sign_flip_attack, sign_flip_attack_trained,
                                classifier_head_flip_attack, gaussian_attack,
                                gaussian_attack_trained, zero_gradient_attack)

if USE_KRUM:
    from defences.krum import multi_krum

if USE_ADAPTIVE_KRUM or USE_HE_KRUM_HYBRID:
    # Experiment 2 uses adaptive Krum on the plaintext slice, for direct
    # comparability with Experiment 1's already-completed adaptive-Krum
    # results. krum_dp_sweep also lands here (USE_ADAPTIVE_KRUM=True).
    from defences.krum import adaptive_multi_krum

# he_local is needed by all three encryption-touching branches: USE_HE
# (revision 25 fix -- routes through the SAME correct pipeline
# USE_HE_KRUM_HYBRID uses), USE_HE_KRUM_HYBRID (Experiment 2), and USE_ZKP
# (revision 25 -- needs real ciphertext to bind its proof to). NOT imported
# under krum_dp_sweep (all three flags False there) -- this run needs no
# TenSEAL/CKKS dependency at all.
if USE_HE or USE_HE_KRUM_HYBRID or USE_ZKP:
    from defences import he_local

# zkp is needed whenever the head-norm guard actually runs: standalone
# (USE_ZKP) or as USE_HE_KRUM_HYBRID's pre-filter. NOT imported under
# krum_dp_sweep.
if USE_ZKP or (USE_HE_KRUM_HYBRID and USE_HEAD_NORM_GUARD):
    from defences import zkp

if USE_DP:
    try:
        from opacus import PrivacyEngine
        _OPACUS_AVAILABLE = True
    except ImportError:
        warnings.warn("Opacus not installed -- USE_DP will be skipped. "
                      "Install with: pip install opacus")
        _OPACUS_AVAILABLE = False
else:
    _OPACUS_AVAILABLE = False

if USE_HE or USE_HE_KRUM_HYBRID or USE_ZKP:
    try:
        import tenseal as ts
        _TENSEAL_AVAILABLE = True
    except ImportError:
        raise ImportError("TenSEAL required for USE_HE/USE_HE_KRUM_HYBRID/"
                          "USE_ZKP=True. Install with Python 3.11: "
                          "pip install tenseal")
# NOTE: _TENSEAL_AVAILABLE is intentionally left UNDEFINED when none of
# USE_HE/USE_HE_KRUM_HYBRID/USE_ZKP are True (e.g. krum_dp_sweep). Every
# later reference to it is of the form
# "(USE_HE or USE_HE_KRUM_HYBRID or USE_ZKP) and _TENSEAL_AVAILABLE", and
# Python's `and` short-circuits on a False left operand -- _TENSEAL_AVAILABLE
# is never evaluated in that case, so this is safe, not an oversight.

_noise_multiplier_cache = {}


# ---------------------------------------------------------------------------
# ROUND-LEVEL LEARNING RATE DECAY
# ---------------------------------------------------------------------------

def get_round_lr(base_lr, round_num, num_rounds, min_lr_frac=0.15):
    progress = round_num / num_rounds
    decay = 0.5 * (1 + np.cos(np.pi * progress))
    return base_lr * (min_lr_frac + (1 - min_lr_frac) * decay)


def _apply_dp_safe_prox_step(real_model, global_dict, mu, lr):
    """
    Applies FedProx's proximal pull as a SEPARATE, non-privatized
    parameter update -- not via loss.backward(). Opacus's DPOptimizer
    builds its update entirely from .grad_sample, which the prox term
    never populates -- so adding it to the loss under DP-SGD silently
    does nothing. This applies mu*(w - w_global) as a deterministic SGD
    step, decoupled from the clipped/noised data-gradient step. Safe:
    the prox term depends only on current params + last round's public
    global model, never on client data, so it costs zero privacy budget
    applied this way.
    """
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
# PARALLEL / SEQUENTIAL CLIENT TRAINING
# ---------------------------------------------------------------------------

def _pool_worker_init():
    """
    Runs once per worker process at pool startup -- CPU-only path.
    Never invoked on GPU runs since no pool exists there.
    """
    import torch as _torch
    _torch.set_num_threads(_THREADS_PER_WORKER)


def _train_one_client(client_idx, X_tr, y_tr, global_params, client_cfg):
    """
    Called either via ProcessPoolExecutor (CPU) or directly in-process (GPU).
    Signature/behavior identical either way.

    Returns (client_idx, params, dp_eps_spent, dp_noise_multiplier).
    """
    device = client_cfg.get("device", "cpu")

    model = get_model(num_features=client_cfg["sample_features"],
                      num_classes=client_cfg["num_classes"],
                      dp_safe=client_cfg["dp_safe"])
    set_model_parameters(model, global_params)
    model = model.to(device)

    dp_eps_spent = None
    dp_noise_multiplier = None

    if client_cfg["use_byzantine_attack"] and client_idx in client_cfg["byzantine_clients"]:
        # Revision 25: widened to include use_zkp -- the standalone ZKP
        # ablation also operates on the classifier-head CKKS slice (via
        # he_local.encrypt_params_with_norm_guard), so a client attacking
        # under that ablation needs the SAME "train first, then poison only
        # the head" stealthy path USE_HE_KRUM_HYBRID already uses -- not a
        # full-model attack, which wouldn't test the guard at all.
        if (client_cfg["use_he"] or client_cfg["use_he_hybrid"] or client_cfg["use_zkp"]) \
                and client_cfg["byzantine_head_only"]:
            # Stealthy variant: train normally on the FULL model first, so
            # the bulk/backbone slice looks like a real locally-computed
            # update -- ONLY THEN overwrite the classifier-head slice with
            # the poisoned values.
            criterion = client_cfg["criterion"]
            train(model, X_tr, y_tr, criterion,
                  epochs=client_cfg["local_epochs"],
                  lr=client_cfg["learning_rate"],
                  global_params=global_params,
                  mu=client_cfg["prox_mu"],
                  device=device)
            trained_params    = get_model_parameters(model)
            model_state_keys  = list(model.state_dict().keys())
            params = classifier_head_flip_attack(
                trained_params, model_state_keys, scale=client_cfg["attack_scale"]
            )
        else:
            # Literature-standard attack -- train normally first, THEN
            # corrupt the RESULT. This is the path krum_dp_sweep takes for
            # its Byzantine clients (use_he/use_he_hybrid/use_zkp are all
            # False in that mode, so the condition above is always False
            # regardless of byzantine_head_only's value).
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
                else:  # "sign_flip", the default
                    params = sign_flip_attack_trained(trained_params,
                                                      scale=client_cfg["attack_scale"])

    else:
        if client_cfg["use_dp"] and _OPACUS_AVAILABLE:
            import torch
            import torch.utils.data as tud
            from opacus import PrivacyEngine

            criterion = client_cfg["criterion"]

            X_t = torch.FloatTensor(X_tr)
            y_t = torch.LongTensor(y_tr)
            loader = tud.DataLoader(
                tud.TensorDataset(X_t, y_t),
                batch_size=client_cfg["dp_batch_size"],
                shuffle=True,
            )
            optimizer = torch.optim.Adam(
                model.parameters(), lr=client_cfg["learning_rate"]
            )
            privacy_engine = PrivacyEngine(accountant="rdp")

            cache_key = (
                client_idx, client_cfg["dp_epsilon"], client_cfg["dp_delta"],
                client_cfg["local_epochs"], client_cfg["dp_batch_size"],
                client_cfg["dp_max_grad_norm"], len(X_tr),
            )
            cached_sigma = _noise_multiplier_cache.get(cache_key)

            if cached_sigma is None:
                model, optimizer, loader = privacy_engine.make_private_with_epsilon(
                    module=model,
                    optimizer=optimizer,
                    data_loader=loader,
                    target_epsilon=client_cfg["dp_epsilon"],
                    target_delta=client_cfg["dp_delta"],
                    epochs=client_cfg["local_epochs"],
                    max_grad_norm=client_cfg["dp_max_grad_norm"],
                )
                dp_noise_multiplier = getattr(optimizer, "noise_multiplier", None)
                if dp_noise_multiplier is not None:
                    _noise_multiplier_cache[cache_key] = dp_noise_multiplier
            else:
                model, optimizer, loader = privacy_engine.make_private(
                    module=model,
                    optimizer=optimizer,
                    data_loader=loader,
                    noise_multiplier=cached_sigma,
                    max_grad_norm=client_cfg["dp_max_grad_norm"],
                )
                dp_noise_multiplier = cached_sigma

            real_model_for_prox = model._module if hasattr(model, "_module") else model
            _model_state_keys = list(real_model_for_prox.state_dict().keys())
            _global_dict = (
                dict(zip(_model_state_keys, global_params))
                if client_cfg["prox_mu"] else None
            )

            model.train()
            for _ in range(client_cfg["local_epochs"]):
                for X_b, y_b in loader:
                    X_b = X_b.to(device)
                    y_b = y_b.to(device)
                    optimizer.zero_grad()
                    loss_val = criterion(model(X_b), y_b)
                    loss_val.backward()
                    optimizer.step()
                    _apply_dp_safe_prox_step(real_model_for_prox, _global_dict,
                                             client_cfg["prox_mu"], client_cfg["learning_rate"])

            dp_eps_spent = privacy_engine.get_epsilon(client_cfg["dp_delta"])

            real_model = model._module if hasattr(model, "_module") else model
            params = get_model_parameters(real_model)
        else:
            criterion = client_cfg["criterion"]
            train(model, X_tr, y_tr, criterion,
                  epochs=client_cfg["local_epochs"],
                  lr=client_cfg["learning_rate"],
                  global_params=global_params,
                  mu=client_cfg["prox_mu"],
                  device=device)
            params = get_model_parameters(model)

    return client_idx, params, dp_eps_spent, dp_noise_multiplier


def _eval_one_client(client_idx, global_params, X_te, y_te, eval_cfg):
    """
    Called either via ProcessPoolExecutor (CPU) or directly in-process (GPU).

    Returns (client_idx, loss, accuracy, per_class_f1).
    """
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
    Runs _train_one_client() for all clients this round, either through the
    persistent ProcessPoolExecutor (CPU path) or as a plain sequential
    in-process loop (GPU path -- executor is None).

    Returns a dict {client_idx: (params, dp_eps_spent, dp_noise_mult)}.
    """
    if executor is None:
        results_by_client = {}
        for i, (X_tr, y_tr, X_te, y_te) in enumerate(clients_data):
            client_idx, params, dp_eps_spent, dp_noise_mult = _train_one_client(
                i, X_tr, y_tr, global_params, round_client_cfg
            )
            results_by_client[client_idx] = (params, dp_eps_spent, dp_noise_mult)
        return results_by_client

    futures = {
        executor.submit(
            _train_one_client, i, X_tr, y_tr, global_params, round_client_cfg
        ): i
        for i, (X_tr, y_tr, X_te, y_te) in enumerate(clients_data)
    }
    results_by_client = {}
    for future in as_completed(futures):
        client_idx, params, dp_eps_spent, dp_noise_mult = future.result()
        results_by_client[client_idx] = (params, dp_eps_spent, dp_noise_mult)
    return results_by_client


def _run_eval_wave(executor, clients_data, global_params, eval_cfg):
    """
    Mirrors _run_training_wave() for the evaluation step.
    Returns a dict {client_idx: (loss, accuracy, per_class_f1)}.
    """
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
    total  = sum(weights)
    result = []
    for layer_idx in range(len(all_params[0])):
        layer_avg = sum(
            p[layer_idx] * (w / total)
            for p, w in zip(all_params, weights)
        )
        result.append(layer_avg)
    return result


# NOTE (revision 25): the old local he_aggregate() function that lived here
# has been DELETED. It never decrypted before returning and averaged
# unweighted, meaning global_params ended up as still-encrypted CKKS
# vectors -- the very next set_model_parameters() call in eval would have
# crashed. USE_HE now routes through he_local.aggregate_encrypted() +
# he_local.decrypt_params() instead, in the round loop below -- the same,
# already-validated pipeline USE_HE_KRUM_HYBRID uses. Do not re-add a local
# he_aggregate() function -- if you need raw ciphertext summation, it lives
# in defences/he_aggregation.py, wrapped correctly by he_local.py.


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

def save_best_checkpoint(global_params: list, round_num: int, f1_macro: float):
    """
    Separate checkpoint saved only when this round beats every prior round's
    F1-Macro this run -- so the best round stays recoverable even if a later
    round degrades and overwrites the per-round checkpoint.
    """
    np.savez(CHECKPOINT_BEST_PARAMS, *global_params)
    with open(CHECKPOINT_BEST_PROGRESS, "w") as f:
        json.dump({"best_round": round_num, "best_f1_macro": float(f1_macro)}, f)

# ---------------------------------------------------------------------------
# CSV LOGGING
# ---------------------------------------------------------------------------

_CSV_HEADER = (
    ["round", "client", "loss", "accuracy"]
    + ATTACK_NAMES
    + ["zkp_rejected", "krum_selected", "krum_detected_byzantine",
       "dp_epsilon_spent", "round_time_s",
       "dp_epsilon_target", "dp_noise_multiplier",
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
                   per_class_f1, zkp_rejected, krum_selected,
                   krum_detected, dp_eps, round_time, is_mean: bool = False,
                   dp_epsilon_target=None, dp_noise_multiplier=None,
                   krum_scores_byzantine_mean=None, krum_scores_honest_mean=None,
                   krum_score_ratio=None, nan_this_round=None):
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
        + [int(zkp_rejected),
           krum_selected_field,
           krum_detected_field,
           f"{dp_eps:.4f}" if dp_eps is not None else "N/A",
           f"{round_time:.2f}",
           _fmt(dp_epsilon_target, ".2f"),
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
    # Computed once, up front, so both the console banner and the
    # experiment_config JSON reflect the SAME actual attack path.
    # Revision 25: widened to include use_zkp.
    if (USE_HE or USE_HE_KRUM_HYBRID or USE_ZKP) and BYZANTINE_HEAD_ONLY:
        _attack_function_label = "classifier_head_flip_attack"
    else:
        _attack_function_label = {
            "sign_flip":     "sign_flip_attack_trained",
            "gaussian":      "gaussian_attack_trained",
            "zero_gradient": "zero_gradient_attack",
        }[ATTACK_TYPE]

    print(f"\n{'='*65}")
    print(f"  FL-IDS Unified Loop -- MODEL: {MODEL_TYPE.upper()}")
    print(f"  Ablation mode: {ABLATION_MODE}")
    if SANITY_CHECK:
        print(f"  *** SANITY_CHECK MODE -- {NUM_ROUNDS} rounds only ***")
    print(f"  Rounds={NUM_ROUNDS}  Clients={NUM_CLIENTS}  Epochs={LOCAL_EPOCHS}")
    print(f"  Device={_DEVICE}  (CUDA available: {_CUDA_AVAILABLE})")
    print(f"  Byzantine={NUM_BYZANTINE} (clients "
          f"{[c+1 for c in BYZANTINE_CLIENTS]}, matching the console "
          f"output's 'Client N' numbering)  "
          f"Attack={'ON' if USE_BYZANTINE_ATTACK else 'OFF'}"
          f"{'  [--byzantine override]' if _args.byzantine is not None else ''}")
    print(f"  Attack function: {_attack_function_label}"
          f"{f'  (std={GAUSSIAN_STD})' if _attack_function_label == 'gaussian_attack_trained' else ''}")
    print(f"  USE_KRUM={USE_KRUM}  USE_ADAPTIVE_KRUM={USE_ADAPTIVE_KRUM}  "
          f"USE_HE={USE_HE}  USE_HE_KRUM_HYBRID={USE_HE_KRUM_HYBRID}  "
          f"USE_DP={USE_DP}  USE_ZKP={USE_ZKP}")
    if _CUDA_AVAILABLE:
        print(f"  Client training: SEQUENTIAL, in-process (no worker pool -- "
              f"avoids fork+CUDA hang)")
    else:
        print(f"  Parallel client training: {CLIENT_POOL_WORKERS} worker(s), "
              f"{_THREADS_PER_WORKER} threads/worker "
              f"({_CPU_COUNT} cores detected)")
    if USE_DP:
        print(f"  DP: eps={DP_EPSILON}  delta={DP_DELTA}  "
              f"max_grad_norm={DP_MAX_GRAD_NORM}  batch_size={DP_BATCH_SIZE}  "
              f"accountant=rdp")
    if USE_KRUM:
        print(f"  Krum (fixed-m): selecting {KRUM_M} of {NUM_CLIENTS} clients "
              f"(discarding {NUM_CLIENTS - KRUM_M}: "
              f"{NUM_BYZANTINE} confirmed Byzantine + "
              f"{NUM_CLIENTS - KRUM_M - NUM_BYZANTINE} safety margin)")
    if USE_ADAPTIVE_KRUM:
        print(f"  Adaptive Krum: method={ADAPTIVE_KRUM_METHOD}  k={ADAPTIVE_KRUM_K}  "
              f"min_keep_fraction={ADAPTIVE_KRUM_MIN_KEEP_FRACTION} "
              f"(clients dropped per round is DYNAMIC, not fixed)")
    if USE_HE:
        print(f"  HE (standalone, no Krum): partial CKKS on classifier head, "
              f"poly_degree={HE_POLY_DEGREE}. ALL accepted clients "
              f"encrypted + averaged unconditionally (via he_local.py).")
    if USE_HE_KRUM_HYBRID:
        print(f"  HE+Krum Hybrid: adaptive Krum (method={ADAPTIVE_KRUM_METHOD}  "
              f"k={ADAPTIVE_KRUM_K}  assumed_f={ADAPTIVE_KRUM_HYBRID_ASSUMED_F}"
              f"{' [ground-truth NUM_BYZANTINE=' + str(NUM_BYZANTINE) + ']' if ADAPTIVE_KRUM_HYBRID_ASSUMED_F != NUM_BYZANTINE else ''}"
              f") scores the PLAINTEXT (bulk) slice only; "
              f"classifier-head slice (CKKS, poly_degree={HE_POLY_DEGREE}) is "
              f"aggregated only over whichever clients that scoring selects.")
        print(f"  Byzantine head-only attack: {BYZANTINE_HEAD_ONLY}")
        print(f"  Head-norm guard (Layer 2 extension): {USE_HEAD_NORM_GUARD} "
              f"(k={HEAD_NORM_GUARD_K}, min_keep_fraction="
              f"{HEAD_NORM_GUARD_MIN_KEEP_FRACTION}) -- ciphertext-bound "
              f"MAD threshold on classifier-head delta norms, runs BEFORE "
              f"Krum each round")
    if USE_ZKP:
        print(f"  ZKP head-norm guard (STANDALONE, no Krum call at all): "
              f"k={HEAD_NORM_GUARD_K}  "
              f"min_keep_fraction={HEAD_NORM_GUARD_MIN_KEEP_FRACTION}  "
              f"-- ciphertext-bound HMAC MAD threshold on classifier-head "
              f"delta norms is the ONLY defence active this run.")
        print(f"  Byzantine head-only attack: {BYZANTINE_HEAD_ONLY} "
              f"(should be True -- this is the whole point of the "
              f"pure_zkp ablation: does the guard alone catch a "
              f"classifier-head-only attacker with no Krum backing it up?)")
    print(f"{'='*65}\n")

    torch.set_num_threads(_CPU_COUNT)

    print("Loading data partitions...")
    clients_data = []
    for i in range(NUM_CLIENTS):
        print(f"  Partition {i+1}/{NUM_CLIENTS}...", end="\r")
        clients_data.append(load_partition(i, NUM_CLIENTS))
    sample_features = clients_data[0][0].shape[1]
    print(f"\nFeature count (measured, not assumed): {sample_features}")
    print(f"All {NUM_CLIENTS} clients loaded.\n")

    def print_data_split():
        """
        Per-client train-partition sample counts, broken down by class.
        Printed at the start of every round so it sits right next to that
        round's aggregation decision in the log.
        """
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
    precomputed_criterion = build_criterion().to(_DEVICE)
    print("Criterion built -- workers will reuse this, no per-round reload.\n")

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
        "use_he":                USE_HE,
        "use_he_hybrid":        USE_HE_KRUM_HYBRID,
        "use_zkp":               USE_ZKP,
        "byzantine_head_only":  BYZANTINE_HEAD_ONLY,
        "use_dp":                USE_DP,
        "dp_epsilon":           DP_EPSILON,
        "dp_delta":             DP_DELTA,
        "dp_max_grad_norm":     DP_MAX_GRAD_NORM,
        "dp_batch_size":        DP_BATCH_SIZE,
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

    # Unified CKKS context init -- revision 25: USE_HE now shares the SAME
    # he_local-backed context/pipeline as USE_HE_KRUM_HYBRID and USE_ZKP,
    # instead of the old broken standalone ts.context()/ts.ckks_vector() path.
    # Stays None under krum_dp_sweep (no HE/ZKP flag active there).
    he_context = None
    if (USE_HE or USE_HE_KRUM_HYBRID or USE_ZKP) and _TENSEAL_AVAILABLE:
        he_context = he_local.create_ckks_context(HE_POLY_DEGREE)
        print(f"CKKS context initialised via he_local.create_ckks_context "
              f"(poly_degree={HE_POLY_DEGREE}, partial/classifier-head-only "
              f"encryption).\n")

    # Model state_dict key order, needed to split each client's flat param
    # list into "sensitive" (classifier.*) vs "bulk" layers. Built once
    # here, off a throwaway model instance. Stays None under krum_dp_sweep.
    MODEL_STATE_KEYS = None
    if USE_HE or USE_HE_KRUM_HYBRID or USE_ZKP:
        _keys_model = get_model(num_features=sample_features,
                                num_classes=NUM_CLASSES, dp_safe=DP_SAFE)
        MODEL_STATE_KEYS = list(_keys_model.state_dict().keys())
        del _keys_model

    global_params, start_round = load_checkpoint()
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
        print("  NOTE: if you changed ABLATION_MODE, DP_EPSILON, USE_KRUM, "
              "USE_ADAPTIVE_KRUM, USE_HE, or any other experiment flag since "
              f"the last run, delete {CHECKPOINT_PARAMS} and "
              f"{CHECKPOINT_PROGRESS} before continuing -- resuming across "
              "different experiment conditions silently contaminates "
              "round-1 comparability.\n")

    resume = start_round > 0
    init_log_csv(resume=resume)
    best_f1_macro = -1.0
    if resume and os.path.exists(CHECKPOINT_BEST_PROGRESS):
        with open(CHECKPOINT_BEST_PROGRESS) as f:
            best_f1_macro = json.load(f).get("best_f1_macro", -1.0)
        print(f"  Resuming best-F1 tracking: {best_f1_macro:.4f} so far.\n")

    meta_path = f"experiment_config_{_TAG}.json"
    with open(meta_path, "w") as f:
        json.dump({
            "ablation_mode": ABLATION_MODE,
            "model_type": MODEL_TYPE,
            "sanity_check": SANITY_CHECK,
            "num_rounds": NUM_ROUNDS,
            "num_clients": NUM_CLIENTS,
            "num_features_measured": sample_features,
            "local_epochs": LOCAL_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "prox_mu": PROX_MU,
            "byzantine_attack": USE_BYZANTINE_ATTACK,
            "num_byzantine": NUM_BYZANTINE,
            "byzantine_clients": BYZANTINE_CLIENTS,
            "attack_scale": ATTACK_SCALE,
            "attack_type": ATTACK_TYPE,
            "gaussian_std": GAUSSIAN_STD if ATTACK_TYPE == "gaussian" else None,
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
            "use_he": USE_HE,
            "use_he_krum_hybrid": USE_HE_KRUM_HYBRID,
            "use_zkp": USE_ZKP,
            "use_head_norm_guard": USE_HEAD_NORM_GUARD,
            "head_norm_guard_k": HEAD_NORM_GUARD_K if (USE_HEAD_NORM_GUARD or USE_ZKP) else None,
            "head_norm_guard_min_keep_fraction": HEAD_NORM_GUARD_MIN_KEEP_FRACTION if (USE_HEAD_NORM_GUARD or USE_ZKP) else None,
            "he_poly_degree": HE_POLY_DEGREE if (USE_HE or USE_HE_KRUM_HYBRID or USE_ZKP) else None,
            "use_dp": USE_DP,
            "dp_epsilon": DP_EPSILON,
            "dp_delta": DP_DELTA,
            "dp_max_grad_norm": DP_MAX_GRAD_NORM,
            "dp_batch_size": DP_BATCH_SIZE,
            "dp_accountant": "rdp",
            "byzantine_head_only": BYZANTINE_HEAD_ONLY,
            "dp_safe": DP_SAFE,
            "device": str(_DEVICE),
            "cuda_available": _CUDA_AVAILABLE,
            "client_pool_workers": CLIENT_POOL_WORKERS,
            "threads_per_worker": _THREADS_PER_WORKER,
            "framework": "custom Python simulation (direct, parallel client training)",
        }, f, indent=2)

    # ========================================================================
    # ROUND LOOP
    # GPU: no pool at all (executor stays None throughout).
    # CPU: original persistent 4-way ProcessPoolExecutor, unchanged.
    # ========================================================================
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

            round_client_cfg = client_cfg

            accepted_params          = []
            accepted_weights         = []
            accepted_client_indices  = []

            zkp_rejected_this_round  = []
            dp_eps_spent_this_round  = []
            dp_noise_mult_this_round = []

            _train_wave_start = time.time()
            results_by_client = _run_training_wave(
                executor, clients_data, global_params, round_client_cfg
            )
            _train_wave_elapsed = time.time() - _train_wave_start
            print(f"  [Timing] Training wave (all {NUM_CLIENTS} clients): "
                  f"{_train_wave_elapsed:.1f}s")

            for i, (X_tr, y_tr, X_te, y_te) in enumerate(clients_data):
                params, dp_eps_spent, dp_noise_mult = results_by_client[i]

                if USE_BYZANTINE_ATTACK and i in BYZANTINE_CLIENTS:
                    if (USE_HE or USE_HE_KRUM_HYBRID or USE_ZKP) and BYZANTINE_HEAD_ONLY:
                        tag = "head-only"
                    elif ATTACK_TYPE == "gaussian":
                        tag = "gaussian (trained)"
                    elif ATTACK_TYPE == "zero_gradient":
                        tag = "zero-gradient"
                    else:
                        tag = "sign-flip (trained)"
                    print(f"  Client {i+1:2d}  [BYZANTINE -- {tag} x{ATTACK_SCALE}]")

                # Revision 25: the old early zkp_verify_norm() rejection
                # gate that used to live here has been REMOVED. USE_ZKP's
                # proof needs a real ciphertext to bind to (it can't exist
                # before encryption), so rejection now happens AFTER
                # encryption, at aggregation time -- same structural
                # position as USE_HE_KRUM_HYBRID's norm guard.

                if (USE_HE or USE_HE_KRUM_HYBRID or USE_ZKP) and _TENSEAL_AVAILABLE and he_context is not None:
                    if (USE_HE_KRUM_HYBRID or USE_ZKP) and USE_HEAD_NORM_GUARD:
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
                              f"plaintext (bulk) -- measured once, client {i+1}.")
                    accepted_params.append(client_enc)
                else:
                    # krum_dp_sweep always lands here -- accepted_params
                    # holds raw plaintext parameter lists, exactly what
                    # adaptive_multi_krum() expects (same shape Experiment
                    # 1's original sign-flip sweep already validated).
                    accepted_params.append(params)

                accepted_weights.append(len(X_tr))
                accepted_client_indices.append(i)
                if dp_eps_spent is not None:
                    dp_eps_spent_this_round.append(dp_eps_spent)
                if dp_noise_mult is not None:
                    dp_noise_mult_this_round.append(dp_noise_mult)

            krum_selected_ids   = set()
            krum_discarded_ids  = set()
            krum_detected_byz   = set()
            krum_score_diag     = None
            krum_scored_client_indices = None

            if len(accepted_params) == 0:
                print("  WARNING: All clients rejected -- skipping round.")
                save_checkpoint(global_params, round_num)
                continue

            if USE_HE and _TENSEAL_AVAILABLE:
                # Revision 25: routes through he_local, ALL accepted clients
                # encrypted + averaged unconditionally -- no Krum call.
                enc_aggregate = he_local.aggregate_encrypted(
                    accepted_params, accepted_weights, he_context
                )
                global_params = he_local.decrypt_params(enc_aggregate)
                agg_label = ("HE (partial, classifier-head-only -- "
                             "full-client average, no Krum)")

            elif USE_HE_KRUM_HYBRID and _TENSEAL_AVAILABLE:
                # accepted_params here is a list of he_local.encrypt_params()
                # output dicts (one per accepted client): {"sensitive_enc":
                # <encrypted classifier head>, "bulk": <plaintext everything
                # else>, "sensitive_idx"/"bulk_idx": layer index mappings}.

                # -- Layer 2 extension: ciphertext-bound head-norm guard --
                if USE_HEAD_NORM_GUARD:
                    verified_positions = []
                    verified_norms = []
                    norm_guard_rejected_ids = set()
                    for pos, c in enumerate(accepted_params):
                        proof = c.get("head_norm_proof")
                        chunks = c["sensitive_enc"]["chunks"]
                        is_valid, reason = (
                            zkp.verify_head_norm_proof(proof, chunks)
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
                        zkp.mad_threshold_head_norms(
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
                                 "norm-guard-surviving clients for "
                                 "plaintext-slice Krum; all survivors "
                                 "included in both slices)")
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

                krum_selected_ids  = {
                    hybrid_accepted_client_indices[pos] for pos in selected_positions
                }
                krum_discarded_ids = (
                    {idx for idx in hybrid_accepted_client_indices
                     if idx not in krum_selected_ids}
                    | norm_guard_rejected_ids
                )
                krum_detected_byz = krum_discarded_ids & set(BYZANTINE_CLIENTS)

                selected_enc_clients = [hybrid_accepted_params[pos] for pos in selected_positions]
                selected_weights     = [hybrid_accepted_weights[pos] for pos in selected_positions]
                enc_aggregate = he_local.aggregate_encrypted(
                    selected_enc_clients, selected_weights, he_context
                )
                global_params = he_local.decrypt_params(enc_aggregate)

                if agg_label is None:
                    agg_label = (
                        f"HE+Krum hybrid (adaptive, {ADAPTIVE_KRUM_METHOD}, "
                        f"k={ADAPTIVE_KRUM_K})  plaintext-slice "
                        f"selected={sorted(krum_selected_ids)}  "
                        f"discarded={sorted(krum_discarded_ids)}"
                        f"{'  (incl. norm-guard-rejected: ' + str(sorted(norm_guard_rejected_ids)) + ')' if norm_guard_rejected_ids else ''}"
                        f"  detected_byz={sorted(krum_detected_byz)}  "
                        f"(encrypted classifier-head slice aggregated over "
                        f"selected clients only)"
                    )

            elif USE_ZKP and _TENSEAL_AVAILABLE:
                # Revision 25 (new): isolates the ciphertext-bound HMAC
                # head-norm guard as the SOLE defence on the classifier-head
                # slice -- NO Krum call at all. Direct component test of
                # Experiment 2's hybrid pipeline's guard stage, decoupled
                # from the plaintext-slice Krum stage.
                verified_positions = []
                verified_norms = []
                zkp_rejected_ids = set()

                for pos, c in enumerate(accepted_params):
                    proof  = c.get("head_norm_proof")
                    chunks = c["sensitive_enc"]["chunks"]
                    is_valid, reason = (
                        zkp.verify_head_norm_proof(proof, chunks)
                        if proof is not None else (False, "PROOF_MISSING")
                    )
                    if is_valid:
                        verified_positions.append(pos)
                        verified_norms.append(proof["norm"])
                    else:
                        zkp_rejected_ids.add(accepted_client_indices[pos])
                        print(f"  [ZKP head-norm guard] Client "
                              f"{accepted_client_indices[pos]+1} REJECTED "
                              f"at verification: {reason}")

                guard_kept_rel, guard_dropped_rel, norm_guard_diag = \
                    zkp.mad_threshold_head_norms(
                        verified_norms, k=HEAD_NORM_GUARD_K,
                        min_keep_fraction=HEAD_NORM_GUARD_MIN_KEEP_FRACTION,
                    )
                survivor_positions = [verified_positions[i] for i in guard_kept_rel]
                for i in guard_dropped_rel:
                    zkp_rejected_ids.add(accepted_client_indices[verified_positions[i]])

                krum_selected_ids  = {accepted_client_indices[pos] for pos in survivor_positions}
                krum_discarded_ids = zkp_rejected_ids
                krum_detected_byz  = zkp_rejected_ids & set(BYZANTINE_CLIENTS)
                zkp_rejected_this_round = sorted(zkp_rejected_ids)

                print(f"  [ZKP head-norm guard] {norm_guard_diag} "
                      f"kept={len(survivor_positions)}/{len(accepted_params)}  "
                      f"rejected_ids={sorted(zkp_rejected_ids)}  "
                      f"detected_byz={sorted(krum_detected_byz)}")

                if len(survivor_positions) == 0:
                    print("  WARNING: ZKP head-norm guard rejected ALL clients "
                          "this round -- skipping round.")
                    save_checkpoint(global_params, round_num)
                    continue

                survivor_enc     = [accepted_params[pos] for pos in survivor_positions]
                survivor_weights = [accepted_weights[pos] for pos in survivor_positions]
                enc_aggregate = he_local.aggregate_encrypted(
                    survivor_enc, survivor_weights, he_context
                )
                global_params = he_local.decrypt_params(enc_aggregate)

                agg_label = (f"ZKP head-norm guard only (no Krum)  "
                             f"selected={sorted(krum_selected_ids)}  "
                             f"rejected={sorted(zkp_rejected_ids)}  "
                             f"detected_byz={sorted(krum_detected_byz)}")

            elif USE_KRUM:
                effective_m = min(KRUM_M, len(accepted_params) - 1)
                if effective_m < 1:
                    global_params = fedprox_aggregate(accepted_params,
                                                      accepted_weights)
                    agg_label = "FedProx (Krum fallback)"
                else:
                    global_params, selected_positions = multi_krum(
                        accepted_params,
                        accepted_weights,
                        num_byzantine=NUM_BYZANTINE,
                        m=effective_m,
                    )
                    krum_selected_ids  = {
                        accepted_client_indices[pos]
                        for pos in selected_positions
                    }
                    krum_discarded_ids = {
                        idx for idx in accepted_client_indices
                        if idx not in krum_selected_ids
                    }
                    krum_detected_byz = krum_discarded_ids & set(BYZANTINE_CLIENTS)

                    agg_label = (f"Multi-Krum (m={effective_m})  "
                                 f"selected={sorted(krum_selected_ids)}  "
                                 f"discarded={sorted(krum_discarded_ids)}  "
                                 f"detected_byz={sorted(krum_detected_byz)}")

            elif USE_ADAPTIVE_KRUM:
                # krum_dp_sweep lands here.
                if len(accepted_params) - NUM_BYZANTINE - 2 < 1:
                    global_params = fedprox_aggregate(accepted_params,
                                                      accepted_weights)
                    agg_label = "FedProx (Adaptive-Krum fallback -- too few accepted clients)"
                else:
                    global_params, selected_positions, krum_score_diag = adaptive_multi_krum(
                        accepted_params,
                        accepted_weights,
                        num_byzantine=NUM_BYZANTINE,
                        k=ADAPTIVE_KRUM_K,
                        method=ADAPTIVE_KRUM_METHOD,
                        min_keep_fraction=ADAPTIVE_KRUM_MIN_KEEP_FRACTION,
                        return_diagnostics=True,
                    )
                    krum_selected_ids  = {
                        accepted_client_indices[pos]
                        for pos in selected_positions
                    }
                    krum_discarded_ids = {
                        idx for idx in accepted_client_indices
                        if idx not in krum_selected_ids
                    }
                    krum_detected_byz = krum_discarded_ids & set(BYZANTINE_CLIENTS)
                    krum_scored_client_indices = accepted_client_indices

                    agg_label = (f"Adaptive Multi-Krum ({ADAPTIVE_KRUM_METHOD}, "
                                 f"k={ADAPTIVE_KRUM_K})  "
                                 f"selected={sorted(krum_selected_ids)}  "
                                 f"discarded={sorted(krum_discarded_ids)}  "
                                 f"detected_byz={sorted(krum_detected_byz)}")

            else:
                global_params = fedprox_aggregate(accepted_params,
                                                  accepted_weights)
                agg_label = "FedProx"

            print(f"  Aggregation: {agg_label}")
            if zkp_rejected_this_round:
                print(f"  ZKP rejected: {zkp_rejected_this_round}")

            _krum_active = USE_KRUM or USE_ADAPTIVE_KRUM or USE_HE_KRUM_HYBRID or USE_ZKP

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

                is_zkp_rejected  = i in zkp_rejected_this_round
                is_krum_selected = (i in krum_selected_ids) if _krum_active else False
                is_krum_detected = (i in krum_detected_byz) if _krum_active else False

                append_log_row(
                    round_num=round_num,
                    client_label=i + 1,
                    loss=loss_v,
                    accuracy=acc_v,
                    per_class_f1=f1_per_class,
                    zkp_rejected=is_zkp_rejected,
                    krum_selected=is_krum_selected,
                    krum_detected=is_krum_detected,
                    dp_eps=None,
                    round_time=0.0,
                    is_mean=False,
                )

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
                             else "ZKP head-norm guard (classifier-head only, no Krum)" if USE_ZKP
                             else "Adaptive Krum")
                print(f"  [{krum_label}] Detection rate this round: "
                      f"{krum_detection_rate:.2%}  "
                      f"({len(krum_detected_byz)}/{NUM_BYZANTINE} Byzantine detected, "
                      f"{len(krum_selected_ids)}/{NUM_CLIENTS - len(zkp_rejected_this_round)} "
                      f"legitimate-eligible clients selected)")

            mean_dp_eps = (
                float(np.mean(dp_eps_spent_this_round))
                if dp_eps_spent_this_round else None
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

            append_log_row(
                round_num=round_num,
                client_label="MEAN",
                loss=mean_loss,
                accuracy=mean_acc,
                per_class_f1=mean_f1,
                zkp_rejected=len(zkp_rejected_this_round),
                krum_selected=len(krum_selected_ids) if _krum_active else None,
                krum_detected=krum_detection_rate,
                dp_eps=mean_dp_eps,
                round_time=round_time,
                is_mean=True,
                dp_epsilon_target=(DP_EPSILON if USE_DP else None),
                dp_noise_multiplier=mean_dp_noise_mult,
                krum_scores_byzantine_mean=krum_byz_mean,
                krum_scores_honest_mean=krum_honest_mean,
                krum_score_ratio=krum_ratio,
                nan_this_round=nan_this_round,
            )

            save_checkpoint(global_params, round_num)

    print("\n" + "="*65)
    print(f"  Training complete -- {NUM_ROUNDS} rounds  [{MODEL_TYPE.upper()}]  "
          f"[ABLATION_MODE={ABLATION_MODE}]")
    if SANITY_CHECK:
        print(f"  *** This was a SANITY_CHECK run ({NUM_ROUNDS} rounds). ***")
        print(f"  *** Set SANITY_CHECK=False and delete the checkpoint before ***")
        print(f"  *** starting the real sweep. ***")
    print(f"  Results logged to:     {LOG_CSV}")
    print(f"  Checkpoint:            {CHECKPOINT_PARAMS} (round {NUM_ROUNDS})")
    if USE_KRUM or USE_ADAPTIVE_KRUM or USE_HE_KRUM_HYBRID or USE_HE or USE_ZKP:
        print(f"\n  Reminder: delete checkpoint before changing flags")
        print(f"  (Krum/Adaptive-Krum/HE/HE-Krum-Hybrid/ZKP/DP flags change the")
        print(f"  experiment -- old checkpoint params will give misleading")
        print(f"  results if reused.)")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
