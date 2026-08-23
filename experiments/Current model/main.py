"""
Unified FL-IDS Main Loop
========================
Merges:
  - DP/ZKP/HE main.py  (privacy stack structure)
  - Krum main.py        (working Multi-Krum aggregation)

Four aggregation branches, selected by flags:
  1. USE_HE=True                   → CKKS homomorphic aggregation (no Krum possible)
  2. USE_KRUM=True, USE_HE=False   → Multi-Krum, fixed m (plaintext, Byzantine-robust)
  3. USE_ADAPTIVE_KRUM=True, USE_HE=False → Adaptive Multi-Krum, dynamic MAD/Z-score
                                       threshold instead of a fixed m (plaintext)
  4. All of the above False        → plain FedAvg / FedProx

Bug fixed: ZKP-rejected clients are removed from accepted_params before Krum
is called, so accepted_params is a COMPACTED list. Multi-Krum returns positions
within that compacted list. We track accepted_client_indices in parallel so we
can map positions back to original 0-indexed client IDs before comparing against
BYZANTINE_CLIENTS for detection-rate logging. Adaptive Multi-Krum uses the exact
same compaction/translation logic — see its branch below.

Run:
    python src/main.py network      # network-layer model
    python src/main.py application  # application-layer model

--------------------------------------------------------------------------
CHANGELOG (this revision)
--------------------------------------------------------------------------
1-16. (see previous revisions — krum_detected truthy fix, KRUM_M=7,
      measured feature count logging, DP_MAX_GRAD_NORM=1.5, params
      extraction UnboundLocalError fix, parallel client training,
      adaptive Multi-Krum / Condition 5, criterion built once,
      eval parallelized, EMA removed, noise_multiplier caching)

17. GPU DEVICE SUPPORT — added (see prior revision).

18. SANITY_CHECK toggle added (see prior revision).

19. FIX — fork+CUDA hang (see prior revision — sequential in-process
    training/eval on GPU runs instead of a forked ProcessPoolExecutor).

20. FIX — sign_flip_attack was non-standard versus the literature (see
    prior revision — sign_flip_attack_trained() now used, trains the
    attacking client first, then negates the result).

21. ADDED — Gaussian noise attack, trains-first version
    (gaussian_attack_trained), wired in as a second selectable attack
    type alongside sign-flip via the new --attack-type CLI flag.
    Genuinely different attack geometry from sign-flip: no consistent
    direction, and — unlike either attack's untrained/naive version —
    two Byzantine clients under Gaussian noise are NOT bitwise-
    identical to each other (independent noise draws), giving Krum's
    distance-based scoring a meaningfully different shape to contend
    with than a coordinated negation. --attack-type zero_gradient also
    added as a third option (zero_gradient_attack was already defined
    in defences/byzantine.py but never wired into main.py's dispatch).

22. FIX — "attack_function" in experiment_config_*.json was hardcoded
    to "sign_flip_attack_trained" regardless of which attack actually
    ran (including when BYZANTINE_HEAD_ONLY routed to
    classifier_head_flip_attack instead) — misleading experiment
    provenance. Now computed dynamically from the actual attack path
    taken, including the new attack-type dispatch from #21.

23. FIX — stale comment on USE_DP. Read "Experiment 2 isolates HE x
    Krum only" (which implies USE_DP should be False) directly above
    USE_DP=True — a leftover from an earlier Experiment-2 config that
    no longer matches this file's actual current experiment (adaptive
    Krum + DP epsilon sweep, USE_HE_KRUM_HYBRID=False). The True value
    was already correct for what's actually running; only the
    misleading comment text is fixed here.

24. FIX — GAUSSIAN_STD had a flat default (10.0) with no model-type
    split, unlike ATTACK_SCALE (5.0 network / 2.0 application).
    Measured via measure_param_scale.py against this codebase's ACTUAL
    trained-delta magnitude (network: mean delta_std ~= 4.17,
    application: mean delta_std ~= 2.64) — the old default of 10.0 was
    only ~2.4x the honest signal on network and ~3.8x on application,
    weaker than intended relative to how dominant ATTACK_SCALE is for
    sign-flip. RSA's cited sigma=10000 is NOT directly portable here:
    unlike sign-flip's multiplicative scale (which is automatically
    proportional to whatever a model's parameter magnitudes are),
    additive Gaussian noise requires knowing the ACTUAL parameter
    scale in THIS codebase, which a different paper's number cannot
    supply. New model-aware defaults (network=50.0, application=30.0)
    sit in the "aggressive" tier (~10-12x measured delta std),
    matching how dominant ATTACK_SCALE already is for sign-flip. Still
    fully overridable via --gaussian-std. NOT YET NaN/overflow-tested
    at these new defaults -- run a short sanity check on both models
    before committing to a full Gaussian sweep, same caution
    ATTACK_SCALE itself originally needed.
--------------------------------------------------------------------------
KNOWN OPEN ITEMS — NOT YET RESOLVED, FLAGGED FOR NEXT REVISION
--------------------------------------------------------------------------
- PROX_MU is 0.02 here (user-confirmed intended value).
- LR decay disabled (user-confirmed decision) — get_round_lr() kept but unused.
- USE_ADAPTIVE_KRUM=True is a deliberate deviation from the master planning
  doc's "Experiment 1 must use fixed-m Krum" instruction (user decision) —
  any comparison against a fixed-m Condition 3 anchor is not apples-to-apples.
- task.py has been patched (separately) to register FocalLoss's weight via
  register_buffer() and accept a `device` kwarg on train()/test() — confirm
  the version on disk matches before running; this file's calls assume it.
- DP_BATCH_SIZE=512 was tuned for CPU. DGX Spark's unified CPU/GPU memory
  means an Opacus per-sample-gradient OOM here can degrade the WHOLE
  system rather than cleanly killing the job — watch `free -h` on the
  first real (non-sanity-check) DP round; drop DP_BATCH_SIZE if memory
  pressure shows up.
- Any epsilon-sweep results collected BEFORE the sign-flip fix (revision
  20), including Experiment 1's original ε=3/9/15 anchors, used the
  non-standard sign_flip_attack — not directly comparable to new runs.
- Gaussian attack results collected BEFORE revision 24 used the old flat
  std=10.0 default -- weaker than the new model-aware defaults intended.
  Re-run any prior Gaussian-attack condition under the new defaults
  before comparing against sign-flip results.
- Gaussian noise draws are UNSEEDED (np.random.normal, no explicit seed)
  -- two runs at identical config will get different attacker noise
  each time. This is a deliberate open question, not yet decided: seed
  it for exact reproducibility of a specific run, or leave it unseeded
  to argue any "detection stays flat" finding is robust across draws,
  not a single lucky/unlucky seed. Pick one and state it explicitly in
  the write-up before publishing any Gaussian-attack result.
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
# Path setup — allow running from project root OR from src/
# ---------------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ---------------------------------------------------------------------------
# ─── CONFIGURATION ──────────────────────────────────────────────────────────
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
                     help="Suffix on every output filename — "
                          "e.g. --tag dp15 → results_network_dp15.csv, "
                          "replaces manual mv-archiving between sweep runs")
_parser.add_argument("--byzantine", type=str, default=None,
                     help="Comma-separated client numbers to make Byzantine, "
                          "using the SAME 1-indexed numbering the console "
                          "output prints everywhere ('Client 4', 'Client 10', "
                          "etc.) — e.g. --byzantine 4,10 to attack the "
                          "clients labelled 'Client 4' and 'Client 10' in the "
                          "logs. Default (no flag) is clients 1,2. Overrides "
                          "both BYZANTINE_CLIENTS and NUM_BYZANTINE below — "
                          "NUM_BYZANTINE becomes len(this list). Numbers must "
                          "be in [1, NUM_CLIENTS].")
_parser.add_argument("--krum-k", type=float, default=None,
                     help="Override ADAPTIVE_KRUM_K (default 2.5) — the MAD "
                          "sensitivity multiplier. Larger k -> more "
                          "permissive (fewer clients dropped). Raise this "
                          "if adaptive Krum is excluding honest clients that "
                          "aren't actually attackers (likely non-IID "
                          "variance, not malice) — try 3.5-4.5 as a start.")
_parser.add_argument("--attack-type", type=str, default="sign_flip",
                     choices=["sign_flip", "gaussian", "zero_gradient"],
                     help="Which Byzantine attack the malicious clients use "
                          "(ignored when BYZANTINE_HEAD_ONLY triggers the "
                          "classifier_head_flip_attack path instead — that "
                          "path is orthogonal to this flag, since it targets "
                          "the HE-hybrid classifier-head slice specifically, "
                          "not a full-model attack). Default sign_flip.")
_parser.add_argument("--gaussian-std", type=float, default=None,
                     help="Standard deviation for --attack-type gaussian. "
                          "Ignored for other attack types. Default is "
                          "model-aware (network=50.0, application=30.0), "
                          "measured via measure_param_scale.py against "
                          "this codebase's actual trained-delta magnitude "
                          "-- NOT RSA's cited sigma=10000, which is a "
                          "different model's units (additive noise doesn't "
                          "transfer across models the way a multiplicative "
                          "scale factor does).")
_args = _parser.parse_args()

MODEL_TYPE = _args.model_type

# ── Sanity-check toggle ──────────────────────────────────────────────────
SANITY_CHECK = False

# FL hyperparameters
NUM_ROUNDS    = 2 if SANITY_CHECK else 25
NUM_CLIENTS   = 10
LOCAL_EPOCHS  = 5
LEARNING_RATE = 0.001
PROX_MU       = 0.02       # FedProx proximal coefficient (0 = plain FedAvg)

# Byzantine attack
USE_BYZANTINE_ATTACK = True

if _args.byzantine is not None:
    # --byzantine "4,10" -> the clients printed as "Client 4" and "Client 10"
    # everywhere in the console output (score tables, [BYZANTINE] tags, etc.)
    # are attacked. Internally BYZANTINE_CLIENTS/client_idx are 0-indexed
    # (they're array positions), so this is where that translation happens
    # -- one place, so the mismatch between "what the logs show" and "what
    # the code stores" can't leak out and cause an off-by-one mistake at
    # every call site the way it did before this flag existed.
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
    # Default, unchanged from before -- clients 0 and 1 (0-indexed) are
    # malicious, i.e. "Client 1" and "Client 2" in the console output.
    NUM_BYZANTINE     = 2
    BYZANTINE_CLIENTS = list(range(NUM_BYZANTINE))

ATTACK_SCALE  = 5.0 if MODEL_TYPE == "network" else 2.0
ATTACK_TYPE   = _args.attack_type

# FIX (changelog #24): model-aware default, same philosophy as
# ATTACK_SCALE -- measured via measure_param_scale.py against this
# codebase's actual trained-delta magnitude (network: delta_std~4.17,
# application: delta_std~2.64; measured on this DGX). NOT ported from
# RSA's sigma=10000 (different model/units -- additive noise doesn't
# transfer the way a multiplicative scale does). Picked from the
# "aggressive" tier (~10-12x measured delta std) so the attack clearly
# dominates honest signal, mirroring how dominant ATTACK_SCALE=5.0/2.0
# already is for sign-flip. Still fully overridable via --gaussian-std.
_GAUSSIAN_STD_DEFAULT = 50.0 if MODEL_TYPE == "network" else 30.0
GAUSSIAN_STD  = _args.gaussian_std if _args.gaussian_std is not None else _GAUSSIAN_STD_DEFAULT

# ─── Defence flags ──────────────────────────────────────────────────────────
USE_KRUM            = False
USE_ADAPTIVE_KRUM   = True
USE_HE              = False
USE_HE_KRUM_HYBRID  = False   # Experiment 2 — plaintext-slice Krum + encrypted-slice HE

# NOTE (fixed comment, revision 23): this flag's value (True) is correct
# for THIS file's current active experiment (adaptive Krum + DP epsilon
# sweep, USE_HE_KRUM_HYBRID=False). The old comment here implied
# Experiment 2's "isolate HE x Krum only, DP off" design (which needs
# USE_DP=False) — that comment belonged to a different config than the
# one actually running now and has been removed to avoid misleading
# whoever reads this next. If/when USE_HE_KRUM_HYBRID=True is flipped
# back on for an actual Experiment-2 run, set USE_DP=False accordingly.
USE_DP   = True
USE_ZKP  = False

assert sum([USE_KRUM, USE_ADAPTIVE_KRUM, USE_HE, USE_HE_KRUM_HYBRID]) <= 1, \
    "USE_KRUM, USE_ADAPTIVE_KRUM, USE_HE, and USE_HE_KRUM_HYBRID are mutually " \
    "exclusive aggregation branches — pick at most one."

DP_SAFE = USE_DP

# Experiment 2's whole premise requires the head-only variant — a full-model
# sign-flip would poison the plaintext ("bulk") slice too, and Krum would
# just catch it the normal way, telling us nothing new.
BYZANTINE_HEAD_ONLY = True

# CKKS parameters for the partial (classifier-head-only) HE path, used by
# USE_HE_KRUM_HYBRID. Matches the "standard, non-RAM-constrained" config
# he_local.py already defines (n=8192, [60,40,40,60], scale=2**40) — same
# parameters main.py's old full-model USE_HE path used, so HE timing/
# security stays comparable to the earlier ablation numbers.
HE_POLY_DEGREE = 8192

# Layer 2 extension (Experiment 2 mitigation) — see defences/zkp.py Part 2.
# Only meaningful when USE_HE_KRUM_HYBRID=True; ignored otherwise.
USE_HEAD_NORM_GUARD = True
HEAD_NORM_GUARD_K = 2.5   # same default/semantics as ADAPTIVE_KRUM_K
HEAD_NORM_GUARD_MIN_KEEP_FRACTION = 0.5

DP_EPSILON       = _args.epsilon if _args.epsilon is not None else 15.0
DP_DELTA         = 1e-5
DP_MAX_GRAD_NORM = 1.5
DP_BATCH_SIZE    = 512

ZKP_MAX_NORM = 10.0

KRUM_M = NUM_CLIENTS - NUM_BYZANTINE - 1

ADAPTIVE_KRUM_K                 = _args.krum_k if _args.krum_k is not None else 2.5
# Raised from 2.5 -> 3.5. Confirmed via print_data_split() that the honest
# clients Krum was persistently excluding (network: clients 4, 10) simply
# hold 3-6x more data than the fleet median, with a heavily skewed class
# mix on top (clients 4+10 alone hold ~73% of the network model's entire
# DDoS_ICMP class) -- a legitimate, data-driven deviation, not an attack
# signature. 3.5 is a starting point to loosen the threshold past that
# effect while still catching the much larger deviation an actual
# sign-flip/scale attack produces -- verify against real run logs, not
# assumed correct on paper. Still overridable via --krum-k.
ADAPTIVE_KRUM_METHOD             = "mad"
ADAPTIVE_KRUM_MIN_KEEP_FRACTION  = 0.5

# In USE_HE_KRUM_HYBRID, Krum only ever scores clients that ALREADY survived
# the head-norm guard (see zkp.py Part 2) -- by the time Krum runs, the
# obvious ciphertext-visible attackers are gone. Scoring with
# num_byzantine=NUM_BYZANTINE (the GROUND-TRUTH attacker count, still needed
# elsewhere for detection-rate bookkeeping) means Krum's neighbour count
# (n-f-2) stays sized for a threat level that's already been mostly
# addressed by Layer 2 -- fewer neighbours per client sharpens sensitivity
# to any deviation, including ordinary non-IID variance among honest
# clients, which is the likely cause of the persistent 2-honest-client
# exclusion seen in every hybrid run so far (see RUN_NOTES for
# network_he_krum_hybrid_v1 / _norm_guard_v1). Lower this to reflect the
# smaller RESIDUAL threat the hybrid branch's Krum call actually needs to
# assume -- e.g. 1, to still catch an adaptive attacker that somehow slips
# past the norm guard, without being as aggressive as f=2 against a
# population that's already mostly been screened. Does NOT affect
# detection-rate bookkeeping (BYZANTINE_CLIENTS/NUM_BYZANTINE, used for
# ground truth, are untouched) -- only Krum's own internal neighbour math.
# Lowered from NUM_BYZANTINE (2) -> 1: the norm guard is now confirmed
# (both mitigated runs, 100% detection every round) to catch every
# ciphertext-visible attacker before Krum ever scores anything, so Krum's
# own math only needs to defensively assume ONE residual/adaptive threat
# might have slipped past it, not the full original attacker count.
# min(1, NUM_BYZANTINE) keeps this sane if NUM_BYZANTINE is ever 0.
ADAPTIVE_KRUM_HYBRID_ASSUMED_F  = min(1, NUM_BYZANTINE)

# ---------------------------------------------------------------------------
# Device / parallelization settings
# ---------------------------------------------------------------------------
_CPU_COUNT      = os.cpu_count() or 4
_CUDA_AVAILABLE = torch.cuda.is_available()
_DEVICE         = torch.device("cuda" if _CUDA_AVAILABLE else "cpu")

# GPU note: see changelog #19. When CUDA is available, no
# ProcessPoolExecutor is created at all — client training/eval runs
# sequentially in-process (see _run_training_wave/_run_eval_wave).
# CLIENT_POOL_WORKERS is kept as a reported/logged value (still 1 on
# GPU) even though no pool actually exists in that case.
CLIENT_POOL_WORKERS = 1 if _CUDA_AVAILABLE else min(4, NUM_CLIENTS)
_THREADS_PER_WORKER = max(1, _CPU_COUNT // CLIENT_POOL_WORKERS)

# ---------------------------------------------------------------------------
# Output paths — one set per model type so both can run simultaneously
# ---------------------------------------------------------------------------
_TAG               = MODEL_TYPE if _args.tag is None else f"{MODEL_TYPE}_{_args.tag}"
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
    # results — see master doc, Experiment 2 Prerequisites #4.
    from defences.krum import adaptive_multi_krum

if USE_HE_KRUM_HYBRID:
    from defences import he_local

if USE_HE_KRUM_HYBRID and USE_HEAD_NORM_GUARD:
    from defences import zkp

if USE_DP:
    try:
        from opacus import PrivacyEngine
        _OPACUS_AVAILABLE = True
    except ImportError:
        warnings.warn("Opacus not installed — USE_DP will be skipped. "
                      "Install with: pip install opacus")
        _OPACUS_AVAILABLE = False
else:
    _OPACUS_AVAILABLE = False

if USE_HE or USE_HE_KRUM_HYBRID:
    try:
        import tenseal as ts
        _TENSEAL_AVAILABLE = True
    except ImportError:
        raise ImportError("TenSEAL required for USE_HE/USE_HE_KRUM_HYBRID=True. "
                          "Install with Python 3.11: pip install tenseal")

_noise_multiplier_cache = {}


# ---------------------------------------------------------------------------
# ─── ROUND-LEVEL LEARNING RATE DECAY ────────────────────────────────────────
# ---------------------------------------------------------------------------

def get_round_lr(base_lr, round_num, num_rounds, min_lr_frac=0.15):
    progress = round_num / num_rounds
    decay = 0.5 * (1 + np.cos(np.pi * progress))
    return base_lr * (min_lr_frac + (1 - min_lr_frac) * decay)


def _apply_dp_safe_prox_step(real_model, global_dict, mu, lr):
    """
    Applies FedProx's proximal pull as a SEPARATE, non-privatized
    parameter update — not via loss.backward(). See changelog #20:
    Opacus's DPOptimizer builds its update entirely from .grad_sample,
    which the prox term never populates (it's a direct function of the
    parameter, not of any per-sample activation a hooked layer would
    capture) — so adding it to the loss under DP-SGD silently does
    nothing. This applies mu*(w - w_global) as a deterministic SGD
    step, decoupled from the clipped/noised data-gradient step. Safe:
    the prox term depends only on current params + last round's public
    global model, never on client data, so it costs zero privacy
    budget applied this way.
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
# ─── PARALLEL / SEQUENTIAL CLIENT TRAINING ──────────────────────────────────
# ---------------------------------------------------------------------------

def _pool_worker_init():
    """
    Runs once per worker process at pool startup — CPU-only path.
    Never invoked on GPU runs since no pool exists there (see #19).
    """
    import torch as _torch
    _torch.set_num_threads(_THREADS_PER_WORKER)


def _train_one_client(client_idx, X_tr, y_tr, global_params, client_cfg):
    """
    Called either via ProcessPoolExecutor (CPU) or directly in-process
    (GPU — see changelog #19). Signature/behavior identical either way.

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
        if (client_cfg["use_he"] or client_cfg["use_he_hybrid"]) and client_cfg["byzantine_head_only"]:
            # Stealthy variant: train normally on the FULL model first, so
            # the bulk/backbone slice looks like a real locally-computed
            # update -- not a frozen, unmodified copy of last round's
            # global model (which is what the old version returned, and
            # which Krum can trivially spot regardless of HE, since it's
            # just "this client didn't train," not "this client's
            # encrypted head is hiding something"). ONLY THEN overwrite
            # the classifier-head slice with the poisoned values. This is
            # what actually tests whether an encrypted classifier head
            # creates a blind spot for an otherwise-normal-looking client.
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
            # FIX (changelog #20): literature-standard attack -- train
            # normally first (same call an honest client would make: same
            # criterion, epochs, lr, global_params for the FedProx
            # proximal term, mu), THEN corrupt the RESULT. Previously this
            # called sign_flip_attack(global_params, ...) directly on the
            # untouched global model -- non-standard versus every
            # literature formulation checked. Mirrors
            # classifier_head_flip_attack's already-correct train-first
            # pattern above.
            #
            # ADDED (changelog #21): attack type is now selectable via
            # client_cfg["attack_type"] instead of sign-flip being the
            # only option. zero_gradient is the one exception to
            # "train first" -- it doesn't need the trained result at all
            # (a lazy client sending zeros IS the attack, independent of
            # what it would have computed), so it's applied directly to
            # global_params without a wasted training pass, per
            # zero_gradient_attack()'s own docstring.
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

            # Computed ONCE here, after BOTH branches above (cached or not)
            # have finished wrapping model/optimizer/loader — not duplicated
            # inside just one branch, since round 1 always takes the
            # "cached_sigma is None" path first.
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
    Called either via ProcessPoolExecutor (CPU) or directly in-process
    (GPU — see changelog #19).

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
    Runs _train_one_client() for all clients this round, either through
    the persistent ProcessPoolExecutor (CPU path) or as a plain
    sequential in-process loop (GPU path — executor is None). See
    changelog #19 for why the GPU path avoids the pool entirely.

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
# ─── AGGREGATION HELPERS ────────────────────────────────────────────────────
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


def he_aggregate(encrypted_params_list, context):
    if not _TENSEAL_AVAILABLE:
        raise RuntimeError("TenSEAL not available.")

    n = len(encrypted_params_list)
    summed = []
    for layer_idx in range(len(encrypted_params_list[0])):
        acc = encrypted_params_list[0][layer_idx].copy()
        for client_idx in range(1, n):
            acc += encrypted_params_list[client_idx][layer_idx]
        summed.append(acc)

    averaged = [layer * (1.0 / n) for layer in summed]
    return averaged


def zkp_verify_norm(params: list, max_norm: float = ZKP_MAX_NORM) -> bool:
    flat = np.concatenate([p.flatten() for p in params])
    norm = float(np.linalg.norm(flat))
    return norm <= max_norm


# ---------------------------------------------------------------------------
# ─── CHECKPOINT HELPERS ─────────────────────────────────────────────────────
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
    Separate checkpoint saved only when this round beats every prior
    round's F1-Macro this run — so the best round stays recoverable
    even if a later round degrades and overwrites the per-round
    checkpoint. This is exactly what was lost for the original locked
    baselines (round 20/22) — not repeating that here.
    """
    np.savez(CHECKPOINT_BEST_PARAMS, *global_params)
    with open(CHECKPOINT_BEST_PROGRESS, "w") as f:
        json.dump({"best_round": round_num, "best_f1_macro": float(f1_macro)}, f)

# ---------------------------------------------------------------------------
# ─── CSV LOGGING ────────────────────────────────────────────────────────────
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
# ─── MAIN TRAINING LOOP ─────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def main():
    # FIX (changelog #22): computed once, up front, so both the console
    # banner and the experiment_config JSON reflect the SAME actual attack
    # path -- previously the JSON hardcoded "sign_flip_attack_trained"
    # regardless of what really ran.
    if (USE_HE or USE_HE_KRUM_HYBRID) and BYZANTINE_HEAD_ONLY:
        _attack_function_label = "classifier_head_flip_attack"
    else:
        _attack_function_label = {
            "sign_flip":     "sign_flip_attack_trained",
            "gaussian":      "gaussian_attack_trained",
            "zero_gradient": "zero_gradient_attack",
        }[ATTACK_TYPE]

    print(f"\n{'='*65}")
    print(f"  FL-IDS Unified Loop — MODEL: {MODEL_TYPE.upper()}")
    if SANITY_CHECK:
        print(f"  *** SANITY_CHECK MODE — {NUM_ROUNDS} rounds only ***")
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
        print(f"  Client training: SEQUENTIAL, in-process (no worker pool — "
              f"see changelog #19, avoids fork+CUDA hang)")
    else:
        print(f"  Parallel client training: {CLIENT_POOL_WORKERS} worker(s), "
              f"{_THREADS_PER_WORKER} threads/worker "
              f"({_CPU_COUNT} cores detected)")
    if USE_DP:
        print(f"  DP: ε={DP_EPSILON}  δ={DP_DELTA}  "
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
    if USE_HE_KRUM_HYBRID:
        print(f"  HE+Krum Hybrid: adaptive Krum (method={ADAPTIVE_KRUM_METHOD}  "
              f"k={ADAPTIVE_KRUM_K}  assumed_f={ADAPTIVE_KRUM_HYBRID_ASSUMED_F}"
              f"{' [ground-truth NUM_BYZANTINE=' + str(NUM_BYZANTINE) + ']' if ADAPTIVE_KRUM_HYBRID_ASSUMED_F != NUM_BYZANTINE else ''}"
              f") scores the PLAINTEXT (bulk) slice only; "
              f"classifier-head slice (CKKS, poly_degree={HE_POLY_DEGREE}) is "
              f"aggregated only over whichever clients that scoring selects.")
        print(f"  Byzantine head-only attack: {BYZANTINE_HEAD_ONLY} "
              f"(should be True — this is the whole point of Experiment 2)")
        print(f"  Head-norm guard (Layer 2 extension): {USE_HEAD_NORM_GUARD} "
              f"(k={HEAD_NORM_GUARD_K}, min_keep_fraction="
              f"{HEAD_NORM_GUARD_MIN_KEEP_FRACTION}) — ciphertext-bound "
              f"MAD threshold on classifier-head delta norms, runs BEFORE "
              f"Krum each round")
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
        Printed at the start of every round (not just once) so it sits
        right next to that round's Krum decision in the log — no
        scrolling back to correlate "did client X get excluded because
        its partition is small/skewed" with what Krum actually did. The
        partition itself is fixed at load time and doesn't change
        round-to-round; reprinting every round is a deliberate log-
        readability choice, not new computation of any real cost.
        """
        print("  ── Data split (train partition, per client) ──")
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
    print("Criterion built — workers will reuse this, no per-round reload.\n")

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

    he_context = None
    if USE_HE and _TENSEAL_AVAILABLE:
        he_context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        he_context.global_scale = 2 ** 40
        he_context.generate_galois_keys()
        print("TenSEAL CKKS context initialised.\n")

    if USE_HE_KRUM_HYBRID and _TENSEAL_AVAILABLE:
        # he_local.create_ckks_context() holds the secret key — fine here
        # since this is a single-process simulation with no untrusted
        # server/client network boundary (same reasoning as he_local.py's
        # own docstring). Only the classifier-head slice will ever be
        # encrypted under it — see split_sensitive_bulk()/SENSITIVE_PREFIX.
        he_context = he_local.create_ckks_context(HE_POLY_DEGREE)
        print(f"Partial-HE (classifier-head-only) CKKS context initialised "
              f"via he_local.create_ckks_context (poly_degree={HE_POLY_DEGREE}).\n")

    # Model state_dict key order, needed to split each client's flat param
    # list into "sensitive" (classifier.*) vs "bulk" layers for the hybrid
    # branch's encryption/Krum split. Built once here, off a throwaway
    # model instance — identical order to every client's own model since
    # architecture + dp_safe are fixed for the whole run.
    MODEL_STATE_KEYS = None
    if USE_HE_KRUM_HYBRID:
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
        print("  NOTE: if you changed DP_EPSILON, USE_KRUM, USE_ADAPTIVE_KRUM, "
              "USE_HE, or any other experiment flag since the last run, delete "
              f"{CHECKPOINT_PARAMS} and {CHECKPOINT_PROGRESS} before "
              "continuing — resuming across different experiment "
              "conditions silently contaminates round-1 comparability.\n")

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
            "use_head_norm_guard": USE_HEAD_NORM_GUARD,
            "head_norm_guard_k": HEAD_NORM_GUARD_K if USE_HEAD_NORM_GUARD else None,
            "head_norm_guard_min_keep_fraction": HEAD_NORM_GUARD_MIN_KEEP_FRACTION if USE_HEAD_NORM_GUARD else None,
            "he_poly_degree": HE_POLY_DEGREE if (USE_HE or USE_HE_KRUM_HYBRID) else None,
            "use_dp": USE_DP,
            "dp_epsilon": DP_EPSILON,
            "dp_delta": DP_DELTA,
            "dp_max_grad_norm": DP_MAX_GRAD_NORM,
            "dp_batch_size": DP_BATCH_SIZE,
            "dp_accountant": "rdp",
            "use_zkp": USE_ZKP,
            "zkp_max_norm": ZKP_MAX_NORM,
            "byzantine_head_only": BYZANTINE_HEAD_ONLY,
            "dp_safe": DP_SAFE,
            "device": str(_DEVICE),
            "cuda_available": _CUDA_AVAILABLE,
            "client_pool_workers": CLIENT_POOL_WORKERS,
            "threads_per_worker": _THREADS_PER_WORKER,
            "framework": "custom Python simulation (direct, parallel client training)",
        }, f, indent=2)

    # ════════════════════════════════════════════════════════════════════════
    # ─── ROUND LOOP ──────────────────────────────────────────────────────────
    # GPU: no pool at all (executor stays None throughout — see #19).
    # CPU: original persistent 4-way ProcessPoolExecutor, unchanged.
    # ════════════════════════════════════════════════════════════════════════
    pool_cm = (
        contextlib.nullcontext()
        if _CUDA_AVAILABLE
        else ProcessPoolExecutor(max_workers=CLIENT_POOL_WORKERS,
                                 initializer=_pool_worker_init)
    )

    with pool_cm as executor:
        # nullcontext()'s __enter__ returns None by default — executor
        # is None on GPU runs, a real ProcessPoolExecutor on CPU runs.
        # _run_training_wave/_run_eval_wave branch on this.

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
                    if (USE_HE or USE_HE_KRUM_HYBRID) and BYZANTINE_HEAD_ONLY:
                        tag = "head-only"
                    elif ATTACK_TYPE == "gaussian":
                        tag = "gaussian (trained)"
                    elif ATTACK_TYPE == "zero_gradient":
                        tag = "zero-gradient"
                    else:
                        tag = "sign-flip (trained)"
                    print(f"  Client {i+1:2d}  [BYZANTINE — {tag} ×{ATTACK_SCALE}]")

                if USE_ZKP:
                    passes = zkp_verify_norm(params, max_norm=ZKP_MAX_NORM)
                    if not passes:
                        print(f"  Client {i+1:2d}  [ZKP REJECTED — norm too large]")
                        zkp_rejected_this_round.append(i)
                        continue

                if USE_HE and _TENSEAL_AVAILABLE and he_context is not None:
                    enc_params = [
                        ts.ckks_vector(he_context, p.flatten().tolist())
                        for p in params
                    ]
                    accepted_params.append(enc_params)
                elif USE_HE_KRUM_HYBRID and _TENSEAL_AVAILABLE and he_context is not None:
                    # Only the classifier-head ("sensitive") layers get
                    # CKKS-encrypted; everything else ("bulk") stays plain
                    # numpy, exactly what the Krum branch below needs to
                    # be able to score. Returns a dict — see he_local.
                    if USE_HEAD_NORM_GUARD:
                        client_enc = he_local.encrypt_params_with_norm_guard(
                            params, MODEL_STATE_KEYS, he_context, HE_POLY_DEGREE,
                            global_params
                        )
                    else:
                        client_enc = he_local.encrypt_params(
                            params, MODEL_STATE_KEYS, he_context, HE_POLY_DEGREE
                        )
                    if round_num == start_round + 1 and len(accepted_params) == 0:
                        print(f"  [HE hybrid] {client_enc['pct_encrypted']:.1f}% of "
                              f"params encrypted (classifier head), rest plaintext "
                              f"(bulk) — measured once, client {i+1}.")
                    accepted_params.append(client_enc)
                else:
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
                print("  WARNING: All clients rejected — skipping round.")
                save_checkpoint(global_params, round_num)
                continue

            if USE_HE and _TENSEAL_AVAILABLE:
                global_params = he_aggregate(accepted_params, he_context)
                agg_label = "HE"

            elif USE_HE_KRUM_HYBRID and _TENSEAL_AVAILABLE:
                # accepted_params here is a list of he_local.encrypt_params()
                # output dicts (one per accepted client): {"sensitive_enc":
                # <encrypted classifier head>, "bulk": <plaintext everything
                # else>, "sensitive_idx"/"bulk_idx": layer index mappings}.

                # ── Layer 2 extension: ciphertext-bound head-norm guard ──
                # Pre-filters BEFORE Krum ever runs, on LOCAL copies —
                # never mutates the shared accepted_params/weights/indices
                # lists other branches/logging rely on. See defences/zkp.py
                # Part 2 for the full design and its disclosed limits.
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
                    agg_label = ("HE+Krum hybrid (fallback — too few "
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
                if len(accepted_params) - NUM_BYZANTINE - 2 < 1:
                    global_params = fedprox_aggregate(accepted_params,
                                                      accepted_weights)
                    agg_label = "FedProx (Adaptive-Krum fallback — too few accepted clients)"
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

            _krum_active = USE_KRUM or USE_ADAPTIVE_KRUM or USE_HE_KRUM_HYBRID

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
                    f"(round {round_num}) → {CHECKPOINT_BEST_PARAMS}")
            round_time = time.time() - round_start

            print(f"  Loss: {mean_loss:.4f}  Acc: {mean_acc:.4f}  "
                  f"F1-Macro: {mean_f1.mean():.4f}  [{round_time:.1f}s]")
            print("  Per-class F1:")
            for name, f1 in zip(ATTACK_NAMES, mean_f1):
                bar = "█" * int(f1 * 20)
                print(f"    {name:<28} {f1:.4f}  {bar}")
            print()

            krum_detection_rate = (
                len(krum_detected_byz) / NUM_BYZANTINE
                if (_krum_active and NUM_BYZANTINE > 0) else None
            )

            if _krum_active and krum_detection_rate is not None:
                krum_label = ("Krum" if USE_KRUM
                             else "HE+Krum Hybrid (plaintext-slice)" if USE_HE_KRUM_HYBRID
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
    print(f"  Training complete — {NUM_ROUNDS} rounds  [{MODEL_TYPE.upper()}]")
    if SANITY_CHECK:
        print(f"  *** This was a SANITY_CHECK run ({NUM_ROUNDS} rounds). ***")
        print(f"  *** Set SANITY_CHECK=False and delete the checkpoint before ***")
        print(f"  *** starting the real sweep. ***")
    print(f"  Results logged to:     {LOG_CSV}")
    print(f"  Checkpoint:            {CHECKPOINT_PARAMS} (round {NUM_ROUNDS})")
    if USE_KRUM or USE_ADAPTIVE_KRUM or USE_HE_KRUM_HYBRID:
        print(f"\n  Reminder: delete checkpoint before changing flags")
        print(f"  (Krum/Adaptive-Krum/HE/HE-Krum-Hybrid/DP flags change the")
        print(f"  experiment — old checkpoint params will give misleading")
        print(f"  results if reused.)")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()