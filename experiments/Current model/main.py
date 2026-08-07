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

17. GPU DEVICE SUPPORT — added.
    - `torch` is now imported at module level (was previously only
      imported lazily inside main()/_train_one_client()) so device
      detection can happen before CLIENT_POOL_WORKERS is decided.
    - `_DEVICE` is resolved once via torch.cuda.is_available().
    - CLIENT_POOL_WORKERS now defaults to 1 (sequential) whenever CUDA
      is available, instead of the CPU-oriented 4-way ProcessPoolExecutor
      pool. Rationale: 4 separate processes each opening their own CUDA
      context on a single GPU causes memory contention and context-switch
      overhead — this is typically SLOWER than sequential GPU training,
      not faster. CPU-only runs keep the original 4-way pool unchanged.
    - `device` is threaded through client_cfg / eval_cfg and both
      _train_one_client() and _eval_one_client() now move the model
      (and, for DP-SGD, each batch) onto that device.
    - `build_criterion()` is now called with `device=_DEVICE`.

    ASSUMPTION THIS RELIES ON, NOT YET VERIFIED HERE: task.py's
    `train()`, `test()`, `build_criterion_network()`, and
    `build_criterion_application()` accept a `device` kwarg and move
    their internal tensors (e.g. FocalLoss class-weight tensor) onto
    it. task.py was not included in the files provided for this edit —
    if it does NOT yet have device-aware signatures, the calls below
    will raise a TypeError (unexpected keyword argument 'device') or,
    worse, silently run with a CPU-resident weight tensor against
    GPU-resident logits and raise a device-mismatch RuntimeError at
    loss computation. Send task.py over to get it patched to match.

18. SANITY_CHECK toggle added — flip to True for a quick 2-round
    end-to-end run (confirm nvidia-smi shows GPU utilization, confirm
    no device-mismatch crashes, get a real round-time number) before
    committing to the full 25-round / 4-epsilon sweep. Flip back to
    False for the real run — do not leave this True by accident.

--------------------------------------------------------------------------
KNOWN OPEN ITEMS — NOT YET RESOLVED, FLAGGED FOR NEXT REVISION
--------------------------------------------------------------------------
- PROX_MU is 0.02 here (user-confirmed intended value).
- LR decay disabled (user-confirmed decision) — get_round_lr() kept but unused.
- USE_ADAPTIVE_KRUM=True is a deliberate deviation from the master planning
  doc's "Experiment 1 must use fixed-m Krum" instruction (user decision) —
  any comparison against a fixed-m Condition 3 anchor is not apples-to-apples.
- Prerequisites 4-6 from the Experiment-1 checklist were manually verified
  against model_defs.py / task.py / data_loader.py in conversation — not
  re-derived from this diff alone.
- task.py device-awareness is ASSUMED, not verified in this revision — see
  changelog #17 above. Confirm before running on GPU.
- Whether ProcessPoolExecutor should be dropped entirely in favor of plain
  sequential in-process training when CUDA is available (rather than a
  1-worker pool) is still an open question — a 1-worker pool avoids a
  second CUDA context but still pays process-spawn/IPC overhead per
  client per round that a plain for-loop wouldn't. Left as a pool with
  max_workers=1 for now since it's a minimal, low-risk change; revisit
  if per-client IPC overhead turns out to matter at these round times.
--------------------------------------------------------------------------
"""

import os
import sys
import csv
import json
import time
import warnings
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

MODEL_TYPE    = sys.argv[1] if len(sys.argv) > 1 else "network"
assert MODEL_TYPE in ("network", "application"), \
    "Usage: python main.py [network|application]"

# ── Sanity-check toggle ──────────────────────────────────────────────────
# Set to True to force a short 2-round run. Recommended before committing
# to a full 25-round GPU run: confirms nvidia-smi shows GPU utilization
# climbing during training, confirms no device-mismatch crashes, and gives
# a real round-time number cheaply (minutes, not hours) before starting
# the actual Experiment 1 sweep. Set back to False for real runs.
SANITY_CHECK = True

# FL hyperparameters
NUM_ROUNDS    = 2 if SANITY_CHECK else 25
NUM_CLIENTS   = 10
LOCAL_EPOCHS  = 5
LEARNING_RATE = 0.001
PROX_MU       = 0.02       # FedProx proximal coefficient (0 = plain FedAvg)
# NOTE: if this doesn't match the mu used to generate your reference
# baselines, the gap you attribute to DP noise (or anything else) in
# the writeup partly reflects a mu mismatch. Confirm against
# experiment_config_*.json from the run you're comparing against.

# Byzantine attack
USE_BYZANTINE_ATTACK = True
NUM_BYZANTINE        = 2
BYZANTINE_CLIENTS    = list(range(NUM_BYZANTINE))   # clients 0 and 1 are malicious
ATTACK_SCALE         = 5.0 if MODEL_TYPE == "network" else 2.0

# ─── Defence flags ──────────────────────────────────────────────────────────
# Experiment 1 (Krum path):          USE_KRUM=True,          USE_HE=False
# Experiment 1b (Adaptive-Krum path): USE_ADAPTIVE_KRUM=True, USE_HE=False
# Experiment 2 (HE path):             USE_HE=True
# Ablation (no defence):              all three False
USE_KRUM          = False
USE_ADAPTIVE_KRUM = True   # USER DECISION: running the epsilon sweep with
                            # adaptive (MAD-threshold) Krum instead of fixed-m.
                            # See defences/krum.py::adaptive_multi_krum.
USE_HE   = False          # CKKS via TenSEAL — set True for Experiment 2

USE_DP   = True           # Opacus per-round DP-SGD — epsilon sweep active
USE_ZKP  = False          # lightweight norm-bound ZKP gate

assert sum([USE_KRUM, USE_ADAPTIVE_KRUM, USE_HE]) <= 1, \
    "USE_KRUM, USE_ADAPTIVE_KRUM, and USE_HE are mutually exclusive aggregation " \
    "branches — pick at most one."

# DP_SAFE must match USE_DP — architecture (BatchNorm→GroupNorm, LSTM→DPLSTM
# with dp_safe=True) must be consistent across checkpoint init, training,
# and eval or set_model_parameters will fail on mismatched state_dict keys.
DP_SAFE = USE_DP

# Head-only attack: flips only classifier weights, stays within ZKP norm bound.
BYZANTINE_HEAD_ONLY = False   # set True for Experiment 2 Condition B

# DP settings (per-round; not composition-tracked — see epsilon sweep for
# the ε study). Accountant is RDP throughout — see PrivacyEngine below.
DP_EPSILON       = 15.0        # set per condition: 15.0 / 9.0 / 3.0
DP_DELTA         = 1e-5
DP_MAX_GRAD_NORM = 1.5
DP_BATCH_SIZE    = 512

# ZKP settings
ZKP_MAX_NORM = 10.0       # reject clients whose update L2-norm exceeds this

# Fixed-m Krum settings
KRUM_M = NUM_CLIENTS - NUM_BYZANTINE - 1   # e.g. 7 of 10 selected, 3 discarded

# Adaptive Multi-Krum settings
ADAPTIVE_KRUM_K                 = 2.5
ADAPTIVE_KRUM_METHOD             = "mad"   # "mad" (robust) or "zscore" (ablation only)
ADAPTIVE_KRUM_MIN_KEEP_FRACTION  = 0.5

# ---------------------------------------------------------------------------
# Device / parallelization settings
# ---------------------------------------------------------------------------
_CPU_COUNT      = os.cpu_count() or 4
_CUDA_AVAILABLE = torch.cuda.is_available()
_DEVICE         = torch.device("cuda" if _CUDA_AVAILABLE else "cpu")

# GPU note: ProcessPoolExecutor's 4-worker pool was designed for
# CPU-parallel client training (each of 4 processes uses its own CPU
# threads). On a single GPU, 4 separate processes would each open their
# own CUDA context on the same device — this causes memory contention
# and context-switch overhead and is typically SLOWER than sequential
# GPU training, not faster. Default to sequential (1 worker) client
# training whenever CUDA is available; CPU-only runs keep the original
# 4-way pool.
CLIENT_POOL_WORKERS = 1 if _CUDA_AVAILABLE else min(4, NUM_CLIENTS)
_THREADS_PER_WORKER = max(1, _CPU_COUNT // CLIENT_POOL_WORKERS)

# ---------------------------------------------------------------------------
# Output paths — one set per model type so both can run simultaneously
# ---------------------------------------------------------------------------
_TAG               = MODEL_TYPE
CHECKPOINT_PARAMS  = f"checkpoint_{_TAG}.npz"
CHECKPOINT_PROGRESS= f"checkpoint_{_TAG}_progress.json"
LOG_CSV            = f"results_{_TAG}.csv"

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

from defences.byzantine import sign_flip_attack

if USE_KRUM:
    from defences.krum import multi_krum

if USE_ADAPTIVE_KRUM:
    from defences.krum import adaptive_multi_krum

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

if USE_HE:
    try:
        import tenseal as ts
        _TENSEAL_AVAILABLE = True
    except ImportError:
        raise ImportError("TenSEAL required for USE_HE=True. "
                          "Install with Python 3.11: pip install tenseal")

# ── Per-worker-process noise_multiplier cache — see prior revision #16 ─────
# Keyed by (client_idx, dp_epsilon, dp_delta, local_epochs, dp_batch_size,
# dp_max_grad_norm, dataset_size). Lives at module level so it persists
# across rounds WITHIN one worker process (the pool is created once and
# reused for the whole run) but is naturally fresh in each new run.
_noise_multiplier_cache = {}


# ---------------------------------------------------------------------------
# ─── ROUND-LEVEL LEARNING RATE DECAY ────────────────────────────────────────
# ---------------------------------------------------------------------------

def get_round_lr(base_lr, round_num, num_rounds, min_lr_frac=0.15):
    """
    Cosine-decays the CLIENT-SIDE learning rate across FL rounds.
    Currently UNUSED — LR decay disabled per user decision (see
    KNOWN OPEN ITEMS at top of file). Left defined in case a future
    experiment wants it back.
    """
    progress = round_num / num_rounds
    decay = 0.5 * (1 + np.cos(np.pi * progress))
    return base_lr * (min_lr_frac + (1 - min_lr_frac) * decay)


# ---------------------------------------------------------------------------
# ─── PARALLEL CLIENT TRAINING ───────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _pool_worker_init():
    """
    Runs once per worker process at pool startup (not per task). Caps
    this worker's torch thread usage so CLIENT_POOL_WORKERS processes
    don't each independently try to claim every core. On GPU runs
    CLIENT_POOL_WORKERS is 1, so this mainly matters for CPU-only runs.
    """
    import torch as _torch
    _torch.set_num_threads(_THREADS_PER_WORKER)


def _train_one_client(client_idx, X_tr, y_tr, global_params, client_cfg):
    """
    Top-level (module-level) function — REQUIRED for Windows
    multiprocessing (spawn pickles this callable).

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

    # ── Byzantine attack injection ───────────────────────────────────
    # (operates on the numpy global_params directly — never touches the
    # model object above, so no device handling needed on this branch)
    if client_cfg["use_byzantine_attack"] and client_idx in client_cfg["byzantine_clients"]:
        if client_cfg["use_he"] and client_cfg["byzantine_head_only"]:
            from defences.byzantine import classifier_head_flip_attack
            model_state_keys = list(model.state_dict().keys())
            params = classifier_head_flip_attack(
                global_params, model_state_keys, scale=client_cfg["attack_scale"]
            )
        else:
            params = sign_flip_attack(global_params, scale=client_cfg["attack_scale"])

    else:
        # ── DP-SGD training (Opacus, RDP accountant) ──────────────────
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

            # ── Noise-multiplier calibration cache ─────────────────────
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

            model.train()
            for _ in range(client_cfg["local_epochs"]):
                for X_b, y_b in loader:
                    # DataLoader always yields CPU tensors regardless of
                    # what device the source TensorDataset was built
                    # from — move each batch explicitly.
                    X_b = X_b.to(device)
                    y_b = y_b.to(device)
                    optimizer.zero_grad()
                    loss_val = criterion(model(X_b), y_b)
                    loss_val.backward()
                    optimizer.step()

            dp_eps_spent = privacy_engine.get_epsilon(client_cfg["dp_delta"])

            real_model = model._module if hasattr(model, "_module") else model
            params = get_model_parameters(real_model)  # already .cpu().numpy()'d

        else:
            # Standard FedProx training (no DP)
            # NOTE: assumes task.py's train() accepts a `device` kwarg —
            # see changelog #17 at top of file. Not verified here since
            # task.py wasn't included with this edit.
            criterion = client_cfg["criterion"]
            train(model, X_tr, y_tr, criterion,
                  epochs=client_cfg["local_epochs"],
                  lr=client_cfg["learning_rate"],
                  global_params=global_params,
                  mu=client_cfg["prox_mu"],
                  device=device)
            params = get_model_parameters(model)  # already .cpu().numpy()'d

    return client_idx, params, dp_eps_spent, dp_noise_multiplier


def _eval_one_client(client_idx, global_params, X_te, y_te, eval_cfg):
    """
    Top-level (module-level) function, mirrors _train_one_client's
    picklability requirement. Rebuilds the eval model fresh in the
    worker and runs test() there instead of sequentially in the main
    process. Model is moved to eval_cfg["device"] before evaluation.

    Returns (client_idx, loss, accuracy, per_class_f1).
    """
    device = eval_cfg.get("device", "cpu")

    model = get_model(num_features=eval_cfg["sample_features"],
                      num_classes=eval_cfg["num_classes"],
                      dp_safe=eval_cfg["dp_safe"])
    set_model_parameters(model, global_params)
    model = model.to(device)

    # NOTE: assumes task.py's test() accepts a `device` kwarg — see
    # changelog #17 at top of file. Not verified here since task.py
    # wasn't included with this edit.
    loss_v, acc_v, f1_per_class = test(model, X_te, y_te,
                                       eval_cfg["num_classes"],
                                       device=device)
    return client_idx, loss_v, acc_v, f1_per_class


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
    print(f"\n{'='*65}")
    print(f"  FL-IDS Unified Loop — MODEL: {MODEL_TYPE.upper()}")
    if SANITY_CHECK:
        print(f"  *** SANITY_CHECK MODE — {NUM_ROUNDS} rounds only ***")
    print(f"  Rounds={NUM_ROUNDS}  Clients={NUM_CLIENTS}  Epochs={LOCAL_EPOCHS}")
    print(f"  Device={_DEVICE}  (CUDA available: {_CUDA_AVAILABLE})")
    print(f"  Byzantine={NUM_BYZANTINE} (clients {BYZANTINE_CLIENTS})  "
          f"Attack={'ON' if USE_BYZANTINE_ATTACK else 'OFF'}")
    print(f"  USE_KRUM={USE_KRUM}  USE_ADAPTIVE_KRUM={USE_ADAPTIVE_KRUM}  "
          f"USE_HE={USE_HE}  USE_DP={USE_DP}  USE_ZKP={USE_ZKP}")
    print(f"  Parallel client training: {CLIENT_POOL_WORKERS} worker(s), "
          f"{_THREADS_PER_WORKER} threads/worker "
          f"({_CPU_COUNT} cores detected)"
          + ("  [sequential — GPU run]" if _CUDA_AVAILABLE else ""))
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
    print(f"{'='*65}\n")

    torch.set_num_threads(_CPU_COUNT)

    # ── Load data partitions ─────────────────────────────────────────────────
    print("Loading data partitions...")
    clients_data = []
    for i in range(NUM_CLIENTS):
        print(f"  Partition {i+1}/{NUM_CLIENTS}...", end="\r")
        clients_data.append(load_partition(i, NUM_CLIENTS))
    sample_features = clients_data[0][0].shape[1]
    print(f"\nFeature count (measured, not assumed): {sample_features}")
    print(f"All {NUM_CLIENTS} clients loaded.\n")

    # Criterion (FocalLoss + class weights) never changes across rounds or
    # clients — build it ONCE here, in the main process, and hand it to
    # every worker via client_cfg. Passed device=_DEVICE so its internal
    # class-weight tensor lives on the same device the model/batches will
    # be on — see changelog #17's ASSUMPTION note re: task.py.
    print("Building criterion once (class weights, FocalLoss)...")
    precomputed_criterion = build_criterion().to(_DEVICE)
    print("Criterion built — workers will reuse this, no per-round reload.\n")

    # ── Static per-client config, built once, passed to every worker call ──
    client_cfg = {
        "sample_features":      sample_features,
        "num_classes":          NUM_CLASSES,
        "dp_safe":              DP_SAFE,
        "use_byzantine_attack": USE_BYZANTINE_ATTACK,
        "criterion":            precomputed_criterion,
        "byzantine_clients":    BYZANTINE_CLIENTS,
        "attack_scale":         ATTACK_SCALE,
        "use_he":                USE_HE,
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

    # ── HE context (only if USE_HE) ──────────────────────────────────────────
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

    # ── Checkpoint / resume ──────────────────────────────────────────────────
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
            "use_krum": USE_KRUM,
            "krum_m": KRUM_M,
            "krum_discards": NUM_CLIENTS - KRUM_M,
            "use_adaptive_krum": USE_ADAPTIVE_KRUM,
            "adaptive_krum_k": ADAPTIVE_KRUM_K,
            "adaptive_krum_method": ADAPTIVE_KRUM_METHOD,
            "adaptive_krum_min_keep_fraction": ADAPTIVE_KRUM_MIN_KEEP_FRACTION,
            "use_he": USE_HE,
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
    # ─── ROUND LOOP — pool created ONCE, reused for every round ─────────────
    # ════════════════════════════════════════════════════════════════════════
    with ProcessPoolExecutor(max_workers=CLIENT_POOL_WORKERS,
                              initializer=_pool_worker_init) as executor:

        for round_num in range(start_round + 1, NUM_ROUNDS + 1):
            round_start = time.time()
            print(f"[ROUND {round_num}/{NUM_ROUNDS}]")

            # LR decay disabled (user decision) — flat client_cfg every round.
            round_client_cfg = client_cfg

            accepted_params          = []
            accepted_weights         = []
            accepted_client_indices  = []

            zkp_rejected_this_round  = []
            dp_eps_spent_this_round  = []
            dp_noise_mult_this_round = []

            # ── Submit all clients to the persistent pool ───────────────────
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

            # ── Sequential bookkeeping, in original client order ────────────
            for i, (X_tr, y_tr, X_te, y_te) in enumerate(clients_data):
                params, dp_eps_spent, dp_noise_mult = results_by_client[i]

                if USE_BYZANTINE_ATTACK and i in BYZANTINE_CLIENTS:
                    tag = "head-only" if (USE_HE and BYZANTINE_HEAD_ONLY) else "sign-flip"
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
                else:
                    accepted_params.append(params)

                accepted_weights.append(len(X_tr))
                accepted_client_indices.append(i)
                if dp_eps_spent is not None:
                    dp_eps_spent_this_round.append(dp_eps_spent)
                if dp_noise_mult is not None:
                    dp_noise_mult_this_round.append(dp_noise_mult)

            # ── Aggregation branch ───────────────────────────────────────────
            krum_selected_ids   = set()
            krum_discarded_ids  = set()
            krum_detected_byz   = set()
            krum_score_diag     = None

            if len(accepted_params) == 0:
                print("  WARNING: All clients rejected — skipping round.")
                save_checkpoint(global_params, round_num)
                continue

            if USE_HE and _TENSEAL_AVAILABLE:
                global_params = he_aggregate(accepted_params, he_context)
                agg_label = "HE"

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

            # ── Evaluation — PARALLELIZED across the persistent pool ────────
            # Each worker rebuilds its own eval model from global_params and
            # moves it to eval_cfg["device"].
            _krum_active = USE_KRUM or USE_ADAPTIVE_KRUM

            eval_futures = {
                executor.submit(
                    _eval_one_client, i, global_params, X_te, y_te, eval_cfg
                ): i
                for i, (X_tr, y_tr, X_te, y_te) in enumerate(clients_data)
            }
            eval_results_by_client = {}
            for future in as_completed(eval_futures):
                client_idx, loss_v, acc_v, f1_per_class = future.result()
                eval_results_by_client[client_idx] = (loss_v, acc_v, f1_per_class)

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
                krum_label = "Krum" if USE_KRUM else "Adaptive Krum"
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
                byz_scores, honest_scores = [], []
                for pos, orig_id in enumerate(accepted_client_indices):
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

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"  Training complete — {NUM_ROUNDS} rounds  [{MODEL_TYPE.upper()}]")
    if SANITY_CHECK:
        print(f"  *** This was a SANITY_CHECK run ({NUM_ROUNDS} rounds). ***")
        print(f"  *** Set SANITY_CHECK=False and delete the checkpoint before ***")
        print(f"  *** starting the real sweep. ***")
    print(f"  Results logged to:     {LOG_CSV}")
    print(f"  Checkpoint:            {CHECKPOINT_PARAMS} (round {NUM_ROUNDS})")
    if USE_KRUM or USE_ADAPTIVE_KRUM:
        print(f"\n  Reminder: delete checkpoint before changing flags")
        print(f"  (Krum/Adaptive-Krum/HE/DP flags change the experiment — old")
        print(f"  checkpoint params will give misleading results if reused.)")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()