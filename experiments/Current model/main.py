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

19. FIX — fork+CUDA hang. The GPU sanity-check run hung indefinitely at
    round 1 with 0% CPU and 0% GPU utilization on the worker process —
    confirmed via `top` (worker process essentially idle, not doing
    kernel-JIT-compile CPU work) rather than crashing outright. Root
    cause: ProcessPoolExecutor's worker was still being created via
    Linux's default 'fork' start method, AFTER CUDA had already been
    initialized in the main process (torch.cuda.is_available() runs at
    module import time, before the pool exists). Forking a process
    that already holds an active CUDA context hands the child a
    half-initialized, unsafe context — a well-known PyTorch/CUDA
    footgun that hangs rather than erroring.

    FIX: when CUDA is available, the ProcessPoolExecutor is no longer
    created at all — client training/eval for each round now runs via
    a plain sequential in-process loop (see _run_training_wave() /
    _run_eval_wave() below), calling _train_one_client()/
    _eval_one_client() directly with no subprocess involved. This is
    strictly safer than trying to force 'spawn' as an alternative fix,
    and also resolves the "revisit if per-client IPC overhead turns
    out to matter" open item from revision 17 — at CLIENT_POOL_WORKERS=1
    there was zero parallelism benefit from the pool anyway, only
    IPC/pickling overhead and, as it turned out, an actual hang risk.
    CPU-only runs are UNCHANGED — still use the original 4-way
    ProcessPoolExecutor pool (fork is safe there since no CUDA context
    ever exists in the parent process).

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
NUM_BYZANTINE        = 2
BYZANTINE_CLIENTS    = list(range(NUM_BYZANTINE))   # clients 0 and 1 are malicious
ATTACK_SCALE         = 5.0 if MODEL_TYPE == "network" else 2.0

# ─── Defence flags ──────────────────────────────────────────────────────────
USE_KRUM          = False
USE_ADAPTIVE_KRUM = True
USE_HE   = False

USE_DP   = True
USE_ZKP  = False

assert sum([USE_KRUM, USE_ADAPTIVE_KRUM, USE_HE]) <= 1, \
    "USE_KRUM, USE_ADAPTIVE_KRUM, and USE_HE are mutually exclusive aggregation " \
    "branches — pick at most one."

DP_SAFE = USE_DP

BYZANTINE_HEAD_ONLY = False

DP_EPSILON       = _args.epsilon if _args.epsilon is not None else 15.0
DP_DELTA         = 1e-5
DP_MAX_GRAD_NORM = 1.5
DP_BATCH_SIZE    = 512

ZKP_MAX_NORM = 10.0

KRUM_M = NUM_CLIENTS - NUM_BYZANTINE - 1

ADAPTIVE_KRUM_K                 = 2.5
ADAPTIVE_KRUM_METHOD             = "mad"
ADAPTIVE_KRUM_MIN_KEEP_FRACTION  = 0.5

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
        if client_cfg["use_he"] and client_cfg["byzantine_head_only"]:
            from defences.byzantine import classifier_head_flip_attack
            model_state_keys = list(model.state_dict().keys())
            params = classifier_head_flip_attack(
                global_params, model_state_keys, scale=client_cfg["attack_scale"]
            )
        else:
            params = sign_flip_attack(global_params, scale=client_cfg["attack_scale"])

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

            _krum_active = USE_KRUM or USE_ADAPTIVE_KRUM

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
