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
1-5. (see previous revisions — krum_detected truthy fix, KRUM_M=7,
      measured feature count logging, DP_MAX_GRAD_NORM=1.5, params
      extraction UnboundLocalError fix)

6. Client training parallelized across processes.
   Per-client training logic moved out of the round loop into a
   top-level function, _train_one_client(). This is REQUIRED (not a
   style choice) for Windows: the default multiprocessing start method
   there is 'spawn', which pickles the target callable to send to each
   worker process, and Python cannot pickle a function defined inside
   another function (a closure) — only top-level, module-level
   functions are picklable. The round loop now submits all NUM_CLIENTS
   client-training jobs to a process pool and collects results before
   doing any ZKP/HE/accepted-list bookkeeping.

7. Pool is created ONCE, outside the round loop, not per-round.
   An earlier version of this change put `with ProcessPoolExecutor(...)`
   INSIDE the round loop, meaning 4 worker processes were spawned and
   torn down on every single round (20x for a 25-round run). Each fresh
   spawn on Windows has to reimport torch/opacus/numpy/data_loader from
   scratch, which is genuinely expensive (multiple seconds per process
   just for `import torch`) — that overhead was eating most of the
   expected speedup. Fixed: the executor now wraps the ENTIRE round
   loop and its 4 workers are reused for all NUM_ROUNDS rounds.

8. Result ordering preserved exactly. Futures are collected via
   as_completed() (workers finish in whatever order they finish in —
   client training times vary with local data size), but results are
   then processed in ORIGINAL client index order (0..NUM_CLIENTS-1)
   before any ZKP/HE/accepted-list logic runs. This matters because
   Multi-Krum's bug fix (see header) depends on accepted_client_indices
   being built in a stable, deterministic order — nothing about that
   logic or its correctness changes here, only the (independent, CPU-
   bound) client training step is now parallel.

9. Thread tuning. The main process uses all detected cores
   (torch.set_num_threads(_CPU_COUNT)) for its own work (evaluation,
   aggregation). Each of the 4 worker processes is capped at
   cores // CLIENT_POOL_WORKERS via a pool initializer, so 4 processes
   aren't each independently trying to claim every core and fighting
   each other for cache/scheduling — this is set once per worker at
   spawn time, not per task.

10. NEW — Adaptive Multi-Krum added as a separate, mutually-exclusive
    aggregation branch (USE_ADAPTIVE_KRUM). Unlike fixed-m multi_krum(),
    which always keeps exactly KRUM_M clients, adaptive_multi_krum()
    computes each client's standard Krum distance score, then drops
    only clients whose score exceeds median(scores) + k * MAD(scores)
    (or mean + k*std, via ADAPTIVE_KRUM_METHOD). On an all-honest round
    this drops ~0 clients regardless of non-IID variance; on a round
    with a cluster of extreme Byzantine clients it drops all of them,
    not a fixed count. This is run as a SEPARATE condition from
    USE_KRUM (fixed m=7) for direct comparison — see
    defences/krum.py::adaptive_multi_krum docstring for the full
    algorithm and tuning notes on k. Wiring mirrors the USE_KRUM branch
    exactly: same accepted_client_indices translation, same
    krum_selected_ids / krum_detected_byz bookkeeping, same CSV
    columns (krum_selected, krum_detected_byzantine) reused rather than
    adding new ones, so results_*.csv stays comparable across both
    Krum variants without a schema change. Only one of USE_KRUM /
    USE_ADAPTIVE_KRUM / USE_HE may be True at a time — asserted below.

Nothing about the aggregation math, fixed-Krum logic, DP accounting, or
HE handling changed in this revision — this adds a new, independently
selected aggregation branch alongside the existing ones.
--------------------------------------------------------------------------
"""

import os
import sys
import csv
import json
import time
import warnings
import numpy as np
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

# FL hyperparameters
NUM_ROUNDS    = 25
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
USE_ADAPTIVE_KRUM = False  # dynamic MAD/Z-score threshold instead of fixed m
                            # — see defences/krum.py::adaptive_multi_krum.
                            # Run as a SEPARATE condition from USE_KRUM, not
                            # a replacement for it — the comparison between
                            # fixed-m and adaptive-threshold Krum is the point.
USE_HE   = False          # CKKS via TenSEAL — set True for Experiment 2

USE_DP   = False          # Opacus per-round DP-SGD
USE_ZKP  = False          # lightweight norm-bound ZKP gate

assert sum([USE_KRUM, USE_ADAPTIVE_KRUM, USE_HE]) <= 1, \
    "USE_KRUM, USE_ADAPTIVE_KRUM, and USE_HE are mutually exclusive aggregation " \
    "branches — pick at most one. (Fixed-m Krum and adaptive-threshold Krum are " \
    "separate conditions to compare against each other, not a combinable pair. " \
    "Experiment 2's Krum+partial-HE pipeline is a separate restructured branch " \
    "— not implemented in this file.)"

# DP_SAFE must match USE_DP — architecture (BatchNorm→GroupNorm, LSTM→DPLSTM
# with dp_safe=True) must be consistent across checkpoint init, training,
# and eval or set_model_parameters will fail on mismatched state_dict keys.
DP_SAFE = USE_DP

# Head-only attack: flips only classifier weights, stays within ZKP norm bound.
# Only meaningful when USE_HE=True (full model encrypted, subtle attack needed).
# When USE_HE=False, sign_flip_attack is always used regardless of this flag.
BYZANTINE_HEAD_ONLY = False   # set True for Experiment 2 Condition B

# DP settings (per-round; not composition-tracked — see epsilon sweep for
# the ε study). Accountant is RDP throughout — see PrivacyEngine below.
DP_EPSILON       = 15.0        # set per condition: 15.0 / 9.0 / 3.0
DP_DELTA         = 1e-5
DP_MAX_GRAD_NORM = 1.5
DP_BATCH_SIZE    = 256

# ZKP settings
ZKP_MAX_NORM = 10.0       # reject clients whose update L2-norm exceeds this

# Fixed-m Krum settings
KRUM_M = NUM_CLIENTS - NUM_BYZANTINE - 1   # e.g. 7 of 10 selected, 3 discarded

# Adaptive Multi-Krum settings — see defences/krum.py::adaptive_multi_krum
# docstring for what each does. k is the equivalent tuning knob to KRUM_M
# above; start around 2.5-3.0 and sweep the same way KRUM_M=6 vs 7 was swept.
ADAPTIVE_KRUM_K                 = 2.5
ADAPTIVE_KRUM_METHOD             = "mad"   # "mad" (robust) or "zscore" (ablation only)
ADAPTIVE_KRUM_MIN_KEEP_FRACTION  = 0.5     # safety floor — never drop more than
                                            # half the accepted clients in one round

# ---------------------------------------------------------------------------
# Parallelization settings
# ---------------------------------------------------------------------------
_CPU_COUNT = os.cpu_count() or 4
CLIENT_POOL_WORKERS = min(4, NUM_CLIENTS)
_THREADS_PER_WORKER = max(1, _CPU_COUNT // CLIENT_POOL_WORKERS)

# ---------------------------------------------------------------------------
# Output paths — one set per model type so both can run simultaneously
# ---------------------------------------------------------------------------
_TAG               = MODEL_TYPE
CHECKPOINT_PARAMS  = f"checkpoint_{_TAG}.npz"
CHECKPOINT_PROGRESS= f"checkpoint_{_TAG}_progress.json"
LOG_CSV            = f"results_{_TAG}.csv"

# EMA (exponential moving average) of the global model across rounds,
# saved separately from the raw last-round checkpoint. WHY:
# confusion_matrix.py showed a single degraded round (network model,
# round 25) causing Ransomware to become a false-positive "attractor"
# for several other classes (16.9-69% of Normal/DDoS_HTTP/
# Vulnerability_scanner/MITM misclassified as Ransomware). FedAvg/
# FedProx has no memory between rounds — one unlucky round's aggregate
# fully overwrites everything before it. EMA blends each round's
# aggregate with the accumulated trajectory of all prior rounds, so a
# single bad round is heavily diluted rather than becoming the entire
# reported/deployed model. This does not fix WHY a round degrades
# (that's addressed separately by weight clamping in task.py) — it
# limits the BLAST RADIUS when one still happens.
EMA_DECAY = 0.9  # higher = more smoothing, slower to reflect real improvement
EMA_CHECKPOINT_PARAMS = f"checkpoint_{_TAG}_ema.npz"

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


# ---------------------------------------------------------------------------
# ─── ROUND-LEVEL LEARNING RATE DECAY ────────────────────────────────────────
# ---------------------------------------------------------------------------

def get_round_lr(base_lr, round_num, num_rounds, min_lr_frac=0.15):
    """
    Cosine-decays the CLIENT-SIDE learning rate across FL rounds, from
    base_lr down to base_lr * min_lr_frac by the final round.

    WHY THIS IS NEEDED: task.py's train() creates a fresh Adam
    optimizer at a fixed lr on every call — every client, every round
    restarts at full learning rate regardless of how far training has
    progressed globally. task.py's own StepLR only decays WITHIN a
    single round's 5 local epochs and resets next round. The result:
    by round 20+, when the global model should mostly be fine-tuning,
    every client is still taking round-2-sized gradient steps, which
    is a strong candidate for the recurring late-round instability
    (e.g. round 20's loss spike from ~0.7 to 1.5, round 24's dip after
    round 23's peak) — a big step from one non-IID client's local
    optimum can knock an otherwise-converging global model backward.

    This is DELIBERATELY separate from PROX_MU: mu bounds how FAR a
    client's local model can drift from the global one; this bounds
    HOW BIG each individual gradient step is. Both affect stability,
    but they are not the same knob — tune/test them independently.
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
    don't each independently try to claim every core — see changelog
    entry 9.
    """
    import torch
    torch.set_num_threads(_THREADS_PER_WORKER)


def _train_one_client(client_idx, X_tr, y_tr, global_params, client_cfg):
    """
    Top-level (module-level) function — REQUIRED for Windows
    multiprocessing. The 'spawn' start method pickles this callable to
    send to the worker process; closures/nested functions cannot be
    pickled, only top-level functions can. See changelog entry 6.

    Runs in a worker process. Re-imports of this module's top-level
    state (build_criterion, get_model, etc.) happen automatically when
    the worker process starts, since spawn re-executes the module up
    to (but not including, thanks to the __main__ guard) the main()
    call — sys.argv is inherited, so MODEL_TYPE and the conditional
    build_criterion/import block above resolve identically in the
    worker as they did in the parent process.

    Returns (client_idx, params, dp_eps_spent) so the caller can match
    results back to their original client index after collecting them
    via as_completed() (which returns futures in COMPLETION order, not
    submission order — the caller is responsible for re-sorting by
    client_idx before doing any ZKP/HE/Krum bookkeeping, exactly as
    the original sequential loop did).
    """
    model = get_model(num_features=client_cfg["sample_features"],
                      num_classes=client_cfg["num_classes"],
                      dp_safe=client_cfg["dp_safe"])
    set_model_parameters(model, global_params)

    dp_eps_spent = None

    # ── Byzantine attack injection ───────────────────────────────────
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

            criterion = build_criterion()

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
            model, optimizer, loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=loader,
                target_epsilon=client_cfg["dp_epsilon"],
                target_delta=client_cfg["dp_delta"],
                epochs=client_cfg["local_epochs"],
                max_grad_norm=client_cfg["dp_max_grad_norm"],
            )
            model.train()
            for _ in range(client_cfg["local_epochs"]):
                for X_b, y_b in loader:
                    optimizer.zero_grad()
                    loss_val = criterion(model(X_b), y_b)
                    loss_val.backward()
                    optimizer.step()

            dp_eps_spent = privacy_engine.get_epsilon(client_cfg["dp_delta"])

            real_model = model._module if hasattr(model, "_module") else model
            params = get_model_parameters(real_model)

        else:
            # Standard FedProx training (no DP)
            criterion = build_criterion()
            train(model, X_tr, y_tr, criterion,
                  epochs=client_cfg["local_epochs"],
                  lr=client_cfg["learning_rate"],
                  global_params=global_params,
                  mu=client_cfg["prox_mu"])
            params = get_model_parameters(model)

    return client_idx, params, dp_eps_spent


# ---------------------------------------------------------------------------
# ─── AGGREGATION HELPERS ────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def fedprox_aggregate(all_params: list, weights: list) -> list:
    """
    Weighted-average aggregation (server side).
    FedProx vs FedAvg difference is entirely in the client training loss
    (proximal term in task.py::train). The server always does weighted average.

    DEFENCE HOOK — swap this call for multi_krum() / adaptive_multi_krum()
    in the Krum branches below.
    """
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
    """
    CKKS homomorphic aggregation via TenSEAL.
    Server sums encrypted vectors without ever decrypting.
    Returns a list of plaintext numpy arrays after client-side decryption.

    NOTE: Neither Krum variant is compatible with this path — distance
    computation requires plaintext. See literature: Lancelot (arXiv
    2408.06197), PBFL (COCOON 2024) for encrypted Byzantine-robust
    alternatives.

    NOTE: this path is not exercised while USE_KRUM or USE_ADAPTIVE_KRUM
    is True (asserted above). Known open issue carried over from
    previous review: this function does not decrypt before returning,
    and averaging is unweighted. Not touched in this revision —
    flagging so it isn't forgotten before Experiment 2.
    """
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
    """
    Lightweight ZKP gate: rejects clients whose flattened parameter update
    L2-norm exceeds max_norm. This is a norm-bound check, not a full ZKP
    (which would require a proving system like Bulletproofs or STARK).

    Returns True if the client PASSES (should be accepted).
    """
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
       "dp_epsilon_spent", "round_time_s"]
)
# NOTE: krum_selected / krum_detected_byzantine columns are shared between
# USE_KRUM (fixed m) and USE_ADAPTIVE_KRUM (dynamic threshold) runs — since
# the two are mutually exclusive per run (see assert above), the CSV schema
# doesn't need separate columns per variant. Check experiment_config_*.json
# for which variant produced a given results_*.csv.


def init_log_csv(resume: bool = False):
    if not resume and os.path.exists(LOG_CSV):
        os.remove(LOG_CSV)
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline="") as f:
            csv.writer(f).writerow(_CSV_HEADER)


def append_log_row(round_num, client_label, loss, accuracy,
                   per_class_f1, zkp_rejected, krum_selected,
                   krum_detected, dp_eps, round_time, is_mean: bool = False):
    if is_mean:
        krum_selected_field = krum_selected
        krum_detected_field = (
            f"{krum_detected:.4f}" if krum_detected is not None else "N/A"
        )
    else:
        krum_selected_field = 1 if krum_selected else 0
        krum_detected_field = 1 if krum_detected else 0

    row = (
        [round_num, client_label,
         f"{loss:.6f}", f"{accuracy:.6f}"]
        + [f"{v:.6f}" for v in per_class_f1]
        + [int(zkp_rejected),
           krum_selected_field,
           krum_detected_field,
           f"{dp_eps:.4f}" if dp_eps is not None else "N/A",
           f"{round_time:.2f}"]
    )
    with open(LOG_CSV, "a", newline="") as f:
        csv.writer(f).writerow(row)


# ---------------------------------------------------------------------------
# ─── MAIN TRAINING LOOP ─────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*65}")
    print(f"  FL-IDS Unified Loop — MODEL: {MODEL_TYPE.upper()}")
    print(f"  Rounds={NUM_ROUNDS}  Clients={NUM_CLIENTS}  Epochs={LOCAL_EPOCHS}")
    print(f"  Byzantine={NUM_BYZANTINE} (clients {BYZANTINE_CLIENTS})  "
          f"Attack={'ON' if USE_BYZANTINE_ATTACK else 'OFF'}")
    print(f"  USE_KRUM={USE_KRUM}  USE_ADAPTIVE_KRUM={USE_ADAPTIVE_KRUM}  "
          f"USE_HE={USE_HE}  USE_DP={USE_DP}  USE_ZKP={USE_ZKP}")
    print(f"  Parallel client training: {CLIENT_POOL_WORKERS} workers, "
          f"{_THREADS_PER_WORKER} threads/worker "
          f"({_CPU_COUNT} cores detected)")
    if USE_DP:
        print(f"  DP: ε={DP_EPSILON}  δ={DP_DELTA}  "
              f"max_grad_norm={DP_MAX_GRAD_NORM}  accountant=rdp")
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

    import torch
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

    # ── Static per-client config, built once, passed to every worker call ──
    client_cfg = {
        "sample_features":      sample_features,
        "num_classes":          NUM_CLASSES,
        "dp_safe":              DP_SAFE,
        "use_byzantine_attack": USE_BYZANTINE_ATTACK,
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

    # EMA starts as a copy of the initial global_params (fresh run) or
    # is reloaded from disk if resuming — see EMA_DECAY comment above.
    if resume and os.path.exists(EMA_CHECKPOINT_PARAMS):
        ema_data = np.load(EMA_CHECKPOINT_PARAMS)
        ema_params = [ema_data[f"arr_{i}"] for i in range(len(ema_data.files))]
    else:
        ema_params = [p.copy() for p in global_params]

    meta_path = f"experiment_config_{_TAG}.json"
    with open(meta_path, "w") as f:
        json.dump({
            "model_type": MODEL_TYPE,
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
            "dp_accountant": "rdp",
            "use_zkp": USE_ZKP,
            "zkp_max_norm": ZKP_MAX_NORM,
            "byzantine_head_only": BYZANTINE_HEAD_ONLY,
            "dp_safe": DP_SAFE,
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

            # Round-aware LR decay — see get_round_lr() docstring for why
            # this exists separately from task.py's within-round StepLR.
            round_lr = get_round_lr(LEARNING_RATE, round_num, NUM_ROUNDS)
            round_client_cfg = {**client_cfg, "learning_rate": round_lr}
            print(f"  Client LR this round: {round_lr:.6f} "
                  f"(base={LEARNING_RATE})")

            accepted_params          = []
            accepted_weights         = []
            accepted_client_indices  = []

            zkp_rejected_this_round  = []
            dp_eps_spent_this_round  = []

            # ── Submit all clients to the persistent pool ───────────────────
            futures = {
                executor.submit(
                    _train_one_client, i, X_tr, y_tr, global_params, round_client_cfg
                ): i
                for i, (X_tr, y_tr, X_te, y_te) in enumerate(clients_data)
            }

            # Collect as they finish (fastest clients first), but store
            # keyed by client index so we can process in ORIGINAL order
            # below — this preserves the exact bookkeeping behavior the
            # Krum index-mapping fix depends on (see changelog entry 8).
            results_by_client = {}
            for future in as_completed(futures):
                client_idx, params, dp_eps_spent = future.result()
                results_by_client[client_idx] = (params, dp_eps_spent)

            # ── Sequential bookkeeping, in original client order ────────────
            for i, (X_tr, y_tr, X_te, y_te) in enumerate(clients_data):
                params, dp_eps_spent = results_by_client[i]

                if USE_BYZANTINE_ATTACK and i in BYZANTINE_CLIENTS:
                    tag = "head-only" if (USE_HE and BYZANTINE_HEAD_ONLY) else "sign-flip"
                    print(f"  Client {i+1:2d}  [BYZANTINE — {tag} ×{ATTACK_SCALE}]")

                # ── ZKP norm-bound gate ──────────────────────────────────────
                if USE_ZKP:
                    passes = zkp_verify_norm(params, max_norm=ZKP_MAX_NORM)
                    if not passes:
                        print(f"  Client {i+1:2d}  [ZKP REJECTED — norm too large]")
                        zkp_rejected_this_round.append(i)
                        continue

                # ── HE encryption (if USE_HE) ────────────────────────────────
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

            # ── Aggregation branch ───────────────────────────────────────────
            krum_selected_ids   = set()
            krum_discarded_ids  = set()
            krum_detected_byz   = set()

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
                        num_byzantine=NUM_BYZANTINE
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

                    agg_label = (f"Multi-Krum  selected={sorted(krum_selected_ids)}  "
                                 f"discarded={sorted(krum_discarded_ids)}  "
                                 f"detected_byz={sorted(krum_detected_byz)}")

            elif USE_ADAPTIVE_KRUM:
                # Same "too few accepted clients" fallback spirit as the
                # fixed-Krum branch's effective_m check — adaptive_multi_krum
                # needs n - f - 2 >= 1 to even compute neighbour scores.
                if len(accepted_params) - NUM_BYZANTINE - 2 < 1:
                    global_params = fedprox_aggregate(accepted_params,
                                                      accepted_weights)
                    agg_label = "FedProx (Adaptive-Krum fallback — too few accepted clients)"
                else:
                    global_params, selected_positions = adaptive_multi_krum(
                        accepted_params,
                        accepted_weights,
                        num_byzantine=NUM_BYZANTINE,
                        k=ADAPTIVE_KRUM_K,
                        method=ADAPTIVE_KRUM_METHOD,
                        min_keep_fraction=ADAPTIVE_KRUM_MIN_KEEP_FRACTION,
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

            # ── EMA update — smooths across rounds, see EMA_DECAY comment ──
            ema_params = [
                EMA_DECAY * e + (1 - EMA_DECAY) * g
                for e, g in zip(ema_params, global_params)
            ]

            # ── Evaluation ───────────────────────────────────────────────────
            eval_model = get_model(num_features=sample_features,
                                   num_classes=NUM_CLASSES,
                                   dp_safe=DP_SAFE)
            set_model_parameters(eval_model, global_params)

            _krum_active = USE_KRUM or USE_ADAPTIVE_KRUM

            round_losses, round_accs, round_f1s = [], [], []
            for i, (X_tr, y_tr, X_te, y_te) in enumerate(clients_data):
                loss_v, acc_v, f1_per_class = test(eval_model, X_te, y_te, NUM_CLASSES)
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

            # ── EMA evaluation — quick check, pooled test set only (not
            # per-client) since this is a sanity comparison against the
            # raw round, not the primary logged metric.
            ema_model = get_model(num_features=sample_features,
                                  num_classes=NUM_CLASSES, dp_safe=DP_SAFE)
            set_model_parameters(ema_model, ema_params)
            X_te_all = np.concatenate([c[2] for c in clients_data], axis=0)
            y_te_all = np.concatenate([c[3] for c in clients_data], axis=0)
            _, ema_acc, ema_f1 = test(ema_model, X_te_all, y_te_all, NUM_CLASSES)
            print(f"  [EMA]   Acc: {ema_acc:.4f}  F1-Macro: {ema_f1.mean():.4f}  "
                  f"(decay={EMA_DECAY}, smoothed across all rounds so far)\n")

            np.savez(EMA_CHECKPOINT_PARAMS, *ema_params)

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
            )

            save_checkpoint(global_params, round_num)

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"  Training complete — {NUM_ROUNDS} rounds  [{MODEL_TYPE.upper()}]")
    print(f"  Results logged to:     {LOG_CSV}")
    print(f"  Last-round checkpoint: {CHECKPOINT_PARAMS} (raw, round {NUM_ROUNDS} — ")
    print(f"                         may reflect a single bad round, see EMA below)")
    print(f"  EMA checkpoint:        {EMA_CHECKPOINT_PARAMS} (decay={EMA_DECAY}, "
          f"smoothed across all rounds)")
    if USE_KRUM or USE_ADAPTIVE_KRUM:
        print(f"\n  Reminder: delete checkpoint before changing flags")
        print(f"  (Krum/Adaptive-Krum/HE/DP flags change the experiment — old")
        print(f"  checkpoint params will give misleading results if reused.)")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
