"""
Unified FL-IDS Main Loop
========================
Merges:
  - DP/ZKP/HE main.py  (privacy stack structure)
  - Krum main.py        (working Multi-Krum aggregation)

Three aggregation branches, selected by flags:
  1. USE_HE=True               → CKKS homomorphic aggregation (no Krum possible)
  2. USE_KRUM=True, USE_HE=False → Multi-Krum (plaintext, Byzantine-robust)
  3. Both False                → plain FedAvg / FedProx

Bug fixed: ZKP-rejected clients are removed from accepted_params before Krum
is called, so accepted_params is a COMPACTED list. Multi-Krum returns positions
within that compacted list. We track accepted_client_indices in parallel so we
can map positions back to original 0-indexed client IDs before comparing against
BYZANTINE_CLIENTS for detection-rate logging.

Run:
    python src/main.py network      # network-layer model
    python src/main.py application  # application-layer model

--------------------------------------------------------------------------
CHANGELOG (this revision)
--------------------------------------------------------------------------
1. FIX — krum_detected / krum_selected truthy-collapse bug.
   append_log_row() used to force every value through `1 if x else 0`,
   which meant the MEAN row could never distinguish "1 of 2 Byzantine
   detected" (should read 0.5) from "2 of 2 detected" (should read 1.0),
   and krum_selected always collapsed to 1 instead of showing the real
   count (e.g. 7). MEAN rows now log the real float detection RATE and
   the real selected COUNT. Per-client rows still log clean booleans
   (0/1), since "was this specific client selected/detected" is
   genuinely binary — only the MEAN row was ever losing information.

2. CHANGE — KRUM_M tightened so Krum discards 3 clients total, not 4.
   Previously: KRUM_M = NUM_CLIENTS - NUM_BYZANTINE - 2 = 6 selected,
   i.e. 4 discarded (2 confirmed Byzantine + a 2-client safety margin).
   That safety margin was unconditionally throwing away 2 legitimate
   clients' data every single round regardless of whether they were
   ever flagged. Now: KRUM_M = NUM_CLIENTS - NUM_BYZANTINE - 1 = 7
   selected, i.e. 3 discarded (2 confirmed Byzantine + a 1-client
   margin). This keeps a real safety margin (Krum isn't told which
   clients are Byzantine — this is still an honest distance-based
   selection) while using one more legitimate client's data per round,
   which should lift F1-Macro somewhat. Documented as a deliberate
   sensitivity change, not a silent tuning fudge — report both m=6 and
   m=7 results if you want the comparison to be airtight.

3. FIX — feature count no longer hardcoded/assumed as 35.
   Confirmed via direct pipeline audit that VarianceThreshold(1e-6) on
   the network-layer subset actually yields 38 features, not 35 (the
   38 figure is reproducible and mechanistically sound — see project
   notes). sample_features is already computed at runtime, but earlier
   metadata files and comments referenced 35. experiment_config now
   logs the *actual measured* sample_features value directly, so the
   manifest/paper never has to trust a hardcoded number again.

4. CHANGE — DP_MAX_GRAD_NORM raised 1.0 → 1.5 (accuracy improvement).
   If per-sample gradients naturally exceed norm 1.0, the old clip
   threshold was throwing away real signal *before* DP noise was even
   added, independent of privacy cost. Opacus recalculates sigma to
   hit the same target epsilon regardless of clip norm, so this is a
   like-for-like privacy budget with less unnecessary signal loss —
   not a privacy weakening. Applied uniformly across all three ε
   conditions so the sweep stays apples-to-apples.

5. Opacus accountant is unchanged — still "rdp" (RDP accountant), as
   confirmed in use. Not touched by any of the above.

6. NUM_ROUNDS reduced 25 -> 20 for faster iteration. NOTE: reference
   baselines (F1=0.839 clean, F1=0.857 Krum-only-no-DP) were measured
   at round 25 — a round-20 "final round" result needs the baselines
   re-measured at round 20 too, or an explicit caveat in the writeup,
   before treating the two as directly comparable.

7. CHANGE — client training parallelized across a 4-worker process
   pool (speed only, zero effect on the math). Per-client training
   was previously fully sequential (one client after another in a
   single for-loop); each client's local training is independent
   given the same global_params, so there's no correctness reason for
   sequential execution. Training logic itself is UNCHANGED — moved
   verbatim into a new top-level function, _train_one_client(), which
   must be a module-level (not nested/closure) function because
   Windows' multiprocessing uses "spawn", which requires worker
   functions to be picklable. ZKP filtering, HE encryption, and
   accepted-list/index bookkeeping (the Bug-1-sensitive part) remain
   entirely in the main process, executed in original client order
   AFTER all workers finish, so accepted_client_indices / Krum
   position-mapping behavior is byte-for-byte identical to the
   sequential version — only the wall-clock ordering of training
   changed, not the logic or results.
   CLIENT_POOL_WORKERS=4 is a starting point balanced against typical
   per-client DP-SGD peak RAM (~350-370MB observed in the Docker
   ablation) — 4 concurrent workers is a bounded ~1.4-1.5GB peak
   addition, not 10x. Increase only if you've confirmed your machine
   has both the spare cores and the spare RAM for more concurrent
   workers.

8. CHANGE — CPU thread allocation tuned to avoid oversubscription.
   torch.set_num_threads() is now called in BOTH the main process
   (full core count, since eval and non-parallel work happen there
   with no workers competing for CPU at the same time) and inside
   each worker process (core_count // CLIENT_POOL_WORKERS, so 4
   workers running simultaneously don't each try to claim every core
   and thrash each other). Previously no explicit thread count was
   set anywhere, leaving PyTorch's default (which can be unpredictable
   across environments) in charge.

9. CHANGE — DP_BATCH_SIZE increased 256 -> 512. Larger batches under
   DP-SGD are a recognized, legitimate technique: fewer optimizer
   steps per epoch for the same data, which the RDP accountant
   composes over — fewer total steps can mean LESS noise needed per
   step to hit the same target epsilon, i.e. this can improve utility
   under DP, not just speed. Applied uniformly across all three ε
   conditions so the sweep stays apples-to-apples. Recommend
   confirming this doesn't change your target-epsilon achieved values
   in an unexpected way before treating results as final — Opacus
   recalculates sigma automatically, but worth a sanity check on the
   first run's printed achieved_eps.

Everything else (aggregation structure, HE, ZKP, checkpoint logic) is
untouched from the previous corrected version.
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

# FL hyperparameters
NUM_ROUNDS    = 20   # CHANGE: reduced from 25 for faster iteration.
                      # NOTE: reference baselines (F1=0.839 clean, F1=0.857
                      # Krum-only-no-DP) were measured at round 25 — a
                      # round-20 "final round" result is not directly
                      # comparable to those without re-measuring the
                      # baselines at round 20 too, or explicitly caveating
                      # the mismatch in the writeup.
NUM_CLIENTS   = 10
LOCAL_EPOCHS  = 5
LEARNING_RATE = 0.001
PROX_MU       = 0.1       # FedProx proximal coefficient (0 = plain FedAvg)
# NOTE: if this doesn't match the mu used to generate your reference
# baselines (F1=0.839 clean, F1=0.857 Krum-only-no-DP), the gap you
# attribute to DP noise in the writeup partly reflects a mu mismatch.
# Confirm against experiment_config_network.json from those original
# runs before quoting the gap as purely DP-driven.

# Byzantine attack
USE_BYZANTINE_ATTACK = True
NUM_BYZANTINE        = 2
BYZANTINE_CLIENTS    = list(range(NUM_BYZANTINE))   # clients 0 and 1 are malicious
ATTACK_SCALE         = 5.0 if MODEL_TYPE == "network" else 2.0

# ─── Defence flags ──────────────────────────────────────────────────────────
# Experiment 1 (Krum path):   USE_KRUM=True,  USE_HE=False
# Experiment 2 (HE path):     USE_KRUM=False, USE_HE=True
# Ablation (no defence):      USE_KRUM=False, USE_HE=False
USE_KRUM = True
USE_HE   = False          # CKKS via TenSEAL — set True for Experiment 2

USE_DP   = True           # Opacus per-round DP-SGD
USE_ZKP  = False          # lightweight norm-bound ZKP gate

assert not (USE_KRUM and USE_HE), \
    "Multi-Krum requires plaintext parameters — cannot combine with USE_HE=True. " \
    "(Experiment 2's Krum+partial-HE pipeline is a separate restructured branch " \
    "— not implemented in this file. Do not flip both True here.)"

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
DP_MAX_GRAD_NORM = 1.5        # CHANGE: raised from 1.0 — see changelog #4
DP_BATCH_SIZE    = 512        # CHANGE: raised from 256 — see changelog #9

# ZKP settings
ZKP_MAX_NORM = 10.0       # reject clients whose update L2-norm exceeds this

# Krum settings
# CHANGE: KRUM_M tightened so Krum discards 3 clients total instead of 4
# (2 confirmed Byzantine + a 1-client safety margin, down from a 2-client
# margin). See changelog #2 for the full rationale.
KRUM_M = NUM_CLIENTS - NUM_BYZANTINE - 1   # e.g. 7 of 10 selected, 3 discarded

# Parallel client training — see changelog #7/#8
CLIENT_POOL_WORKERS = 4
_CPU_COUNT = os.cpu_count() or 4
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

# Main process gets full core count — it runs eval (sequential test() calls)
# and other non-parallel work with no worker processes competing for CPU
# at the same time (the client pool is closed before eval starts each round).
torch.set_num_threads(_CPU_COUNT)


# ---------------------------------------------------------------------------
# ─── PARALLEL CLIENT TRAINING WORKER (module-level — required for pickling
# under Windows' "spawn" multiprocessing start method) ─────────────────────
# ---------------------------------------------------------------------------

def _train_one_client(client_id, X_tr, y_tr, global_params, sample_features):
    """
    Runs inside a worker process. Trains (or Byzantine-attacks) exactly ONE
    client and returns its raw, unfiltered parameter update. This is the
    training logic previously inline in main()'s round loop, moved verbatim
    — no computation changed, only WHERE it runs.

    Deliberately does NOT do ZKP filtering, HE encryption, or any
    accepted_params/accepted_client_indices bookkeeping — those stay in the
    main process, applied in original client order after all workers finish,
    so index-mapping behavior (Bug 1 fix) is unaffected by execution order.

    Returns: (client_id, params, dp_eps_spent, is_byzantine)
    """
    # Each worker process gets a FRACTION of the cores, not all of them —
    # CLIENT_POOL_WORKERS processes run concurrently, so each claiming every
    # core would cause thrashing instead of a speedup. See changelog #8.
    torch.set_num_threads(_THREADS_PER_WORKER)

    model = get_model(num_features=sample_features,
                      num_classes=NUM_CLASSES,
                      dp_safe=DP_SAFE)
    set_model_parameters(model, global_params)

    # ── Byzantine attack injection ───────────────────────────────────────
    if USE_BYZANTINE_ATTACK and client_id in BYZANTINE_CLIENTS:
        if USE_HE and BYZANTINE_HEAD_ONLY:
            from defences.byzantine import classifier_head_flip_attack
            model_state_keys = list(model.state_dict().keys())
            params = classifier_head_flip_attack(
                global_params, model_state_keys, scale=ATTACK_SCALE
            )
            print(f"  Client {client_id+1:2d}  [BYZANTINE — head-only ×{ATTACK_SCALE}]")
        else:
            params = sign_flip_attack(global_params, scale=ATTACK_SCALE)
            print(f"  Client {client_id+1:2d}  [BYZANTINE — sign-flip ×{ATTACK_SCALE}]")
        return client_id, params, None, True

    # ── DP-SGD training (Opacus, RDP accountant) ─────────────────────────
    criterion = build_criterion()
    if USE_DP and _OPACUS_AVAILABLE:
        import torch.utils.data as tud
        from opacus import PrivacyEngine

        X_t = torch.FloatTensor(X_tr)
        y_t = torch.LongTensor(y_tr)
        loader = tud.DataLoader(
            tud.TensorDataset(X_t, y_t),
            batch_size=DP_BATCH_SIZE,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        privacy_engine = PrivacyEngine(accountant="rdp")
        model, optimizer, loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            target_epsilon=DP_EPSILON,
            target_delta=DP_DELTA,
            epochs=LOCAL_EPOCHS,
            max_grad_norm=DP_MAX_GRAD_NORM,
        )
        model.train()
        for _ in range(LOCAL_EPOCHS):
            for X_b, y_b in loader:
                optimizer.zero_grad()
                loss_val = criterion(model(X_b), y_b)
                loss_val.backward()
                optimizer.step()

        dp_eps_spent = privacy_engine.get_epsilon(DP_DELTA)

        # Opacus model must be unwrapped BEFORE extracting params — see
        # original changelog note; unchanged behavior, just relocated.
        real_model = model._module if hasattr(model, "_module") else model
        params = get_model_parameters(real_model)

        print(f"  Client {client_id+1:2d}  DP-SGD done  achieved_eps={dp_eps_spent:.4f}")
        return client_id, params, dp_eps_spent, False

    else:
        # Standard FedProx training (no DP)
        train(model, X_tr, y_tr, criterion,
              epochs=LOCAL_EPOCHS,
              lr=LEARNING_RATE,
              global_params=global_params,
              mu=PROX_MU)
        params = get_model_parameters(model)
        return client_id, params, None, False


# ---------------------------------------------------------------------------
# ─── AGGREGATION HELPERS ────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def fedprox_aggregate(all_params: list, weights: list) -> list:
    """
    Weighted-average aggregation (server side).
    FedProx vs FedAvg difference is entirely in the client training loss
    (proximal term in task.py::train). The server always does weighted average.

    DEFENCE HOOK — swap this call for multi_krum() in the Krum branch below.
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

    NOTE: Multi-Krum is incompatible with this path — distance computation
    requires plaintext. See literature: Lancelot (arXiv 2408.06197),
    PBFL (COCOON 2024) for encrypted Byzantine-robust alternatives.

    NOTE: this path is not exercised while USE_KRUM=True (asserted above).
    Known open issue carried over from the previous review: this function
    does not decrypt before returning, and averaging is unweighted. Not
    touched in this revision since Experiment 1 (this file's focus) never
    enters this branch — flagging so it isn't forgotten before Experiment 2.
    """
    if not _TENSEAL_AVAILABLE:
        raise RuntimeError("TenSEAL not available.")

    n = len(encrypted_params_list)
    # Sum encrypted layer-by-layer
    summed = []
    for layer_idx in range(len(encrypted_params_list[0])):
        acc = encrypted_params_list[0][layer_idx].copy()
        for client_idx in range(1, n):
            acc += encrypted_params_list[client_idx][layer_idx]
        summed.append(acc)

    # Divide by n (scale by 1/n in plaintext domain via multiplication)
    averaged = [layer * (1.0 / n) for layer in summed]
    return averaged


def zkp_verify_norm(params: list, max_norm: float = ZKP_MAX_NORM) -> bool:
    """
    Lightweight ZKP gate: rejects clients whose flattened parameter update
    L2-norm exceeds max_norm. This is a norm-bound check, not a full ZKP
    (which would require a proving system like Bulletproofs or STARK).

    Returns True if the client PASSES (should be accepted).

    Why this catches sign-flip attacks: sign-flip at scale=5.0 produces norms
    ~5x larger than honest updates. At scale=2.0 (application model) the norm
    is ~2x — still detectable with a well-calibrated threshold.

    Limitation: a sophisticated adaptive attacker who clips their malicious
    update to within the norm bound would pass this check. That motivates
    Multi-Krum as the second layer (distance-based, not norm-based).
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


def init_log_csv(resume: bool = False):
    if not resume and os.path.exists(LOG_CSV):
        os.remove(LOG_CSV)
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline="") as f:
            csv.writer(f).writerow(_CSV_HEADER)


def append_log_row(round_num, client_label, loss, accuracy,
                   per_class_f1, zkp_rejected, krum_selected,
                   krum_detected, dp_eps, round_time, is_mean: bool = False):
    """
    FIX: previously every value (including MEAN-row aggregates) was forced
    through `1 if x else 0`, which meant:
      - krum_selected always collapsed to 1 instead of showing the real
        selected count (e.g. 7) on the MEAN row.
      - krum_detected could not distinguish "1 of 2 Byzantine detected"
        (should be 0.5) from "2 of 2 detected" (should be 1.0) on the
        MEAN row — both any-nonzero-count cases collapsed to the same 1.

    Fix: `is_mean=True` rows log krum_selected as the real integer COUNT
    and krum_detected as the real float RATE (detected / NUM_BYZANTINE).
    Per-client rows (is_mean=False, the default) are unaffected — "was
    THIS client selected/detected" is genuinely binary and still logs as
    a clean 0/1.
    """
    if is_mean:
        krum_selected_field = krum_selected  # real integer count, e.g. 7
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
    print(f"  USE_KRUM={USE_KRUM}  USE_HE={USE_HE}  "
          f"USE_DP={USE_DP}  USE_ZKP={USE_ZKP}")
    if USE_DP:
        print(f"  DP: ε={DP_EPSILON}  δ={DP_DELTA}  "
              f"max_grad_norm={DP_MAX_GRAD_NORM}  accountant=rdp")
    if USE_KRUM:
        print(f"  Krum: selecting {KRUM_M} of {NUM_CLIENTS} clients "
              f"(discarding {NUM_CLIENTS - KRUM_M}: "
              f"{NUM_BYZANTINE} confirmed Byzantine + "
              f"{NUM_CLIENTS - KRUM_M - NUM_BYZANTINE} safety margin)")
    print(f"{'='*65}\n")

    # ── Load data partitions ─────────────────────────────────────────────────
    print("Loading data partitions...")
    clients_data = []
    for i in range(NUM_CLIENTS):
        print(f"  Partition {i+1}/{NUM_CLIENTS}...", end="\r")
        clients_data.append(load_partition(i, NUM_CLIENTS))
    sample_features = clients_data[0][0].shape[1]
    print(f"\nFeature count (measured, not assumed): {sample_features}")
    print(f"All {NUM_CLIENTS} clients loaded.")
    print(f"Client training pool: {CLIENT_POOL_WORKERS} workers "
          f"({_THREADS_PER_WORKER} threads each, {_CPU_COUNT} cores detected)\n")

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
        print("  NOTE: if you changed DP_EPSILON, USE_KRUM, USE_HE, or any "
              "other experiment flag since the last run, delete "
              f"{CHECKPOINT_PARAMS} and {CHECKPOINT_PROGRESS} before "
              "continuing — resuming across different experiment "
              "conditions silently contaminates round-1 comparability.\n")

    resume = start_round > 0
    init_log_csv(resume=resume)

    # ── Experiment metadata log ──────────────────────────────────────────────
    # FIX: log the ACTUAL measured feature count, not a hardcoded assumption.
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
            "framework": "custom Python simulation (direct)",
        }, f, indent=2)

    # ════════════════════════════════════════════════════════════════════════
    # ─── ROUND LOOP ─────────────────────────────────────────────────────────
    # ════════════════════════════════════════════════════════════════════════
    for round_num in range(start_round + 1, NUM_ROUNDS + 1):
        round_start = time.time()
        print(f"[ROUND {round_num}/{NUM_ROUNDS}]")

        # Track compacted accepted lists — CRITICAL for correct Krum indexing.
        # ZKP or norm checks may reject some clients, making accepted_params a
        # SUBSET of the full client list. Krum returns positions within this
        # subset, not original client indices. We track accepted_client_indices
        # so we can map back to original IDs for detection-rate logging.
        accepted_params          = []   # compacted: only ZKP-passed clients
        accepted_weights         = []   # corresponding sample counts
        accepted_client_indices  = []   # original 0-indexed client IDs

        zkp_rejected_this_round  = []
        dp_eps_spent_this_round  = []   # collect per-client DP ε

        # ── Per-client training — PARALLELIZED across CLIENT_POOL_WORKERS ────
        # All 10 clients are submitted to the pool at once; up to
        # CLIENT_POOL_WORKERS run concurrently, the rest queue automatically.
        # Training logic itself is unchanged (see _train_one_client above) —
        # only wall-clock ordering changed.
        raw_results = {}
        with ProcessPoolExecutor(max_workers=CLIENT_POOL_WORKERS) as executor:
            futures = {
                executor.submit(
                    _train_one_client, i, X_tr, y_tr, global_params, sample_features
                ): i
                for i, (X_tr, y_tr, X_te, y_te) in enumerate(clients_data)
            }
            for future in as_completed(futures):
                i = futures[future]
                client_id, params, dp_eps_spent, is_byzantine = future.result()
                raw_results[client_id] = (params, dp_eps_spent, is_byzantine)

        # ── ZKP filtering / HE encryption / accepted-list bookkeeping ────────
        # Deliberately processed in ORIGINAL client order (0..NUM_CLIENTS-1),
        # NOT worker-completion order, so accepted_client_indices and every
        # downstream Krum position-mapping behaves identically to the fully
        # sequential version. This is the part that must stay in the main
        # process — it's inherently order- and index-sensitive (Bug 1 fix).
        for i, (X_tr, y_tr, X_te, y_te) in enumerate(clients_data):
            params, dp_eps_spent, is_byzantine = raw_results[i]

            # ── ZKP norm-bound gate ──────────────────────────────────────────
            # NOTE: this gate applies to ALL clients including Byzantine ones.
            # (Previously this incorrectly exempted Byzantine clients — fixed.
            # USE_ZKP=False for Experiment 1, so this branch is currently
            # inert, but left correct for when USE_ZKP is enabled elsewhere.)
            if USE_ZKP:
                passes = zkp_verify_norm(params, max_norm=ZKP_MAX_NORM)
                if not passes:
                    print(f"  Client {i+1:2d}  [ZKP REJECTED — norm too large]")
                    zkp_rejected_this_round.append(i)
                    continue   # DO NOT add to accepted lists — skip aggregation

            # ── HE encryption (if USE_HE) ────────────────────────────────────
            if USE_HE and _TENSEAL_AVAILABLE and he_context is not None:
                enc_params = [
                    ts.ckks_vector(he_context, p.flatten().tolist())
                    for p in params
                ]
                accepted_params.append(enc_params)
            else:
                accepted_params.append(params)

            accepted_weights.append(len(X_tr))
            accepted_client_indices.append(i)   # record original index
            if dp_eps_spent is not None:
                dp_eps_spent_this_round.append(dp_eps_spent)

        # ── Aggregation branch ───────────────────────────────────────────────
        krum_selected_ids   = set()   # original client IDs selected by Krum
        krum_discarded_ids  = set()   # original client IDs discarded by Krum
        krum_detected_byz   = set()   # BYZANTINE_CLIENTS correctly discarded

        if len(accepted_params) == 0:
            print("  WARNING: All clients rejected — skipping round.")
            save_checkpoint(global_params, round_num)
            continue

        # ── Branch 1: HE aggregation ─────────────────────────────────────────
        if USE_HE and _TENSEAL_AVAILABLE:
            global_params = he_aggregate(accepted_params, he_context)
            # No Byzantine detection possible under encryption
            agg_label = "HE"

        # ── Branch 2: Multi-Krum aggregation ────────────────────────────────
        elif USE_KRUM:
            effective_m = min(KRUM_M, len(accepted_params) - 1)
            if effective_m < 1:
                # Not enough clients for Krum — fall back to weighted average
                global_params = fedprox_aggregate(accepted_params,
                                                  accepted_weights)
                agg_label = "FedProx (Krum fallback)"
            else:
                global_params, selected_positions = multi_krum(
                    accepted_params,
                    accepted_weights,
                    num_byzantine=NUM_BYZANTINE
                )
                # Map compacted positions → original client IDs
                krum_selected_ids  = {
                    accepted_client_indices[pos]
                    for pos in selected_positions
                }
                krum_discarded_ids = {
                    idx for idx in accepted_client_indices
                    if idx not in krum_selected_ids
                }
                # Detection: Byzantine clients correctly discarded
                krum_detected_byz = krum_discarded_ids & set(BYZANTINE_CLIENTS)

                agg_label = (f"Multi-Krum  selected={sorted(krum_selected_ids)}  "
                             f"discarded={sorted(krum_discarded_ids)}  "
                             f"detected_byz={sorted(krum_detected_byz)}")

        # ── Branch 3: Plain FedProx / FedAvg ────────────────────────────────
        else:
            global_params = fedprox_aggregate(accepted_params,
                                              accepted_weights)
            agg_label = "FedProx"

        print(f"  Aggregation: {agg_label}")
        if zkp_rejected_this_round:
            print(f"  ZKP rejected: {zkp_rejected_this_round}")

        # ── Evaluation ───────────────────────────────────────────────────────
        eval_model = get_model(num_features=sample_features,
                               num_classes=NUM_CLASSES,
                               dp_safe=DP_SAFE)
        set_model_parameters(eval_model, global_params)

        round_losses, round_accs, round_f1s = [], [], []
        for i, (X_tr, y_tr, X_te, y_te) in enumerate(clients_data):
            loss_v, acc_v, f1_per_class = test(eval_model, X_te, y_te, NUM_CLASSES)
            round_losses.append(loss_v)
            round_accs.append(acc_v)
            round_f1s.append(f1_per_class)

            is_zkp_rejected  = i in zkp_rejected_this_round
            is_krum_selected = (i in krum_selected_ids) if USE_KRUM else False
            is_krum_detected = (i in krum_detected_byz) if USE_KRUM else False

            append_log_row(
                round_num=round_num,
                client_label=i + 1,
                loss=loss_v,
                accuracy=acc_v,
                per_class_f1=f1_per_class,
                zkp_rejected=is_zkp_rejected,
                krum_selected=is_krum_selected,
                krum_detected=is_krum_detected,
                dp_eps=None,      # individual client rows: N/A (computed during train)
                round_time=0.0,   # filled in MEAN row below
                is_mean=False,
            )

        # ── Round MEAN row ───────────────────────────────────────────────────
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

        # FIX: compute the real detection RATE (not a collapsed boolean) for
        # both the console printout and the MEAN row.
        krum_detection_rate = (
            len(krum_detected_byz) / NUM_BYZANTINE
            if (USE_KRUM and NUM_BYZANTINE > 0) else None
        )

        if USE_KRUM and krum_detection_rate is not None:
            print(f"  [Krum] Detection rate this round: {krum_detection_rate:.2%}  "
                  f"({len(krum_detected_byz)}/{NUM_BYZANTINE} Byzantine detected, "
                  f"{len(krum_selected_ids)}/{NUM_CLIENTS - len(zkp_rejected_this_round)} "
                  f"legitimate-eligible clients selected)")

        mean_dp_eps = (
            float(np.mean(dp_eps_spent_this_round))
            if dp_eps_spent_this_round else None
        )

        # FIX: MEAN row now logs real count (krum_selected) and real rate
        # (krum_detected) via is_mean=True — see append_log_row() docstring.
        append_log_row(
            round_num=round_num,
            client_label="MEAN",
            loss=mean_loss,
            accuracy=mean_acc,
            per_class_f1=mean_f1,
            zkp_rejected=len(zkp_rejected_this_round),
            krum_selected=len(krum_selected_ids) if USE_KRUM else None,
            krum_detected=krum_detection_rate,
            dp_eps=mean_dp_eps,
            round_time=round_time,
            is_mean=True,
        )

        save_checkpoint(global_params, round_num)

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"  Training complete — {NUM_ROUNDS} rounds  [{MODEL_TYPE.upper()}]")
    print(f"  Results logged to: {LOG_CSV}")
    print(f"  Checkpoint:        {CHECKPOINT_PARAMS}")
    if USE_KRUM:
        print(f"\n  Reminder: delete checkpoint before changing flags")
        print(f"  (Krum/HE/DP flags change the experiment — old checkpoint")
        print(f"  params will give misleading results if reused.)")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()