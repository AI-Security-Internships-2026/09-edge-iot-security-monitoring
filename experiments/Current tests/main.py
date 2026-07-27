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
"""

import os
import sys
import csv
import json
import time
import warnings
import numpy as np

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
PROX_MU       = 0.1       # FedProx proximal coefficient (0 = plain FedAvg)

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
USE_ZKP  = False           # lightweight norm-bound ZKP gate

assert not (USE_KRUM and USE_HE), \
    "Multi-Krum requires plaintext parameters — cannot combine with USE_HE=True."

# DP_SAFE must match USE_DP — architecture (BatchNorm→GroupNorm, LSTM→LSTM
# with dp_safe=True) must be consistent across checkpoint init, training,
# and eval or set_model_parameters will fail on mismatched state_dict keys.
DP_SAFE = USE_DP

# Head-only attack: flips only classifier weights, stays within ZKP norm bound.
# Only meaningful when USE_HE=True (full model encrypted, subtle attack needed).
# When USE_HE=False, sign_flip_attack is always used regardless of this flag.
BYZANTINE_HEAD_ONLY = False   # set True for Experiment 2 Condition B

# DP settings (per-round; not composition-tracked — see RQ3 sweep for ε study)
DP_EPSILON      = 15.0
DP_DELTA        = 1e-5
DP_MAX_GRAD_NORM = 1.0
DP_BATCH_SIZE   = 256

# ZKP settings
ZKP_MAX_NORM = 10.0       # reject clients whose update L2-norm exceeds this

# Krum settings
KRUM_M = NUM_CLIENTS - NUM_BYZANTINE - 2   # clients to SELECT (e.g. 6 from 10)

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
                   krum_detected, dp_eps, round_time):
    row = (
        [round_num, client_label,
         f"{loss:.6f}", f"{accuracy:.6f}"]
        + [f"{v:.6f}" for v in per_class_f1]
        + [int(zkp_rejected),
           1 if krum_selected else 0,
           1 if krum_detected else 0,
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
              f"max_grad_norm={DP_MAX_GRAD_NORM}")
    if USE_KRUM:
        print(f"  Krum: selecting {KRUM_M} of {NUM_CLIENTS} clients "
              f"(f={NUM_BYZANTINE})")
    print(f"{'='*65}\n")

    # ── Load data partitions ─────────────────────────────────────────────────
    print("Loading data partitions...")
    clients_data = []
    for i in range(NUM_CLIENTS):
        print(f"  Partition {i+1}/{NUM_CLIENTS}...", end="\r")
        clients_data.append(load_partition(i, NUM_CLIENTS))
    sample_features = clients_data[0][0].shape[1]
    print(f"\nFeature count: {sample_features}")
    print(f"All {NUM_CLIENTS} clients loaded.\n")

    # ── Build shared criterion ───────────────────────────────────────────────
    criterion = build_criterion()

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

    resume = start_round > 0
    init_log_csv(resume=resume)

    # ── Experiment metadata log ──────────────────────────────────────────────
    meta_path = f"experiment_config_{_TAG}.json"
    with open(meta_path, "w") as f:
        json.dump({
            "model_type": MODEL_TYPE,
            "num_rounds": NUM_ROUNDS,
            "num_clients": NUM_CLIENTS,
            "local_epochs": LOCAL_EPOCHS,
            "prox_mu": PROX_MU,
            "byzantine_attack": USE_BYZANTINE_ATTACK,
            "num_byzantine": NUM_BYZANTINE,
            "byzantine_clients": BYZANTINE_CLIENTS,
            "attack_scale": ATTACK_SCALE,
            "use_krum": USE_KRUM,
            "krum_m": KRUM_M,
            "use_he": USE_HE,
            "use_dp": USE_DP,
            "dp_epsilon": DP_EPSILON,
            "dp_delta": DP_DELTA,
            "dp_max_grad_norm": DP_MAX_GRAD_NORM,
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
        dp_eps_spent_this_round  = []   # Bug 3 fix: collect per-client DP ε

        # ── Per-client training ──────────────────────────────────────────────
        for i, (X_tr, y_tr, X_te, y_te) in enumerate(clients_data):
            model = get_model(num_features=sample_features,
                              num_classes=NUM_CLASSES,
                              dp_safe=DP_SAFE)
            set_model_parameters(model, global_params)

            # ── Byzantine attack injection ───────────────────────────────────
            if USE_BYZANTINE_ATTACK and i in BYZANTINE_CLIENTS:
                if USE_HE and BYZANTINE_HEAD_ONLY:
                    # Head-only attack: only flip classifier weights.
                    # Stays within ZKP norm bound — more realistic under HE
                    # since the full model is encrypted and a sign-flip at
                    # ATTACK_SCALE=5.0 would have a huge norm detectable by ZKP.
                    from defences.byzantine import classifier_head_flip_attack
                    params = classifier_head_flip_attack(global_params,
                                                         scale=ATTACK_SCALE)
                    print(f"  Client {i+1:2d}  [BYZANTINE — head-only ×{ATTACK_SCALE}]")
                else:
                    params = sign_flip_attack(global_params, scale=ATTACK_SCALE)
                    print(f"  Client {i+1:2d}  [BYZANTINE — sign-flip ×{ATTACK_SCALE}]")
                dp_eps_spent = None

            else:
                # ── DP-SGD training (Opacus) ─────────────────────────────────
                if USE_DP and _OPACUS_AVAILABLE:
                    import torch
                    import torch.utils.data as tud
                    from opacus import PrivacyEngine

                    X_t = torch.FloatTensor(X_tr)
                    y_t = torch.LongTensor(y_tr)
                    loader = tud.DataLoader(
                        tud.TensorDataset(X_t, y_t),
                        batch_size=DP_BATCH_SIZE,
                        shuffle=True,
                    )
                    optimizer = __import__("torch").optim.Adam(
                        model.parameters(), lr=LEARNING_RATE
                    )
                    privacy_engine = PrivacyEngine()
                    model, optimizer, loader = privacy_engine.make_private_with_epsilon(
                        module=model,
                        optimizer=optimizer,
                        data_loader=loader,
                        target_epsilon=DP_EPSILON,
                        target_delta=DP_DELTA,
                        epochs=LOCAL_EPOCHS,
                        max_grad_norm=DP_MAX_GRAD_NORM,
                    )
                    # Manual epoch loop (Opacus wraps the loader)
                    model.train()
                    for _ in range(LOCAL_EPOCHS):
                        for X_b, y_b in loader:
                            optimizer.zero_grad()
                            loss_val = criterion(model(X_b), y_b)
                            loss_val.backward()
                            optimizer.step()

                    dp_eps_spent = privacy_engine.get_epsilon(DP_DELTA)

                    # Bug 2 fix: unwrap Opacus GradSampleModule BEFORE
                    # extracting params. Calling state_dict() on the wrapped
                    # object produces different keys than a plain model, which
                    # corrupts set_model_parameters() in the next round.
                    real_model = model._module if hasattr(model, "_module") else model
                    params = get_model_parameters(real_model)

                else:
                    # Standard FedProx training (no DP)
                    train(model, X_tr, y_tr, criterion,
                          epochs=LOCAL_EPOCHS,
                          lr=LEARNING_RATE,
                          global_params=global_params,
                          mu=PROX_MU)
                    dp_eps_spent = None

            # ── ZKP norm-bound gate ──────────────────────────────────────────
            if USE_ZKP and not (USE_BYZANTINE_ATTACK and i in BYZANTINE_CLIENTS):
                passes = zkp_verify_norm(params, max_norm=ZKP_MAX_NORM)
                if not passes:
                    print(f"  Client {i+1:2d}  [ZKP REJECTED — norm too large]")
                    zkp_rejected_this_round.append(i)
                    continue   # DO NOT add to accepted lists — skip aggregation

            # ── HE encryption (if USE_HE) ────────────────────────────────────
            if USE_HE and _TENSEAL_AVAILABLE and he_context is not None:
                import torch
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
            loss_v, acc_v, f1_per_class = test(eval_model, X_te, y_te)
            round_losses.append(loss_v)
            round_accs.append(acc_v)
            round_f1s.append(f1_per_class)

            is_zkp_rejected  = i in zkp_rejected_this_round
            is_krum_selected = i in krum_selected_ids if USE_KRUM else None
            is_krum_detected = i in krum_detected_byz if USE_KRUM else None
            # Bug 3 fix: dp_eps_spent was being overwritten with None here,
            # causing dp_epsilon_spent column to always read N/A in the CSV.
            # We don't have per-client dp_eps_spent available at eval time
            # (it was computed during training), so log None for individual
            # client rows and the true spent epsilon in the MEAN row below.

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

        if USE_KRUM and krum_detected_byz:
            pdr = len(krum_detected_byz) / max(NUM_BYZANTINE, 1)
            print(f"  [Krum] PDR this round: {pdr:.2%}  "
                  f"({len(krum_detected_byz)}/{NUM_BYZANTINE} Byzantine detected)")

        # Bug 3 fix: log the actual mean DP epsilon spent this round,
        # not None. This is what appears in dp_epsilon_spent in the CSV.
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
            krum_selected=len(krum_selected_ids),
            krum_detected=len(krum_detected_byz),
            dp_eps=mean_dp_eps,
            round_time=round_time,
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