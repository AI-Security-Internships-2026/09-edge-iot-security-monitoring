"""
FL-IDS — Full 3-Layer Privacy Stack
=====================================
Layer 1: Local DP       — intra-client user privacy (quantum-safe, info-theoretic)
Layer 2: ZKP Commitment — Byzantine protection via norm-bound proof (hash-based, quantum-safe)
Layer 3: CKKS HE        — server-side gradient blindness (lattice-based, post-quantum)

Usage:
  python src/main.py network        → network-layer model, full privacy stack
  python src/main.py application    → application-layer model, full privacy stack

Condition flags (set below):
  BYZANTINE_ATTACK: inject sign-flip poison from Byzantine clients
  USE_LOCAL_DP:     Layer 1 — Gaussian noise on local gradients
  USE_ZKP:          Layer 2 — commitment + norm proof, rejects if proof fails
  USE_HE:           Layer 3 — CKKS encryption, server aggregates on ciphertext

Data pipeline per round:
  Train → [DP] → [ZKP proof] → [CKKS encrypt] → server
  Server: [verify ZKP] → [HE aggregate] → broadcast enc(global)
  Gateway: [decrypt] → new global model

NOTE ON HE (local run vs Docker):
  This local, single-process run uses the SAME partial-HE design as
  the Docker client: only the classifier-head layers (~6% of params)
  are CKKS-encrypted; the CNN+LSTM feature-extraction layers (~94%)
  are sent DP-noised but in plaintext and weighted-averaged normally.
  This keeps local results directly comparable to the Docker numbers.
  See he_local.py's docstring for the exact CKKS parameters (n=8192,
  standard 128-bit security chain — larger than Docker's n=4096 since
  there's no RAM ceiling here, but the same partial-layer split).
"""

import os
import csv
import json
import sys
import time
import numpy as np

from data_loader import (
    load_partition_network, load_partition_application,
    NETWORK_NAMES, NUM_NETWORK_CLASSES,
    APP_NAMES,     NUM_APP_CLASSES,
)
from task import (
    get_model, get_model_parameters, get_model_parameter_keys,
    set_model_parameters,
    train, test,
    build_criterion_network, build_criterion_application,
)

# ── Model ─────────────────────────────────────────────────────────────
MODEL_TYPE = sys.argv[1] if len(sys.argv) > 1 else "network"
if MODEL_TYPE not in ("network", "application"):
    print(f"Usage: python src/main.py [network|application]")
    sys.exit(1)

# ── FL Hyperparameters ────────────────────────────────────────────────
NUM_CLIENTS     = 10
NUM_ROUNDS      = 25
LOCAL_EPOCHS    = 5
LEARNING_RATE   = 0.001
DIRICHLET_ALPHA = 0.7
SEED            = 42
PROX_MU         = 0.01

# ── Byzantine Attack ──────────────────────────────────────────────────
BYZANTINE_ATTACK  = False   # set True to inject poisoned clients
NUM_BYZANTINE     = 2
BYZANTINE_CLIENTS = [0, 1]
BYZANTINE_SCALE   = 5.0 if MODEL_TYPE == "network" else 2.0

# ── Layer 1: Local DP ─────────────────────────────────────────────────
USE_LOCAL_DP = True
DP_EPSILON   = 3.0     # privacy budget: 1.0=strong, 3.0=moderate, 10.0=weak
DP_DELTA     = 1e-5    # failure probability (standard)
DP_CLIP_NORM = 1.0     # L2 clipping bound

# ── Layer 2: ZKP Commitment ───────────────────────────────────────────
USE_ZKP = True
# Clients that fail ZKP verification are DROPPED before HE aggregation
# This is the primary Byzantine protection mechanism in this stack

# ── Layer 3: CKKS Homomorphic Encryption ─────────────────────────────
USE_HE             = True
HE_POLY_DEGREE     = 8192   # 128-bit post-quantum security (RLWE)
# Set to 16384 for 256-bit security at ~4x compute cost

# ── Derive from MODEL_TYPE ────────────────────────────────────────────
if MODEL_TYPE == "network":
    ATTACK_NAMES    = NETWORK_NAMES
    NUM_CLASSES     = NUM_NETWORK_CLASSES
    load_partition  = load_partition_network
    build_criterion = build_criterion_network
else:
    ATTACK_NAMES    = APP_NAMES
    NUM_CLASSES     = NUM_APP_CLASSES
    load_partition  = load_partition_application
    build_criterion = build_criterion_application

# ── Output files ──────────────────────────────────────────────────────
parts = [MODEL_TYPE]
if USE_LOCAL_DP: parts.append("dp")
if USE_ZKP:      parts.append("zkp")
if USE_HE:       parts.append("he")
if BYZANTINE_ATTACK: parts.append(f"byz{NUM_BYZANTINE}")
CONDITION = "_".join(parts)

CHECKPOINT_PATH = f"fl_checkpoint_{CONDITION}.npz"
PROGRESS_PATH   = f"fl_progress_{CONDITION}.json"
LOG_PATH        = f"fl_results_{CONDITION}.csv"
PRIVACY_LOG     = f"fl_privacy_{CONDITION}.csv"


# ── Standard FedAvg (fallback when USE_HE=False) ─────────────────────

def fedavg(all_params, weights):
    total  = sum(weights)
    result = []
    for layer_idx in range(len(all_params[0])):
        layer_avg = sum(
            p[layer_idx] * (w / total)
            for p, w in zip(all_params, weights)
        )
        result.append(layer_avg)
    return result


# ── Checkpointing ─────────────────────────────────────────────────────

def save_checkpoint(params, round_num):
    np.savez(CHECKPOINT_PATH, *params)
    with open(PROGRESS_PATH, "w") as f:
        json.dump({"last_round": round_num}, f)


def load_checkpoint():
    if not (os.path.exists(CHECKPOINT_PATH)
            and os.path.exists(PROGRESS_PATH)):
        return None, 0
    data   = np.load(CHECKPOINT_PATH)
    params = [data[f"arr_{i}"] for i in range(len(data.files))]
    with open(PROGRESS_PATH) as f:
        last = json.load(f)["last_round"]
    return params, last


# ── Logging ───────────────────────────────────────────────────────────

def init_logs():
    with open(LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow(
            ["round", "client", "loss", "accuracy"] + ATTACK_NAMES
        )
    with open(PRIVACY_LOG, "w", newline="") as f:
        csv.writer(f).writerow([
            "round", "client",
            "dp_epsilon", "dp_delta", "dp_clip_norm",
            "dp_actual_norm", "dp_noise_sigma",
            "zkp_norm", "zkp_threshold", "zkp_passed",
            "he_mode", "he_n_chunks",
            "round_time_s"
        ])


def log_result(round_num, client_id, loss, acc, f1s):
    import time as _time
    for attempt in range(5):
        try:
            with open(LOG_PATH, "a", newline="") as f:
                csv.writer(f).writerow(
                    [round_num, client_id,
                     f"{loss:.4f}", f"{acc:.4f}"]
                    + [f"{v:.4f}" for v in f1s]
                )
            return
        except PermissionError:
            if attempt == 4:
                print(f"  [WARNING] Cannot write to {LOG_PATH} — close in Excel")
                return
            _time.sleep(1)


def log_privacy(round_num, client_id, dp_info, zkp_proof, zkp_passed,
                he_mode, he_n_chunks, round_time):
    pi = zkp_proof["norm_proof"] if zkp_proof else {}
    with open(PRIVACY_LOG, "a", newline="") as f:
        csv.writer(f).writerow([
            round_num, client_id,
            dp_info.get("epsilon", ""),
            dp_info.get("delta", ""),
            dp_info.get("clip_norm", ""),
            f"{dp_info.get('actual_norm', 0):.4f}",
            f"{dp_info.get('noise_sigma', 0):.4f}",
            f"{pi.get('norm', 0):.4f}",
            f"{pi.get('threshold', 0):.4f}",
            int(zkp_passed),
            he_mode,
            he_n_chunks,
            f"{round_time:.2f}",
        ])


# ── Display ───────────────────────────────────────────────────────────

def print_partition(cid, X_tr, y_tr, X_te, y_te):
    print(f"\n  Client {cid+1:>2} │ "
          f"train={len(X_tr):>7,}  test={len(X_te):>6,}")
    tr_c = np.bincount(y_tr.astype(int), minlength=NUM_CLASSES)
    te_c = np.bincount(y_te.astype(int), minlength=NUM_CLASSES)
    for i, name in enumerate(ATTACK_NAMES):
        bar  = "█" * min(30, tr_c[i] // 100)
        flag = " ← missing" if tr_c[i] == 0 else ""
        print(f"             {name:<25} "
              f"train={tr_c[i]:>6,}  test={te_c[i]:>5,}  {bar}{flag}")


def print_round_summary(rnd, losses, accs, all_f1s):
    valid    = [l for l in losses if np.isfinite(l)]
    mean_loss = float(np.mean(valid)) if valid else float('nan')
    mean_acc  = float(np.mean(accs))
    mean_f1   = np.mean(all_f1s, axis=0)
    macro     = float(np.mean(mean_f1))

    print(f"\n── Round {rnd} summary ──")
    loss_str = f"{mean_loss:.4f}" if np.isfinite(mean_loss) else "nan"
    print(f"  Loss: {loss_str}  │  Accuracy: {mean_acc:.4f}  │  "
          f"F1-Macro: {macro:.4f}")
    for name, f1 in zip(ATTACK_NAMES, mean_f1):
        bar  = "█" * int(f1 * 20)
        flag = " ◄ low" if f1 < 0.3 else ""
        print(f"    {name:<25} F1: {f1:.4f}  {bar}{flag}")
    return mean_loss, mean_acc, mean_f1


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print(f"FL-IDS  │  MODEL={MODEL_TYPE.upper()}  │  Condition: {CONDITION}")
    print(f"FedProx μ={PROX_MU}  │  α={DIRICHLET_ALPHA}  │  Seed={SEED}")
    print()

    # Privacy stack summary
    if USE_LOCAL_DP:
        print(f"🔒 Layer 1 — Local DP: ε={DP_EPSILON}, δ={DP_DELTA}, "
              f"clip={DP_CLIP_NORM}  [quantum-safe: info-theoretic]")
    if USE_ZKP:
        print(f"🔐 Layer 2 — ZKP Commitment: HMAC-SHA256 norm proof  "
              f"[quantum-safe: hash-based]")
    if USE_HE:
        print(f"🛡  Layer 3 — CKKS HE: poly_degree={HE_POLY_DEGREE}  "
              f"[quantum-safe: RLWE lattice]")
    if BYZANTINE_ATTACK:
        print(f"⚠  Byzantine: clients {[c+1 for c in BYZANTINE_CLIENTS]} "
              f"poisoned (scale={BYZANTINE_SCALE})")
    print("=" * 70)

    # ── Imports conditioned on flags ──────────────────────────────────
    if USE_LOCAL_DP:
        from defences.local_dp import apply_local_dp

    if USE_ZKP:
        from defences.zkp import (generate_proof, verify_proof,
                                   print_verification)

    if USE_HE:
        # he_local.py bridges main.py's expected API onto
        # he_aggregation.py's primitives, configured for a local,
        # full-model (not partial) encryption run — see this file's
        # top-of-module note and he_local.py's docstring.
        from he_local import (
            create_ckks_context, encrypt_params,
            aggregate_encrypted, decrypt_params, benchmark_he
        )

    if BYZANTINE_ATTACK:
        from defences.byzantine import sign_flip_attack

    # ── Load data ─────────────────────────────────────────────────────
    print("\nLOADING DATA PARTITIONS")
    print("-" * 70)

    clients_data = []
    criterion    = build_criterion()

    for i in range(NUM_CLIENTS):
        print(f"\nPartition {i+1}/{NUM_CLIENTS}...")
        data = load_partition(
            partition_id   = i,
            num_partitions = NUM_CLIENTS,
            alpha          = DIRICHLET_ALPHA,
            seed           = SEED
        )
        clients_data.append(data)
        X_tr, y_tr, X_te, y_te = data
        print_partition(i, X_tr, y_tr, X_te, y_te)

    num_features = clients_data[0][0].shape[1]

    print(f"\n{'='*70}")
    print(f"Features: {num_features}  │  Classes: {NUM_CLASSES}  │  "
          f"Clients: {NUM_CLIENTS}  │  Rounds: {NUM_ROUNDS}")
    print(f"Epochs: {LOCAL_EPOCHS}/round  │  Output: {LOG_PATH}")
    print(f"{'='*70}\n")

    # ── HE context (created once, reused every round) ─────────────────
    # Parameter key names are the same for every client (identical
    # architecture) — computed once here and reused every round to
    # split each update into sensitive (classifier head) vs bulk.
    param_keys = get_model_parameter_keys(
        get_model(num_features=num_features, num_classes=NUM_CLASSES)
    )

    he_context = None
    if USE_HE:
        print("Initialising CKKS context...")
        he_context = create_ckks_context(HE_POLY_DEGREE)

        # Benchmark on a dummy model to set expectations
        dummy = get_model(num_features=num_features, num_classes=NUM_CLASSES)
        dummy_params = get_model_parameters(dummy)
        benchmark_he(he_context, dummy_params, param_keys, HE_POLY_DEGREE, NUM_CLIENTS)

    # ── Checkpoint ────────────────────────────────────────────────────
    global_params, last_round = load_checkpoint()
    if global_params is not None:
        print(f"Resuming from checkpoint — last round: {last_round}")
    else:
        last_round = 0
        init_logs()
        global_params = get_model_parameters(
            get_model(num_features=num_features, num_classes=NUM_CLASSES)
        )

    # ── Training loop ─────────────────────────────────────────────────
    for rnd in range(last_round + 1, NUM_ROUNDS + 1):
        round_start = time.time()

        print(f"\n{'='*70}")
        print(f"ROUND {rnd}/{NUM_ROUNDS}  [{MODEL_TYPE.upper()} | {CONDITION}]")
        print(f"{'='*70}")

        # ── Local training + privacy pipeline ────────────────────────
        print("\n── Local training + privacy pipeline ──")

        accepted_params   = []   # params to aggregate (plaintext or encrypted)
        accepted_weights  = []
        rejected_clients  = []
        dp_logs           = {}
        zkp_logs          = {}
        he_meta           = {"mode": "none", "n_chunks": 0}

        for i, (X_tr, y_tr, _, _) in enumerate(clients_data):
            cid = i + 1

            # Byzantine injection (skips real training)
            if BYZANTINE_ATTACK and i in BYZANTINE_CLIENTS:
                raw_params = sign_flip_attack(global_params, BYZANTINE_SCALE)
                print(f"  Client {cid:>2} ✗  BYZANTINE — "
                      f"sign-flip scale={BYZANTINE_SCALE}")
                dp_info = {}
            else:
                # Real local training
                model = get_model(num_features=num_features,
                                  num_classes=NUM_CLASSES)
                set_model_parameters(model, global_params)
                train(model, X_tr, y_tr, criterion,
                      epochs        = LOCAL_EPOCHS,
                      lr            = LEARNING_RATE,
                      global_params = global_params,
                      mu            = PROX_MU)
                raw_params = get_model_parameters(model)
                dp_info    = {}
                print(f"  Client {cid:>2} ✓  n={len(X_tr):>7,}", end="")

            # ── Layer 1: Local DP ────────────────────────────────────
            if USE_LOCAL_DP:
                raw_params, dp_info = apply_local_dp(
                    raw_params,
                    epsilon  = DP_EPSILON,
                    delta    = DP_DELTA,
                    clip_norm= DP_CLIP_NORM
                )
                if not BYZANTINE_ATTACK or i not in BYZANTINE_CLIENTS:
                    print(f"  [DP] ε={dp_info['epsilon']} "
                          f"σ={dp_info['noise_sigma']:.3f} "
                          f"‖g‖={dp_info['actual_norm']:.3f}", end="")

            dp_logs[cid] = dp_info

            # ── Layer 2: ZKP Commitment ──────────────────────────────
            zkp_proof  = None
            zkp_passed = True

            if USE_ZKP:
                noise_sigma = dp_info.get("noise_sigma", 0.0)
                zkp_proof   = generate_proof(
                    raw_params,
                    clip_norm    = DP_CLIP_NORM,
                    noise_sigma  = noise_sigma
                )
                # Verify — in real deployment server does this
                # Here we simulate server verification after receipt
                zkp_passed, reason = verify_proof(
                    zkp_proof,
                    params               = raw_params,
                    clip_norm            = DP_CLIP_NORM,
                    verify_commitment_flag = True
                )
                zkp_logs[cid] = (zkp_proof, zkp_passed, reason)

                if not BYZANTINE_ATTACK or i not in BYZANTINE_CLIENTS:
                    print()  # newline after DP stats
                print_verification(cid, zkp_proof, zkp_passed, reason)

                if not zkp_passed:
                    rejected_clients.append(cid)
                    log_privacy(rnd, cid, dp_info, zkp_proof, False,
                                "rejected", 0, 0)
                    continue  # DROP this client — never reaches aggregation

            # ── Layer 3: CKKS Encryption (classifier head only) ──────
            if USE_HE:
                enc = encrypt_params(raw_params, param_keys, he_context, HE_POLY_DEGREE)
                accepted_params.append(enc)
                he_meta = {
                    "mode":     enc["mode"],
                    "n_chunks": enc.get("n_chunks", 1)
                }
            else:
                accepted_params.append(raw_params)

            accepted_weights.append(len(X_tr))
            log_privacy(rnd, cid, dp_info, zkp_proof, zkp_passed,
                        he_meta["mode"], he_meta["n_chunks"], 0)

        # ── Aggregation ───────────────────────────────────────────────
        if not accepted_params:
            print("\n  [WARNING] All clients rejected by ZKP — "
                  "using previous global model")
        else:
            if rejected_clients:
                print(f"\n  ZKP rejected {len(rejected_clients)} clients: "
                      f"{rejected_clients}")
            print(f"  Aggregating {len(accepted_params)} accepted clients "
                  f"(total weight: {sum(accepted_weights):,})")

            if USE_HE:
                print("  Homomorphic aggregation (server blind)...")
                t0            = time.time()
                enc_aggregate = aggregate_encrypted(
                    accepted_params, accepted_weights, he_context
                )
                t_agg = time.time() - t0

                print("  Decrypting aggregate (gateway-side)...")
                t0            = time.time()
                global_params = decrypt_params(enc_aggregate)
                t_dec         = time.time() - t0

                print(f"  HE aggregate: {t_agg:.1f}s | "
                      f"decrypt: {t_dec:.1f}s")
            else:
                global_params = fedavg(accepted_params, accepted_weights)

        # ── Evaluation ────────────────────────────────────────────────
        print(f"\n── Per-client evaluation ──")
        losses, accs, all_f1s = [], [], []

        for i, (_, _, X_te, y_te) in enumerate(clients_data):
            model = get_model(num_features=num_features,
                              num_classes=NUM_CLASSES)
            set_model_parameters(model, global_params)
            loss, acc, f1s = test(model, X_te, y_te, NUM_CLASSES)

            losses.append(loss)
            accs.append(acc)
            all_f1s.append(f1s)

            cid = i + 1
            print(f"\n  Client {cid:>2} │ loss={loss:.4f}  acc={acc:.4f}")
            for name, f1 in zip(ATTACK_NAMES, f1s):
                bar  = "█" * int(f1 * 20)
                flag = " ◄ low" if f1 < 0.3 else ""
                print(f"             {name:<25} F1: {f1:.4f}  {bar}{flag}")
            log_result(rnd, cid, loss, acc, f1s)

        mean_loss, mean_acc, mean_f1 = print_round_summary(
            rnd, losses, accs, all_f1s
        )
        log_result(rnd, "MEAN", mean_loss, mean_acc, mean_f1)

        round_time = time.time() - round_start
        print(f"\n  Round time: {round_time:.1f}s  "
              f"({round_time/60:.1f} min)")

        save_checkpoint(global_params, rnd)
        print(f"  [Checkpoint saved — round {rnd}/{NUM_ROUNDS}]")

    print(f"\n{'='*70}")
    print(f"Complete.")
    print(f"  Results  → {LOG_PATH}")
    print(f"  Privacy  → {PRIVACY_LOG}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
