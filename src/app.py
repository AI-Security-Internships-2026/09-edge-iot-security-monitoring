"""
FL-IDS Dual-Model Training — Byzantine Attack + Multi-Krum Defence
===================================================================
Usage:
  python src/main.py network      → network-layer model
  python src/main.py application  → application-layer model

EXPERIMENT CONDITIONS (set flags below):
  BYZANTINE_ATTACK=False, USE_KRUM=False  → clean baseline
  BYZANTINE_ATTACK=True,  USE_KRUM=False  → attack, no defence
  BYZANTINE_ATTACK=True,  USE_KRUM=True   → attack + Multi-Krum
  BYZANTINE_ATTACK=False, USE_KRUM=True   → Krum on clean data

BYZANTINE SCALE:
  Network model:     BYZANTINE_SCALE = 5.0  (causes NaN collapse in FedAvg)
  Application model: BYZANTINE_SCALE = 2.0  (measurable without NaN overflow)
"""

import os
import csv
import json
import sys
import numpy as np

from data_loader import (
    load_partition_network, load_partition_application,
    NETWORK_NAMES, NUM_NETWORK_CLASSES,
    APP_NAMES,     NUM_APP_CLASSES,
)
from task import (
    get_model, get_model_parameters, set_model_parameters,
    train, test,
    build_criterion_network, build_criterion_application,
)

# ── Model selection ───────────────────────────────────────────────────
MODEL_TYPE = sys.argv[1] if len(sys.argv) > 1 else "network"

if MODEL_TYPE not in ("network", "application"):
    print(f"Unknown model type: {MODEL_TYPE}")
    print("Usage: python src/main.py [network|application]")
    sys.exit(1)

# ── Hyperparameters ───────────────────────────────────────────────────
NUM_CLIENTS     = 10
NUM_ROUNDS      = 25
LOCAL_EPOCHS    = 5
LEARNING_RATE   = 0.001
DIRICHLET_ALPHA = 0.7
SEED            = 42
PROX_MU         = 0.01

# ── Attack / defence flags ────────────────────────────────────────────
BYZANTINE_ATTACK  = True
USE_KRUM          = True
NUM_BYZANTINE     = 1
BYZANTINE_CLIENTS = [0, 1]    # 0-indexed → Client 1 and Client 2

# Scale of sign-flip attack:
#   network model     → 5.0 (causes NaN collapse in undefended FedAvg)
#   application model → 2.0 (prevents NaN overflow in smaller gradients)
BYZANTINE_SCALE = 5.0 if MODEL_TYPE == "network" else 2.0

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

# ── Output file naming ────────────────────────────────────────────────
if BYZANTINE_ATTACK and USE_KRUM:
    CONDITION = "attack_krum"
elif BYZANTINE_ATTACK:
    CONDITION = "attack_nodefence"
elif USE_KRUM:
    CONDITION = "krum_clean"
else:
    CONDITION = "baseline"

CHECKPOINT_PATH = f"fl_checkpoint_{MODEL_TYPE}_{CONDITION}.npz"
PROGRESS_PATH   = f"fl_progress_{MODEL_TYPE}_{CONDITION}.json"
LOG_PATH        = f"fl_results_{MODEL_TYPE}_{CONDITION}.csv"


# ── Aggregation ───────────────────────────────────────────────────────

def fedavg(all_params, weights):
    """Standard FedAvg — weighted average of all client updates."""
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

def init_log():
    with open(LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow(
            ["round", "client", "loss", "accuracy"] + ATTACK_NAMES
        )


def log_row(round_num, client_id, loss, acc, f1s):
    import time
    for attempt in range(5):
        try:
            with open(LOG_PATH, "a", newline="") as f:
                csv.writer(f).writerow(
                    [round_num, client_id,
                     f"{loss:.4f}" if loss == loss else "nan",
                     f"{acc:.4f}"]
                    + [f"{v:.4f}" for v in f1s]
                )
            return
        except PermissionError:
            if attempt == 4:
                print(f"  [WARNING] Could not write log — "
                      f"close {LOG_PATH} in Excel")
                return
            time.sleep(1)


# ── Display ───────────────────────────────────────────────────────────

def print_partition_summary(cid, X_tr, y_tr, X_te, y_te):
    print(f"\n  Client {cid+1:>2} │ "
          f"train={len(X_tr):>7,}  test={len(X_te):>6,}")
    tr_c = np.bincount(y_tr.astype(int), minlength=NUM_CLASSES)
    te_c = np.bincount(y_te.astype(int), minlength=NUM_CLASSES)
    for i, name in enumerate(ATTACK_NAMES):
        bar  = "█" * min(30, tr_c[i] // 100)
        flag = " ← missing" if tr_c[i] == 0 else ""
        print(f"             {name:<25} "
              f"train={tr_c[i]:>6,}  test={te_c[i]:>5,}  {bar}{flag}")


def print_aggregation_weights(weights):
    total = sum(weights)
    print(f"\n── FedAvg aggregation weights ──")
    for i, w in enumerate(weights):
        pct = 100 * w / total
        bar = "█" * int(pct / 2)
        tag = " ← BYZANTINE" \
              if BYZANTINE_ATTACK and i in BYZANTINE_CLIENTS else ""
        print(f"  Client {i+1:>2}  {pct:5.1f}%  {bar}{tag}")


def print_round_summary(round_num, losses, accs, all_f1s):
    valid_losses = [l for l in losses if l == l]   # filter NaN
    mean_loss = float(np.mean(valid_losses)) if valid_losses else float('nan')
    mean_acc  = float(np.mean(accs))
    mean_f1   = np.mean(all_f1s, axis=0)
    macro_f1  = float(np.mean(mean_f1))

    print(f"\n── Round {round_num} summary ──")
    loss_str = f"{mean_loss:.4f}" if mean_loss == mean_loss else "nan"
    print(f"  Loss: {loss_str}  │  "
          f"Accuracy: {mean_acc:.4f}  │  "
          f"F1-Macro: {macro_f1:.4f}")
    for name, f1 in zip(ATTACK_NAMES, mean_f1):
        bar  = "█" * int(f1 * 20)
        flag = " ◄ low" if f1 < 0.3 else ""
        print(f"    {name:<25} F1: {f1:.4f}  {bar}{flag}")

    return mean_loss, mean_acc, mean_f1


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print(f"FL-IDS  │  MODEL={MODEL_TYPE.upper()}  │  "
          f"Condition={CONDITION.upper()}")
    print(f"FedProx μ={PROX_MU}  │  "
          f"Dirichlet α={DIRICHLET_ALPHA}  │  Seed={SEED}")
    if BYZANTINE_ATTACK:
        print(f"⚠  BYZANTINE ATTACK: clients "
              f"{[c+1 for c in BYZANTINE_CLIENTS]} "
              f"sign-flip scale={BYZANTINE_SCALE}")
    if USE_KRUM:
        print(f"🛡  DEFENCE: Multi-Krum (f={NUM_BYZANTINE}, "
              f"selecting {NUM_CLIENTS - NUM_BYZANTINE - 2} clients/round)")
    print("=" * 70)

    global_params, last_round = load_checkpoint()
    if global_params is not None:
        print(f"\nResuming from checkpoint — "
              f"last completed round: {last_round}")
    else:
        last_round = 0

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
        print_partition_summary(i, X_tr, y_tr, X_te, y_te)

    num_features = clients_data[0][0].shape[1]

    print(f"\n{'=' * 70}")
    print(f"Features : {num_features}  │  Classes : {NUM_CLASSES}")
    print(f"Clients  : {NUM_CLIENTS}   │  Rounds  : {NUM_ROUNDS}")
    print(f"Epochs   : {LOCAL_EPOCHS} per round")
    print(f"{'=' * 70}\n")

    if global_params is None:
        init_log()
        global_params = get_model_parameters(
            get_model(
                num_features = num_features,
                num_classes  = NUM_CLASSES
            )
        )

    # Import attack/defence modules
    if BYZANTINE_ATTACK:
        from defences.byzantine import sign_flip_attack
    if USE_KRUM:
        from defences.krum import multi_krum

    for rnd in range(last_round + 1, NUM_ROUNDS + 1):
        print(f"\n{'=' * 70}")
        print(f"ROUND {rnd}/{NUM_ROUNDS}  "
              f"[{MODEL_TYPE.upper()} | {CONDITION.upper()}]")
        print(f"{'=' * 70}")

        print("\n── Local training ──")
        all_params = []
        weights    = []

        for i, (X_tr, y_tr, _, _) in enumerate(clients_data):

            if BYZANTINE_ATTACK and i in BYZANTINE_CLIENTS:
                poisoned = sign_flip_attack(
                    global_params, scale=BYZANTINE_SCALE
                )
                all_params.append(poisoned)
                weights.append(len(X_tr))
                print(f"  Client {i+1:>2} ✗  BYZANTINE — "
                      f"sign-flip scale={BYZANTINE_SCALE}")
                continue

            model = get_model(
                num_features = num_features,
                num_classes  = NUM_CLASSES
            )
            set_model_parameters(model, global_params)

            train(
                model, X_tr, y_tr, criterion,
                epochs        = LOCAL_EPOCHS,
                lr            = LEARNING_RATE,
                global_params = global_params,
                mu            = PROX_MU
            )

            all_params.append(get_model_parameters(model))
            weights.append(len(X_tr))
            print(f"  Client {i+1:>2} ✓  n={len(X_tr):>7,}")

        print_aggregation_weights(weights)

        # ── Aggregation ───────────────────────────────────────────────
        if USE_KRUM:
            global_params = multi_krum(
                all_params, weights,
                num_byzantine = NUM_BYZANTINE
            )
        else:
            global_params = fedavg(all_params, weights)

        # ── Evaluation ────────────────────────────────────────────────
        print(f"\n── Per-client evaluation ──")
        losses, accs, all_f1s = [], [], []

        for i, (_, _, X_te, y_te) in enumerate(clients_data):
            model = get_model(
                num_features = num_features,
                num_classes  = NUM_CLASSES
            )
            set_model_parameters(model, global_params)
            loss, acc, f1s = test(model, X_te, y_te, NUM_CLASSES)

            losses.append(loss)
            accs.append(acc)
            all_f1s.append(f1s)

            print(f"\n  Client {i+1:>2} │ "
                  f"loss={loss:.4f}  acc={acc:.4f}")
            for name, f1 in zip(ATTACK_NAMES, f1s):
                bar  = "█" * int(f1 * 20)
                flag = " ◄ low" if f1 < 0.3 else ""
                print(f"             {name:<25} "
                      f"F1: {f1:.4f}  {bar}{flag}")

            log_row(rnd, i + 1, loss, acc, f1s)

        mean_loss, mean_acc, mean_f1 = print_round_summary(
            rnd, losses, accs, all_f1s
        )
        log_row(rnd, "MEAN", mean_loss, mean_acc, mean_f1)
        save_checkpoint(global_params, rnd)
        print(f"\n  [Checkpoint saved — round {rnd}/{NUM_ROUNDS}]")

    print(f"\n{'=' * 70}")
    print(f"Complete.  Results → {LOG_PATH}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()