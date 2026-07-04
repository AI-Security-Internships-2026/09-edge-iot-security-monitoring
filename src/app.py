"""
FL-IDS Dual-Model Training
===========================
Usage:
  python src/main.py network      → Model 1: network-layer attacks
  python src/main.py application  → Model 2: application-layer attacks

Algorithm    : FedProx (mu=0.01)
Partitioning : Dirichlet alpha=0.7, seed=42 (non-IID)
Dataset      : Edge-IIoTset DNN subset, DDoS_TCP capped to 18%

Network model features   : ~40 (after VarianceThreshold)
Application model features: ~52 (no VarianceThreshold — HTTP features kept)
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

# ── Config ────────────────────────────────────────────────────────────
MODEL_TYPE = sys.argv[1] if len(sys.argv) > 1 else "network"

NUM_CLIENTS     = 10
NUM_ROUNDS      = 25
LOCAL_EPOCHS    = 5
LEARNING_RATE   = 0.001
DIRICHLET_ALPHA = 0.7
SEED            = 42
PROX_MU         = 0.01

if MODEL_TYPE == "network":
    ATTACK_NAMES    = NETWORK_NAMES
    NUM_CLASSES     = NUM_NETWORK_CLASSES
    load_partition  = load_partition_network
    build_criterion = build_criterion_network
elif MODEL_TYPE == "application":
    ATTACK_NAMES    = APP_NAMES
    NUM_CLASSES     = NUM_APP_CLASSES
    load_partition  = load_partition_application
    build_criterion = build_criterion_application
else:
    print(f"Unknown model type: {MODEL_TYPE}")
    print("Usage: python src/main.py [network|application]")
    sys.exit(1)

CHECKPOINT_PATH = f"fl_checkpoint_{MODEL_TYPE}.npz"
PROGRESS_PATH   = f"fl_progress_{MODEL_TYPE}.json"
LOG_PATH        = f"fl_results_{MODEL_TYPE}.csv"


# ── Aggregation ───────────────────────────────────────────────────────

def fedavg(all_params, weights):
    """
    Standard FedAvg aggregation (McMahan et al., 2017).

    DEFENCE HOOK — Krum / Multi-Krum:
        from defences.krum import multi_krum
        return multi_krum(all_params, weights, num_byzantine=2)
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
                     f"{loss:.4f}", f"{acc:.4f}"]
                    + [f"{v:.4f}" for v in f1s]
                )
            return
        except PermissionError:
            if attempt == 4:
                print(f"  [WARNING] Could not write log row — "
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
        print(f"  Client {i+1:>2}  {pct:5.1f}%  {bar}")


def print_round_summary(round_num, losses, accs, all_f1s):
    mean_loss = float(np.mean(losses))
    mean_acc  = float(np.mean(accs))
    mean_f1   = np.mean(all_f1s, axis=0)
    macro_f1  = float(np.mean(mean_f1))

    print(f"\n── Round {round_num} summary ──")
    print(f"  Loss: {mean_loss:.4f}  │  "
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
          f"FedProx μ={PROX_MU}  │  "
          f"Dirichlet α={DIRICHLET_ALPHA}  │  Seed={SEED}")
    print(f"Classes : {ATTACK_NAMES}")
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

    for rnd in range(last_round + 1, NUM_ROUNDS + 1):
        print(f"\n{'=' * 70}")
        print(f"ROUND {rnd}/{NUM_ROUNDS}  [{MODEL_TYPE.upper()}]")
        print(f"{'=' * 70}")

        print("\n── Local training ──")
        all_params = []
        weights    = []

        for i, (X_tr, y_tr, _, _) in enumerate(clients_data):

            # DEFENCE HOOK — Byzantine injection:
            # from defences.byzantine import poison_update
            # if i in BYZANTINE_CLIENTS:
            #     all_params.append(poison_update(global_params))
            #     weights.append(len(X_tr))
            #     continue

            model = get_model(
                num_features = num_features,
                num_classes  = NUM_CLASSES
            )
            set_model_parameters(model, global_params)

            # DEFENCE HOOK — DP-SGD:
            # from defences.dp_sgd import make_private
            # model, optimizer = make_private(model, epsilon=3.0)

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

        # DEFENCE HOOK — FLDetector:
        # from defences.fldetector import fldetector_filter
        # all_params, weights = fldetector_filter(
        #     all_params, weights, global_params, round_num=rnd
        # )

        # DEFENCE HOOK — Krum / Multi-Krum:
        # from defences.krum import multi_krum
        # global_params = multi_krum(all_params, weights, num_byzantine=2)

        global_params = fedavg(all_params, weights)

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
    print(f"{MODEL_TYPE.upper()} model complete.  Results → {LOG_PATH}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()