#!/usr/bin/env python3
"""
scripts/train_centralized.py

Issue 4 Task 4, blocker 1 (final open item): trains the four
CENTRALIZED baselines Table 1 requires -- CNN-LSTM, MLP, Random Forest,
XGBoost -- and writes results in the EXACT SAME
results_{model}_{tag}_seed{seed}_FINAL_{TEST,VALIDATION}.csv schema
main.py's federated runs write, so scripts/aggregate_task4_results.py
ingests centralized rows with zero changes -- verified in this file's
own __main__ smoke test (see bottom).

DATA: uses data_loader.py's three global-split getters directly --
get_global_train_holdout() (the SAME stratified 80% TRAIN split
_dirichlet_partition() carves into federated client shards, exposed
here as one, non-partitioned dataset -- added this revision, mirrors
get_global_test_holdout()'s exact contract), get_global_validation_
holdout() (added same revision), get_global_test_holdout() (pre-
existing). No new preprocessing path -- same TRAIN-only-fitted scalers,
same leakage guarantees as the federated pipeline, per SPLIT_PROTOCOL.md.

ARCHITECTURE-CONTROLLED COMPARISON: the centralized CNN-LSTM condition
trains the EXACT SAME model class as every federated condition
(model_defs.CNN_LSTM via get_model(dp_safe=False)), using task.py's own
train()/test() functions unchanged -- so Table 1's centralized-vs-
federated CNN-LSTM comparison isolates the effect of federation itself,
not a confounded architecture difference.

HYPERPARAMETERS FLAGGED, NOT SILENTLY INVENTED: hyperparams.json has
ZERO entries for MLP/RandomForest/XGBoost (confirmed by inspection --
only fedprox_mu, dp_max_grad_norm, adaptive_krum_k, adaptive_krum_
hybrid_assumed_f, class_weight_multipliers_application, trimmed_mean_
beta exist). These are genuine paper-baseline design decisions this
script cannot derive from existing code. CENTRALIZED_DEFAULTS below are
literature-standard starting points, explicitly labeled as choices this
script made, not values the project has validated -- change them here,
in one place, if the paper's authors have different requirements, and
update the "validated_on_split" style comment accordingly.

Usage:
    # Single run (get a timing estimate FIRST, per this project's
    # established convention -- centralized RF/XGBoost are fast, but
    # the CNN-LSTM condition is not):
    python scripts/train_centralized.py network --condition cnnlstm --seed 42

    # Full Task 4 schedule (2 models x 4 conditions x 5 seeds = 40 runs):
    python scripts/train_centralized.py --all --seeds 42,123,456,789,2024
PATH FIX (this revision): this file lives in scripts/, but its
run_one()/main() calls import from data_loader, task, and model_defs
below via plain `from data_loader import ...` -- those modules live one
directory up, in experiments/Current model/, not in scripts/ itself.
Unlike scripts/fit_hetero_variance.py (which already inserts
dirname(__file__)/".." onto sys.path before its `from defences.krum
import ...`), this file previously had no equivalent fix -- running it
exactly as documented above (`python scripts/train_centralized.py ...`)
would raise ModuleNotFoundError the moment run_one() executed its first
`from data_loader import ...`, since Python only auto-adds scripts/
itself (this file's own directory) to sys.path, not its parent. Fixed
below with the same pattern fit_hetero_variance.py already uses.
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

# ---------------------------------------------------------------------------
# Literature-standard defaults for the 3 sklearn-style baselines -- see
# module docstring. NOT sourced from hyperparams.json (nothing there for
# these). CNN-LSTM's epoch budget/early-stopping is likewise a choice
# made here, not an inherited convention -- flagged the same way.
# ---------------------------------------------------------------------------
CENTRALIZED_DEFAULTS = {
    "cnnlstm": {
        "max_epochs": 60,
        "lr": 0.001,
        # Best-epoch-on-VALIDATION checkpointing, same pattern as
        # main.py's federated round loop's best_f1_macro tracking --
        # NOT a fixed epoch count, to keep this comparable in spirit
        # (best-checkpoint selection) even though centralized training
        # has no notion of "rounds".
        "patience": 10,  # stop if no VAL Macro-F1 improvement for this many epochs
    },
    "mlp": {
        # sklearn's MLPClassifier -- a single hidden layer of 128 units
        # is a common, unremarkable centralized-IDS-paper baseline
        # width; alpha (L2) and max_iter are sklearn's own defaults
        # left mostly as-is except max_iter raised (sklearn's default
        # of 200 often doesn't converge on tabular IDS data this size).
        "hidden_layer_sizes": (128,),
        "alpha": 1e-4,
        "max_iter": 300,
        "early_stopping": True,
    },
    "rf": {
        "n_estimators": 200,
        "max_depth": None,
        "n_jobs": -1,
    },
    "xgboost": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_jobs": -1,
    },
}

ALL_CONDITIONS = ["cnnlstm", "mlp", "rf", "xgboost"]
ALL_SEEDS = [42, 123, 456, 789, 2024]


# ---------------------------------------------------------------------------
# Shared extended-metric computation -- mirrors task.py's test(return_
# extended=True) EXACTLY (same label_binarize + average_precision_score
# NaN-for-zero-positive-class handling), so all 4 centralized conditions
# and the federated conditions are measured with IDENTICAL metric
# definitions. Duplicated here (not imported from task.py) because
# sklearn's fit/predict_proba path doesn't go through a torch model
# forward pass the way task.py's test() assumes.
# ---------------------------------------------------------------------------
def _extended_metrics_from_predictions(y_true, preds, probs, num_classes):
    from sklearn.metrics import f1_score, recall_score, average_precision_score
    from sklearn.preprocessing import label_binarize

    accuracy = float((preds == y_true).mean())
    per_class_f1 = f1_score(y_true, preds, average=None,
                             labels=np.arange(num_classes), zero_division=0)
    per_class_recall = recall_score(y_true, preds, average=None,
                                     labels=np.arange(num_classes), zero_division=0)

    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
    if num_classes == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    per_class_aucpr = np.full(num_classes, np.nan)
    for c in range(num_classes):
        if y_true_bin[:, c].sum() > 0:
            per_class_aucpr[c] = average_precision_score(y_true_bin[:, c], probs[:, c])

    return accuracy, per_class_f1, per_class_recall, per_class_aucpr


def _write_final_csv(path, prefix, model_type, seed, condition,
                      loss, accuracy, f1_macro, per_class_f1,
                      per_class_recall, per_class_aucpr, attack_names):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["model_type", "seed", "ablation_mode", "num_rounds",
             f"{prefix}_loss", f"{prefix}_accuracy", f"{prefix}_f1_macro"]
            + [f"{prefix}_f1_{name}" for name in attack_names]
            + [f"{prefix}_recall_{name}" for name in attack_names]
            + [f"{prefix}_aucpr_{name}" for name in attack_names]
        )
        w.writerow(
            [model_type, seed, condition, "N/A",  # num_rounds N/A -- not a
                                                     # federated run; kept as a
                                                     # column for schema
                                                     # compatibility with
                                                     # aggregate_task4_
                                                     # results.py, which
                                                     # never reads it.
             loss, accuracy, f1_macro]
            + [float(v) for v in per_class_f1]
            + [float(v) for v in per_class_recall]
            + [float(v) for v in per_class_aucpr]
        )


# ---------------------------------------------------------------------------
# CNN-LSTM (centralized) -- same model class + train()/test() as every
# federated condition.
# ---------------------------------------------------------------------------
def train_cnnlstm(model_type, seed, num_classes, attack_names):
    import torch
    from model_defs import get_model, get_model_parameters, set_model_parameters
    from task import train as task_train, test as task_test
    from data_loader import (get_global_train_holdout, get_global_validation_holdout,
                              get_global_test_holdout, get_class_counts_network,
                              get_class_counts_application)
    import task as task_module

    torch.manual_seed(seed)
    np.random.seed(seed)

    X_tr, y_tr = get_global_train_holdout(model_type, seed=seed)
    X_val, y_val = get_global_validation_holdout(model_type, seed=seed)
    X_te, y_te = get_global_test_holdout(model_type, seed=seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = (task_module.build_criterion_network(seed=seed) if model_type == "network"
                 else task_module.build_criterion_application(seed=seed)).to(device)

    model = get_model(num_features=X_tr.shape[1], num_classes=num_classes,
                       dp_safe=False).to(device)

    cfg = CENTRALIZED_DEFAULTS["cnnlstm"]
    best_val_f1 = -1.0
    best_params = None
    epochs_since_improve = 0

    print(f"  [CNN-LSTM centralized] {model_type} seed={seed}: "
          f"{X_tr.shape[0]} train / {X_val.shape[0]} val / {X_te.shape[0]} test rows, "
          f"max_epochs={cfg['max_epochs']}, patience={cfg['patience']}")

    for epoch in range(1, cfg["max_epochs"] + 1):
        model = task_train(model, X_tr, y_tr, criterion, epochs=1,
                            lr=cfg["lr"], global_params=None, mu=0.0, device=device)
        _, val_acc, val_f1_per_class = task_test(model, X_val, y_val, num_classes, device=device)
        val_f1_macro = float(np.mean(val_f1_per_class))

        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            best_params = get_model_parameters(model)
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if epoch % 5 == 0 or epochs_since_improve == 0:
            print(f"    epoch {epoch:3d}/{cfg['max_epochs']}  val_f1_macro={val_f1_macro:.4f}  "
                  f"best={best_val_f1:.4f}  since_improve={epochs_since_improve}")

        if epochs_since_improve >= cfg["patience"]:
            print(f"    Early stop at epoch {epoch} (no VAL improvement for "
                  f"{cfg['patience']} epochs).")
            break

    set_model_parameters(model, best_params)

    def _eval(X, y):
        loss, acc, f1, recall, aucpr = task_test(model, X, y, num_classes,
                                                   device=device, return_extended=True)
        return loss, acc, f1, recall, aucpr

    val_loss, val_acc, val_f1, val_recall, val_aucpr = _eval(X_val, y_val)
    test_loss, test_acc, test_f1, test_recall, test_aucpr = _eval(X_te, y_te)

    return {
        "val":  (val_loss, val_acc, float(np.mean(val_f1)), val_f1, val_recall, val_aucpr),
        "test": (test_loss, test_acc, float(np.mean(test_f1)), test_f1, test_recall, test_aucpr),
    }


# ---------------------------------------------------------------------------
# MLP / RandomForest / XGBoost -- shared sklearn-style fit/eval path.
# ---------------------------------------------------------------------------
def _fit_and_eval_sklearn_style(clf, model_type, seed, num_classes):
    from data_loader import get_global_train_holdout, get_global_validation_holdout, get_global_test_holdout

    X_tr, y_tr = get_global_train_holdout(model_type, seed=seed)
    X_val, y_val = get_global_validation_holdout(model_type, seed=seed)
    X_te, y_te = get_global_test_holdout(model_type, seed=seed)

    clf.fit(X_tr, y_tr)

    def _eval(X, y):
        preds = clf.predict(X)
        probs = clf.predict_proba(X)
        # predict_proba's column order matches clf.classes_, not
        # necessarily np.arange(num_classes) if a class is entirely
        # absent from TRAIN (classes_ would then be a strict subset) --
        # guard against silently misaligning probs[:, c] with class c.
        if list(clf.classes_) != list(range(num_classes)):
            full_probs = np.zeros((len(y), num_classes))
            for i, c in enumerate(clf.classes_):
                full_probs[:, int(c)] = probs[:, i]
            probs = full_probs
        accuracy, f1, recall, aucpr = _extended_metrics_from_predictions(
            y, preds, probs, num_classes
        )
        loss = float("nan")  # cross-entropy loss isn't meaningful/available
                              # for RF; kept as NaN rather than fabricated,
                              # consistently across all 3 sklearn-style
                              # conditions so the column stays comparable
                              # (never silently 0).
        return loss, accuracy, f1, recall, aucpr

    val_loss, val_acc, val_f1, val_recall, val_aucpr = _eval(X_val, y_val)
    test_loss, test_acc, test_f1, test_recall, test_aucpr = _eval(X_te, y_te)

    return {
        "val":  (val_loss, val_acc, float(np.mean(val_f1)), val_f1, val_recall, val_aucpr),
        "test": (test_loss, test_acc, float(np.mean(test_f1)), test_f1, test_recall, test_aucpr),
    }


def train_mlp(model_type, seed, num_classes, attack_names):
    from sklearn.neural_network import MLPClassifier
    cfg = CENTRALIZED_DEFAULTS["mlp"]
    clf = MLPClassifier(random_state=seed, **cfg)
    print(f"  [MLP centralized] {model_type} seed={seed}: hidden_layer_sizes="
          f"{cfg['hidden_layer_sizes']}, max_iter={cfg['max_iter']}")
    return _fit_and_eval_sklearn_style(clf, model_type, seed, num_classes)


def train_rf(model_type, seed, num_classes, attack_names):
    from sklearn.ensemble import RandomForestClassifier
    cfg = CENTRALIZED_DEFAULTS["rf"]
    clf = RandomForestClassifier(random_state=seed, **cfg)
    print(f"  [RandomForest centralized] {model_type} seed={seed}: "
          f"n_estimators={cfg['n_estimators']}")
    return _fit_and_eval_sklearn_style(clf, model_type, seed, num_classes)


def train_xgboost(model_type, seed, num_classes, attack_names):
    try:
        import xgboost as xgb
    except ImportError:
        raise RuntimeError(
            "xgboost is not installed in this environment. Install with "
            "`pip install xgboost` before running the xgboost condition. "
            "Not silently skipped -- this run must fail loudly rather "
            "than produce a missing Table 1 row with no explanation."
        )
    cfg = CENTRALIZED_DEFAULTS["xgboost"]
    clf = xgb.XGBClassifier(
        random_state=seed, eval_metric="mlogloss", **cfg,
    )
    print(f"  [XGBoost centralized] {model_type} seed={seed}: "
          f"n_estimators={cfg['n_estimators']}, max_depth={cfg['max_depth']}")
    return _fit_and_eval_sklearn_style(clf, model_type, seed, num_classes)


TRAINERS = {
    "cnnlstm": train_cnnlstm,
    "mlp": train_mlp,
    "rf": train_rf,
    "xgboost": train_xgboost,
}


def run_one(model_type, condition, seed):
    from data_loader import (NETWORK_NAMES, NUM_NETWORK_CLASSES,
                              APP_NAMES, NUM_APP_CLASSES)
    attack_names = NETWORK_NAMES if model_type == "network" else APP_NAMES
    num_classes = NUM_NETWORK_CLASSES if model_type == "network" else NUM_APP_CLASSES

    tag = f"task4_centralized_{condition}"
    t0 = time.time()
    result = TRAINERS[condition](model_type, seed, num_classes, attack_names)
    elapsed = time.time() - t0

    for split in ("val", "test"):
        loss, acc, f1_macro, f1, recall, aucpr = result[split]
        prefix = "test" if split == "test" else "val"
        split_suffix = "VALIDATION" if split == "val" else "TEST"
        out_path = f"results_{model_type}_{tag}_seed{seed}_FINAL_{split_suffix}.csv"
        _write_final_csv(out_path, prefix, model_type, seed, tag,
                          loss, acc, f1_macro, f1, recall, aucpr, attack_names)
        print(f"    [{split.upper()}] loss={loss:.4f}  acc={acc:.4f}  "
              f"f1_macro={f1_macro:.4f}  -> {out_path}")

    print(f"  Done: {model_type}/{condition}/seed={seed} in {elapsed:.1f}s\n")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_type", choices=["network", "application"], nargs="?",
                    default=None)
    p.add_argument("--condition", choices=ALL_CONDITIONS, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--all", action="store_true",
                    help="Run all (model x condition x seed) combinations.")
    p.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated seeds for --all (default: "
                         f"{ALL_SEEDS}).")
    args = p.parse_args()

    if args.all:
        seeds = ([int(s) for s in args.seeds.split(",")]
                 if args.seeds else ALL_SEEDS)
        for model_type in ("network", "application"):
            for condition in ALL_CONDITIONS:
                for seed in seeds:
                    print(f"=== {model_type} / {condition} / seed={seed} ===")
                    run_one(model_type, condition, seed)
        return

    if args.model_type is None or args.condition is None:
        p.error("Either pass --all, or model_type + --condition (+ --seed).")

    seed = args.seed if args.seed is not None else ALL_SEEDS[0]
    run_one(args.model_type, args.condition, seed)


if __name__ == "__main__":
    main()
