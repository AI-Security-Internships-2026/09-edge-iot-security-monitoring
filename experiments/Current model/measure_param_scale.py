#!/usr/bin/env python3
"""
measure_param_scale.py
========================
Answers the question a literature-cited sigma value CANNOT answer on its
own: what is the actual parameter/delta magnitude in THIS codebase's
CNN-LSTM, for THIS model type? Gaussian noise addition (unlike
sign-flip's multiplicative scale) does not auto-scale with whatever
magnitude the model's parameters happen to be -- it needs a real
measurement, not a number ported from a different paper's different
architecture/units.

Runs one round of ordinary (non-Byzantine, non-DP) local training for
every client, measures:
  - L2 norm of the full flattened trained parameter vector
  - L2 norm of the DELTA (trained_params - global_params) -- this is
    the more relevant number, since the Gaussian attack conceptually
    perturbs "how far this client's contribution differs," not the
    absolute parameter scale (which includes a lot of shared,
    unchanging structure from the initial/global model)
  - per-element std of both, for a directly comparable "what should
    sigma roughly be" number

Prints a recommended sigma range: a few multiples of the delta's
per-element std, analogous to how ATTACK_SCALE ends up several times
larger than the honest signal for sign-flip.

Usage:
    python3 measure_param_scale.py --model network
    python3 measure_param_scale.py --model application
"""

import argparse
import os
import sys

import numpy as np
import torch

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["network", "application"], required=True)
    parser.add_argument("--num-clients", type=int, default=10)
    args = parser.parse_args()

    from data_loader import (
        load_partition_network, NUM_NETWORK_CLASSES,
        load_partition_application, NUM_APP_CLASSES,
    )
    from task import (get_model, get_model_parameters, set_model_parameters,
                      train, build_criterion_network, build_criterion_application)

    if args.model == "network":
        load_partition = load_partition_network
        num_classes = NUM_NETWORK_CLASSES
        build_criterion = build_criterion_network
    else:
        load_partition = load_partition_application
        num_classes = NUM_APP_CLASSES
        build_criterion = build_criterion_application

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {args.model}  Device: {device}")

    clients_data = [load_partition(i, args.num_clients) for i in range(args.num_clients)]
    sample_features = clients_data[0][0].shape[1]
    criterion = build_criterion().to(device)

    global_params = get_model_parameters(
        get_model(num_features=sample_features, num_classes=num_classes, dp_safe=False)
    )

    param_norms, delta_norms = [], []
    param_stds, delta_stds = [], []

    print(f"\nTraining {args.num_clients} clients for 1 round (plain, no DP/attack)...")
    for i, (X_tr, y_tr, X_te, y_te) in enumerate(clients_data):
        model = get_model(num_features=sample_features, num_classes=num_classes, dp_safe=False)
        set_model_parameters(model, global_params)
        model = model.to(device)
        train(model, X_tr, y_tr, criterion, epochs=5, lr=0.001,
             global_params=global_params, mu=0.02, device=device)
        trained_params = get_model_parameters(model)

        flat_trained = np.concatenate([p.flatten() for p in trained_params])
        flat_global = np.concatenate([p.flatten() for p in global_params])
        delta = flat_trained - flat_global

        param_norms.append(float(np.linalg.norm(flat_trained)))
        delta_norms.append(float(np.linalg.norm(delta)))
        param_stds.append(float(np.std(flat_trained)))
        delta_stds.append(float(np.std(delta)))

        print(f"  Client {i+1:2d}: param_L2={param_norms[-1]:.4f}  "
             f"delta_L2={delta_norms[-1]:.6f}  param_std={param_stds[-1]:.6f}  "
             f"delta_std={delta_stds[-1]:.6f}")

    mean_param_std = float(np.mean(param_stds))
    mean_delta_std = float(np.mean(delta_stds))
    mean_delta_norm = float(np.mean(delta_norms))

    print(f"\n{'='*60}")
    print(f"Mean per-element param std across clients: {mean_param_std:.6f}")
    print(f"Mean per-element DELTA std across clients: {mean_delta_std:.6f}")
    print(f"Mean delta L2 norm across clients:          {mean_delta_norm:.6f}")
    print(f"{'='*60}")
    print(f"\nCurrent GAUSSIAN_STD default: 10.0")
    print(f"RSA's cited sigma: 10000 (almost certainly a different model's "
         f"units -- not directly portable, see additive-vs-multiplicative "
         f"reasoning)")
    print(f"\nSuggested starting range for THIS model ({args.model}):")
    print(f"  Conservative (comparable to honest variation): "
         f"sigma ~= {mean_delta_std * 3:.4f} - {mean_delta_std * 5:.4f}")
    print(f"  Aggressive (clearly dominates honest signal, matching "
         f"ATTACK_SCALE's ~5x-ish philosophy): "
         f"sigma ~= {mean_delta_std * 10:.4f} - {mean_delta_std * 20:.4f}")
    print(f"\nNeither 10.0 nor 10000 is validated against this measurement --"
         f" pick from the ranges above, or run a 1-round sanity check with "
         f"a candidate --gaussian-std value and confirm no NaN/overflow "
         f"before committing to a full sweep, same caution as ATTACK_SCALE's "
         f"original tuning.")


if __name__ == "__main__":
    main()
