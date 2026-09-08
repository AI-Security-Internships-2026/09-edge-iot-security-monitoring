"""
tests/test_dp_determinism.py

PRV1 Task 4 (script entry, no new production file): two identical
synthetic runs with the same seed (including Opacus's noise sampler,
seeded via torch.manual_seed since Opacus draws noise from torch's
global RNG with no explicit `generator=` passed -- see main.py's
top-of-file seeding comment) must produce per-client final_total_epsilon
within +/- 0.01 of each other.

This exercises src/dp_persistent_client_state.py's real, single-process
code path directly -- the same functions main.py's round loop calls.
Unlike the earlier multiprocessing design, there is no IPC/fork
involved here at all, so determinism only depends on ordinary,
single-process RNG seeding (no fork-vs-spawn start-method sensitivity).

Run with: pytest tests/test_dp_determinism.py -v -s
"""
import os
import sys
import random

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np
import torch

from task import get_model, get_model_parameters, FocalLoss
from dp_persistent_client_state import (
    build_dp_client_states, run_dp_client_round, get_final_epsilons,
)

NUM_FEATURES  = 6
NUM_CLASSES   = 3
DATASET_LEN   = 20
BATCH_SIZE    = 5
LOCAL_EPOCHS  = 2
NUM_ROUNDS    = 3
TARGET_EPS    = 4.0
TARGET_DELTA  = 1e-5
MAX_GRAD_NORM = 1.0
SEED          = 1234


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_once(seed):
    """
    One full synthetic 2-client, NUM_ROUNDS-round DP-active run through
    the real production dp_persistent_client_state.py path. Returns
    {client_idx: final_total_epsilon}.
    """
    _seed_everything(seed)

    rng = np.random.default_rng(seed)
    clients_data = []
    for _ in range(2):
        X = rng.normal(size=(DATASET_LEN, NUM_FEATURES)).astype(np.float32)
        y = rng.integers(0, NUM_CLASSES, size=DATASET_LEN).astype(np.int64)
        clients_data.append((X, y, X, y))

    dp_states = build_dp_client_states(
        clients_data, byzantine_clients=[], use_byzantine_attack=False,
        sample_features=NUM_FEATURES, num_classes=NUM_CLASSES, dp_safe=True,
        learning_rate=0.001, dp_batch_size=BATCH_SIZE,
        dp_target_epsilon=TARGET_EPS, dp_target_delta=TARGET_DELTA,
        dp_max_grad_norm=MAX_GRAD_NORM,
        total_epochs_per_client=NUM_ROUNDS * LOCAL_EPOCHS,
        device=torch.device("cpu"),
    )

    criterion = FocalLoss(gamma=2.0)
    global_params = get_model_parameters(
        get_model(num_features=NUM_FEATURES, num_classes=NUM_CLASSES, dp_safe=True)
    )
    for _round_num in range(NUM_ROUNDS):
        params_by_client = []
        for i in range(len(clients_data)):
            params, _cum_eps = run_dp_client_round(
                i, dp_states, global_params, criterion,
                local_epochs=LOCAL_EPOCHS, learning_rate=0.001,
                prox_mu=0.0, dp_delta=TARGET_DELTA,
            )
            params_by_client.append(params)
        # Simple FedAvg over the 2 synthetic clients, just to advance
        # global_params round-to-round realistically.
        global_params = [
            sum(p[layer] for p in params_by_client) / len(params_by_client)
            for layer in range(len(params_by_client[0]))
        ]

    return get_final_epsilons(dp_states, TARGET_DELTA)


def test_determinism_across_duplicate_seed_runs():
    """
    PRV1 Task 4: two identical runs with the same seed must land within
    +/- 0.01 epsilon per client.
    """
    run1 = _run_once(SEED)
    run2 = _run_once(SEED)

    assert set(run1.keys()) == set(run2.keys()), (
        "Two identical-seed runs produced different sets of DP-active "
        "client indices -- something non-deterministic upstream of "
        "accounting itself."
    )

    for client_idx in sorted(run1.keys()):
        eps1, eps2 = run1[client_idx], run2[client_idx]
        diff = abs(eps1 - eps2)
        print(f"  Client {client_idx}: run1={eps1:.6f}  run2={eps2:.6f}  "
              f"diff={diff:.6f}")
        assert diff <= 0.01, (
            f"Client {client_idx}: final_total_epsilon differs by "
            f"{diff:.6f} between two identical-seed runs (run1={eps1:.6f}, "
            f"run2={eps2:.6f}) -- exceeds the +/-0.01 determinism "
            f"tolerance required by PRV1 Task 4."
        )
