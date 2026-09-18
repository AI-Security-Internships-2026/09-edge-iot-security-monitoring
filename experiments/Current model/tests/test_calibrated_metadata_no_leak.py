"""Regression tests for E6 fix #3: DP metadata must not encode Byzantine labels."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from defences.krum import public_noise_multiplier_map, calibrated_adaptive_multi_krum


def test_missing_clients_get_median_honest_sigma():
    sig = {2: 1.0, 3: 1.2, 4: 1.4}          # clients 0,1 = Byzantine (no DP state)
    m = public_noise_multiplier_map(range(5), sig)
    assert m[0] == m[1] == 1.2
    assert m[2] == 1.0 and m[4] == 1.4


def test_no_dp_stays_none():
    assert public_noise_multiplier_map(range(3), {}) == {0: None, 1: None, 2: None}


def test_sigma_metadata_carries_no_label_information():
    m = public_noise_multiplier_map(range(10), {i: 1.0 + 0.01 * i for i in range(2, 10)})
    byz, honest = [m[0], m[1]], [m[i] for i in range(2, 10)]
    assert min(honest) <= byz[0] <= max(honest)     # attacker indistinguishable by range
    assert all(v is not None for v in m.values())


def test_calibrated_krum_runs_with_public_sigma():
    rng = np.random.default_rng(0)
    base = rng.normal(size=500) * 0.05
    P = [[(base + rng.normal(size=500) * 0.05).astype(np.float32)] for _ in range(10)]
    for b in (0, 1):
        P[b] = [(-base + rng.normal(size=500) * 0.01).astype(np.float32)]
    sig = public_noise_multiplier_map(range(10), {i: 1.1 for i in range(2, 10)})
    meta = {i: {"n_samples": 1000, "class_entropy": 1.0,
                "noise_multiplier": sig[i], "epsilon": 5} for i in range(10)}
    _, sel, _ = calibrated_adaptive_multi_krum(
        P, [1000] * 10, meta, num_byzantine=2, k=2.5, return_diagnostics=True)
    assert 0 not in sel and 1 not in sel
