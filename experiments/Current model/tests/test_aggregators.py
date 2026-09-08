"""
tests/test_aggregators.py

BAS1 (Issue 3) Task 2. Hand-calculated 5-client, 3-dim (all 3 coords
identical per client, so distances/medians/trims reduce to 1-D
arithmetic that's actually checkable by hand) synthetic fixture.

Fixture: client scalar values v = [1, 2, 4, 7, 100], weights all 1.0,
num_byzantine=1 (client index 4, the 100, is the intended outlier).

Pairwise squared distances (3 identical coords per client, so each
distance is 3*(vi-vj)**2):
  (0,1)=3,   (0,2)=27,   (0,3)=108,   (0,4)=29403
  (1,2)=12,  (1,3)=75,   (1,4)=28812
  (2,3)=27,  (2,4)=27648
  (3,4)=25947

Each client's Krum score = sum of its 2 (= n-f-2 = 5-1-2) smallest
distances to other clients:
  c0: distances to others = [3,27,108,29403]   -> smallest 2 = 3+27  = 30
  c1: distances to others = [3,12,75,28812]    -> smallest 2 = 3+12  = 15
  c2: distances to others = [27,12,27,27648]   -> smallest 2 = 12+27 = 39
  c3: distances to others = [108,75,27,25947]  -> smallest 2 = 27+75 = 102
  c4: distances to others = [29403,28812,27648,25947] -> smallest 2 =
      25947+27648 = 53595

Run with: pytest tests/test_aggregators.py -v
(self-locating -- see the sys.path block below; works whether `defences`
lives directly under the repo root, as in this project's real
`experiments/Current model/defences/` layout, or under a `src/`
subdirectory, without needing PYTHONPATH set manually first.)

PATH FIX (this revision): the previous version of this file hardcoded
`_SRC_DIR = _REPO_ROOT/"src"` unconditionally -- there is no `src/` in
this project's real layout (`defences/` sits directly under
`experiments/Current model/`, i.e. directly under `_REPO_ROOT` as
computed below), so that insert silently pointed at a directory that
doesn't exist and `from defences.krum import ...` below would raise
ModuleNotFoundError unless something else had already put the real
directory on sys.path. Fixed to match tests/test_calibrated_krum.py's
already-correct pattern: check both candidate locations for a real
`defences/` subdirectory and use whichever one actually exists.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _candidate in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if os.path.isdir(os.path.join(_candidate, "defences")) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

import numpy as np
import pytest

from defences.krum import (
    fedavg, coordinate_median, trimmed_mean, multi_krum,
    adaptive_multi_krum,
)

V = [1.0, 2.0, 4.0, 7.0, 100.0]
WEIGHTS = [1.0] * 5


def _fixture():
    # One layer, shape (3,), same scalar repeated across all 3 dims.
    return [[np.full(3, v, dtype=np.float64)] for v in V]


def _legacy_fedprox_aggregate(all_params, weights):
    """
    Verbatim-extracted copy of main.py's pre-existing fedprox_aggregate()
    -- NOT reimplemented from memory, copied exactly, because main.py
    itself cannot be safely imported for a hermetic pytest run
    (argparse parses sys.argv at module import time). This is the "1
    legacy aggregator" referenced in the issue's Tests Required
    section -- the only other pre-existing aggregator anywhere in the
    codebase besides adaptive_multi_krum().
    """
    total = sum(weights)
    result = []
    for layer_idx in range(len(all_params[0])):
        layer_avg = sum(
            p[layer_idx] * (w / total)
            for p, w in zip(all_params, weights)
        )
        result.append(layer_avg)
    return result


def test_fedavg_matches_hand_calculation():
    result = fedavg(_fixture(), WEIGHTS)
    np.testing.assert_allclose(result[0], np.full(3, 22.8))


def test_legacy_fedprox_aggregate_matches_new_fedavg():
    """
    Regression guard: the dispatch table's fedavg() must produce
    EXACTLY what the pre-existing, still-in-production
    fedprox_aggregate() has always produced -- i.e. swapping
    ABLATION_MODE='baseline' from its old hardcoded fedprox_aggregate()
    call to the new AGGREGATOR='fedavg' dispatch path changes nothing
    numerically.
    """
    legacy = _legacy_fedprox_aggregate(_fixture(), WEIGHTS)
    new = fedavg(_fixture(), WEIGHTS)
    np.testing.assert_allclose(legacy[0], new[0])
    np.testing.assert_allclose(new[0], np.full(3, 22.8))


def test_coordinate_median_matches_hand_calculation():
    result = coordinate_median(_fixture(), WEIGHTS)
    np.testing.assert_allclose(result[0], np.full(3, 4.0))


def test_trimmed_mean_beta_0_2_matches_hand_calculation():
    # beta=0.1 (the module default) floors to 0 trimmed on n=5 --
    # deliberately using beta=0.2 here so the trim is non-trivial and
    # hand-verifiable (see docstring judgment call in trimmed_mean()).
    # sorted [1,2,4,7,100], trim 1 each tail -> mean(2,4,7) = 13/3.
    result = trimmed_mean(_fixture(), WEIGHTS, beta=0.2)
    np.testing.assert_allclose(result[0], np.full(3, 13.0 / 3.0))


def test_trimmed_mean_beta_0_1_on_n5_is_a_noop():
    """Explicitly documents/locks in the floor(beta*n)=0 edge case."""
    result = trimmed_mean(_fixture(), WEIGHTS, beta=0.1)
    expected = fedavg(_fixture(), WEIGHTS)
    np.testing.assert_allclose(result[0], expected[0])


def test_multi_krum_matches_hand_calculation():
    # scores = [30, 15, 39, 102, 53595] -> 3 lowest = clients {0,1,2}
    # -> mean(1,2,4) = 7/3
    result, selected = multi_krum(
        _fixture(), WEIGHTS, num_byzantine=1, m=3
    )
    assert selected == [0, 1, 2]
    np.testing.assert_allclose(result[0], np.full(3, 7.0 / 3.0))


def test_multi_krum_drops_the_extreme_outlier_when_m_is_tight():
    """Sanity companion: with m=1 (keep only the single lowest-score
    client), client 1 (v=2, score=15, the lowest) must be the survivor."""
    result, selected = multi_krum(_fixture(), WEIGHTS, num_byzantine=1, m=1)
    assert selected == [1]
    np.testing.assert_allclose(result[0], np.full(3, 2.0))


def test_adaptive_multi_krum_matches_hand_calculation():
    """
    adaptive_multi_krum() is UNCHANGED code (per the issue's explicit
    constraint) -- this test is a regression/consistency check against
    the SAME fixture, not a claim that its algorithm changed. Hand
    derivation: scores [30,15,39,102,53595] -> median=39, MAD=24,
    spread=1.4826*24=35.5824, threshold=39+2.5*35.5824=127.956 ->
    only client 4 (score 53595) exceeds it.
    """
    result, kept = adaptive_multi_krum(
        _fixture(), WEIGHTS, num_byzantine=1, k=2.5, method="mad",
        min_keep_fraction=0.5,
    )
    assert kept == [0, 1, 2, 3]
    np.testing.assert_allclose(result[0], np.full(3, 3.5))
