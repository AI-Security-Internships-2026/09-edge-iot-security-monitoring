"""
tests/test_dat1_smoke_run.py

DAT1 acceptance items covered here:
  - "Smoke-run 1-round 10-client experiment: the global TEST holdout ...
    is NEVER opened by any process during client local-train steps."
  - Task 2 config-integrity: class-weight multipliers, FedProx mu, DP
    clipping norm C, and MAD-k must be config-driven, not hardcoded.

Design note on the smoke-run test (read before changing it):
main.py is a script, not an importable library -- its top-level code
parses sys.argv, seeds torch/numpy, and branches on ABLATION_MODE the
moment it is imported, and a real run needs the ~1.2GB Edge-IIoTset
CSV on disk. Neither is compatible with a fast, hermetic pytest run,
and test_data_pipeline_dat1.py already establishes the project's
convention of testing the underlying MACHINERY against small synthetic
data rather than invoking main.py end-to-end. This file follows that
same convention:

  1. A DYNAMIC test exercises the exact function main.py's real
     per-client loop calls (load_partition_network(), for all 10
     clients -- this is identical regardless of NUM_ROUNDS, since
     partitioning happens once, not per round, so "1-round" and
     "25-round" touch exactly the same per-client data) against a
     synthetic dataset primed directly into data_loader._cache, and
     confirms the TEST holdout is never touched by that path, at both
     the call-count level and the row-content level.

  2. A STATIC test (mirroring test_data_pipeline_dat1.py's own
     .fit_transform( grep audit) confirms the ACTUAL shipped main.py,
     not just this test's synthetic stand-in, calls
     get_global_test_holdout( exactly once, textually after the round
     loop closes -- i.e. the real code has the right shape, not just
     a hypothetical one this test constructed.

  3. Config-integrity tests confirm hyperparams.json exists with the
     required keys/provenance fields, and that main.py/task.py
     actually read from it (a positive wiring check -- we deliberately
     do NOT grep for the absence of the old literals, since changelog
     comments and docstrings in main.py legitimately still mention
     historical values like "DP_MAX_GRAD_NORM=1.5" in prose, which
     would make a naive "must not contain 1.5" check false-positive).

Run with: pytest tests/test_dat1_smoke_run.py -v
"""
import json
import os
import re
import shutil
import tempfile

import numpy as np
import pytest

import data_loader as dl


REPO_ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_PY_PATH       = os.path.join(REPO_ROOT, "main.py")
TASK_PY_PATH       = os.path.join(REPO_ROOT, "task.py")
HYPERPARAMS_PATH   = os.path.abspath(os.path.join(REPO_ROOT, "..", "configs", "hyperparams.json"))


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def isolated_splits_dir(monkeypatch):
    """Same isolation pattern as test_data_pipeline_dat1.py: points
    data_loader.SPLITS_DIR at a fresh temp dir so these tests never
    touch real split artifacts."""
    tmp_dir = tempfile.mkdtemp(prefix="dat1_smoke_splits_")
    monkeypatch.setattr(dl, "SPLITS_DIR", tmp_dir)
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def _clear_data_loader_cache():
    """
    data_loader._cache is a module-level dict keyed by (model_type,
    seed) -- clear it before AND after every test in this file so
    tests never see each other's primed synthetic entries (a real run
    never hits this because every real (model_type, seed) pair is
    used at most once per process).
    """
    dl._cache.clear()
    yield
    dl._cache.clear()


def _prime_synthetic_cache(model_type, seed, n_total=3000, n_features=10,
                           num_classes=8):
    """
    Directly populates data_loader._cache[(model_type, seed)] with an
    already-split synthetic TRAIN/VAL/TEST dataset, so
    load_and_preprocess() -- called internally by both
    load_partition_network()/load_partition_application() and
    get_global_test_holdout() -- finds the key already cached (its
    very first check is `if key in _cache: continue`) and never
    touches _load_raw() / the real Edge-IIoTset CSV at all.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, num_classes, size=n_total)
    X = rng.normal(size=(n_total, n_features)).astype(np.float64)

    n_train = int(n_total * 0.8)
    n_val   = int(n_total * 0.1)
    idx = rng.permutation(n_total)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    cache_entry = {
        "X_train": X[train_idx], "y_train": y[train_idx],
        "X_val":   X[val_idx],   "y_val":   y[val_idx],
        "X_test":  X[test_idx],  "y_test":  y[test_idx],
    }
    dl._cache[(model_type, seed)] = cache_entry
    return cache_entry


# ── Test 1: dynamic -- TEST holdout untouched during client partitioning ──

def test_test_holdout_never_touched_during_client_partitioning(
    isolated_splits_dir, monkeypatch
):
    """
    Simulates main.py's real per-client data-loading loop
    (`for i in range(NUM_CLIENTS): clients_data.append(load_partition(i, ...))`)
    for all 10 clients against a synthetic, pre-cached dataset.

    Asserts:
      (a) get_global_test_holdout is never called anywhere in that
          loop (wrapped with a call counter);
      (b) no client's local-train or local-val rows are bit-identical
          to any TEST-holdout row -- a row-content-level check, not
          just "the function wasn't called", so it would also catch
          leakage introduced via some other code path.
    """
    model_type, seed = "network", 555
    cache_entry = _prime_synthetic_cache(model_type, seed)
    test_rows_actual = cache_entry["X_test"]

    call_count = {"n": 0}
    _real_get_holdout = dl.get_global_test_holdout

    def _counting_get_holdout(*args, **kwargs):
        call_count["n"] += 1
        return _real_get_holdout(*args, **kwargs)

    monkeypatch.setattr(dl, "get_global_test_holdout", _counting_get_holdout)

    NUM_CLIENTS = 10
    all_local_rows = []
    for i in range(NUM_CLIENTS):
        X_tr, y_tr, X_te_local, y_te_local = dl.load_partition_network(
            i, num_partitions=NUM_CLIENTS, seed=seed
        )
        all_local_rows.append(X_tr)
        all_local_rows.append(X_te_local)   # client-LOCAL val, not global TEST

    assert call_count["n"] == 0, (
        "get_global_test_holdout() was called during client "
        "partitioning -- the TEST holdout must never be touched "
        "during client local-train steps."
    )

    local_rows_flat = np.concatenate(all_local_rows, axis=0)
    for test_row in test_rows_actual:
        matches = np.all(np.isclose(local_rows_flat, test_row), axis=1)
        assert not matches.any(), (
            "A TEST-holdout row appears among client local-train/"
            "local-val rows -- TEST-holdout leakage into client "
            "partitioning."
        )


def test_test_holdout_matches_global_test_when_evaluated_separately(
    isolated_splits_dir
):
    """
    Simulates the NEW final-evaluation step main.py now runs exactly
    once, after the round loop: calling get_global_test_holdout()
    directly (never via any client-partitioning path) and confirming
    it returns precisely the cached TEST split.
    """
    model_type, seed = "network", 777
    cache_entry = _prime_synthetic_cache(model_type, seed)

    X_test, y_test = dl.get_global_test_holdout(model_type, seed=seed)

    np.testing.assert_array_equal(X_test, cache_entry["X_test"])
    np.testing.assert_array_equal(y_test, cache_entry["y_test"])


# ── Test 2: static -- shape check on the REAL main.py source ───────────

def test_main_py_calls_test_holdout_exactly_once_after_round_loop():
    """
    Static, dataset-free check (mirrors
    test_data_pipeline_dat1.py's .fit_transform( grep audit) that the
    ACTUAL shipped main.py calls get_global_test_holdout( exactly
    once, and that call site is textually after the round loop's
    `with pool_cm as executor:` block has closed -- i.e. not nested
    inside the per-round loop or inside
    _train_one_client/_eval_one_client, where it would run every
    round or every client instead of once.
    """
    with open(MAIN_PY_PATH) as f:
        lines = f.readlines()

    call_lines = [
        i for i, l in enumerate(lines)
        if re.search(r"get_global_test_holdout\(", l)
        and "def get_global_test_holdout" not in l
        and "import get_global_test_holdout" not in l
        and not l.strip().startswith("#")
    ]

    assert len(call_lines) == 1, (
        f"Expected exactly one get_global_test_holdout( call site in "
        f"main.py, found {len(call_lines)} at lines "
        f"{[n + 1 for n in call_lines]} -- it must be called exactly "
        f"once, after the round loop, per DAT1 Task 1.10."
    )

    call_line = call_lines[0]

    with_line = next(
        i for i, l in enumerate(lines) if "with pool_cm as executor:" in l
    )
    with_indent = len(lines[with_line]) - len(lines[with_line].lstrip())

    assert call_line > with_line, (
        "get_global_test_holdout( appears before the round loop even "
        "starts -- it must run after training completes."
    )

    call_indent = len(lines[call_line]) - len(lines[call_line].lstrip())
    assert call_indent <= with_indent, (
        f"get_global_test_holdout( at main.py:{call_line + 1} is "
        f"indented ({call_indent} spaces) deeper than the round "
        f"loop's own 'with pool_cm as executor:' block "
        f"({with_indent} spaces) -- this means it is still nested "
        f"inside the round loop and would run every round/client "
        f"instead of exactly once after training completes."
    )


def test_final_test_csv_is_distinct_from_progress_log_csv():
    """
    Static check that main.py writes the final TEST-holdout result to
    a filename distinct from LOG_CSV (the per-round progress file),
    so the two can never be confused with each other downstream.
    """
    with open(MAIN_PY_PATH) as f:
        main_src = f.read()

    assert 'FINAL_TEST_CSV = f"results_{_TAG}_FINAL_TEST.csv"' in main_src, (
        "main.py no longer writes a distinctly-named FINAL_TEST_CSV -- "
        "the paper-citable result must be written somewhere other "
        "than LOG_CSV's per-round progress rows."
    )


# ── Test 3: config integrity (DAT1 Task 2) ──────────────────────────────

def test_hyperparams_config_exists_and_has_required_keys():
    assert os.path.exists(HYPERPARAMS_PATH), (
        f"{HYPERPARAMS_PATH} does not exist -- DAT1 Task 2 requires "
        f"fedprox_mu, dp_max_grad_norm, adaptive_krum_k, "
        f"adaptive_krum_hybrid_assumed_f, and the application "
        f"class-weight multipliers to be read from a config file, "
        f"not hardcoded."
    )
    with open(HYPERPARAMS_PATH) as f:
        cfg = json.load(f)

    required = ("fedprox_mu", "dp_max_grad_norm", "adaptive_krum_k",
               "adaptive_krum_hybrid_assumed_f",
               "class_weight_multipliers_application",
               "fedprox_mu_sweep_default", "mad_k_sweep_default",
               "byzantine_f_sweep_default")
    missing = [k for k in required if k not in cfg]
    assert not missing, f"hyperparams.json missing required key(s): {missing}"

    for key in ("fedprox_mu", "dp_max_grad_norm", "adaptive_krum_k",
               "adaptive_krum_hybrid_assumed_f"):
        assert "validated_on_split" in cfg[key], (
            f"{key} is missing a validated_on_split provenance field -- "
            f"every tunable value must state which split (if any) "
            f"justifies it."
        )

    for cls in ("Uploading", "XSS", "Fingerprinting"):
        assert cls in cfg["class_weight_multipliers_application"], (
            f"class_weight_multipliers_application missing {cls!r}."
        )


def test_main_and_task_actually_read_from_config():
    """
    Positive wiring check: confirms main.py/task.py actually call
    config_loader.get_value(...)/read the config keys they're
    supposed to, rather than grepping for the ABSENCE of the old
    literals (which would false-positive against changelog comments
    and docstrings in main.py that legitimately still mention
    historical values like "DP_MAX_GRAD_NORM=1.5" in prose).
    """
    with open(MAIN_PY_PATH) as f:
        main_src = f.read()
    with open(TASK_PY_PATH) as f:
        task_src = f.read()

    assert "from config_loader import" in main_src, (
        "main.py does not import config_loader."
    )
    for key in ("fedprox_mu", "dp_max_grad_norm", "adaptive_krum_k",
               "adaptive_krum_hybrid_assumed_f"):
        assert f'get_value(_HP_CONFIG, "{key}")' in main_src, (
            f"main.py does not call get_value(_HP_CONFIG, {key!r}) -- "
            f"this parameter does not appear to be config-driven."
        )

    assert "from config_loader import load_hyperparams_config" in task_src, (
        "task.py does not import config_loader."
    )
    assert 'cfg["class_weight_multipliers_application"]' in task_src or \
           "_hp_cfg[\"class_weight_multipliers_application\"]" in task_src, (
        "task.py does not read class_weight_multipliers_application "
        "from the loaded config."
    )


def test_config_loader_rejects_missing_required_keys(tmp_path):
    """
    config_loader.load_hyperparams_config must fail loudly, not
    silently proceed, if a required key is missing from the config
    file it's given -- otherwise a future edit could quietly drop a
    required parameter's provenance without any test catching it.
    """
    import config_loader as cl

    bad_cfg_path = tmp_path / "bad_hyperparams.json"
    bad_cfg_path.write_text(json.dumps({"fedprox_mu": {"value": 0.02,
                                                       "validated_on_split": "UNVALIDATED"}}))

    with pytest.raises(AssertionError, match="missing required key"):
        cl.load_hyperparams_config(path=str(bad_cfg_path))
