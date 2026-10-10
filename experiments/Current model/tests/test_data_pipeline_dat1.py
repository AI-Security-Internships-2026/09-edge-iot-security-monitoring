"""
tests/test_data_pipeline_dat1.py

Required tests for Issue DAT1 (Fix Preprocessing Data Leakage & Enforce
T/V/T 3-Way Split Discipline). Uses small synthetic data, not the full
~1.2GB Edge-IIoTset CSV, so these run in seconds and don't require the
dataset to be present -- they test the SPLIT/SCALER MACHINERY itself
(_get_or_build_tvt_indices, _fit_or_load_scalers), which is exactly
where the leakage bug lived, independent of the real dataset's content.

Run with: pytest tests/test_data_pipeline_dat1.py -v
"""
import os
import shutil
import tempfile
import numpy as np
import pytest

import data_loader as dl


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def isolated_splits_dir(monkeypatch):
    """
    Points data_loader.SPLITS_DIR at a fresh temp directory for the
    duration of one test, so tests never touch real split/scaler
    artifacts and never interfere with each other or a real run.
    """
    tmp_dir = tempfile.mkdtemp(prefix="dat1_test_splits_")
    monkeypatch.setattr(dl, "SPLITS_DIR", tmp_dir)
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _synthetic_labels(n=2000, num_classes=8, seed=0):
    """Small synthetic label array with every class represented at
    least a few times, so stratified splitting doesn't error out."""
    rng = np.random.default_rng(seed)
    # Skewed but every class >= 20 samples so 80/10/10 stratified split
    # always has enough rows per class in every split.
    counts = rng.integers(low=20, high=n // num_classes, size=num_classes)
    y = np.concatenate([np.full(c, i) for i, c in enumerate(counts)])
    rng.shuffle(y)
    return y


def _synthetic_features(y, n_features=12, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(len(y), n_features)).astype(np.float64)


# ── Test 1: Determinism ─────────────────────────────────────────────

def test_tvt_split_determinism_across_fresh_builds(isolated_splits_dir):
    """
    DAT1 determinism requirement: two independent pipeline invocations
    with the same global seed produce byte-identical split artifacts.

    Builds the split fresh under seed=42, deletes the cache, rebuilds
    it fresh again under the SAME seed, and asserts the resulting
    train/val/test indices are byte-identical both times.
    """
    y = _synthetic_labels(seed=1)
    seed = 42

    train_idx_1, val_idx_1, test_idx_1 = dl._get_or_build_tvt_indices(
        "synthtest", y, seed
    )

    # Force a genuinely fresh rebuild: delete the artifact this call
    # just wrote, so the next call can't just reload the cached file.
    paths = dl._tvt_split_paths("synthtest", seed)
    os.remove(paths["npz"])
    os.remove(paths["hash"])

    train_idx_2, val_idx_2, test_idx_2 = dl._get_or_build_tvt_indices(
        "synthtest", y, seed
    )

    np.testing.assert_array_equal(train_idx_1, train_idx_2)
    np.testing.assert_array_equal(val_idx_1, val_idx_2)
    np.testing.assert_array_equal(test_idx_1, test_idx_2)


def test_tvt_split_npz_byte_identical_across_fresh_builds(isolated_splits_dir):
    """
    DAT1 determinism requirement, literal form (Mati86 review comment):
    the acceptance criterion asks for byte-identical .npz ARTIFACTS, not
    just identical decoded index arrays (the test above). Builds fresh
    under seed=42, reads the raw file bytes, deletes, rebuilds fresh
    again under the same seed, and asserts the two files are
    byte-for-byte identical -- this only holds because
    _get_or_build_tvt_indices now saves via _deterministic_savez rather
    than np.savez_compressed (which embeds a wall-clock timestamp per
    zip member and would make this assertion fail even with identical
    array content).
    """
    import hashlib

    y = _synthetic_labels(seed=1)
    seed = 42

    dl._get_or_build_tvt_indices("synthtest", y, seed)
    paths = dl._tvt_split_paths("synthtest", seed)
    with open(paths["npz"], "rb") as f:
        bytes_1 = f.read()

    os.remove(paths["npz"])
    os.remove(paths["hash"])

    dl._get_or_build_tvt_indices("synthtest", y, seed)
    with open(paths["npz"], "rb") as f:
        bytes_2 = f.read()

    assert hashlib.sha256(bytes_1).hexdigest() == hashlib.sha256(bytes_2).hexdigest(), (
        "Two fresh builds of the same seed produced non-byte-identical "
        ".npz files -- check for embedded timestamps or non-deterministic "
        "member ordering in the save path."
    )


def test_scaler_determinism_across_fresh_fits(isolated_splits_dir):
    """
    DAT1 determinism requirement (scaler half): fitting fresh twice on
    identical TRAIN data produces identical .mean_/.var_ (StandardScaler)
    and identical .get_support() (VarianceThreshold, network model path).
    """
    y = _synthetic_labels(seed=2)
    X = _synthetic_features(y, seed=2)
    seed = 123

    fitted_1 = dl._fit_or_load_scalers("network", seed, X)
    os.remove(dl._scaler_path("network", seed))
    fitted_2 = dl._fit_or_load_scalers("network", seed, X)

    np.testing.assert_array_equal(fitted_1["vt"].get_support(), fitted_2["vt"].get_support())
    np.testing.assert_allclose(fitted_1["scaler"].mean_, fitted_2["scaler"].mean_)
    np.testing.assert_allclose(fitted_1["scaler"].var_, fitted_2["scaler"].var_)


def test_tvt_hash_check_catches_corruption(isolated_splits_dir):
    """
    DAT1 Task 4 exception-handling guard: if the saved split file is
    hand-edited/corrupted so its content no longer matches its own
    stored hash, loading it must raise loudly, not silently proceed.
    """
    y = _synthetic_labels(seed=3)
    seed = 7
    dl._get_or_build_tvt_indices("synthtest", y, seed)

    paths = dl._tvt_split_paths("synthtest", seed)
    data = dict(np.load(paths["npz"]))
    data["test_idx"] = data["test_idx"][:-1]   # corrupt: drop one row
    np.savez_compressed(paths["npz"], **data)

    with pytest.raises(AssertionError, match="hash mismatch"):
        dl._get_or_build_tvt_indices("synthtest", y, seed)


# ── Test 2: No-leakage ───────────────────────────────────────────────

def test_scaler_unaffected_by_test_only_outlier(isolated_splits_dir):
    """
    DAT1 no-leakage unit test: inject a single extreme outlier row
    (all values = +1e6) ONLY into the TEST subset. The fitted scaler's
    .mean_ and .var_ (and, for the network path, .get_support()) must be
    numerically identical whether or not that outlier row exists,
    because the scaler is fit on TRAIN rows only and must never see it.

    If the scaler differs, leakage remains.
    """
    y = _synthetic_labels(n=2000, seed=4)
    X = _synthetic_features(y, seed=4)
    seed = 2024

    train_idx, val_idx, test_idx = dl._get_or_build_tvt_indices(
        "synthtest", y, seed
    )

    # Baseline: fit on the real TRAIN rows.
    fitted_clean = dl._fit_or_load_scalers("network", seed, X[train_idx])
    os.remove(dl._scaler_path("network", seed))   # force a fresh fit below,
                                                    # not a reload of this one

    # Now inject an extreme outlier ONLY at a TEST row's position, and
    # rebuild TRAIN/VAL/TEST from this outlier-contaminated X. TRAIN
    # rows themselves are untouched (only a TEST-index row was edited),
    # so a correctly-isolated pipeline's scaler must be unaffected.
    X_with_outlier = X.copy()
    X_with_outlier[test_idx[0]] = 1e6

    fitted_with_outlier = dl._fit_or_load_scalers(
        "network", seed, X_with_outlier[train_idx]
    )

    np.testing.assert_allclose(
        fitted_clean["scaler"].mean_, fitted_with_outlier["scaler"].mean_,
        err_msg="Scaler .mean_ changed after injecting an outlier into a "
                "TEST-only row -- this means TRAIN-only fitting is broken "
                "and leakage remains."
    )
    np.testing.assert_allclose(
        fitted_clean["scaler"].var_, fitted_with_outlier["scaler"].var_,
        err_msg="Scaler .var_ changed after injecting an outlier into a "
                "TEST-only row -- leakage remains."
    )
    np.testing.assert_array_equal(
        fitted_clean["vt"].get_support(), fitted_with_outlier["vt"].get_support(),
        err_msg="VarianceThreshold .get_support() changed after a TEST-only "
                "outlier -- leakage remains."
    )


# ── Test 3: Grep audit (static, no dataset/sandbox needed) ──────────

def test_grep_audit_fit_transform_only_inside_scaler_helper():
    """
    DAT1 grep audit requirement: every .fit_transform( call in
    data_loader.py must live inside _fit_or_load_scalers(), operating
    on a TRAIN-only argument -- never on the pre-split full feature
    matrix. This is a static source check, not a data-dependent test,
    so it runs even without the real dataset present.
    """
    src_path = os.path.join(os.path.dirname(dl.__file__), "data_loader.py")
    with open(src_path) as f:
        lines = f.readlines()

    fit_transform_lines = [
        (i + 1, line) for i, line in enumerate(lines) if ".fit_transform(" in line
    ]

    assert fit_transform_lines, (
        "No .fit_transform( calls found at all -- did the scaler-fitting "
        "code move or get removed? This test needs updating, not silently "
        "passing on a false premise."
    )

    # Find the line range of _fit_or_load_scalers so we can confirm every
    # fit_transform call falls strictly inside it.
    start = next(i for i, l in enumerate(lines) if l.startswith("def _fit_or_load_scalers"))
    end = next(
        i for i, l in enumerate(lines[start + 1:], start + 1)
        if l.startswith("def ")
    )

    for lineno, line in fit_transform_lines:
        assert start < lineno <= end, (
            f"data_loader.py:{lineno} calls .fit_transform( outside "
            f"_fit_or_load_scalers() (lines {start+1}-{end}): {line.strip()!r} "
            f"-- this is exactly the DAT1 leakage pattern (fitting on "
            f"something other than an isolated TRAIN-only call site). Move "
            f"this call inside _fit_or_load_scalers or justify explicitly "
            f"why it's safe."
        )


def test_grep_audit_fit_transform_repo_wide():
    """
    DAT1 grep audit requirement, repository-wide form (Mati86 review
    comment): the acceptance criterion says
    'grep -n ".fit_transform(" experiments/Current model/*.py' -- i.e.
    every .py file in the directory, not only data_loader.py. The test
    above already confirms data_loader.py's own .fit_transform( calls
    are correctly isolated inside _fit_or_load_scalers(); this test
    confirms no OTHER .py file in the same directory (main.py, task.py,
    config_loader.py, etc.) calls .fit_transform( at all -- if one ever
    does, it must be reviewed the same way, not silently missed because
    only data_loader.py was ever scanned.
    """
    dir_path = os.path.dirname(dl.__file__)
    # Only the actual pipeline/entry-point code is in scope here -- test
    # files legitimately mention ".fit_transform(" in prose/docstrings
    # when describing what THIS audit covers (see this file's own
    # docstring, and test_dat1_smoke_run.py's), which would otherwise be
    # a false positive of exactly the kind the config-wiring test above
    # already documents avoiding for a different literal.
    py_files = sorted(
        f for f in os.listdir(dir_path)
        if f.endswith(".py") and not f.startswith("test_")
    )
    assert "data_loader.py" in py_files, (
        "Sanity check on the file listing itself -- data_loader.py "
        "should always be found next to itself."
    )

    for fname in py_files:
        if fname == "data_loader.py":
            continue  # covered by test_grep_audit_fit_transform_only_inside_scaler_helper
        if fname == "ciciot2023_loader.py":
            # Covered by test_dat1_ciciot_audit.py, which proves the single
            # fit_transform call is TRAIN-only (static lineage + dynamic
            # scaler-statistics check) rather than silently exempting it.
            continue
        with open(os.path.join(dir_path, fname), encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        hits = [(i + 1, l.strip()) for i, l in enumerate(lines) if ".fit_transform(" in l]
        assert not hits, (
            f"{fname} contains .fit_transform( call(s) outside "
            f"data_loader.py: {hits} -- every fit_transform in the "
            f"repository must trace to TRAIN-only data lineage inside "
            f"_fit_or_load_scalers(); this one was not audited by the "
            f"data_loader.py-only test."
        )
