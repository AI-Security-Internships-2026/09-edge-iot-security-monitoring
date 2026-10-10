"""
tests/test_dat1_ciciot_audit.py

DAT1 audit of ciciot2023_loader.py, which the repo-wide grep audit
(test_grep_audit_fit_transform_repo_wide) flagged at its
`X_train = scaler.fit_transform(X_train)` line.

Rather than silently exempting the file, this module proves the call is
TRAIN-only in two ways:

  1. STATIC: exactly one .fit_transform( call exists, its argument is
     X_train, it appears AFTER the first train_test_split( call, nothing
     is ever fit on X / X_rest / X_val / X_test, and VAL/TEST are only
     ever .transform()ed.

  2. DYNAMIC: runs the real load_and_preprocess_ciciot2023() on a tiny
     synthetic parquet and checks the scaling statistics. TRAIN must be
     exactly zero-mean / unit-variance (it was the fit set), while TEST
     must NOT be (it was never seen by the scaler). If the scaler had
     been fit on all rows, the pooled TRAIN+VAL+TEST mean would be zero.

Run with: python -m pytest tests/test_dat1_ciciot_audit.py -v
"""
import os
import re

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")
import ciciot2023_loader as cl


LOADER_PATH = os.path.join(os.path.dirname(cl.__file__), "ciciot2023_loader.py")


# ── 1. Static lineage audit ─────────────────────────────────────────

def test_ciciot_fit_transform_is_train_only_static():
    with open(LOADER_PATH, encoding="utf-8") as f:
        lines = f.read().splitlines()

    code = [(i, l.strip()) for i, l in enumerate(lines)
            if not l.strip().startswith("#")]

    ft = [(i, l) for i, l in code if ".fit_transform(" in l]
    assert len(ft) == 1, f"expected exactly one .fit_transform( call, found {ft}"
    ft_idx, ft_line = ft[0]
    assert ft_line == "X_train = scaler.fit_transform(X_train)", (
        f"fit_transform call changed and must be re-audited: {ft_line!r}"
    )

    split_idx = next(
        i for i, l in code
        if "train_test_split(" in l and not l.startswith(("from ", "import "))
    )
    assert split_idx < ft_idx, (
        "the scaler is fit BEFORE the first train_test_split( call -- "
        "this is the DAT1 leakage pattern."
    )

    # Never fit on anything other than X_train.
    bad = re.compile(r"\.fit(_transform)?\(\s*(X|X_rest|X_val|X_test)\s*\)")
    offenders = [(i + 1, l) for i, l in code if bad.search(l)]
    assert not offenders, f"scaler fit on non-TRAIN data: {offenders}"

    # VAL/TEST must only be transformed, and only after the fit.
    for needle in ("scaler.transform(X_val)", "scaler.transform(X_test)"):
        hits = [i for i, l in code if needle in l]
        assert hits and min(hits) > ft_idx, (
            f"{needle} missing or placed before the TRAIN-only fit"
        )


# ── 2. Dynamic check on the real function ───────────────────────────

CATEGORIES = ["Benign", "DDoS", "DoS", "Mirai", "Recon",
              "Spoofing", "BruteForce", "Web"]


@pytest.fixture
def tiny_parquet(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    frames = []
    for k, c in enumerate(CATEGORIES):
        f = pd.DataFrame(rng.normal(loc=k, size=(100, 6)),
                         columns=[f"f{i}" for i in range(6)])
        f["label"] = c
        f["category"] = c
        f["binary_label"] = int(c != "Benign")
        frames.append(f)
    df = pd.concat(frames).sample(frac=1, random_state=0).reset_index(drop=True)
    path = tmp_path / "ciciot2023.parquet"
    df.to_parquet(path, row_group_size=200)

    monkeypatch.setattr(cl, "_ENV_PATH", str(path))
    cl._STATE.clear()
    cl._PARTITION_CACHE.clear()
    yield path
    cl._STATE.clear()
    cl._PARTITION_CACHE.clear()


def test_ciciot_scaler_fit_on_train_only_dynamic(tiny_parquet):
    st = cl.load_and_preprocess_ciciot2023(seed=1, subset_fraction=1.0)
    Xtr, Xva, Xte = st["X_train"], st["X_val"], st["X_test"]

    # TRAIN is the fit set: exactly standardized.
    np.testing.assert_allclose(Xtr.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(Xtr.std(axis=0), 1.0, atol=1e-9)

    # TEST/VAL were only transformed: NOT standardized themselves.
    assert not np.allclose(Xte.mean(axis=0), 0.0, atol=1e-6), (
        "TEST is exactly zero-mean -- the scaler appears to have seen it."
    )
    assert not np.allclose(Xva.mean(axis=0), 0.0, atol=1e-6)

    # If the scaler had been fit on everything, the pooled mean would be 0.
    pooled = np.vstack([Xtr, Xva, Xte])
    assert not np.allclose(pooled.mean(axis=0), 0.0, atol=1e-6), (
        "pooled TRAIN+VAL+TEST mean is zero -- scaler was fit on all rows."
    )
