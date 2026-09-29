"""
ciciot2023_loader.py -- CICIoT2023 data loader, CLEANED-PARQUET EDITION.

Rewritten for the single cleaned `ciciot2023.parquet` distribution
(cleaning_report.json: 46,776,697 rows, 3 malformed source rows
dropped, 1,037 infinite `rate` values capped, `Tot size` dropped as an
exact duplicate of AVG, duplicates deliberately RETAINED). This is
NOT the same file layout as the original 169-part-CSV loader -- do
not mix the two.

HOW THE SWAP WORKS (unchanged from the original loader -- see
main.py's `if _args.dataset == "ciciot2023":` block, which MUST run
before `task.py` (or anything importing `data_loader`) is imported,
since Python caches sys.modules on first import):

    import ciciot2023_loader
    sys.modules["data_loader"] = ciciot2023_loader

Everything below this point implements the same public surface a
`data_loader`-compatible module is expected to provide. THE EXACT
CALL SIGNATURE OF `load_partition_network()` HAS NOT BEEN VERIFIED
AGAINST task.py's ACTUAL CALL SITE -- this reproduces the one call
confirmed from main.py itself
(`load_and_preprocess_ciciot2023(seed=..., subset_fraction=...,
use_five_feature_slice=...)`) and a best-effort reconstruction of
`get_global_test_holdout` / `load_partition_network` from the
run-log conventions observed elsewhere in this codebase. Before
trusting this end to end, run:

    grep -n "data_loader\.\|from data_loader import" task.py

and reconcile any mismatch (arg names/order, return shape) with what
is implemented below.

MEMORY SAFETY: the source file is ~700MB / 46.7M rows. This loader
NEVER reads the whole table into memory. It streams row groups via
pyarrow, and for each of the 8 `category` values keeps a bounded
reservoir (Algorithm R) sized to that category's own target subset
count -- so a rare category (e.g. ~1,252 raw `Uploading_Attack` rows,
which is folded into the Web category) is sampled at the SAME rate
as DDoS, not swamped by it. This is what "stratified subset,
attack-class representation preserved" (Issue 5, E7) means in code.

DAT1 DISCIPLINE: split is computed once per (seed, subset_fraction,
five_feature_slice) and cached to disk. The StandardScaler is fit on
TRAIN ONLY and applied unchanged to VAL/TEST. A split_hash (sha256 of
the sorted, stringified TEST-row feature+label content) is exposed so
main.py's existing "[Task 6] split_hash backfilled..." logging works
unchanged for CICIoT2023 runs too.
"""

import hashlib
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler

# ── Locating the dataset ─────────────────────────────────────────────
# CICIOT2023_DATASET_DIR may point at a directory containing
# ciciot2023.parquet, OR directly at a .parquet file.
_ENV_PATH = os.environ.get("CICIOT2023_DATASET_DIR")
_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "datasets", "ciciot2023.parquet"
)


def _resolve_parquet_path():
    p = _ENV_PATH or _DEFAULT_PATH
    if os.path.isdir(p):
        cand = os.path.join(p, "ciciot2023.parquet")
        if os.path.isfile(cand):
            return cand
        raise FileNotFoundError(
            f"CICIOT2023_DATASET_DIR={p!r} is a directory but does not "
            f"contain ciciot2023.parquet. Point CICIOT2023_DATASET_DIR "
            f"at the file itself, or place it at that path. Download: "
            f"https://www.unb.ca/cic/datasets/iotdataset-2023.html "
            f"(or your cleaned-Parquet source)."
        )
    if os.path.isfile(p):
        return p
    raise FileNotFoundError(
        f"CICIoT2023 parquet not found at {p!r}. Set CICIOT2023_DATASET_DIR "
        f"to the file or its containing directory."
    )


# ── Schema ────────────────────────────────────────────────────────────
# Non-feature columns present in the cleaned Parquet (see
# cleaning_report.json's "schema" list). Everything else in the file
# is treated as a numeric feature -- computed from the actual Parquet
# schema at load time (not hardcoded to a specific count), so this
# stays correct even if the cleaned release adds/drops a feature
# column in a future version.
_METADATA_COLUMNS = {
    "label", "category", "binary_label", "source_file", "rate_was_infinite",
}

DEFAULT_SUBSET_FRACTION = 0.10

# Expected 8-category CICIoT2023 grouping (paper convention). Verified
# against the ACTUAL unique values in the `category` column at load
# time in _load_raw() -- if the real data disagrees (different
# spelling/grouping), this raises loudly rather than silently
# mis-assigning classes. Benign is forced to index 0 to match the
# "Normal"-first convention used elsewhere in this codebase for
# Edge-IIoTset; the remaining 7 are sorted alphabetically for a
# deterministic, arbitrary-but-fixed ordering.
_EXPECTED_CATEGORIES = {
    "benign", "ddos", "dos", "mirai", "recon", "spoofing",
    "bruteforce", "web",
}

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.80, 0.10, 0.10

# module-level state populated by load_and_preprocess_ciciot2023();
# every other public function reads from this rather than re-loading.
_STATE = {}


# ── Cache ────────────────────────────────────────────────────────────

def _cache_path(seed, subset_fraction, use_five_feature_slice, parquet_path):
    src_stat = os.stat(parquet_path)
    key = (
        f"seed={seed}|frac={subset_fraction}|5f={int(use_five_feature_slice)}"
        f"|src={parquet_path}|size={src_stat.st_size}|mtime={int(src_stat.st_mtime)}"
    )
    h = hashlib.sha256(key.encode()).hexdigest()[:20]
    cache_dir = os.path.join(os.path.dirname(parquet_path), "_ciciot_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"ciciot_split_{h}.pkl")


# ── Streaming, per-category stratified reservoir sample ────────────────

def _reservoir_stratified_sample(parquet_path, subset_fraction, seed, feature_cols):
    """
    Two passes over the Parquet file, neither of which materializes
    the full table:

      Pass 1 -- read ONLY the `category` column (one column, all row
      groups) to get exact per-category row counts. Cheap: this is a
      few MB even at 46.7M rows.

      Pass 2 -- stream row groups reading [feature_cols + category],
      and for each category maintain a reservoir of size
      ceil(category_count * subset_fraction) using Algorithm R, so
      every row in that category has equal retention probability
      regardless of which row group it lands in. A rare category
      (smallest count in the file) is therefore sampled at the SAME
      rate as the largest one -- this is what keeps attack-class
      representation preserved in the subsample.
    """
    pf = pq.ParquetFile(parquet_path)
    rng = np.random.default_rng(seed)

    # -- Pass 1: exact per-category counts --
    counts = {}
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=["category"])
        vals, cnts = np.unique(tbl.column("category").to_numpy(zero_copy_only=False),
                                return_counts=True)
        for v, c in zip(vals, cnts):
            counts[v] = counts.get(v, 0) + int(c)

    found_norm = {str(k).strip().lower() for k in counts}
    unexpected = found_norm - _EXPECTED_CATEGORIES
    missing = _EXPECTED_CATEGORIES - found_norm
    if unexpected or missing:
        raise ValueError(
            f"CICIoT2023 `category` column does not match the expected "
            f"8-category grouping.\n  Found:    {sorted(counts.keys())}\n"
            f"  Expected (case-insensitive): {sorted(_EXPECTED_CATEGORIES)}\n"
            f"  Unexpected: {sorted(unexpected) or 'none'}\n"
            f"  Missing:    {sorted(missing) or 'none'}\n"
            f"Fix _EXPECTED_CATEGORIES (or the mapping) to match the real "
            f"data before proceeding -- do not silently drop rows."
        )

    target_per_cat = {cat: max(2, int(np.ceil(n * subset_fraction)))
                       for cat, n in counts.items()}
    print(f"  [ciciot2023_loader] Full-dataset category counts and "
          f"{subset_fraction:.1%} stratified targets:")
    for cat in sorted(counts):
        print(f"    {cat:14s} full={counts[cat]:>10,}  "
              f"target={target_per_cat[cat]:>9,}")

    # -- Pass 2: reservoir sample per category, streaming row groups --
    reservoirs = {cat: [] for cat in counts}
    seen = {cat: 0 for cat in counts}
    read_cols = list(feature_cols) + ["category"]

    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=read_cols)
        df = tbl.to_pandas()
        for cat, group in df.groupby("category", sort=False):
            k = target_per_cat[cat]
            rows = group.to_dict("records")
            for row in rows:
                seen[cat] += 1
                if len(reservoirs[cat]) < k:
                    reservoirs[cat].append(row)
                else:
                    j = rng.integers(0, seen[cat])
                    if j < k:
                        reservoirs[cat][j] = row

    all_rows = [r for cat_rows in reservoirs.values() for r in cat_rows]
    out = pd.DataFrame(all_rows)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    subset_counts = out["category"].value_counts().to_dict()
    print(f"  [ciciot2023_loader] Stratified subset built: "
          f"{len(out):,} rows total.")
    for cat in sorted(subset_counts):
        print(f"    {cat:14s} subset={subset_counts[cat]:>9,}  "
              f"(target was {target_per_cat.get(cat, 'n/a')})")

    return out


# ── Public API ───────────────────────────────────────────────────────

def load_and_preprocess_ciciot2023(seed=42, subset_fraction=DEFAULT_SUBSET_FRACTION,
                                    use_five_feature_slice=False):
    """
    Builds (or loads from cache) the leakage-safe TRAIN/VAL/TEST split
    for CICIoT2023 and populates module-level _STATE. Idempotent per
    (seed, subset_fraction, use_five_feature_slice, source file).
    """
    if use_five_feature_slice:
        warnings.warn(
            "--ciciot-five-feature-slice has no effect with the cleaned-"
            "Parquet loader: this schema already has far fewer columns "
            "than the original 46-feature part-CSV layout, so there is "
            "no separate reduced-feature slice to switch to. Proceeding "
            "with the full cleaned-Parquet feature set.",
            stacklevel=2,
        )

    parquet_path = _resolve_parquet_path()
    cache_file = _cache_path(seed, subset_fraction, use_five_feature_slice, parquet_path)

    if os.path.isfile(cache_file):
        with open(cache_file, "rb") as f:
            state = pickle.load(f)
        _STATE.clear()
        _STATE.update(state)
        _sync_module_globals()
        print(f"  [ciciot2023_loader] Cached split loaded (seed={seed}, "
              f"subset_fraction={subset_fraction}): "
              f"{len(_STATE['y_train']):,} train / {len(_STATE['y_val']):,} val "
              f"/ {len(_STATE['y_test']):,} test rows, "
              f"{_STATE['num_features']} features")
        return _STATE

    schema_names = pq.ParquetFile(parquet_path).schema_arrow.names
    feature_cols = [c for c in schema_names if c not in _METADATA_COLUMNS]

    df = _reservoir_stratified_sample(parquet_path, subset_fraction, seed, feature_cols)

    cats_present = sorted(df["category"].unique(), key=lambda c: str(c).lower())
    benign_key = next(c for c in cats_present if str(c).strip().lower() == "benign")
    other_cats = sorted([c for c in cats_present if c != benign_key])
    category_names = [benign_key] + other_cats
    cat_to_idx = {c: i for i, c in enumerate(category_names)}

    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["category"].map(cat_to_idx).to_numpy(dtype=np.int64)

    if not np.isfinite(X).all():
        bad = int((~np.isfinite(X)).sum())
        raise ValueError(
            f"{bad} non-finite feature values survived into the sampled "
            f"subset despite cleaning_report.json claiming 0 unresolved "
            f"numeric values -- do not silently coerce these; investigate "
            f"the source Parquet."
        )

    # Stratified 80/10/10 TRAIN/VAL/TEST split (DAT1 discipline: no
    # leakage, reproducible given `seed`).
    from sklearn.model_selection import train_test_split
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=(VAL_FRAC + TEST_FRAC), stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
        stratify=y_rest, random_state=seed
    )

    # Fit scaler on TRAIN ONLY; apply unchanged to VAL/TEST.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Task 6 traceability: split_hash over the TEST set's exact content.
    test_repr = "|".join(
        f"{row.tobytes().hex()}:{label}"
        for row, label in zip(np.round(X_test, 6), y_test)
    )
    split_hash = hashlib.sha256(test_repr.encode()).hexdigest()

    state = dict(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        num_features=X_train.shape[1],
        category_names=category_names,
        split_hash=split_hash,
        seed=seed,
        subset_fraction=subset_fraction,
    )
    with open(cache_file, "wb") as f:
        pickle.dump(state, f)

    _STATE.clear()
    _STATE.update(state)
    _sync_module_globals()
    print(f"  [ciciot2023_loader] Split built and cached (seed={seed}, "
          f"subset_fraction={subset_fraction}): "
          f"{len(y_train):,} train / {len(y_val):,} val / {len(y_test):,} test "
          f"rows, {X_train.shape[1]} features, split_hash={split_hash[:16]}...")
    return _STATE


NUM_NETWORK_CLASSES = 8  # fixed by the CICIoT2023 8-category grouping
NETWORK_NAMES = None      # populated by load_and_preprocess_ciciot2023()/cache load

# ── task.py compatibility shims ─────────────────────────────────────
# task.py does `from data_loader import (..., APP_NAMES, NUM_APP_CLASSES,
# get_class_counts_application, ...)` UNCONDITIONALLY at module-import
# time -- these names must exist in this module or that import
# statement itself fails, even though CICIoT2023 never actually uses
# the application-model path (main.py's own --dataset ciciot2023 guard
# already refuses model_type="application" before any of this could be
# reached at runtime). Stubs, not real implementations.
APP_NAMES = []
NUM_APP_CLASSES = 0


def get_class_counts_application(seed=42):
    raise NotImplementedError(
        "CICIoT2023 has no 'application' model equivalent -- this stub "
        "should be unreachable because main.py's --dataset ciciot2023 "
        "guard already refuses model_type='application' before any code "
        "path could call this."
    )


def _sync_module_globals():
    """`from data_loader import NETWORK_NAMES` (in task.py) copies the
    CURRENT VALUE at import time -- later mutation of this module's
    global does not propagate to already-executed `from X import Y`
    statements. So NETWORK_NAMES must be set here, INSIDE
    load_and_preprocess_ciciot2023()/the cache-load branch, not only
    lazily in _ensure_loaded() -- main.py calls
    load_and_preprocess_ciciot2023() directly (see its --dataset
    ciciot2023 block), never through _ensure_loaded(), so relying on
    the latter alone left NETWORK_NAMES permanently None for any
    caller (task.py included) that imports it before something else
    happens to touch _ensure_loaded() first."""
    global NETWORK_NAMES
    NETWORK_NAMES = _STATE.get("category_names")


def _ensure_loaded(seed):
    if not _STATE or _STATE.get("seed") != seed:
        load_and_preprocess_ciciot2023(seed=seed)
    _sync_module_globals()
    return _STATE


def get_class_counts_network(seed=42):
    """Per-class TRAIN-partition row counts, in the same index order as
    NETWORK_NAMES/category indices -- task.py's build_criterion_network()
    uses this directly to build inverse-sqrt class weights for FocalLoss,
    so the order here MUST match the y_train label encoding exactly."""
    state = _ensure_loaded(seed)
    counts = np.bincount(state["y_train"], minlength=NUM_NETWORK_CLASSES)
    return counts.tolist()


def get_global_test_holdout(model_type, seed=None):
    if model_type != "network":
        raise ValueError(
            f"CICIoT2023 has no '{model_type}' model equivalent -- only "
            f"'network' is supported (see main.py's --dataset ciciot2023 "
            f"guard)."
        )
    state = _ensure_loaded(seed if seed is not None else 42)
    return state["X_test"], state["y_test"]


def get_global_validation_holdout(model_type, seed=None):
    """NOTE: real name confirmed from main.py's own
    `from data_loader import get_global_test_holdout,
    get_global_validation_holdout` -- NOT get_global_val_holdout, which
    was this function's name in an earlier draft and would have raised
    ImportError."""
    if model_type != "network":
        raise ValueError(f"CICIoT2023 has no '{model_type}' model equivalent.")
    state = _ensure_loaded(seed if seed is not None else 42)
    return state["X_val"], state["y_val"]


# Fraction of each client's Dirichlet-allocated TRAIN rows held out as
# that client's own LOCAL eval split (the X_te/y_te main.py unpacks
# from load_partition()'s 4-tuple and feeds to _eval_one_client() every
# round -- used for per-round/per-client metrics and Krum diagnostics,
# NOT the official Table numbers, which come from
# get_global_test_holdout() exactly once at the end).
# CONFIRMED against the real Edge-IIoTset data_loader.py's own
# _dirichlet_partition(local_val_size=0.1) default -- 90/10, not the
# 80/20 this file originally guessed.
CLIENT_LOCAL_EVAL_FRAC = 0.1

# Cache of the full Dirichlet partitioning, keyed by (seed, alpha,
# num_clients) -- built ONCE on the first client's call, since
# main.py calls load_partition_network() once PER CLIENT (see its
# `for i in range(NUM_CLIENTS): clients_data.append(load_partition(i,
# NUM_CLIENTS, ...))` loop) and re-running the Dirichlet split from
# scratch on every one of those calls would be both wasteful and
# (worse) give each client an INCONSISTENT view of the other clients'
# allocations if the RNG state weren't shared across calls.
_PARTITION_CACHE = {}


def _build_client_partitions(num_clients, alpha, seed):
    key = (seed, alpha, num_clients)
    if key in _PARTITION_CACHE:
        return _PARTITION_CACHE[key]

    state = _ensure_loaded(seed)
    X_train, y_train = state["X_train"], state["y_train"]
    rng = np.random.default_rng(seed)

    classes = np.unique(y_train)
    client_indices = [[] for _ in range(num_clients)]
    for c in classes:
        idx_c = np.where(y_train == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        splits = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        parts = np.split(idx_c, splits)
        for i, part in enumerate(parts):
            client_indices[i].extend(part.tolist())

    from sklearn.model_selection import train_test_split as _tts
    local_rng_seed = seed
    partitions = []
    for idx in client_indices:
        idx = np.array(idx, dtype=np.int64)
        X_part, y_part = X_train[idx], y_train[idx]

        # CONFIRMED against real data_loader.py's _dirichlet_partition():
        # drop any class with <2 rows in THIS client's shard before the
        # local train/eval split, since stratify=y_part would otherwise
        # raise on a singleton class. Matches the real function exactly
        # (including using the run seed, not a client-shifted seed, for
        # the local split -- data_loader.py passes the same `seed` to
        # every client's train_test_split call).
        counts_part = np.bincount(y_part.astype(int), minlength=NUM_NETWORK_CLASSES)
        valid = np.isin(y_part, np.where(counts_part >= 2)[0])
        X_part, y_part = X_part[valid], y_part[valid]
        use_stratify = len(y_part) > 0 and np.bincount(y_part.astype(int)).min() >= 2

        X_tr, X_ev, y_tr, y_ev = _tts(
            X_part, y_part, test_size=CLIENT_LOCAL_EVAL_FRAC,
            random_state=local_rng_seed,
            stratify=y_part if use_stratify else None,
        )
        partitions.append((X_tr, y_tr, X_ev, y_ev))

    _PARTITION_CACHE[key] = partitions
    return partitions


def load_partition_network(partition_id, num_partitions=10, local_val_size=0.1,
                            alpha=0.7, seed=42):
    """
    REAL signature confirmed against both main.py's call site
    (`load_partition(i, NUM_CLIENTS, seed=_args.seed, alpha=ALPHA_DIRICHLET)`,
    positional) and the real Edge-IIoTset data_loader.py's own
    load_partition_network(partition_id, num_partitions=10,
    local_val_size=0.1, alpha=0.7, seed=42) -- parameter names/order/
    defaults now match that file exactly, including local_val_size as
    a pass-through kwarg (CLIENT_LOCAL_EVAL_FRAC above is only the
    module-level default used when the partition CACHE is first built;
    if a caller ever passes a non-default local_val_size, note that
    _build_client_partitions() caches by (seed, alpha, num_clients)
    only -- NOT by local_val_size -- matching the fact that main.py
    itself never varies local_val_size across a run, so this has not
    been made cache-key-aware).

    Returns a 4-tuple (X_train_i, y_train_i, X_local_eval_i, y_local_eval_i)
    for client `partition_id`.
    """
    partitions = _build_client_partitions(num_partitions, alpha, seed)
    if not (0 <= partition_id < num_partitions):
        raise ValueError(
            f"partition_id={partition_id} out of range for "
            f"num_partitions={num_partitions}"
        )
    return partitions[partition_id]
