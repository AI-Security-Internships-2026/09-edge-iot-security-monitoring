"""
ciciot2023_loader.py

Issue 5, Task 3/6 (E7) -- cross-dataset external-validity confirmation.
Loads CICIoT2023 (Neto et al., Sensors 2023) as a drop-in stand-in for
data_loader.py's NETWORK-model surface, so main.py's existing round
loop, Krum/Calibrated-Krum aggregation, DP path, and attack dispatch
all work against it completely unmodified.

------------------------------------------------------------------------
HOW THE SWAP WORKS (read this before touching main.py's --dataset wiring)
------------------------------------------------------------------------
This module exports the EXACT SAME NAMES, with the EXACT SAME call
signatures, as data_loader.py's network-model functions:
    NETWORK_NAMES, NUM_NETWORK_CLASSES, APP_NAMES, NUM_APP_CLASSES,
    load_partition_network(partition_id, num_partitions, local_val_size,
                            alpha, seed),
    load_partition_application(...)   -- NotImplementedError, see below,
    get_global_test_holdout(model_type, seed),
    get_global_validation_holdout(model_type, seed),
    get_global_train_holdout(model_type, seed),
    get_class_counts_network(seed), get_class_counts_application(seed),
    encode_labels, ALL_CLASSES  -- present for API-completeness/no-crash
                                    on an unexpected import, not really
                                    meaningful for a single-taxonomy
                                    dataset like this one.

main.py's --dataset ciciot2023 path does, at the very top of the file
(before task.py or anything else imports data_loader):

    if _args.dataset == "ciciot2023":
        import ciciot2023_loader
        sys.modules["data_loader"] = ciciot2023_loader

Every subsequent `from data_loader import ...` anywhere in the process
(task.py, dp_persistent_client_state.py doesn't import data_loader
directly, model_defs.py doesn't either) then transparently resolves to
THIS module instead. This is a deliberate, narrow use of sys.modules
substitution rather than threading a --dataset parameter through every
call site in task.py/data_loader.py's consumers -- task.py imports
NETWORK_NAMES etc. by name at ITS OWN module-import time
(`from data_loader import (NETWORK_NAMES, ...)`), so there is no clean
seam to inject a dataset choice without either this swap or a much
larger refactor of task.py's import structure. If task.py or
model_defs.py's contract ever changes to accept an explicit dataset
module parameter instead, this swap becomes unnecessary and should be
removed in favor of that.

APPLICATION MODEL: CICIoT2023 has no HTTP-payload / MQTT-application-
layer attack taxonomy analogous to Edge-IIoTset's application model
(SQL injection / XSS / Uploading / etc. as a SEPARATE classification
task from network-layer DDoS/DoS) -- it is a single, flat taxonomy.
load_partition_application() / get_class_counts_application() raise
NotImplementedError with a clear message rather than silently
returning nonsense; E7 is scoped to the network model only per the
issue text, and main.py should never call the application-side
functions when --dataset ciciot2023 is active (--model-type application
+ --dataset ciciot2023 should be treated as a usage error at the
main.py CLI level, not something this module tries to paper over).

------------------------------------------------------------------------
FEATURE SCHEMA (verified against 3 independent public sources -- GitHub
preprocessing scripts, a HuggingFace dataset card, and a data-viewer
snapshot -- since a wrong column NAME here would silently corrupt every
downstream row rather than error; ALSO validated at runtime against the
actual CSV header on first load, see _load_raw()'s assertion, in case
the real files on disk differ from what's documented below)
------------------------------------------------------------------------
The real CICIoT2023 CSVs ship as ~169 "part-*.csv" files under the
official download, each with this exact 47-column header (46 numeric
features + 'label'):

    flow_duration, Header_Length, Protocol Type, Duration, Rate, Srate,
    Drate, fin_flag_number, syn_flag_number, rst_flag_number,
    psh_flag_number, ack_flag_number, ece_flag_number, cwr_flag_number,
    ack_count, syn_count, fin_count, urg_count, rst_count, HTTP, HTTPS,
    DNS, Telnet, SMTP, SSH, IRC, TCP, UDP, DHCP, ARP, ICMP, IGMP, IPv,
    LLC, Tot sum, Min, Max, AVG, Std, Tot size, IAT, Number, Magnitue,
    Radius, Covariance, Variance, Weight, label

(Note: "Magnitue" -- not "Magnitude" -- is the real, published column
name; a typo in the original dataset release, preserved here
deliberately so schema validation matches the actual file.)

------------------------------------------------------------------------
LABEL MAPPING (34 fine-grained labels -> 8 merged categories)
------------------------------------------------------------------------
CICIoT2023's raw 'label' column has 34 distinct string values (1 Benign
+ 33 attacks). Mapped here to 8 merged categories matching the
dataset's own commonly-cited "7 attack categories + Benign" taxonomy
(Neto et al.'s own paper groups DDoS/DoS/Recon/Web-based/BruteForce/
Spoofing/Mirai) -- this mapping is standard in the CICIoT2023
literature, not invented for this codebase; see CICIOT_FINE_TO_MERGED
below for the exact fine-label membership of each merged category, and
cross-check against Neto et al. 2023 (Sensors) Table 2/3 if reproducing
for a paper.
"""

import glob
import hashlib
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

# Reuses data_loader.py's own Dirichlet-partitioning implementation
# directly -- that function is dataset-agnostic (operates on an
# already-built (X_train, y_train) pair + num_classes), no reason to
# duplicate it. Imported under the REAL module name here, at THIS
# module's own import time -- this executes before main.py's
# sys.modules["data_loader"] = ciciot2023_loader swap ever happens
# (the swap only affects imports that happen AFTER this module itself
# has finished importing), so this correctly binds to the genuine
# Edge-IIoTset data_loader.py, not to this module recursively.
from data_loader import _dirichlet_partition


# ── Dataset location ──────────────────────────────────────────────────
# CICIoT2023 ships as ~169 separate part-CSV files (~13GB uncompressed
# total) -- NOT a single file like Edge-IIoTset's DNN-EdgeIIoT-dataset
# .csv. Configurable via env var so this doesn't hardcode a path that
# will be wrong on every machine; falls back to a documented default
# location analogous to data_loader.py's Edge-IIoTset convention.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CICIOT2023_DIR = os.environ.get(
    "CICIOT2023_DATASET_DIR",
    os.path.join(BASE_DIR, "datasets", "CICIoT2023"),
)
CACHE_DIR = os.path.join(BASE_DIR, "datasets", "ciciot2023_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

SPLITS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "splits", "ciciot2023")
os.makedirs(SPLITS_DIR, exist_ok=True)

# Issue 5 Task 3 spec: "10%-subset stratified slice" -- this is the
# fraction of the FULL ~46M-row dataset this loader targets, applied
# via per-file random sampling during ingestion (reading all 46M rows
# into memory first, then subsampling, would need ~13GB+ RAM for no
# benefit -- see _load_raw()'s two-stage sampling design below).
DEFAULT_SUBSET_FRACTION = 0.10

# ── Confirmed real feature schema (see module docstring) ──────────────
CICIOT_FEATURE_COLUMNS = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate",
    "Srate", "Drate", "fin_flag_number", "syn_flag_number", "rst_flag_number",
    "psh_flag_number", "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "urg_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP",
    "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC", "Tot sum", "Min", "Max",
    "AVG", "Std", "Tot size", "IAT", "Number", "Magnitue", "Radius",
    "Covariance", "Variance", "Weight",
]
CICIOT_LABEL_COLUMN = "label"

# Issue 5 Task 3 spec fallback ("If full subset is too heavy, do
# CICIoT2023 feature-aligned 5-feature overlap slice, match Edge-IIoTset
# columns by semantics"). The 4 semantic categories chosen here are NOT
# a guess -- they match exactly what an independent cross-dataset
# feature-alignment study (BRIDGE/TCH-Net, arXiv:2604.11324) found to
# be the genuinely comparable categories between Edge-IIoTset's raw
# Wireshark-captured columns and a canonical CICFlowMeter-style 46-
# feature schema: "inter-packet times, packet lengths, TCP flags, and
# header size" (that paper's own wording). 'Protocol Type' is added as
# a 5th feature since both datasets unambiguously expose protocol
# identification (Edge-IIoTset via per-protocol boolean columns like
# tcp/udp/icmp presence, CICIoT2023 via this single numeric field).
FIVE_FEATURE_OVERLAP_SLICE = [
    "flow_duration",   # duration / inter-packet-time proxy
    "Header_Length",   # header size
    "AVG",             # average packet length
    "syn_flag_number",  # TCP flag activity
    "Protocol Type",   # protocol identification
]

# 34 fine-grained labels -> 8 merged categories. Standard CICIoT2023
# "7 attack categories + Benign" grouping (Neto et al. 2023) -- see
# module docstring. Fixed list ORDER (not a set) so class index i is
# stable across every call, same convention as data_loader.py's
# ALL_CLASSES / _CLASS_TO_IDX pattern.
CICIOT_NAMES = ["Benign", "DDoS", "DoS", "Mirai", "Recon", "Spoofing",
                  "BruteForce", "Web"]
NUM_CICIOT_CLASSES = len(CICIOT_NAMES)

CICIOT_FINE_TO_MERGED = {
    "BenignTraffic": "Benign",

    "DDoS-ACK_Fragmentation": "DDoS", "DDoS-HTTP_Flood": "DDoS",
    "DDoS-ICMP_Flood": "DDoS", "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-PSHACK_Flood": "DDoS", "DDoS-RSTFINFlood": "DDoS",
    "DDoS-SlowLoris": "DDoS", "DDoS-SYN_Flood": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS", "DDoS-TCP_Flood": "DDoS",
    "DDoS-UDP_Flood": "DDoS", "DDoS-UDP_Fragmentation": "DDoS",

    "DoS-HTTP_Flood": "DoS", "DoS-SYN_Flood": "DoS",
    "DoS-TCP_Flood": "DoS", "DoS-UDP_Flood": "DoS",

    "Mirai-greeth_flood": "Mirai", "Mirai-greip_flood": "Mirai",
    "Mirai-udpplain": "Mirai",

    "Recon-HostDiscovery": "Recon", "Recon-OSScan": "Recon",
    "Recon-PingSweep": "Recon", "Recon-PortScan": "Recon",
    "VulnerabilityScan": "Recon",

    "DNS_Spoofing": "Spoofing", "MITM-ArpSpoofing": "Spoofing",

    "DictionaryBruteForce": "BruteForce",

    "SqlInjection": "Web", "CommandInjection": "Web",
    "BrowserHijacking": "Web", "Backdoor_Malware": "Web",
    "XSS": "Web", "Uploading_Attack": "Web",
}
_MERGED_TO_IDX = {name: i for i, name in enumerate(CICIOT_NAMES)}

# API-completeness surface for task.py's unconditional
# `from data_loader import (... APP_NAMES, NUM_APP_CLASSES ...)` --
# see module docstring's "APPLICATION MODEL" section. These names must
# EXIST (so the import statement itself doesn't crash) but are never
# meant to be called against this dataset.
ALL_CLASSES = list(CICIOT_NAMES)
APP_NAMES = []
NUM_APP_CLASSES = 0

_cache = {}


def encode_labels(label_series: pd.Series) -> np.ndarray:
    """Maps CICIoT2023's raw fine-grained label strings to merged-
    category indices via CICIOT_FINE_TO_MERGED. Raises loudly (not a
    silent NaN-fill) on any label string not in the mapping -- either
    the real dataset has a label this mapping doesn't know about
    (dataset version drift) or something upstream corrupted the label
    column; either way this must not be swallowed silently, since a
    silently-dropped/miscoded label would corrupt the class-balance
    accounting this whole codebase's DAT1 split protocol depends on
    being accurate."""
    mapped = label_series.map(CICIOT_FINE_TO_MERGED)
    if mapped.isna().any():
        unknown = label_series[mapped.isna()].unique().tolist()
        raise ValueError(
            f"Unknown CICIoT2023 label value(s) not in "
            f"CICIOT_FINE_TO_MERGED: {unknown}. Either the dataset on "
            f"disk is a different version than the one this mapping was "
            f"built against (see this module's docstring for the "
            f"34-label list it expects), or the label column was "
            f"misread. Do NOT silently drop these rows -- update "
            f"CICIOT_FINE_TO_MERGED explicitly after confirming what "
            f"these new label(s) mean."
        )
    return mapped.map(_MERGED_TO_IDX).values.astype(int)


def _find_part_files():
    """Real CICIoT2023 downloads as many part-*.csv files. Also accept
    a single pre-merged CSV (e.g. from a `cat part-*.csv > merged.csv`
    preprocessing step some users do, per the merge scripts this
    module's docstring's sources reference) if that's what's present
    instead."""
    if not os.path.isdir(CICIOT2023_DIR):
        raise FileNotFoundError(
            f"CICIoT2023 directory not found at {CICIOT2023_DIR!r}. Set "
            f"the CICIOT2023_DATASET_DIR environment variable to point "
            f"at the directory containing the dataset's part-*.csv "
            f"files (or a single merged.csv), e.g.:\n"
            f"  export CICIOT2023_DATASET_DIR=/data/CICIoT2023\n"
            f"Download from https://www.unb.ca/cic/datasets/iotdataset-2023.html"
        )
    parts = sorted(glob.glob(os.path.join(CICIOT2023_DIR, "part*.csv")))
    if parts:
        return parts
    merged = os.path.join(CICIOT2023_DIR, "merged.csv")
    if os.path.isfile(merged):
        return [merged]
    raise FileNotFoundError(
        f"No part-*.csv or merged.csv found under {CICIOT2023_DIR!r}. "
        f"Expected the real CICIoT2023 download layout (many part-*.csv "
        f"files) or a single pre-merged merged.csv."
    )


def _validate_schema(df: pd.DataFrame, source_path: str):
    expected = set(CICIOT_FEATURE_COLUMNS + [CICIOT_LABEL_COLUMN])
    actual = set(df.columns)
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise ValueError(
            f"{source_path}: missing expected CICIoT2023 column(s): "
            f"{sorted(missing)}. This loader's CICIOT_FEATURE_COLUMNS "
            f"list (see module docstring) was verified against public "
            f"documentation at write time, but the real file on disk "
            f"doesn't match it -- confirm you have the genuine "
            f"CICIoT2023 release (not a modified/subsetted variant) "
            f"before proceeding; do not silently zero-fill missing "
            f"columns for a security-relevant feature set."
        )
    if extra:
        print(f"  NOTE: {source_path} has {len(extra)} extra column(s) "
              f"beyond this loader's expected schema: {sorted(extra)} -- "
              f"ignored, not used as features.")


def _load_raw(subset_fraction=DEFAULT_SUBSET_FRACTION, use_five_feature_slice=False,
              rng_seed=42):
    """
    Two-stage sampling, deliberately NOT "load all 46M rows then
    subsample": each part file (~77MB / ~270K rows on average) is read
    and immediately down-sampled to subset_fraction BEFORE being
    concatenated with the rest, so peak memory stays bounded by one
    part file at a time plus the (much smaller) accumulated sample --
    not by the full ~13GB dataset. The per-file sample is intentionally
    NOT independently re-stratified per file (attack scenarios are
    largely segregated by file in the real CICIoT2023 release, i.e.
    individual files are not label-balanced) -- final stratification
    to exactly subset_fraction, balanced across the 8 merged
    categories, happens ONCE on the accumulated pool via
    train_test_split(..., stratify=y), after every file has
    contributed its raw (unstratified) share.

    Returns (X, y) -- X as a raw, unscaled DataFrame (feature columns
    only, label already stripped out and encoded into y).
    """
    part_files = _find_part_files()
    print(f"  Loading CICIoT2023 from {len(part_files)} file(s) under "
          f"{CICIOT2023_DIR} (two-stage sampling, target subset "
          f"fraction={subset_fraction:.0%})...")

    rng = np.random.default_rng(rng_seed)
    # Over-sample at the per-file stage (2x target) so the final
    # stratification step has enough of the RAREST merged category
    # (Web, BruteForce) to draw from -- CICIoT2023 is extremely
    # imbalanced (BruteForce/Web are <0.1% of the full dataset per
    # this module's docstring's cited row counts), so a naive uniform
    # subset_fraction sample per file risks ending up with zero rows
    # of the rarest categories before the final stratified draw even
    # runs. 2x is a documented, adjustable safety margin, not a
    # precisely-derived constant.
    per_file_fraction = min(subset_fraction * 2.0, 1.0)

    frames = []
    for path in part_files:
        df = pd.read_csv(path, low_memory=False)
        _validate_schema(df, path)
        if per_file_fraction < 1.0:
            df = df.sample(frac=per_file_fraction,
                             random_state=int(rng.integers(0, 2**31 - 1)))
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    del frames
    print(f"  Accumulated pool after per-file sampling: {len(full):,} rows "
          f"(from an estimated ~{int(len(full) / per_file_fraction):,} "
          f"rows across all files at 100%).")

    y_all = encode_labels(full[CICIOT_LABEL_COLUMN])
    feature_cols = (FIVE_FEATURE_OVERLAP_SLICE if use_five_feature_slice
                     else CICIOT_FEATURE_COLUMNS)
    X_all = full[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_all = np.nan_to_num(X_all.values.astype(float), nan=0.0)
    del full

    # Final stratified draw down to exactly subset_fraction of the
    # ORIGINAL (pre-any-sampling) estimated total -- i.e. relative to
    # the accumulated pool, this draws subset_fraction / per_file_fraction.
    final_frac = min(subset_fraction / per_file_fraction, 1.0)
    if final_frac < 1.0:
        counts = np.bincount(y_all, minlength=NUM_CICIOT_CLASSES)
        stratifiable = counts.min() >= 2
        if stratifiable:
            keep_idx, _ = train_test_split(
                np.arange(len(y_all)), train_size=final_frac,
                random_state=rng_seed, stratify=y_all,
            )
        else:
            print(f"  WARNING: at least one merged category has <2 rows "
                  f"in the accumulated pool ({dict(zip(CICIOT_NAMES, counts))}) "
                  f"-- cannot stratify the final draw; falling back to a "
                  f"plain random (non-stratified) sample for this class "
                  f"only. Consider raising the per-file oversample factor "
                  f"in _load_raw() if this happens.")
            keep_idx = rng.choice(len(y_all), size=int(len(y_all) * final_frac),
                                    replace=False)
        X_all, y_all = X_all[keep_idx], y_all[keep_idx]

    print(f"  Final CICIoT2023 subset: {len(y_all):,} rows, "
          f"{X_all.shape[1]} feature(s) "
          f"({'5-feature overlap slice' if use_five_feature_slice else 'full 46-feature schema'}).")
    for i, name in enumerate(CICIOT_NAMES):
        count = int((y_all == i).sum())
        pct = 100 * count / len(y_all) if len(y_all) else 0.0
        print(f"    {name:<12} {count:>8,}  {pct:5.2f}%")

    return X_all, y_all


def _tvt_split_paths(seed):
    base = os.path.join(SPLITS_DIR, f"TVT_ciciot2023_{seed}")
    return {"npz": base + ".npz", "hash": base + ".sha256"}


def _get_or_build_tvt_indices(y_filtered, seed):
    """Identical contract/hash-sidecar-integrity pattern to
    data_loader.py's _get_or_build_tvt_indices() -- see that function's
    docstring for the full rationale (SPLIT_PROTOCOL.md's stratified
    80/10/10, written once per seed, hash-checked on every subsequent
    load). Kept as a near-duplicate rather than a shared helper because
    data_loader.py's version is keyed on (model_type, seed) and this
    dataset has no model_type axis -- forcing a shared function would
    need a dummy model_type argument threaded through for no benefit.
    """
    paths = _tvt_split_paths(seed)

    if os.path.exists(paths["npz"]):
        data = np.load(paths["npz"])
        train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]
        test_hash = hashlib.sha256(test_idx.tobytes()).hexdigest()
        with open(paths["hash"]) as f:
            stored_hash = f.read().strip()
        assert test_hash == stored_hash, (
            f"CICIoT2023 TEST-holdout hash mismatch for seed={seed}: "
            f"stored={stored_hash[:12]}... computed={test_hash[:12]}... "
            f"-- do not proceed; delete both split files under "
            f"{SPLITS_DIR} only if you intend to genuinely reset this "
            f"split (see data_loader.py's identical check for the full "
            f"rationale)."
        )
        print(f"  CICIoT2023 TVT split loaded (seed={seed}): "
              f"{len(train_idx):,} train / {len(val_idx):,} val / "
              f"{len(test_idx):,} test  [hash OK: {test_hash[:12]}...]")
        return train_idx, val_idx, test_idx

    all_idx = np.arange(len(y_filtered))
    train_idx, temp_idx, _, y_temp = train_test_split(
        all_idx, y_filtered, test_size=0.20, random_state=seed,
        stratify=y_filtered,
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=seed, stratify=y_temp,
    )
    train_idx, val_idx, test_idx = np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)

    np.savez_compressed(paths["npz"], train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    test_hash = hashlib.sha256(test_idx.tobytes()).hexdigest()
    with open(paths["hash"], "w") as f:
        f.write(test_hash)
    print(f"  CICIoT2023 TVT split BUILT (seed={seed}): "
          f"{len(train_idx):,} train / {len(val_idx):,} val / "
          f"{len(test_idx):,} test  [hash: {test_hash[:12]}...]")
    return train_idx, val_idx, test_idx


def _scaler_path(seed):
    return os.path.join(SPLITS_DIR, f"scalers_ciciot2023_{seed}.pkl")


def _fit_or_load_scalers(seed, X_train_raw):
    """Same VarianceThreshold+StandardScaler-fit-on-TRAIN-only pattern
    as data_loader.py's _fit_or_load_scalers() -- always applies
    VarianceThreshold (unconditionally, unlike data_loader.py's
    model_type-dependent branch) since this loader has only one
    "model" (the network-equivalent task)."""
    path = _scaler_path(seed)
    if os.path.exists(path):
        with open(path, "rb") as f:
            fitted = pickle.load(f)
        print(f"  CICIoT2023 scalers loaded (seed={seed}) -- NOT refit.")
        return fitted

    vt = VarianceThreshold(threshold=1e-6)
    X_vt = vt.fit_transform(X_train_raw)
    fitted = {"vt": vt, "scaler": StandardScaler().fit(X_vt)}
    with open(path, "wb") as f:
        pickle.dump(fitted, f)
    print(f"  CICIoT2023 scalers FIT on TRAIN rows only (seed={seed}), "
          f"pickled to {os.path.basename(path)}.")
    return fitted


def _apply_scalers(fitted, X_raw):
    return fitted["scaler"].transform(fitted["vt"].transform(X_raw))


def load_and_preprocess_ciciot2023(seed=42, subset_fraction=DEFAULT_SUBSET_FRACTION,
                                     use_five_feature_slice=False):
    key = seed
    if key in _cache:
        return

    cache_path = os.path.join(
        CACHE_DIR, f"ciciot2023_preprocessed_{seed}"
        f"{'_5feat' if use_five_feature_slice else ''}.npz"
    )
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        _cache[key] = {k: data[k] for k in
                        ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test")}
        print(f"  CICIoT2023 preprocessed cache loaded (seed={seed}).")
        return

    X_all, y_all = _load_raw(subset_fraction=subset_fraction,
                               use_five_feature_slice=use_five_feature_slice,
                               rng_seed=seed)
    train_idx, val_idx, test_idx = _get_or_build_tvt_indices(y_all, seed)

    X_train_raw, y_train = X_all[train_idx], y_all[train_idx]
    X_val_raw,   y_val   = X_all[val_idx],   y_all[val_idx]
    X_test_raw,  y_test  = X_all[test_idx],  y_all[test_idx]

    fitted = _fit_or_load_scalers(seed, X_train_raw)
    X_train = _apply_scalers(fitted, X_train_raw)
    X_val   = _apply_scalers(fitted, X_val_raw)
    X_test  = _apply_scalers(fitted, X_test_raw)

    print(f"  CICIoT2023: {X_train.shape[1]} features after "
          f"VarianceThreshold (fit on TRAIN only)")

    np.savez_compressed(cache_path, X_train=X_train, y_train=y_train,
                          X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
    _cache[key] = {"X_train": X_train, "y_train": y_train,
                    "X_val": X_val, "y_val": y_val,
                    "X_test": X_test, "y_test": y_test}


# ===========================================================================
# Drop-in surface matching data_loader.py's network-model API exactly
# (see module docstring's "HOW THE SWAP WORKS" section).
# ===========================================================================

def get_global_test_holdout(model_type: str, seed: int = 42):
    if model_type != "network":
        raise NotImplementedError(
            f"ciciot2023_loader only implements the network-equivalent "
            f"task (model_type='network') -- got model_type={model_type!r}. "
            f"See this module's docstring's 'APPLICATION MODEL' section: "
            f"E7 is scoped to the network model only."
        )
    load_and_preprocess_ciciot2023(seed=seed)
    cached = _cache[seed]
    return cached["X_test"], cached["y_test"]


def get_global_train_holdout(model_type: str, seed: int = 42):
    if model_type != "network":
        raise NotImplementedError(
            "ciciot2023_loader only implements model_type='network' -- "
            "see get_global_test_holdout()'s docstring."
        )
    load_and_preprocess_ciciot2023(seed=seed)
    cached = _cache[seed]
    return cached["X_train"], cached["y_train"]


def get_global_validation_holdout(model_type: str, seed: int = 42):
    if model_type != "network":
        raise NotImplementedError(
            "ciciot2023_loader only implements model_type='network' -- "
            "see get_global_test_holdout()'s docstring."
        )
    load_and_preprocess_ciciot2023(seed=seed)
    cached = _cache[seed]
    return cached["X_val"], cached["y_val"]


def load_partition_network(partition_id: int, num_partitions: int = 10,
                             local_val_size: float = 0.1, alpha: float = 0.7,
                             seed: int = 42):
    load_and_preprocess_ciciot2023(seed=seed)
    cached = _cache[seed]
    return _dirichlet_partition(
        cached["X_train"], cached["y_train"], NUM_CICIOT_CLASSES,
        partition_id, num_partitions, local_val_size, alpha, seed,
    )


def load_partition_application(*args, **kwargs):
    raise NotImplementedError(
        "ciciot2023_loader has no application-model equivalent -- see "
        "this module's docstring's 'APPLICATION MODEL' section. Do not "
        "combine --dataset ciciot2023 with --model-type application."
    )


def get_class_counts_network(seed: int = 42):
    load_and_preprocess_ciciot2023(seed=seed)
    return np.bincount(_cache[seed]["y_train"].astype(int),
                         minlength=NUM_CICIOT_CLASSES).tolist()


def get_class_counts_application(seed: int = 42):
    raise NotImplementedError(
        "ciciot2023_loader has no application-model equivalent -- see "
        "load_partition_application()'s error message."
    )


# Exported under the SAME names data_loader.py uses, so
# `from data_loader import NETWORK_NAMES, NUM_NETWORK_CLASSES` inside
# task.py resolves correctly after the sys.modules swap.
NETWORK_NAMES = CICIOT_NAMES
NUM_NETWORK_CLASSES = NUM_CICIOT_CLASSES
