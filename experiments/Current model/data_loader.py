import os
import re
import hashlib
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH = os.path.join(
    BASE_DIR, "datasets", "Edge-IIoTset dataset",
    "Selected dataset for ML and DL", "DNN-EdgeIIoT-dataset.csv"
)
CACHE_PATH = os.path.join(
    BASE_DIR, "datasets", "dnn_preprocessed_cache.npz"
)

# ── DAT1: Train/Val/Test split artifacts + fitted-scaler pickles ─────
# Per Issue DAT1 Task 1: experiments/Current model/splits/. This module
# lives at experiments/Current model/data_loader.py, so "splits" next
# to it resolves to the exact path the issue specifies.
SPLITS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits")
os.makedirs(SPLITS_DIR, exist_ok=True)

# ── Columns dropped outright (no numeric or engineerable signal) ─────
# NOTE: http.request.full_uri, http.request.uri.query, http.file_data,
# http.referer, and tcp.payload used to be in this list. They are now
# engineered into numeric features below (application model only) and
# then dropped from the raw frame afterward — see engineer_text_features().
DROP_COLS = [
    'frame.time', 'ip.src_host', 'ip.dst_host',
    'tcp.options',
    'mqtt.conack.flags', 'mqtt.msg',
    'mqtt.protoname', 'mqtt.topic'
]

# Text columns engineered into numeric features for the application model.
# mqtt.msg, mqtt.topic, dns.qry.name were tested (verify_text_signal.py)
# and showed flat ~3.0 avg_len across every class — no evidence of signal,
# so they are NOT engineered; mqtt.msg/mqtt.topic stay in DROP_COLS above,
# dns.qry.name is left alone (numeric-coerced to 0 like before, harmless).
TEXT_FEATURE_COLS = [
    'tcp.payload',              # hex-encoded; decoded before analysis
    'http.request.full_uri',
    'http.request.uri.query',
    'http.file_data',
    'http.referer',
]

# Handled separately from TEXT_FEATURE_COLS (categorical, not free text)
# but ALSO needs removing from the generic numeric-coercion block below
# for the same reason — it's a string column that was silently getting
# zeroed by pd.to_numeric otherwise. This was overlooked in the original
# text-feature pass; found via inspect_weak_class_payloads.py, which
# showed request METHOD (GET vs POST) correlating with attack type.
METHOD_COL = 'http.request.method'

SQLI_PATTERN = re.compile(
    r"(?:union\s+select|select\s+.*\s+from|or\s+1\s*=\s*1|'\s*or\s*'|--|;--|drop\s+table)",
    re.IGNORECASE,
)
XSS_PATTERN = re.compile(
    r"(?:<script|onerror\s*=|onload\s*=|javascript:|<img\s|alert\()",
    re.IGNORECASE,
)

# ── Added after inspect_weak_class_payloads.py findings ──────────────
# Raw samples showed the real distinguishing signal for XSS/Password/
# Uploading in THIS dataset is tool fingerprints in the HTTP User-Agent
# header (e.g. "Mozilla/5.0 (Hydra)" for a Password/brute-force row,
# "--reverse-check" for an XSS recon row) and request METHOD/coarse URL
# category — NOT injected exploit syntax like <script> or UNION SELECT,
# which is what SQLI_PATTERN/XSS_PATTERN above were built to catch and
# which scored near-zero importance in check_feature_signal.py.
#
# OVERFITTING TRADEOFF, stated explicitly:
#   - ua_non_browser is the GENERALIZABLE end of this: "not a standard
#     browser token" is broad and would transfer to attack tools never
#     seen in this dataset, not just the ones named below.
#   - ua_known_attack_tool is NARROWER — a short named list of specific
#     pentest tools. More dataset-coupled (only catches tools on this
#     list), kept deliberately small and separate from ua_non_browser
#     so its contribution can be evaluated independently rather than
#     silently inflating a combined score.
#   - URI keyword buckets use coarse SEMANTIC categories (xss, sqli,
#     upload, login, admin) rather than exact DVWA page paths (e.g. NOT
#     matching the literal string "/dvwa/vulnerabilities/xss/"). This
#     is still somewhat dataset-coupled — this IS a DVWA-based capture,
#     so "xss" appearing in a URL is partly a label leak from DVWA's
#     own naming convention — but it is meaningfully less brittle than
#     one-hot-encoding every exact known page path, and the keywords
#     themselves are common enough to plausibly transfer to other web
#     apps with similar page-naming conventions.
BROWSER_UA_PATTERN = re.compile(
    r"(?:mozilla|chrome|safari|firefox|edge|opera|msie|trident)",
    re.IGNORECASE,
)
ATTACK_TOOL_UA_PATTERNS = re.compile(
    r"(?:hydra|sqlmap|nikto|nmap|metasploit|burp|gobuster|wpscan|dirbuster|acunetix|nessus|openvas)",
    re.IGNORECASE,
)
USER_AGENT_LINE = re.compile(r"user-agent:\s*(.*)", re.IGNORECASE)

URI_KEYWORD_BUCKETS = {
    "xss":    re.compile(r"xss|script", re.IGNORECASE),
    "sqli":   re.compile(r"sql|union|select", re.IGNORECASE),
    "upload": re.compile(r"upload|file", re.IGNORECASE),
    "login":  re.compile(r"login|auth|password|passwd", re.IGNORECASE),
    "admin":  re.compile(r"admin|vulnerabilit", re.IGNORECASE),
}


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = pd.Series(list(s)).value_counts(normalize=True)
    return float(-(counts * np.log2(counts)).sum())


def _hex_decode_safe(s: str) -> str:
    """tcp.payload is hex-encoded raw bytes. Decode to ASCII/UTF-8 where
    possible so keyword/entropy features see the real content (e.g. the
    literal string 'password') instead of hex digit noise."""
    if not s or len(s) % 2 != 0:
        return ""
    try:
        return bytes.fromhex(s).decode("utf-8", errors="ignore")
    except (ValueError, TypeError):
        return ""


def _text_stats(series: pd.Series, prefix: str, is_hex: bool = False) -> pd.DataFrame:
    s = series.fillna("").astype(str)
    s = s.replace({"0.0": "", "0": ""})  # these placeholders mean "absent" in this dataset
    if is_hex:
        s = s.apply(_hex_decode_safe)

    out = pd.DataFrame(index=series.index)
    out[f"{prefix}_len"] = s.str.len()
    out[f"{prefix}_entropy"] = s.apply(_shannon_entropy)
    out[f"{prefix}_has_sqli"] = s.str.contains(SQLI_PATTERN).astype(int)
    out[f"{prefix}_has_xss"] = s.str.contains(XSS_PATTERN).astype(int)
    out[f"{prefix}_special_char_count"] = s.apply(
        lambda x: sum(c in "'\"<>%=;()" for c in x)
    )
    return out


def _extract_user_agent(decoded_text: str) -> str:
    """Pulls the User-Agent header value out of decoded raw HTTP text
    (tcp.payload after hex-decoding). Returns "" if no User-Agent line
    is present."""
    if not decoded_text:
        return ""
    m = USER_AGENT_LINE.search(decoded_text)
    return m.group(1).strip() if m else ""


HTTP_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS', 'PATCH', 'TRACE', 'CONNECT']


def engineer_method_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encodes http.request.method into a fixed, small set of
    known HTTP methods plus an OTHER bucket. This column was previously
    neither in DROP_COLS nor TEXT_FEATURE_COLS — it was silently
    zeroed by the generic pd.to_numeric coercion, the same failure
    mode as the original preprocessing bug, just on a column that had
    been overlooked. Found via inspect_weak_class_payloads.py, which
    showed request method correlating with attack type (e.g. login/
    brute-force attacks are POST, recon/XSS probes are often GET).
    """
    out = pd.DataFrame(index=df.index)
    if METHOD_COL not in df.columns:
        return out
    s = df[METHOD_COL].fillna("").astype(str).str.upper()
    for method in HTTP_METHODS:
        out[f"method_{method}"] = (s == method).astype(int)
    known_or_empty = s.isin(HTTP_METHODS) | s.isin(["", "0", "0.0"])
    out["method_OTHER"] = (~known_or_empty).astype(int)
    return out


def engineer_ua_and_uri_keyword_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    User-Agent and coarse URI-category features — see the module-level
    comment above ATTACK_TOOL_UA_PATTERNS for the overfitting tradeoff
    discussion for each feature added here.
    """
    out = pd.DataFrame(index=df.index)

    if 'tcp.payload' in df.columns:
        decoded = df['tcp.payload'].fillna("").astype(str).apply(_hex_decode_safe)
        ua = decoded.apply(_extract_user_agent)
        out['ua_len'] = ua.str.len()
        out['ua_non_browser'] = (
            (~ua.str.contains(BROWSER_UA_PATTERN)) & (ua.str.len() > 0)
        ).astype(int)
        out['ua_known_attack_tool'] = ua.str.contains(ATTACK_TOOL_UA_PATTERNS).astype(int)

    uri_source_cols = [c for c in ('http.request.full_uri', 'http.request.uri.query')
                        if c in df.columns]
    if uri_source_cols:
        combined_uri = df[uri_source_cols[0]].fillna("").astype(str)
        for c in uri_source_cols[1:]:
            combined_uri = combined_uri + " " + df[c].fillna("").astype(str)
        for bucket_name, pattern in URI_KEYWORD_BUCKETS.items():
            out[f"uri_kw_{bucket_name}"] = combined_uri.str.contains(pattern).astype(int)

    return out


def engineer_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts numeric signal from text columns that would otherwise be
    zeroed out by pd.to_numeric(errors='coerce') + nan_to_num, or were
    dropped outright. Verified against verify_text_signal.py output:
      - tcp.payload: rich signal for Backdoor, Password (decoded hex
        showed a literal login form for Password rows)
      - http.request.full_uri / uri.query: length signal for SQL_injection
        (avg len 18.9 / 14.3 vs ~3.0 baseline)
      - http.file_data: length signal for XSS (22.2) and Password (77.4)
      - http.referer: weak signal for XSS (1.2 vs 3.0 baseline), kept anyway

    ALSO includes method one-hot and UA/URI-keyword features (added
    after inspect_weak_class_payloads.py — see comments above
    ATTACK_TOOL_UA_PATTERNS and engineer_method_features for what these
    are and the overfitting tradeoffs of each).

    Only called from the application-model path — network model relies on
    VarianceThreshold instead and never sees these columns meaningfully.
    """
    engineered = pd.DataFrame(index=df.index)

    for col in TEXT_FEATURE_COLS:
        if col not in df.columns:
            continue
        prefix = col.replace(".", "_")
        is_hex = (col == "tcp.payload")
        engineered = pd.concat(
            [engineered, _text_stats(df[col], prefix, is_hex=is_hex)], axis=1
        )

    engineered = pd.concat([engineered, engineer_method_features(df)], axis=1)
    engineered = pd.concat([engineered, engineer_ua_and_uri_keyword_features(df)], axis=1)

    return engineered


# ── Full 15-class list — explicit index mapping, NOT LabelEncoder ────
# LabelEncoder.fit() sorts classes ALPHABETICALLY regardless of list
# order, which silently broke every index assumption in this file
# (confirmed via verify_label_bug.py: 14/15 classes mismatched, e.g.
# y==7 was assumed to be DDoS_TCP but sklearn actually assigned it to
# Normal, meaning the "DDoS_TCP cap" was capping Normal instead).
# This explicit dict guarantees ALL_CLASSES[i] and y==i always agree.
ALL_CLASSES = [
    'Normal', 'DDoS_UDP', 'DDoS_ICMP', 'Ransomware', 'DDoS_HTTP',
    'SQL_injection', 'Uploading', 'DDoS_TCP', 'Backdoor',
    'Vulnerability_scanner', 'Port_Scanning', 'XSS', 'Password',
    'MITM', 'Fingerprinting'
]
_CLASS_TO_IDX = {name: i for i, name in enumerate(ALL_CLASSES)}


def encode_labels(attack_type_series: pd.Series) -> np.ndarray:
    mapped = attack_type_series.map(_CLASS_TO_IDX)
    if mapped.isna().any():
        unknown = attack_type_series[mapped.isna()].unique().tolist()
        raise ValueError(f"Unknown Attack_type values not in ALL_CLASSES: {unknown}")
    return mapped.values.astype(int)


# ── Model 1: Network-layer attacks ────────────────────────────────────
NETWORK_ORIG_IDX = [0, 1, 2, 3, 4, 7, 9, 13]
NETWORK_NAMES    = [ALL_CLASSES[i] for i in NETWORK_ORIG_IDX]
NUM_NETWORK_CLASSES = len(NETWORK_ORIG_IDX)

# ── Model 2: Application-layer attacks ───────────────────────────────
APP_ORIG_IDX = [0, 5, 6, 8, 10, 11, 12, 14]
APP_NAMES    = [ALL_CLASSES[i] for i in APP_ORIG_IDX]
NUM_APP_CLASSES = len(APP_ORIG_IDX)

# NOTE: NETWORK_COUNTS / APP_COUNTS hardcoded tables removed deliberately.
# They were derived under the broken LabelEncoder mapping and are no
# longer trustworthy. load_and_preprocess() below prints live counts
# per corrected class on every cache rebuild — use that output to
# regenerate these tables if a static reference table is needed later.

# DAT1: cache is now keyed by (model_type, seed), since the TVT split
# (and therefore everything downstream of it) depends on seed. Each
# entry holds X_train/y_train/X_val/y_val/X_test/y_test together.
_cache = {}


def _load_raw():
    """
    Loads the raw CSV, applies the corrected label encoding, and caps
    DDoS_TCP (now genuinely DDoS_TCP, not Normal) to 18% of total.
    Returns the raw dataframe (minus dropped/label columns) and y.
    This does NOT do model-specific column selection or scaling —
    that happens separately for network vs application so the two
    models can have different feature sets (VarianceThreshold vs
    text-feature engineering).
    """
    print("  Loading DNN dataset (~1.2 GB, first run only)...")
    df = pd.read_csv(DATASET_PATH, low_memory=False)

    y = encode_labels(df['Attack_type'])

    drop = [c for c in DROP_COLS + ['Attack_label', 'Attack_type'] if c in df.columns]
    df = df.drop(columns=drop, errors='ignore')

    # NOTE: previously this capped 'DDoS_TCP'. Under the OLD buggy
    # LabelEncoder, index 7 was silently 'Normal' (see verify_label_bug.py),
    # so the old "DDoS_TCP cap" was, by accident, actually capping the
    # real 72.8% majority class the whole time. Now that labels are
    # corrected, 'DDoS_TCP' is genuinely only ~2.3% of the data and is
    # NOT a majority-class problem — 'Normal' is. The cap target below
    # was changed to match reality; the VARS-FL 18% ratio methodology
    # is preserved, only the class it targets has changed.
    print("  Capping Normal to 18% of total (VARS-FL methodology, "
          "corrected target — see comment above)...")
    majority_label_idx = _CLASS_TO_IDX['Normal']
    rng_bal = np.random.default_rng(42)
    majority_idx = np.where(y == majority_label_idx)[0]
    other_idx = np.where(y != majority_label_idx)[0]
    majority_cap = int(len(other_idx) * 18 / 82)
    majority_keep = rng_bal.choice(
        majority_idx, size=min(majority_cap, len(majority_idx)), replace=False
    )
    final_idx = np.concatenate([other_idx, majority_keep])
    rng_bal.shuffle(final_idx)

    df = df.iloc[final_idx].reset_index(drop=True)
    y = y[final_idx]

    total = len(y)
    print(f"  Samples after capping: {total:,}")
    for i, name in enumerate(ALL_CLASSES):
        count = int((y == i).sum())
        pct = 100 * count / total
        print(f"    {name:<25} {count:>8,}  {pct:5.2f}%")

    return df, y


def _build_network_features(df: pd.DataFrame) -> np.ndarray:
    work = df.drop(columns=[c for c in TEXT_FEATURE_COLS if c in df.columns], errors='ignore')
    work = work.apply(pd.to_numeric, errors='coerce')
    X = np.nan_to_num(work.values.astype(float), nan=0.0)
    return X


def _build_application_features(df: pd.DataFrame) -> np.ndarray:
    text_features = engineer_text_features(df)

    cols_to_drop = [c for c in TEXT_FEATURE_COLS + [METHOD_COL] if c in df.columns]
    work = df.drop(columns=cols_to_drop, errors='ignore')
    work = work.apply(pd.to_numeric, errors='coerce')
    work = pd.DataFrame(
        np.nan_to_num(work.values.astype(float), nan=0.0),
        columns=work.columns, index=work.index
    )

    combined = pd.concat([work, text_features], axis=1)
    return combined.values.astype(float)


def _tvt_split_paths(model_type, seed):
    base = os.path.join(SPLITS_DIR, f"TVT_global_{model_type}_{seed}")
    return {"npz": base + ".npz", "hash": base + ".sha256"}


def _scaler_path(model_type, seed):
    return os.path.join(SPLITS_DIR, f"scalers_{model_type}_{seed}.pkl")


def _get_or_build_tvt_indices(model_type, y_filtered, seed):
    """
    Stratified 80/10/10 TRAIN/VAL/TEST split over the model-specific,
    already row-filtered (class-subset + Normal-capped) label array.
    Returns (train_idx, val_idx, test_idx) -- indices INTO y_filtered.

    Written to disk once per (model_type, seed) and reused on every
    subsequent call -- this IS the "no random split drift between
    runs" requirement (DAT1 Task 1.3): once
    TVT_global_{model_type}_{seed}.npz exists, this function never
    recomputes the split for that (model_type, seed) key again, it
    just loads the saved indices. This is also what makes the
    determinism test (DAT1 Task: two successive pipeline invocations
    with the same seed produce byte-identical split artifacts) true
    by construction rather than by coincidence.

    Also computes and checks a SHA-256 hash of the TEST index list
    (DAT1 Task 4 exception-handling guard): on first build, the hash
    is written alongside the split as a sidecar file. On every
    subsequent load, the freshly-loaded TEST indices are re-hashed and
    compared against that stored hash -- if the .npz was ever
    hand-edited, corrupted, or a future code change accidentally
    regenerated a different split under the same (model_type, seed)
    key, this raises loudly instead of silently shipping a different
    TEST set under a name that claims to be reproducible.
    """
    paths = _tvt_split_paths(model_type, seed)

    if os.path.exists(paths["npz"]):
        data = np.load(paths["npz"])
        train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]

        test_hash = hashlib.sha256(test_idx.tobytes()).hexdigest()
        with open(paths["hash"]) as f:
            stored_hash = f.read().strip()
        assert test_hash == stored_hash, (
            f"TEST-holdout hash mismatch for {model_type}/seed={seed}: "
            f"stored={stored_hash[:12]}... computed={test_hash[:12]}... "
            f"The saved split file does not match its own hash sidecar -- "
            f"DO NOT PROCEED, this indicates split-file corruption or an "
            f"accidental split-generation regression. Delete both "
            f"{os.path.basename(paths['npz'])} and "
            f"{os.path.basename(paths['hash'])} and re-run ONLY if you "
            f"intend to genuinely reset this split."
        )
        print(f"  TVT split loaded ({model_type}, seed={seed}): "
              f"{len(train_idx):,} train / {len(val_idx):,} val / "
              f"{len(test_idx):,} test  [hash OK: {test_hash[:12]}...]")
        return train_idx, val_idx, test_idx

    # First time for this (model_type, seed) -- build it.
    all_idx = np.arange(len(y_filtered))

    # 80/10/10: split off 80% train vs 20% temp, then split temp 50/50
    # into val/test (10%/10% of the original total each).
    train_idx, temp_idx, _, y_temp = train_test_split(
        all_idx, y_filtered, test_size=0.20, random_state=seed,
        stratify=y_filtered
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=seed, stratify=y_temp
    )

    train_idx, val_idx, test_idx = np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)

    np.savez_compressed(paths["npz"], train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    test_hash = hashlib.sha256(test_idx.tobytes()).hexdigest()
    with open(paths["hash"], "w") as f:
        f.write(test_hash)

    print(f"  TVT split BUILT ({model_type}, seed={seed}): "
          f"{len(train_idx):,} train / {len(val_idx):,} val / "
          f"{len(test_idx):,} test  [hash: {test_hash[:12]}...]")
    return train_idx, val_idx, test_idx


def _fit_or_load_scalers(model_type, seed, X_train_raw):
    """
    Fits VarianceThreshold (network model only) + StandardScaler on
    TRAIN rows ONLY, exactly once per (model_type, seed), then pickles
    both to disk and reuses them on every subsequent call -- "do NOT
    refit between invocations" (DAT1 Task 1.6). This is also what
    makes the no-leakage unit test meaningful: .mean_/.var_ (and
    .support_ for the network model) are a pure function of TRAIN
    rows, never touched by VAL or TEST rows, anywhere in this pipeline.
    """
    path = _scaler_path(model_type, seed)

    if os.path.exists(path):
        with open(path, "rb") as f:
            fitted = pickle.load(f)
        print(f"  Scalers loaded ({model_type}, seed={seed}) -- NOT refit.")
        return fitted

    fitted = {}
    if model_type == "network":
        vt = VarianceThreshold(threshold=1e-6)
        X_vt = vt.fit_transform(X_train_raw)
        fitted["vt"]     = vt
        fitted["scaler"] = StandardScaler().fit(X_vt)
    else:
        fitted["vt"]     = None
        fitted["scaler"] = StandardScaler().fit(X_train_raw)

    with open(path, "wb") as f:
        pickle.dump(fitted, f)
    print(f"  Scalers FIT on TRAIN rows only ({model_type}, seed={seed}), "
          f"pickled to {os.path.basename(path)}.")
    return fitted


def _apply_scalers(fitted, X_raw):
    if fitted["vt"] is not None:
        X_raw = fitted["vt"].transform(X_raw)
    return fitted["scaler"].transform(X_raw)


def load_and_preprocess(model_type, seed=42):
    """
    DAT1-corrected pipeline order (previously: fit VarianceThreshold +
    StandardScaler on the ENTIRE dataset, THEN partition/split -- see
    Issue DAT1 for why that was invalid data leakage):

        raw data -> model-specific row filter (+ Normal-cap, from
        _load_raw()) -> stratified TRAIN/VAL/TEST split -> fit
        VarianceThreshold + StandardScaler on TRAIN ONLY -> transform
        TRAIN, VAL, and TEST with those SAME fitted objects.

    `seed` controls the TRAIN/VAL/TEST split (see
    _get_or_build_tvt_indices) -- it is INDEPENDENT of _load_raw()'s
    internal Normal-capping seed, which stays pinned at 42 regardless,
    by design (see that function's docstring): every seed shares the
    same underlying capped dataset, only the split/partition/training
    randomness varies.

    Caches TRAIN + VAL + TEST arrays together, keyed by (model_type,
    seed), both in-memory and on disk (splits/preprocessed_<model>_
    <seed>.npz) -- repeated calls for the same key never recompute.
    """
    if model_type not in ('network', 'application', 'both'):
        raise ValueError(
            f"model_type must be 'network', 'application', or 'both', got {model_type!r}"
        )

    for mt in (['network', 'application'] if model_type == 'both' else [model_type]):
        key = (mt, seed)
        if key in _cache:
            continue

        cache_path = os.path.join(SPLITS_DIR, f"preprocessed_{mt}_{seed}.npz")
        if os.path.exists(cache_path):
            data = np.load(cache_path)
            _cache[key] = {k: data[k] for k in
                          ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test")}
            print(f"  Preprocessed cache loaded ({mt}, seed={seed}): "
                  f"{data['X_train'].shape[0]:,} train / "
                  f"{data['X_val'].shape[0]:,} val / "
                  f"{data['X_test'].shape[0]:,} test rows, "
                  f"{data['X_train'].shape[1]} features")
            continue

        df, y = _load_raw()
        orig_idx = NETWORK_ORIG_IDX if mt == 'network' else APP_ORIG_IDX
        mask   = np.isin(y, orig_idx)
        df_sub = df.loc[mask].reset_index(drop=True)
        y_sub  = y[mask]

        label_map = {orig: new for new, orig in enumerate(orig_idx)}
        y_sub = np.array([label_map[yi] for yi in y_sub])

        X_raw = (_build_network_features(df_sub) if mt == 'network'
                 else _build_application_features(df_sub))

        train_idx, val_idx, test_idx = _get_or_build_tvt_indices(mt, y_sub, seed)

        X_train_raw, y_train = X_raw[train_idx], y_sub[train_idx]
        X_val_raw,   y_val   = X_raw[val_idx],   y_sub[val_idx]
        X_test_raw,  y_test  = X_raw[test_idx],  y_sub[test_idx]

        fitted = _fit_or_load_scalers(mt, seed, X_train_raw)

        X_train = _apply_scalers(fitted, X_train_raw)
        X_val   = _apply_scalers(fitted, X_val_raw)
        X_test  = _apply_scalers(fitted, X_test_raw)

        if mt == 'network':
            print(f"  Network model: {X_train.shape[1]} features after "
                  f"VarianceThreshold (fit on TRAIN only)")
        else:
            print(f"  Application model: {X_train.shape[1]} features "
                  f"after text engineering")

        np.savez_compressed(
            cache_path,
            X_train=X_train, y_train=y_train,
            X_val=X_val,     y_val=y_val,
            X_test=X_test,   y_test=y_test,
        )
        _cache[key] = {
            "X_train": X_train, "y_train": y_train,
            "X_val":   X_val,   "y_val":   y_val,
            "X_test":  X_test,  "y_test":  y_test,
        }


def get_global_test_holdout(model_type: str, seed: int = 42):
    """
    Returns (X_test, y_test) -- the untouched global TEST holdout for
    this (model_type, seed). NEVER pass this into any client training
    loop or any per-round decision. Evaluate against the FINAL global
    model EXACTLY ONCE per experiment, after the last FL round -- this
    is the paper's actual reported metric (DAT1 Task 1.10).
    """
    load_and_preprocess(model_type, seed=seed)
    cached = _cache[(model_type, seed)]
    return cached["X_test"], cached["y_test"]


def _dirichlet_partition(X_train, y_train, num_classes,
                         partition_id, num_partitions,
                         local_val_size, alpha, seed):
    """
    Partitions the (already TRAIN-only, already-scaled-on-TRAIN) data
    among clients via Dirichlet(alpha), then splits THIS CLIENT's
    shard local_val_size (default 0.1, i.e. 90/10) into client-local
    train vs client-local validation.

    DAT1 IMPORTANT: the "val" half returned here is for local progress
    logging / early-stopping ONLY -- it is NOT the paper's TEST metric.
    The global TEST holdout is a separate set (get_global_test_holdout())
    that no client ever sees during training, evaluated exactly once
    per experiment, after the final FL round, against the final global
    model. Do not report this function's val split as a paper number.

    alpha=0.7 -> moderate heterogeneity (realistic IoT scenario).
    """
    rng           = np.random.default_rng(seed)
    class_indices = [np.where(y_train == c)[0] for c in range(num_classes)]
    client_idx    = [[] for _ in range(num_partitions)]

    for c_idx in class_indices:
        if len(c_idx) == 0:
            continue
        proportions = rng.dirichlet(np.ones(num_partitions) * alpha)
        counts      = (proportions * len(c_idx)).astype(int)
        counts[-1]  = len(c_idx) - counts[:-1].sum()
        shuffled    = rng.permutation(c_idx)
        start = 0
        for p, count in enumerate(counts):
            client_idx[p].extend(shuffled[start:start + count].tolist())
            start += count

    idx    = np.array(client_idx[partition_id])
    X_part = X_train[idx]
    y_part = y_train[idx]

    counts_part = np.bincount(y_part.astype(int), minlength=num_classes)
    valid       = np.isin(y_part, np.where(counts_part >= 2)[0])
    X_part      = X_part[valid]
    y_part      = y_part[valid]

    use_stratify = np.bincount(y_part.astype(int)).min() >= 2

    X_local_train, X_local_val, y_local_train, y_local_val = train_test_split(
        X_part, y_part,
        test_size    = local_val_size,
        random_state = seed,
        stratify     = y_part if use_stratify else None
    )
    return X_local_train, y_local_train, X_local_val, y_local_val


def load_partition_network(partition_id: int,
                           num_partitions: int   = 10,
                           local_val_size: float = 0.1,
                           alpha: float          = 0.7,
                           seed: int             = 42):
    load_and_preprocess('network', seed=seed)
    cached = _cache[('network', seed)]
    return _dirichlet_partition(
        cached["X_train"], cached["y_train"], NUM_NETWORK_CLASSES,
        partition_id, num_partitions, local_val_size, alpha, seed
    )


def load_partition_application(partition_id: int,
                               num_partitions: int   = 10,
                               local_val_size: float = 0.1,
                               alpha: float          = 0.7,
                               seed: int             = 42):
    load_and_preprocess('application', seed=seed)
    cached = _cache[('application', seed)]
    return _dirichlet_partition(
        cached["X_train"], cached["y_train"], NUM_APP_CLASSES,
        partition_id, num_partitions, local_val_size, alpha, seed
    )


def get_class_counts_network(seed: int = 42):
    """
    Live per-class counts for the network model's 8 remapped classes,
    computed from the (model_type, seed)'s TRAIN split ONLY -- DAT1
    requires all tuning-relevant statistics come from TRAIN/VAL, never
    TEST. NOTE: main.py/task.py must pass the run's actual --seed here
    (not rely on the default) so class-weight tuning matches the same
    TRAIN split the model actually trains on for that seed.
    """
    load_and_preprocess('network', seed=seed)
    return np.bincount(
        _cache[('network', seed)]["y_train"].astype(int),
        minlength=NUM_NETWORK_CLASSES
    ).tolist()


def get_class_counts_application(seed: int = 42):
    """
    Live per-class counts for the application model's 8 remapped
    classes, computed from the (model_type, seed)'s TRAIN split ONLY.
    Same seed-threading requirement as get_class_counts_network().
    """
    load_and_preprocess('application', seed=seed)
    return np.bincount(
        _cache[('application', seed)]["y_train"].astype(int),
        minlength=NUM_APP_CLASSES
    ).tolist()