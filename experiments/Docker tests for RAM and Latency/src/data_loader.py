import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(
    BASE_DIR, "datasets", "Edge-IIoTset dataset",
    "Selected dataset for ML and DL", "DNN-EdgeIIoT-dataset.csv"
)
CACHE_PATH = os.path.join(
    BASE_DIR, "datasets", "dnn_preprocessed_cache.npz"
)

DROP_COLS = [
    'frame.time', 'ip.src_host', 'ip.dst_host',
    'tcp.options', 'tcp.payload',
    'mqtt.conack.flags', 'mqtt.msg',
    'mqtt.protoname', 'mqtt.topic'
]

ALL_CLASSES = [
    'Normal', 'DDoS_UDP', 'DDoS_ICMP', 'Ransomware', 'DDoS_HTTP',
    'SQL_injection', 'Uploading', 'DDoS_TCP', 'Backdoor',
    'Vulnerability_scanner', 'Port_Scanning', 'XSS', 'Password',
    'MITM', 'Fingerprinting'
]
_encoder = LabelEncoder()
_encoder.fit(ALL_CLASSES)

# ── Model 1: Network-layer attacks ────────────────────────────────────
NETWORK_ORIG_IDX = [0, 1, 2, 3, 4, 7, 9, 13]
NETWORK_NAMES    = [
    'Normal', 'DDoS_UDP', 'DDoS_ICMP', 'Ransomware',
    'DDoS_HTTP', 'DDoS_TCP', 'Vulnerability_scanner', 'MITM'
]
NUM_NETWORK_CLASSES = len(NETWORK_ORIG_IDX)

NETWORK_COUNTS = [
     24862, 49911, 116436, 50062, 121568, 132488, 22564, 50110,
]

# ── Model 2: Application-layer attacks ───────────────────────────────
APP_ORIG_IDX = [0, 5, 6, 8, 10, 11, 12, 14]
APP_NAMES    = [
    'Normal', 'SQL_injection', 'Uploading', 'Backdoor',
    'Port_Scanning', 'XSS', 'Password', 'Fingerprinting'
]
NUM_APP_CLASSES = len(APP_ORIG_IDX)

APP_COUNTS = [
    24862, 1001, 1214, 50153, 10925, 51203, 37634, 15915,
]

# Cache is keyed by max_rows so switching --max-rows between runs of
# build_partitions.py doesn't silently reuse a stale, differently-sized
# cache. Keeps the in-memory cache dict keyed the same way.
_cache = {}


def _cache_path_for(max_rows):
    if max_rows is None:
        return CACHE_PATH
    return os.path.join(BASE_DIR, "datasets", f"dnn_preprocessed_cache_{max_rows}.npz")


def load_and_preprocess(max_rows=None):
    """
    Loads and preprocesses the full DNN-EdgeIIoT dataset ONCE, optionally
    capped to a PORTION of it via max_rows.

    No VarianceThreshold applied here — it is applied per-model
    in the partition functions so each model keeps its relevant features.

    Steps:
      1. Drop non-numeric columns
      2. Encode Attack_type into 0-14
      3. Cap DDoS_TCP to 18% of total (VARS-FL methodology)
      4. [NEW] Optionally subsample to max_rows total, STRATIFIED by
         Attack_type — a plain random sample would risk wiping out rare
         classes (SQL_injection is only 1,001 of ~568k rows); stratified
         sampling preserves per-class proportions at any max_rows size.
      5. StandardScaler — zero mean, unit variance
      6. Cache result to disk (cache path includes max_rows, so a full
         run and a --max-rows 100000 run never collide or overwrite
         each other's cache)
    """
    global _cache
    key = max_rows if max_rows is not None else "full"
    if key in _cache:
        return _cache[key]

    cache_path = _cache_path_for(max_rows)
    if os.path.exists(cache_path):
        print(f"  Loading cached preprocessed data ({cache_path})...")
        data = np.load(cache_path)
        X, y = data['X'], data['y']
        print(f"  Cache loaded: {X.shape[0]:,} rows, {X.shape[1]} features")
        _cache[key] = (X, y)
        return X, y

    print("  Loading DNN dataset (~1.2 GB, first run only)...")
    df = pd.read_csv(DATASET_PATH, low_memory=False)

    drop = [c for c in DROP_COLS + ['Attack_label'] if c in df.columns]
    df   = df.drop(columns=drop, errors='ignore')

    y  = _encoder.transform(df['Attack_type'].values)
    df = df.drop(columns=['Attack_type'])
    df = df.apply(pd.to_numeric, errors='coerce')

    X = np.nan_to_num(df.values.astype(float), nan=0.0)
    print(f"  Raw features: {X.shape[1]}")

    print("  Capping DDoS_TCP to 18% of total (VARS-FL methodology)...")
    rng_bal       = np.random.default_rng(42)
    ddos_tcp_idx  = np.where(y == 7)[0]
    other_idx     = np.where(y != 7)[0]
    ddos_tcp_cap  = int(len(other_idx) * 18 / 82)
    ddos_tcp_keep = rng_bal.choice(ddos_tcp_idx, size=ddos_tcp_cap, replace=False)
    final_idx = np.concatenate([other_idx, ddos_tcp_keep])
    rng_bal.shuffle(final_idx)
    X = X[final_idx]
    y = y[final_idx]

    total = len(y)
    print(f"  Samples after capping: {total:,}")

    # ── NEW: optional stratified subsample to a PORTION of the corpus ──
    if max_rows is not None and total > max_rows:
        print(f"  Subsampling to a portion of the dataset: {max_rows:,} of "
              f"{total:,} rows (stratified by Attack_type so rare classes "
              f"like SQL_injection aren't wiped out by a naive random sample)...")
        X, _, y, _ = train_test_split(
            X, y, train_size=max_rows, random_state=42, stratify=y
        )
        total = len(y)
        print(f"  After subsampling: {total:,} rows")

    for i, name in enumerate(ALL_CLASSES):
        count = int((y == i).sum())
        pct   = 100 * count / total
        print(f"    {name:<25} {count:>8,}  {pct:5.2f}%")

    X = StandardScaler().fit_transform(X)

    print(f"\n  Saving cache → {cache_path}")
    np.savez_compressed(cache_path, X=X, y=y)

    _cache[key] = (X, y)
    return X, y


def _dirichlet_partition(X_sub, y_sub, num_classes,
                         partition_id, num_partitions,
                         test_size, alpha, seed):
    rng           = np.random.default_rng(seed)
    class_indices = [np.where(y_sub == c)[0] for c in range(num_classes)]
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
    X_part = X_sub[idx]
    y_part = y_sub[idx]

    counts_part = np.bincount(y_part.astype(int), minlength=num_classes)
    valid       = np.isin(y_part, np.where(counts_part >= 2)[0])
    X_part      = X_part[valid]
    y_part      = y_part[valid]

    use_stratify = np.bincount(y_part.astype(int)).min() >= 2

    X_train, X_test, y_train, y_test = train_test_split(
        X_part, y_part,
        test_size    = test_size,
        random_state = seed,
        stratify     = y_part if use_stratify else None
    )
    return X_train, y_train, X_test, y_test


def load_partition_network(partition_id: int,
                           num_partitions: int = 10,
                           test_size: float    = 0.2,
                           alpha: float        = 0.7,
                           seed: int           = 42,
                           max_rows: int       = None):
    X, y = load_and_preprocess(max_rows=max_rows)

    mask  = np.isin(y, NETWORK_ORIG_IDX)
    X_net = X[mask].copy()
    y_net = y[mask].copy()

    vt    = VarianceThreshold(threshold=1e-6)
    X_net = vt.fit_transform(X_net)

    label_map = {orig: new for new, orig in enumerate(NETWORK_ORIG_IDX)}
    y_net     = np.array([label_map[yi] for yi in y_net])

    return _dirichlet_partition(
        X_net, y_net, NUM_NETWORK_CLASSES,
        partition_id, num_partitions, test_size, alpha, seed
    )


def load_partition_application(partition_id: int,
                               num_partitions: int = 10,
                               test_size: float    = 0.2,
                               alpha: float        = 0.7,
                               seed: int           = 42,
                               max_rows: int       = None):
    X, y = load_and_preprocess(max_rows=max_rows)

    mask  = np.isin(y, APP_ORIG_IDX)
    X_app = X[mask].copy()
    y_app = y[mask].copy()

    label_map = {orig: new for new, orig in enumerate(APP_ORIG_IDX)}
    y_app     = np.array([label_map[yi] for yi in y_app])

    return _dirichlet_partition(
        X_app, y_app, NUM_APP_CLASSES,
        partition_id, num_partitions, test_size, alpha, seed
    )


# ── NEW: offline per-client partition builder (used by build_partitions.py)

def save_client_partitions(model_type, num_clients, out_dir,
                           max_rows=None, test_size=0.2,
                           alpha=0.7, seed=42):
    """
    Build and save one .npz partition file per client, containing ONLY
    that client's local train/test split. Called by build_partitions.py,
    which runs OUTSIDE the RAM-constrained containers. client.py at
    runtime never calls this or imports pandas/sklearn — it only ever
    loads the resulting per-client file, matching what a real gateway
    would actually have access to.
    """
    os.makedirs(out_dir, exist_ok=True)
    fn = (load_partition_network if model_type == "network"
          else load_partition_application)

    for cid in range(num_clients):
        X_tr, y_tr, X_te, y_te = fn(
            partition_id=cid, num_partitions=num_clients,
            test_size=test_size, alpha=alpha, seed=seed,
            max_rows=max_rows,
        )
        out_path = os.path.join(out_dir, f"client_{cid}_{model_type}.npz")
        np.savez_compressed(out_path, X_train=X_tr, y_train=y_tr,
                            X_test=X_te, y_test=y_te)
        print(f"  Saved {out_path}  train={len(X_tr):,} test={len(X_te):,} "
              f"features={X_tr.shape[1]} "
              f"({os.path.getsize(out_path) / 1024 / 1024:.2f}MB)")