import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

DROP_COLS = [
    'frame.time', 'ip.src_host', 'ip.dst_host',
    'tcp.options', 'tcp.payload',
    'mqtt.conack.flags', 'mqtt.msg',
    'mqtt.protoname', 'mqtt.topic'
]

# Full 15-class list — used only for encoding the raw CSV
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
     24862,   # Normal
     49911,   # DDoS_UDP
    116436,   # DDoS_ICMP
     50062,   # Ransomware
    121568,   # DDoS_HTTP
    132488,   # DDoS_TCP  (capped from 1,615,643)
     22564,   # Vulnerability_scanner
     50110,   # MITM
]

# ── Model 2: Application-layer attacks ───────────────────────────────
APP_ORIG_IDX = [0, 5, 6, 8, 10, 11, 12, 14]
APP_NAMES    = [
    'Normal', 'SQL_injection', 'Uploading', 'Backdoor',
    'Port_Scanning', 'XSS', 'Password', 'Fingerprinting'
]
NUM_APP_CLASSES = len(APP_ORIG_IDX)

# Counts within the APPLICATION MODEL SUBSET only
# (verified by check_classes_split.py)
APP_COUNTS = [
    24862,   # Normal           12.89%
     1001,   # SQL_injection     0.52%
     1214,   # Uploading         0.63%
    50153,   # Backdoor         26.00%
    10925,   # Port_Scanning     5.66%
    51203,   # XSS              26.54%
    37634,   # Password         19.51%
    15915,   # Fingerprinting    8.25%
]

_cached_X = None
_cached_y = None


def load_and_preprocess():
    """
    Loads and preprocesses the full DNN-EdgeIIoT dataset ONCE.

    No VarianceThreshold applied here — it is applied per-model
    in the partition functions so each model keeps its relevant features:
      Network model:     applies VarianceThreshold (removes near-zero cols)
      Application model: NO VarianceThreshold (keeps HTTP/UDP features
                         that are critical for app-layer attack separation)

    Steps:
      1. Drop non-numeric columns
      2. Encode Attack_type into 0-14
      3. Cap DDoS_TCP to 18% of total (VARS-FL methodology)
      4. StandardScaler — zero mean, unit variance
      5. Cache result to disk
    """
    global _cached_X, _cached_y

    if _cached_X is not None:
        return _cached_X, _cached_y

    if os.path.exists(CACHE_PATH):
        print("  Loading cached preprocessed data...")
        data      = np.load(CACHE_PATH)
        _cached_X = data['X']
        _cached_y = data['y']
        print(f"  Cache loaded: {_cached_X.shape[0]:,} rows, "
              f"{_cached_X.shape[1]} features")
        return _cached_X, _cached_y

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
    ddos_tcp_keep = rng_bal.choice(
        ddos_tcp_idx, size=ddos_tcp_cap, replace=False
    )
    final_idx = np.concatenate([other_idx, ddos_tcp_keep])
    rng_bal.shuffle(final_idx)
    X = X[final_idx]
    y = y[final_idx]

    total = len(y)
    print(f"  Samples after capping: {total:,}")
    for i, name in enumerate(ALL_CLASSES):
        count = int((y == i).sum())
        pct   = 100 * count / total
        print(f"    {name:<25} {count:>8,}  {pct:5.2f}%")

    X = StandardScaler().fit_transform(X)

    print(f"\n  Saving cache → {CACHE_PATH}")
    np.savez_compressed(CACHE_PATH, X=X, y=y)

    _cached_X, _cached_y = X, y
    return X, y


def _dirichlet_partition(X_sub, y_sub, num_classes,
                         partition_id, num_partitions,
                         test_size, alpha, seed):
    """
    Shared Dirichlet non-IID partitioning logic used by both models.
    alpha=0.7 → moderate heterogeneity (realistic IoT scenario).
    """
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

    # Drop classes with fewer than 2 samples — real non-IID behaviour
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
                           seed: int           = 42):
    """
    Partition for Model 1 (network-layer attacks).
    Applies VarianceThreshold to remove near-zero variance features
    that add noise without helping network-level classification.
    """
    X, y = load_and_preprocess()

    mask      = np.isin(y, NETWORK_ORIG_IDX)
    X_net     = X[mask].copy()
    y_net     = y[mask].copy()

    # Apply VarianceThreshold for network model only
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
                               seed: int           = 42):
    """
    Partition for Model 2 (application-layer attacks).

    NO VarianceThreshold — HTTP and UDP features have low overall
    variance across the full dataset but are the primary discriminative
    signal between Backdoor, XSS, Password, SQL_injection, and Uploading.
    Removing them (as VarianceThreshold was doing) makes these classes
    impossible to separate — they all look identical at TCP level.

    With HTTP features included:
      http.file_data         → distinguishes Backdoor (file upload)
      http.content_length    → distinguishes Uploading vs XSS
      http.request.uri.query → distinguishes SQL_injection
      http.request.method    → GET vs POST vs PUT
      http.referer           → XSS-specific
      http.request.full_uri  → Password attack paths
    """
    X, y = load_and_preprocess()

    mask      = np.isin(y, APP_ORIG_IDX)
    X_app     = X[mask].copy()
    y_app     = y[mask].copy()

    label_map = {orig: new for new, orig in enumerate(APP_ORIG_IDX)}
    y_app     = np.array([label_map[yi] for yi in y_app])

    return _dirichlet_partition(
        X_app, y_app, NUM_APP_CLASSES,
        partition_id, num_partitions, test_size, alpha, seed
    )