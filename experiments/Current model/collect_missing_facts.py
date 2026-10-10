"""Run from the folder containing data_loader.py, with the Edge-IIoTset CSV reachable
(DATASET_PATH env var, as data_loader expects). Prints the facts the manuscript still lacks.
Closes todos: 39 columns per seed, app feature count (90 vs 91), per-model TRAIN class counts
(Table X, rare-class choice), identifier-like retained columns."""
import pickle, pandas as pd, numpy as np
import data_loader as dl
dl.DATASET_PATH = r'C:\Users\Zarawar Khan\09-edge-iot-security-monitoring\datasets\DNN-EdgeIIoT-dataset.csv'
dl.DATASET_PATH = r'C:\Users\Zarawar Khan\09-edge-iot-security-monitoring\datasets\DNN-EdgeIIoT-dataset.csv'

SEEDS = (42, 123, 456, 789, 2024)
hdr = list(pd.read_csv(dl.DATASET_PATH, nrows=0).columns)
print("CSV columns:", len(hdr))
pre = [c for c in hdr if c not in dl.DROP_COLS + ["Attack_label", "Attack_type"]
       and c not in dl.TEXT_FEATURE_COLS]
print("network columns before variance filter:", len(pre))

IDLIKE = ("srcport", "dstport", "udp.port", ".seq", "ack_raw", "checksum", "trans_id", "udp.stream", "tls_port")
for seed in SEEDS:
    dl.load_and_preprocess("network", seed=seed)
    vt = pickle.load(open(dl._scaler_path("network", seed), "rb")).get("vt")
    kept = [c for c, k in zip(pre, vt.get_support())] if vt is None else \
           [c for c, k in zip(pre, vt.get_support()) if k]
    print(f"network seed {seed}: {len(kept)} features; identifier-like kept:",
          [c for c in kept if any(t in c for t in IDLIKE)])
    if seed == 42:
        print("  all kept:", kept)

for seed in SEEDS:
    dl.load_and_preprocess("application", seed=seed)
    X = dl._cache[("application", seed)]["X_train"]
    print(f"application seed {seed}: X_train.shape[1] = {X.shape[1]}")

print("\nTRAIN class counts (seed 42)")
for name, cnt in zip(dl.NETWORK_NAMES, dl.get_class_counts_network(seed=42)):
    print(f"  network     {name:24s}{cnt}")
for name, cnt in zip(dl.APP_NAMES, dl.get_class_counts_application(seed=42)):
    print(f"  application {name:24s}{cnt}")
