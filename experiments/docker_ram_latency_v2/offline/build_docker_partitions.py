"""
Offline partition builder for the Docker RAM/Latency test suite (rerun).

WHY THIS EXISTS / WHAT CHANGED FROM THE OLD RUN
-------------------------------------------------
The old He-Full/He-Partial ablation used 35 features because it ran
against a PRE-text-feature-engineering snapshot of the pipeline (before
engineer_text_features() existed / before the label fix was finalised).
The row-subsample size itself (~100k rows) was not the cause of the
feature-count difference and is being kept unchanged here, per project
decision -- this script reproduces that same data VOLUME, but runs it
through the CURRENT, corrected data_loader.py (correct labels, correct
18%-Normal cap, current feature engineering). The resulting feature
count and total_params are measured directly by this script and printed
/ saved -- NOT assumed to be 39/129352 or any other prior figure, since
a 100k-row subsample's VarianceThreshold outcome is legitimately
data-dependent (see the old ablation's own documented caveat) and the
application model's feature count depends on what the text-engineering
pipeline currently produces, not a hardcoded assumption either.

WHAT THIS SCRIPT DOES
----------------------
1. Loads the raw CSV via data_loader._load_raw() -- same corrected
   labels, same 18%-Normal cap as every other current experiment.
2. Takes a STRATIFIED subsample of TARGET_ROWS total rows (default
   100,000), preserving class proportions from the capped population.
3. Runs the subsample through the model-specific feature pipeline
   (network: VarianceThreshold + StandardScaler; application: text
   feature engineering + StandardScaler) -- same code data_loader.py
   uses for the main experiments, just invoked directly here instead
   of through the full-corpus cache path.
4. Dirichlet-partitions (alpha=0.7, seed=42) into NUM_CLIENTS clients
   (default 2, matching the old Docker ablation's 2-client setup).
5. Saves one .npz per client (X_train, y_train, X_test, y_test) plus a
   manifest.json recording the actual measured feature count, row
   counts per client, and per-class counts -- ground truth for the
   Docker containers to consume, and for documenting what this run
   actually used (no hand-copied numbers into the paper later).

USAGE
-----
    python build_docker_partitions.py --model network --rows 100000 --clients 2
    python build_docker_partitions.py --model application --rows 100000 --clients 2

Run this ONCE per model type before starting the Docker suite. Output
goes to ../partitions/<model>/client_<i>.npz and
../partitions/<model>/manifest.json (read by client_runner.py inside
the containers via a mounted volume -- the containers themselves never
import pandas/sklearn/data_loader, matching model_defs.py's documented
constraint for the constrained subprocess path).
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_loader as dl  # noqa: E402

# data_loader.py's BASE_DIR is computed by walking up 3 directories from
# its own file location, which assumes it lives directly in
# experiments/Current model/ (2 levels below repo root). Our copy lives
# one level deeper (inside offline/), which throws that off by one
# level. Rather than depend on exact directory nesting, allow an
# explicit override via --csv-path or the CSV_PATH env var.
_csv_override = os.environ.get("CSV_PATH")


def stratified_subsample(df, y, target_rows, seed=42):
    """Stratified subsample preserving class proportions of the
    (already label-corrected, already 18%-Normal-capped) population.
    If target_rows >= len(y), returns everything unchanged."""
    n = len(y)
    if target_rows >= n:
        print(f"  target_rows={target_rows:,} >= available {n:,} -- using all rows.")
        return df, y

    frac = target_rows / n
    rng = np.random.default_rng(seed)
    keep_idx = []
    for c in np.unique(y):
        c_idx = np.where(y == c)[0]
        n_keep = max(1, int(round(len(c_idx) * frac)))
        n_keep = min(n_keep, len(c_idx))
        keep_idx.append(rng.choice(c_idx, size=n_keep, replace=False))
    keep_idx = np.concatenate(keep_idx)
    rng.shuffle(keep_idx)
    return df.iloc[keep_idx].reset_index(drop=True), y[keep_idx]


def build_features(df, y, model_type):
    """Mirrors data_loader.py's model-specific feature pipeline exactly,
    just invoked directly on the subsample rather than the full corpus."""
    if model_type == "network":
        orig_idx = dl.NETWORK_ORIG_IDX
        mask = np.isin(y, orig_idx)
        X_raw = dl._build_network_features(df.loc[mask])
        y_sub = y[mask]

        vt = VarianceThreshold(threshold=1e-6)
        X = vt.fit_transform(X_raw)
        X = StandardScaler().fit_transform(X)

        label_map = {orig: new for new, orig in enumerate(orig_idx)}
        y_sub = np.array([label_map[yi] for yi in y_sub])
        names = dl.NETWORK_NAMES
        n_pre_vt = X_raw.shape[1]

    elif model_type == "application":
        orig_idx = dl.APP_ORIG_IDX
        mask = np.isin(y, orig_idx)
        X = dl._build_application_features(df.loc[mask])
        X = StandardScaler().fit_transform(X)
        y_sub = y[mask]

        label_map = {orig: new for new, orig in enumerate(orig_idx)}
        y_sub = np.array([label_map[yi] for yi in y_sub])
        names = dl.APP_NAMES
        n_pre_vt = X.shape[1]  # no VarianceThreshold on this path

    else:
        raise ValueError(f"model_type must be 'network' or 'application', got {model_type!r}")

    print(f"  {model_type} model: {X.shape[1]} features "
          f"(pre-selection: {n_pre_vt}), {X.shape[0]:,} rows, {len(names)} classes")
    return X, y_sub, names


def dirichlet_partition_all_clients(X, y, num_classes, num_clients, alpha, seed, test_size):
    """Same logic as data_loader._dirichlet_partition, but returns every
    client's split in one pass (that function only returns one partition
    id at a time)."""
    rng = np.random.default_rng(seed)
    class_indices = [np.where(y == c)[0] for c in range(num_classes)]
    client_idx = [[] for _ in range(num_clients)]

    for c_idx in class_indices:
        if len(c_idx) == 0:
            continue
        proportions = rng.dirichlet(np.ones(num_clients) * alpha)
        counts = (proportions * len(c_idx)).astype(int)
        counts[-1] = len(c_idx) - counts[:-1].sum()
        shuffled = rng.permutation(c_idx)
        start = 0
        for p, count in enumerate(counts):
            client_idx[p].extend(shuffled[start:start + count].tolist())
            start += count

    partitions = []
    for p in range(num_clients):
        idx = np.array(client_idx[p])
        X_part, y_part = X[idx], y[idx]

        counts_part = np.bincount(y_part.astype(int), minlength=num_classes)
        valid = np.isin(y_part, np.where(counts_part >= 2)[0])
        X_part, y_part = X_part[valid], y_part[valid]

        use_stratify = np.bincount(y_part.astype(int)).min() >= 2 if len(y_part) else False
        X_train, X_test, y_train, y_test = train_test_split(
            X_part, y_part, test_size=test_size, random_state=seed,
            stratify=y_part if use_stratify else None
        )
        partitions.append((X_train, y_train, X_test, y_test))
    return partitions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["network", "application"], required=True)
    ap.add_argument("--rows", type=int, default=100_000,
                     help="Target row count for the stratified subsample "
                          "(default 100000, matching the old Docker ablation's data volume)")
    ap.add_argument("--clients", type=int, default=2,
                     help="Number of Dirichlet partitions to produce (default 2, "
                          "matching the old He-Full/He-Partial/pure_dp Docker tests)")
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None,
                     help="Output dir (default: ../partitions/<model>/ next to this script)")
    ap.add_argument("--csv-path", type=str, default=None,
                     help="Explicit path to DNN-EdgeIIoT-dataset.csv, overriding "
                          "data_loader.py's auto-computed BASE_DIR path. Also settable "
                          "via the CSV_PATH env var. Use this if you get a "
                          "FileNotFoundError from the default path.")
    args = ap.parse_args()

    csv_path = args.csv_path or _csv_override
    if csv_path:
        if not os.path.exists(csv_path):
            print(f"ERROR: --csv-path/CSV_PATH given but file not found: {csv_path}")
            sys.exit(1)
        dl.DATASET_PATH = csv_path
        print(f"Using explicit CSV path: {csv_path}")
    elif not os.path.exists(dl.DATASET_PATH):
        print(f"ERROR: dataset not found at the auto-computed path:\n  {dl.DATASET_PATH}")
        print("Pass --csv-path \"<full path to DNN-EdgeIIoT-dataset.csv>\" to override.")
        sys.exit(1)

    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "partitions", args.model
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"[1/4] Loading + label-correcting + capping raw data...")
    df, y = dl._load_raw()

    print(f"[2/4] Stratified subsample -> target {args.rows:,} rows...")
    df_sub, y_sub = stratified_subsample(df, y, args.rows, seed=args.seed)

    print(f"[3/4] Building {args.model} features on the subsample...")
    X, y_final, names = build_features(df_sub, y_sub, args.model)
    num_classes = len(names)

    print(f"[4/4] Dirichlet-partitioning into {args.clients} clients "
          f"(alpha={args.alpha}, seed={args.seed})...")
    partitions = dirichlet_partition_all_clients(
        X, y_final, num_classes, args.clients, args.alpha, args.seed, args.test_size
    )

    client_summaries = []
    for i, (X_tr, y_tr, X_te, y_te) in enumerate(partitions):
        path = os.path.join(out_dir, f"client_{i}.npz")
        np.savez_compressed(path, X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te)
        summary = {
            "client_id": i,
            "train_rows": int(len(y_tr)),
            "test_rows": int(len(y_te)),
            "class_counts_train": np.bincount(y_tr.astype(int), minlength=num_classes).tolist(),
        }
        client_summaries.append(summary)
        print(f"  client_{i}: {summary['train_rows']:,} train / {summary['test_rows']:,} test rows")

    manifest = {
        "model_type": args.model,
        "num_features": int(X.shape[1]),
        "num_classes": num_classes,
        "class_names": names,
        "subsample_rows_target": args.rows,
        "subsample_rows_actual": int(len(y_final)),
        "num_clients": args.clients,
        "dirichlet_alpha": args.alpha,
        "test_size": args.test_size,
        "seed": args.seed,
        "clients": client_summaries,
        "note": ("Feature count measured directly from this run's "
                 "VarianceThreshold/text-engineering output on a "
                 f"{args.rows}-row stratified subsample of the current, "
                 "label-corrected pipeline -- not assumed from any prior "
                 "ablation's figure."),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. {X.shape[1]} features, {args.clients} clients, "
          f"manifest + .npz written to {out_dir}")


if __name__ == "__main__":
    main()
