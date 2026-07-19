"""
FL Client — runs in constrained IoT gateway container (200-300MB RAM target).

This process NEVER imports torch. Training happens in an isolated
subprocess (train_worker.py) so PyTorch's memory is fully returned to
the OS the moment training finishes, instead of stacking on top of the
DP/ZKP/HE work that follows in the same process (see train_worker.py's
docstring for why that matters).

Privacy/robustness stack per round, in order:
  1. Train        (subprocess, torch — memory isolated)
  2. Local DP      (numpy only  — clip + Gaussian noise, whole model)
  3. ZKP proof      (numpy only — commitment + signed norm bound, whole model)
  4. Split          sensitive (classifier head, ~6% of params) vs.
                     bulk (everything else, ~94% of params)
  5. Encrypt         sensitive layers only, under the server's shared
                     public CKKS context (fixes the invalid-keypair bug —
                     see he_aggregation.py)
  6. Send            sensitive as ciphertext, bulk as raw float32 bytes
                     (base64) instead of JSON float lists — see
                     wire_format.py for why that matters
"""

import os
import sys
import time
import json
import subprocess
import numpy as np
import requests
import psutil

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wire_format import pack_array, unpack_array, pack_param_list, unpack_param_list
import he_aggregation as he

# ── Config ──────────────────────────────────────────────────────────
CLIENT_ID       = int(os.environ.get("CLIENT_ID", 0))
NUM_CLIENTS     = int(os.environ.get("NUM_CLIENTS", 2))
SERVER_URL      = os.environ.get("SERVER_URL", "http://fl_server:5000")
USE_LOCAL_DP    = os.environ.get("USE_LOCAL_DP", "true").lower() == "true"
USE_ZKP         = os.environ.get("USE_ZKP",      "true").lower() == "true"
USE_HE          = os.environ.get("USE_HE",       "true").lower() == "true"
DP_EPSILON      = float(os.environ.get("DP_EPSILON", "3.0"))
MODEL_TYPE      = os.environ.get("MODEL_TYPE", "network")
NUM_ROUNDS      = int(os.environ.get("NUM_ROUNDS", "5"))
LOCAL_EPOCHS    = int(os.environ.get("LOCAL_EPOCHS", "5"))
DIRICHLET_ALPHA = float(os.environ.get("DIRICHLET_ALPHA", "0.7"))
RESULTS_DIR     = "/results"
TMP_DIR         = "/tmp"
APP_PATH        = "/app"

NUM_FEATURES = 40 if MODEL_TYPE == "network" else 52
NUM_CLASSES  = 8

# Sensitive-layer prefix: only the classifier head goes through CKKS.
# The bulk feature-extraction layers (CNN + LSTM, ~94% of params) are
# sent DP-noised but as plaintext — see the "Partial HE" design note.
SENSITIVE_PREFIX = "classifier"


# ── Helpers ─────────────────────────────────────────────────────────

def get_ram():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def log(msg, level="INFO"):
    print(f"[Client {CLIENT_ID}][{level}][{get_ram():.0f}MB] {msg}", flush=True)


def wait_for_server(timeout=60):
    log("Waiting for server...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=5)
            if r.status_code == 200:
                log("Server ready")
                return True
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("Server not available")


def fetch_he_context():
    """Fetch the server's public CKKS context once, at startup."""
    if not USE_HE:
        return None
    log("Fetching shared public HE context from server...")
    for attempt in range(10):
        try:
            r = requests.get(f"{SERVER_URL}/get_he_context", timeout=30)
            if r.status_code == 200:
                public_b64 = r.json()["public_context"]
                context = he.load_client_context(public_b64)
                log("HE public context loaded")
                return context
        except Exception as e:
            log(f"Attempt {attempt + 1}: {e}", "WARN")
        time.sleep(3)
    raise RuntimeError("Could not fetch HE context from server")


# ── Data loading ────────────────────────────────────────────────────

def load_data():
    """
    Load real Edge-IIoTset partition from mounted dataset volume.
    Falls back to synthetic data if dataset not mounted.
    No torch here — safe to run in this process.
    """
    cache_path = "/datasets/dnn_preprocessed_cache.npz"

    if os.path.exists(cache_path):
        log("Loading from cache...")
        from data_loader import (load_partition_network,
                                  load_partition_application)
        fn = (load_partition_network if MODEL_TYPE == "network"
              else load_partition_application)
        try:
            X_tr, y_tr, X_te, y_te = fn(
                partition_id   = CLIENT_ID,
                num_partitions = NUM_CLIENTS,
                alpha          = DIRICHLET_ALPHA,
                seed           = 42
            )
            log(f"Real data loaded: train={len(X_tr):,}, test={len(X_te):,}")
            return X_tr, y_tr, X_te, y_te
        except Exception as e:
            log(f"Cache load failed: {e} — using synthetic", "WARN")

    log("Using synthetic data (dataset not mounted)")
    rng    = np.random.default_rng(42 + CLIENT_ID)
    n      = 40000
    n_feat = NUM_FEATURES
    n_cls  = NUM_CLASSES
    X      = rng.standard_normal((n, n_feat)).astype(np.float32)
    props  = rng.dirichlet(np.ones(n_cls) * DIRICHLET_ALPHA)
    counts = (props * n).astype(int)
    counts[-1] = n - counts[:-1].sum()
    y = np.concatenate([np.full(c, i) for i, c in enumerate(counts)]).astype(np.int64)
    rng.shuffle(y)
    split = int(n * 0.8)
    return X[:split], y[:split], X[split:], y[split:]


# ── Sensitive/bulk split (no torch needed — driven by saved key names) ─

def split_sensitive_bulk(keys, params):
    sensitive_idx = [i for i, k in enumerate(keys) if k.startswith(SENSITIVE_PREFIX)]
    bulk_idx      = [i for i, k in enumerate(keys) if not k.startswith(SENSITIVE_PREFIX)]
    sensitive = [params[i] for i in sensitive_idx]
    bulk      = [params[i] for i in bulk_idx]
    return sensitive, bulk, sensitive_idx, bulk_idx


# ── One FL round ────────────────────────────────────────────────────

def run_round(rnd, global_params, data_path, he_context):
    stage_times = {}
    stage_mem   = {"before_round": get_ram()}

    log(f"─── ROUND {rnd} START ───")

    # ── 1. Train in an isolated subprocess ─────────────────────────
    t0 = time.time()
    global_path = f"{TMP_DIR}/global_r{rnd}_c{CLIENT_ID}.npz"
    out_path    = f"{TMP_DIR}/raw_params_r{rnd}_c{CLIENT_ID}.npz"
    keys_path   = f"{TMP_DIR}/keys_r{rnd}_c{CLIENT_ID}.json"

    # data_path is written ONCE in main(), before the round loop — the
    # training partition never changes round to round, only the model
    # weights do. Re-serializing the whole dataset every round (the old
    # behavior) wasted I/O and RAM proportional to NUM_ROUNDS for no
    # reason, which matters a lot more once this is real sensor data
    # instead of a few-MB synthetic array.
    np.savez(global_path, *global_params)

    log("Training (isolated subprocess)...")
    result = subprocess.run(
        [
            sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_worker.py"),
            "--client-id", str(CLIENT_ID),
            "--model-type", MODEL_TYPE,
            "--num-features", str(NUM_FEATURES),
            "--num-classes", str(NUM_CLASSES),
            "--local-epochs", str(LOCAL_EPOCHS),
            "--global-params-path", global_path,
            "--train-data-path", data_path,
            "--output-path", out_path,
            "--keys-output-path", keys_path,
            "--app-path", APP_PATH,
        ],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        if result.returncode < 0:
            import signal
            try:
                sig_name = signal.Signals(-result.returncode).name
            except ValueError:
                sig_name = str(-result.returncode)
            log(f"train_worker.py was KILLED by signal {sig_name} "
                f"(returncode={result.returncode}). Empty stderr + a negative "
                f"returncode almost always means the container's mem_limit "
                f"was hit and the OOM killer sent SIGKILL — the process never "
                f"got a chance to print anything.", "ERROR")
        else:
            log(f"train_worker.py FAILED (returncode={result.returncode}):\n{result.stderr}", "ERROR")
        raise RuntimeError("Training subprocess failed")
    log(result.stdout.strip().replace("\n", "\n  "))

    raw_npz = np.load(out_path)
    raw_params = [raw_npz[k] for k in raw_npz.files]
    with open(keys_path) as f:
        keys = json.load(f)

    for p in (global_path, out_path, keys_path):
        try:
            os.remove(p)
        except OSError:
            pass

    stage_times["train"] = time.time() - t0
    stage_mem["after_train"] = get_ram()
    log(f"Training done: {stage_times['train']:.1f}s  RAM={stage_mem['after_train']:.0f}MB "
        f"(subprocess memory already reclaimed by OS)")

    # ── 2. Local DP — applied to the WHOLE model ────────────────────
    dp_info = {}
    if USE_LOCAL_DP:
        t0 = time.time()
        log("Applying Local DP...")
        from defences.local_dp import apply_local_dp
        raw_params, dp_info = apply_local_dp(
            raw_params, epsilon=DP_EPSILON, delta=1e-5, clip_norm=1.0
        )
        stage_times["dp"] = time.time() - t0
        stage_mem["after_dp"] = get_ram()
        log(f"DP done: {stage_times['dp']:.3f}s  ε={dp_info['epsilon']}  "
            f"σ={dp_info['noise_sigma']:.4f}  ‖g‖={dp_info['actual_norm']:.4f}  "
            f"RAM={stage_mem['after_dp']:.0f}MB")

    # ── 3. ZKP — commitment + norm proof over the WHOLE model ──────
    zkp_proof = None
    if USE_ZKP:
        t0 = time.time()
        log("Generating ZKP proof...")
        from defences.zkp import generate_proof
        zkp_proof = generate_proof(
            raw_params, clip_norm=1.0,
            noise_sigma=dp_info.get("noise_sigma", 0.0)
        )
        # generate_proof()'s "salt" field is raw bytes (os.urandom), which
        # is correct for its own HMAC/commitment math but is NOT JSON
        # serializable. Convert it to hex here, at the network boundary,
        # rather than changing zkp.py's internal representation.
        if isinstance(zkp_proof.get("salt"), (bytes, bytearray)):
            zkp_proof["salt"] = zkp_proof["salt"].hex()
        stage_times["zkp"] = time.time() - t0
        stage_mem["after_zkp"] = get_ram()
        pi = zkp_proof["norm_proof"]
        log(f"ZKP done: {stage_times['zkp']:.3f}s  norm={pi['norm']:.4f}  "
            f"threshold={pi['threshold']:.4f}  passed={pi['passes']}  "
            f"RAM={stage_mem['after_zkp']:.0f}MB")

    # ── 4/5. Split sensitive/bulk, encrypt sensitive only ───────────
    sensitive, bulk, sensitive_idx, bulk_idx = split_sensitive_bulk(keys, raw_params)
    n_sensitive = sum(p.size for p in sensitive)
    n_bulk      = sum(p.size for p in bulk)
    log(f"Partial HE split: sensitive={n_sensitive:,} params "
        f"({100 * n_sensitive / (n_sensitive + n_bulk):.1f}%), "
        f"bulk={n_bulk:,} params (plaintext, DP-noised)")

    update_payload = {}
    he_info = {}

    if USE_HE and he_context is not None:
        t0 = time.time()
        log(f"CKKS encryption of sensitive layers only "
            f"(poly_degree={he.POLY_MODULUS_DEGREE})...")

        sens_flat = np.concatenate([p.flatten() for p in sensitive]).astype(np.float64)
        chunk_size = he.POLY_MODULUS_DEGREE // 2
        try:
            chunks_b64 = he.encrypt_flat_array(sens_flat, he_context, chunk_size)
            update_payload["sensitive"] = {
                "mode":       "ckks",
                "chunks":     chunks_b64,
                "shapes":     [list(p.shape) for p in sensitive],
                "sizes":      [int(p.size) for p in sensitive],
                "idx":        sensitive_idx,
                "total":      int(sens_flat.size),
                "chunk_size": chunk_size,
                "n_chunks":   len(chunks_b64),
            }
            he_info = {"n_chunks": len(chunks_b64), "oom": False}
            stage_times["he_encrypt"] = time.time() - t0
            stage_mem["after_he"] = get_ram()
            log(f"HE encrypt done: {stage_times['he_encrypt']:.2f}s  "
                f"chunks={len(chunks_b64)}  RAM={stage_mem['after_he']:.0f}MB")
        except MemoryError:
            log("OOM during CKKS — falling back to plaintext for sensitive layers too", "ERROR")
            he_info = {"n_chunks": 0, "oom": True}
            update_payload["sensitive"] = {
                "mode":   "plaintext",
                **pack_param_list(sensitive),
                "idx":    sensitive_idx,
            }
        del sens_flat
    else:
        update_payload["sensitive"] = {
            "mode":   "plaintext",
            **pack_param_list(sensitive),
            "idx":    sensitive_idx,
        }

    # Bulk layers: always plaintext, but as raw float32 bytes (base64),
    # NOT as JSON nested lists — see wire_format.py.
    update_payload["bulk"] = {
        "mode": "plaintext",
        **pack_param_list(bulk),
        "idx":  bulk_idx,
    }
    update_payload["keys"] = keys

    stage_times["total"] = sum(v for k, v in stage_times.items())
    stage_mem["peak"] = get_ram()

    return update_payload, zkp_proof, stage_times, stage_mem, he_info


# ── Main ────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log(f"IoT Client {CLIENT_ID} starting")
    log(f"Stack: DP={USE_LOCAL_DP}  ZKP={USE_ZKP}  HE={USE_HE} (partial: classifier head only)")
    log(f"Model: {MODEL_TYPE}  Server: {SERVER_URL}")
    log("=" * 60)

    wait_for_server(timeout=60)
    he_context = fetch_he_context()
    X_train, y_train, X_test, y_test = load_data()
    n_samples = len(X_train)
    del X_test, y_test  # loaded by load_data() but never actually used in
                         # this file — no reason to hold them in RAM

    # Written ONCE — the training partition doesn't change round to
    # round, only the model weights do. See run_round()'s docstring
    # note for why re-writing this every round was wasteful, especially
    # once this is a real (larger) sensor dataset instead of synthetic.
    data_path = f"{TMP_DIR}/traindata_c{CLIENT_ID}.npz"
    np.savez(data_path, X_train=X_train, y_train=y_train)
    log(f"Training data written once: {n_samples:,} samples "
        f"({os.path.getsize(data_path) / 1024 / 1024:.1f}MB on disk)")
    del X_train, y_train  # parent no longer needs to hold these in RAM —
                           # the subprocess reads them fresh from data_path
                           # each round; keeping a live reference here would
                           # otherwise sit in the parent's RAM for the whole run

    all_timing = {}
    all_memory = {}

    for rnd in range(1, NUM_ROUNDS + 1):
        log(f"\n{'=' * 50}")

        log("Fetching global model from server...")
        global_params = None
        for attempt in range(10):
            try:
                r = requests.get(f"{SERVER_URL}/get_global_model", timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    global_params = unpack_param_list(data["params"])
                    log(f"Global model received: {len(global_params)} layers")
                    break
            except Exception as e:
                log(f"Attempt {attempt + 1}: {e}", "WARN")
            time.sleep(3)
        if global_params is None:
            log("Could not fetch global model — skipping round", "WARN")
            continue

        update, zkp_proof, timing, memory, he_info = run_round(
            rnd, global_params, data_path, he_context
        )
        all_timing[rnd] = timing
        all_memory[rnd] = memory

        log(f"Submitting update to server (sensitive mode={update['sensitive']['mode']})...")
        payload = {
            "client_id": CLIENT_ID,
            "round":     rnd,
            "n_samples": n_samples,
            "update":    update,
            "zkp_proof": zkp_proof,
        }

        t0 = time.time()
        try:
            r = requests.post(f"{SERVER_URL}/submit_update", json=payload, timeout=120)
            timing["http_send"] = time.time() - t0
            log(f"Server response: {r.status_code}  send_time={timing['http_send']:.2f}s")
            if r.status_code == 400:
                log(f"Update REJECTED by server: {r.json()}", "WARN")
                continue
        except Exception as e:
            log(f"Submission failed: {e}", "ERROR")
            continue

        log("Waiting for server aggregate...")
        for poll in range(60):
            time.sleep(5)
            try:
                r = requests.get(f"{SERVER_URL}/get_round_result/{rnd}", timeout=10)
                if r.status_code == 200 and r.json()["status"] == "complete":
                    log(f"Round {rnd} complete on server")
                    break
            except Exception:
                pass
        else:
            log(f"Timeout waiting for round {rnd} result", "WARN")

    # ── Final summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"CLIENT {CLIENT_ID} — RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Rnd':<4} {'Train':>7} {'DP':>7} {'ZKP':>7} {'HE_enc':>9} {'Total':>9} {'PeakRAM':>9}")
    print("-" * 55)
    for rnd, t in all_timing.items():
        m = all_memory.get(rnd, {})
        print(f"{rnd:<4} {t.get('train', 0):>7.1f} {t.get('dp', 0):>7.3f} "
              f"{t.get('zkp', 0):>7.3f} {t.get('he_encrypt', 0):>9.2f} "
              f"{t.get('total', 0):>9.1f} {m.get('peak', 0):>7.0f}MB")

    results = {
        "client_id": CLIENT_ID,
        "timing":    {str(k): v for k, v in all_timing.items()},
        "memory":    {str(k): dict(v) for k, v in all_memory.items()},
        "config": {
            "model_type":    MODEL_TYPE,
            "use_dp":        USE_LOCAL_DP,
            "use_zkp":       USE_ZKP,
            "use_he":        USE_HE,
            "he_scope":      "classifier_head_only",
            "dp_epsilon":    DP_EPSILON,
            "he_poly_degree": he.POLY_MODULUS_DEGREE,
            "mem_limit_mb":  200,
            "local_epochs":  LOCAL_EPOCHS,
        }
    }
    path = f"{RESULTS_DIR}/client_{CLIENT_ID}_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Results saved: {path}")

    try:
        os.remove(data_path)
    except OSError:
        pass
    log("Client complete.")


if __name__ == "__main__":
    main()
