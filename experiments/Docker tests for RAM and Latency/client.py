"""
FL Client — runs in constrained IoT gateway container (200-300MB RAM target).

This process NEVER imports torch, and — as of this version — NEVER
imports pandas/sklearn/data_loader either. It loads a pre-built, real
Edge-IIoTset partition file for its own CLIENT_ID (built offline by
build_partitions.py; see that script's docstring for why). A real IoT
gateway only ever has its own local traffic logs, never the full
federation's data or the ability to compute a Dirichlet partition at
runtime — this now matches that.

Training happens in an isolated subprocess (train_worker.py) so
PyTorch's memory is fully returned to the OS the moment training
finishes, instead of stacking on top of the ZKP/HE work that follows
in the same process (see train_worker.py's docstring).

DP-SGD (CHANGED): differential privacy is no longer a separate
post-training stage in this file. It now happens INSIDE
train_worker.py's training loop via Opacus (per-sample gradient
clipping + noise, standard DP-SGD), because the old approach — noising
the entire flattened ~80k-param model vector once after training
finished — had catastrophic signal-to-noise ratio at this
dimensionality and caused loss to diverge round over round instead of
converging. This file now just passes DP config flags into the
train_worker.py subprocess call and reads back the ACHIEVED epsilon
Opacus reports, instead of calling defences/local_dp.py itself.

Privacy/robustness stack per round, in order:
  1. Train + DP-SGD  (subprocess, torch+opacus — memory isolated;
                       DP noise is now injected per-gradient-step
                       during training, not after)
  2. ZKP proof        (numpy only — commitment + signed norm bound,
                       whole model)
  3. Split            sensitive vs. bulk. Normally: classifier head (~6% of
                       params) vs. everything else (~94%). Set
                       HE_FULL_COVERAGE=true to route 100% of params through
                       the "sensitive" (HE-eligible) path instead — used for
                       a true full-coverage "pure HE" ablation run.
  4. Encrypt          sensitive layers only, under the server's shared
                       public CKKS context
  5. Send             sensitive as ciphertext, bulk as raw float32 bytes
                       (base64) instead of JSON float lists

RAM/LATENCY INSTRUMENTATION: each privacy stage (ZKP, HE) is wrapped
in a RamSampler that polls RSS on a background thread throughout the
stage. The training subprocess (which now includes DP-SGD) reports its
own peak/avg via a JSON file since it's a separate process.
"""

import os
import sys
import time
import json
import threading
import subprocess
import numpy as np
import requests
import psutil

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wire_format import pack_array, unpack_array, pack_param_list, unpack_param_list
import he_aggregation as he

# ── Config ──────────────────────────────────────────────────────────
CLIENT_ID        = int(os.environ.get("CLIENT_ID", 0))
NUM_CLIENTS      = int(os.environ.get("NUM_CLIENTS", 2))
SERVER_URL       = os.environ.get("SERVER_URL", "http://fl_server:5000")
USE_LOCAL_DP     = os.environ.get("USE_LOCAL_DP", "true").lower() == "true"
USE_ZKP          = os.environ.get("USE_ZKP",      "true").lower() == "true"
USE_HE           = os.environ.get("USE_HE",       "true").lower() == "true"
HE_FULL_COVERAGE = os.environ.get("HE_FULL_COVERAGE", "false").lower() == "true"
DP_EPSILON       = float(os.environ.get("DP_EPSILON", "5.0"))
DP_DELTA         = float(os.environ.get("DP_DELTA", "1e-5"))
DP_MAX_GRAD_NORM = float(os.environ.get("DP_MAX_GRAD_NORM", "1.0"))
MODEL_TYPE       = os.environ.get("MODEL_TYPE", "network")
NUM_ROUNDS       = int(os.environ.get("NUM_ROUNDS", "5"))
LOCAL_EPOCHS     = int(os.environ.get("LOCAL_EPOCHS", "5"))
RAM_SAMPLE_INTERVAL_S = float(os.environ.get("RAM_SAMPLE_INTERVAL_S", "0.05"))
PARTITION_DIR    = os.environ.get("PARTITION_DIR", "/datasets/partitions")
RESULTS_DIR      = "/results"
TMP_DIR          = "/tmp"
APP_PATH         = "/app"

NUM_CLASSES  = 8
SENSITIVE_PREFIX = "classifier"


# ── RAM sampler ───────────────────────────────────────────────────────

class RamSampler:
    def __init__(self, interval_s=RAM_SAMPLE_INTERVAL_S):
        self.interval_s = interval_s
        self._samples = []
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        proc = psutil.Process(os.getpid())
        while not self._stop.is_set():
            try:
                self._samples.append(proc.memory_info().rss / 1024 / 1024)
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._stop.clear()
        self._samples = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=1)
        return False

    @property
    def peak(self):
        return max(self._samples) if self._samples else 0.0

    @property
    def avg(self):
        return sum(self._samples) / len(self._samples) if self._samples else 0.0

    @property
    def samples_count(self):
        return len(self._samples)


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


# ── Data loading — REAL DATA ONLY, no synthetic fallback ─────────────

def load_data():
    """
    Load this client's pre-built real-data partition from
    {PARTITION_DIR}/client_{CLIENT_ID}_{MODEL_TYPE}.npz.

    This file is built ONCE, offline, by build_partitions.py. This
    function does NOT do any preprocessing, partitioning, or
    pandas/sklearn work itself — a real IoT gateway only ever has its
    own local data, never the full corpus, so client.py deliberately
    has zero ability to compute a partition at runtime.

    If the file isn't there, this is a HARD ERROR, not a silent
    synthetic-data fallback. Silently substituting synthetic data would
    make RAM/latency numbers meaningless with no indication in the
    logs or results JSON that it happened — exactly the trap this
    project already fell into once.
    """
    partition_path = os.path.join(PARTITION_DIR, f"client_{CLIENT_ID}_{MODEL_TYPE}.npz")

    if not os.path.exists(partition_path):
        raise RuntimeError(
            f"Partition file not found: {partition_path}\n"
            f"Run build_partitions.py first (see its docstring) to generate "
            f"per-client partition files from the real Edge-IIoTset dataset, "
            f"and mount the output directory to {PARTITION_DIR} in "
            f"docker-compose.yml. Refusing to silently fall back to synthetic "
            f"data."
        )

    log(f"Loading real-data partition: {partition_path}")
    data = np.load(partition_path)
    X_tr, y_tr, X_te, y_te = data["X_train"], data["y_train"], data["X_test"], data["y_test"]
    log(f"Partition loaded: train={len(X_tr):,} rows, test={len(X_te):,} rows, "
        f"features={X_tr.shape[1]} "
        f"(size on disk={os.path.getsize(partition_path) / 1024 / 1024:.1f}MB)")
    return X_tr, y_tr, X_te, y_te


# ── Sensitive/bulk split ─────────────────────────────────────────────

def split_sensitive_bulk(keys, params):
    if HE_FULL_COVERAGE:
        sensitive_idx = list(range(len(keys)))
        bulk_idx = []
    else:
        sensitive_idx = [i for i, k in enumerate(keys) if k.startswith(SENSITIVE_PREFIX)]
        bulk_idx      = [i for i, k in enumerate(keys) if not k.startswith(SENSITIVE_PREFIX)]
    sensitive = [params[i] for i in sensitive_idx]
    bulk      = [params[i] for i in bulk_idx]
    return sensitive, bulk, sensitive_idx, bulk_idx


# ── One FL round ────────────────────────────────────────────────────

def run_round(rnd, global_params, num_features, data_path, he_context):
    stage_times    = {}
    stage_mem      = {"before_round": get_ram()}
    stage_ram_peak = {}
    stage_ram_avg  = {}

    log(f"─── ROUND {rnd} START ───")

    t0 = time.time()
    global_path = f"{TMP_DIR}/global_r{rnd}_c{CLIENT_ID}.npz"
    out_path    = f"{TMP_DIR}/raw_params_r{rnd}_c{CLIENT_ID}.npz"
    keys_path   = f"{TMP_DIR}/keys_r{rnd}_c{CLIENT_ID}.json"
    mem_path    = f"{TMP_DIR}/mem_r{rnd}_c{CLIENT_ID}.json"

    np.savez(global_path, *global_params)

    # ── Train (+ DP-SGD inside the subprocess, if enabled) ───────────
    log("Training (isolated subprocess)"
        + (" with DP-SGD..." if USE_LOCAL_DP else "..."))
    dp_flags = (
        [
            "--use-dp",
            "--dp-epsilon", str(DP_EPSILON),
            "--dp-delta", str(DP_DELTA),
            "--dp-max-grad-norm", str(DP_MAX_GRAD_NORM),
        ]
        if USE_LOCAL_DP else []
    )
    result = subprocess.run(
        [
            sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_worker.py"),
            "--client-id", str(CLIENT_ID),
            "--model-type", MODEL_TYPE,
            "--num-features", str(num_features),
            "--num-classes", str(NUM_CLASSES),
            "--local-epochs", str(LOCAL_EPOCHS),
            "--global-params-path", global_path,
            "--train-data-path", data_path,
            "--output-path", out_path,
            "--keys-output-path", keys_path,
            "--memory-output-path", mem_path,
            "--app-path", APP_PATH,
        ] + dp_flags,
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
                f"(returncode={result.returncode}) — likely OOM SIGKILL.", "ERROR")
        else:
            log(f"train_worker.py FAILED (returncode={result.returncode}):\n{result.stderr}", "ERROR")
        raise RuntimeError("Training subprocess failed")
    log(result.stdout.strip().replace("\n", "\n  "))

    raw_npz = np.load(out_path)
    raw_params = [raw_npz[k] for k in raw_npz.files]
    with open(keys_path) as f:
        keys = json.load(f)

    worker_mem = {}
    if os.path.exists(mem_path):
        with open(mem_path) as f:
            worker_mem = json.load(f)

    for p in (global_path, out_path, keys_path, mem_path):
        try:
            os.remove(p)
        except OSError:
            pass

    stage_times["train"] = time.time() - t0
    stage_mem["after_train"] = get_ram()
    stage_ram_peak["train_subprocess"] = worker_mem.get("peak_mb", 0.0)
    stage_ram_avg["train_subprocess"]  = worker_mem.get("avg_mb", 0.0)

    # ── DP-SGD results (NEW) — read back from train_worker.py instead
    # of computing DP here. achieved_epsilon is what Opacus's privacy
    # accountant actually measured given the real number of steps/
    # batches this client trained on, which can differ slightly from
    # the target_epsilon passed in.
    dp_info = {}
    if USE_LOCAL_DP:
        achieved_epsilon = worker_mem.get("achieved_epsilon")
        noise_multiplier = worker_mem.get("noise_multiplier")
        dp_info = {
            "target_epsilon":   DP_EPSILON,
            "achieved_epsilon": achieved_epsilon,
            "delta":            DP_DELTA,
            "max_grad_norm":    DP_MAX_GRAD_NORM,
            "noise_multiplier": noise_multiplier,
        }
        eps_str = f"{achieved_epsilon:.4f}" if achieved_epsilon is not None else "N/A"
        nm_str  = f"{noise_multiplier:.4f}" if noise_multiplier is not None else "N/A"
        log(f"DP-SGD: target_eps={DP_EPSILON}  achieved_eps={eps_str}  "
            f"noise_multiplier={nm_str}")

    log(f"Training done: {stage_times['train']:.1f}s  "
        f"subprocess peak={stage_ram_peak['train_subprocess']:.0f}MB "
        f"avg={stage_ram_avg['train_subprocess']:.0f}MB")

    # ── ZKP proof ─────────────────────────────────────────────────────
    # NOTE: generate_proof() previously took dp_info["noise_sigma"] —
    # the single sigma value from the old one-shot Gaussian mechanism.
    # That concept doesn't map 1:1 onto DP-SGD's per-step noise, so
    # this passes noise_multiplier * max_grad_norm as an approximate
    # stand-in (roughly: the per-step noise scale actually applied
    # during training). VERIFY this against what generate_proof()
    # actually does with the value before trusting proof output —
    # flagging this explicitly rather than guessing silently.
    zkp_proof = None
    if USE_ZKP:
      t0 = time.time()
      log("Generating ZKP proof...")
      from defences.zkp import generate_proof
    # Prove the DELTA (trained - global-at-round-start), not raw
    # trained weights — under DP-SGD the delta is the quantity that
    # was actually clipped+noised per-step; raw trained weights have
    # no natural relationship to clip_norm.
      delta_params = [t - g for t, g in zip(raw_params, global_params)]
      with RamSampler() as sampler:
        zkp_proof = generate_proof(
            delta_params, clip_norm=DP_MAX_GRAD_NORM,
            noise_sigma=dp_info.get("noise_multiplier", 0.0) * DP_MAX_GRAD_NORM
                        if USE_LOCAL_DP else 0.0
        )
        if isinstance(zkp_proof.get("salt"), (bytes, bytearray)):
            zkp_proof["salt"] = zkp_proof["salt"].hex()
        stage_times["zkp"] = time.time() - t0
        stage_mem["after_zkp"] = get_ram()
        stage_ram_peak["zkp"] = sampler.peak
        stage_ram_avg["zkp"]  = sampler.avg
        pi = zkp_proof["norm_proof"]
        log(f"ZKP done: {stage_times['zkp']:.3f}s  norm={pi['norm']:.4f}  "
            f"threshold={pi['threshold']:.4f}  passed={pi['passes']}  "
            f"peak={sampler.peak:.0f}MB avg={sampler.avg:.0f}MB")

    sensitive, bulk, sensitive_idx, bulk_idx = split_sensitive_bulk(keys, raw_params)
    n_sensitive = sum(p.size for p in sensitive)
    n_bulk      = sum(p.size for p in bulk)
    split_label = "Full-coverage HE" if HE_FULL_COVERAGE else "Partial HE"
    log(f"{split_label} split: sensitive={n_sensitive:,} params "
        f"({100 * n_sensitive / max(n_sensitive + n_bulk, 1):.1f}%), "
        f"bulk={n_bulk:,} params (plaintext"
        f"{', DP-SGD-trained' if USE_LOCAL_DP else ''})")

    update_payload = {}
    he_info = {}

    if USE_HE and he_context is not None:
        t0 = time.time()
        log(f"CKKS encryption of {'entire model' if HE_FULL_COVERAGE else 'sensitive layers only'} "
            f"(poly_degree={he.POLY_MODULUS_DEGREE})...")
        sens_flat = np.concatenate([p.flatten() for p in sensitive]).astype(np.float64)
        chunk_size = he.POLY_MODULUS_DEGREE // 2
        try:
            with RamSampler() as sampler:
                chunks_b64 = he.encrypt_flat_array(sens_flat, he_context, chunk_size)
            update_payload["sensitive"] = {
                "mode": "ckks", "chunks": chunks_b64,
                "shapes": [list(p.shape) for p in sensitive],
                "sizes": [int(p.size) for p in sensitive],
                "idx": sensitive_idx, "total": int(sens_flat.size),
                "chunk_size": chunk_size, "n_chunks": len(chunks_b64),
            }
            he_info = {"n_chunks": len(chunks_b64), "oom": False}
            stage_times["he_encrypt"] = time.time() - t0
            stage_mem["after_he"] = get_ram()
            stage_ram_peak["he_encrypt"] = sampler.peak
            stage_ram_avg["he_encrypt"]  = sampler.avg
            log(f"HE encrypt done: {stage_times['he_encrypt']:.2f}s  "
                f"chunks={len(chunks_b64)}  peak={sampler.peak:.0f}MB avg={sampler.avg:.0f}MB")
        except MemoryError:
            log("OOM during CKKS — falling back to plaintext for sensitive layers too", "ERROR")
            he_info = {"n_chunks": 0, "oom": True}
            update_payload["sensitive"] = {
                "mode": "plaintext", **pack_param_list(sensitive), "idx": sensitive_idx,
            }
        del sens_flat
    else:
        update_payload["sensitive"] = {
            "mode": "plaintext", **pack_param_list(sensitive), "idx": sensitive_idx,
        }

    update_payload["bulk"] = {"mode": "plaintext", **pack_param_list(bulk), "idx": bulk_idx}
    update_payload["keys"] = keys

    stage_times["total"] = sum(v for k, v in stage_times.items())

    all_peaks = [v for v in stage_ram_peak.values() if v]
    all_avgs  = [v for v in stage_ram_avg.values() if v]
    snapshot_peak = max(
        stage_mem.get("after_train", 0), stage_mem.get("after_zkp", 0),
        stage_mem.get("after_he", 0)
    )
    stage_mem["peak"] = snapshot_peak
    stage_mem["round_peak_mb"] = max(all_peaks) if all_peaks else snapshot_peak
    stage_mem["round_avg_mb"]  = sum(all_avgs) / len(all_avgs) if all_avgs else 0.0
    stage_mem["stage_peaks"] = stage_ram_peak
    stage_mem["stage_avgs"]  = stage_ram_avg

    return update_payload, zkp_proof, stage_times, stage_mem, he_info, dp_info


# ── Main ────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log(f"IoT Client {CLIENT_ID} starting")
    log(f"Stack: DP-SGD={USE_LOCAL_DP} (eps={DP_EPSILON if USE_LOCAL_DP else 'N/A'})  "
        f"ZKP={USE_ZKP}  HE={USE_HE} "
        f"(coverage={'full_model' if HE_FULL_COVERAGE else 'classifier_head_only'})")
    log(f"Model: {MODEL_TYPE}  Server: {SERVER_URL}")
    log("=" * 60)

    wait_for_server(timeout=60)
    he_context = fetch_he_context()

    X_train, y_train, X_test, y_test = load_data()
    n_samples = len(X_train)

    # NUM_FEATURES is now derived from the ACTUAL loaded data, not
    # hardcoded — VarianceThreshold in data_loader.py can drop columns
    # for the network model, so a hardcoded assumption risks a shape
    # mismatch the moment real data (rather than hand-matched synthetic
    # data) is used. All clients share the same feature count since
    # VarianceThreshold is fit once on the full subset before per-client
    # partitioning, so this is safe and consistent across clients.
    num_features = X_train.shape[1]
    log(f"Derived num_features={num_features} from actual partition data "
        f"(not hardcoded)")

    del X_test, y_test

    data_path = f"{TMP_DIR}/traindata_c{CLIENT_ID}.npz"
    np.savez(data_path, X_train=X_train, y_train=y_train)
    log(f"Training data written once: {n_samples:,} samples "
        f"({os.path.getsize(data_path) / 1024 / 1024:.1f}MB on disk)")
    del X_train, y_train

    all_timing = {}
    all_memory = {}
    all_dp     = {}

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

        update, zkp_proof, timing, memory, he_info, dp_info = run_round(
            rnd, global_params, num_features, data_path, he_context
        )
        all_timing[rnd] = timing
        all_memory[rnd] = memory
        all_dp[rnd]     = dp_info

        log(f"Submitting update to server (sensitive mode={update['sensitive']['mode']})...")
        payload = {
            "client_id": CLIENT_ID, "round": rnd, "n_samples": n_samples,
            "update": update, "zkp_proof": zkp_proof,
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

    print("\n" + "=" * 60)
    print(f"CLIENT {CLIENT_ID} — RESULTS SUMMARY")
    print("=" * 60)
    if USE_LOCAL_DP:
        print(f"{'Rnd':<4} {'Train':>7} {'ZKP':>7} {'HE_enc':>9} {'Total':>9} "
              f"{'PeakRAM':>9} {'AvgRAM':>9} {'AchievedEps':>12}")
        print("-" * 78)
        for rnd, t in all_timing.items():
            m = all_memory.get(rnd, {})
            d = all_dp.get(rnd, {})
            peak = m.get('round_peak_mb', m.get('peak', 0))
            avg  = m.get('round_avg_mb', 0)
            eps  = d.get('achieved_epsilon')
            eps_str = f"{eps:.4f}" if eps is not None else "N/A"
            print(f"{rnd:<4} {t.get('train', 0):>7.1f} "
                  f"{t.get('zkp', 0):>7.3f} {t.get('he_encrypt', 0):>9.2f} "
                  f"{t.get('total', 0):>9.1f} {peak:>7.0f}MB {avg:>7.0f}MB {eps_str:>12}")
    else:
        print(f"{'Rnd':<4} {'Train':>7} {'ZKP':>7} {'HE_enc':>9} {'Total':>9} {'PeakRAM':>9} {'AvgRAM':>9}")
        print("-" * 65)
        for rnd, t in all_timing.items():
            m = all_memory.get(rnd, {})
            peak = m.get('round_peak_mb', m.get('peak', 0))
            avg  = m.get('round_avg_mb', 0)
            print(f"{rnd:<4} {t.get('train', 0):>7.1f} "
                  f"{t.get('zkp', 0):>7.3f} {t.get('he_encrypt', 0):>9.2f} "
                  f"{t.get('total', 0):>9.1f} {peak:>7.0f}MB {avg:>7.0f}MB")

    all_round_peaks = [m.get("round_peak_mb", m.get("peak", 0)) for m in all_memory.values()]
    all_round_avgs  = [m.get("round_avg_mb", 0) for m in all_memory.values() if m.get("round_avg_mb")]

    results = {
        "client_id": CLIENT_ID,
        "timing": {str(k): v for k, v in all_timing.items()},
        "memory": {str(k): dict(v) for k, v in all_memory.items()},
        "dp": {str(k): v for k, v in all_dp.items()},
        "summary_ram": {
            "overall_peak_mb": max(all_round_peaks) if all_round_peaks else 0,
            "overall_avg_mb":  sum(all_round_avgs) / len(all_round_avgs) if all_round_avgs else 0,
        },
        "config": {
            "model_type": MODEL_TYPE, "use_dp": USE_LOCAL_DP, "use_zkp": USE_ZKP,
            "use_he": USE_HE,
            "he_scope": "full_model" if HE_FULL_COVERAGE else "classifier_head_only",
            "dp_epsilon_target": DP_EPSILON, "dp_delta": DP_DELTA,
            "dp_max_grad_norm": DP_MAX_GRAD_NORM,
            "dp_method": "opacus_dpsgd_per_step" if USE_LOCAL_DP else None,
            "he_poly_degree": he.POLY_MODULUS_DEGREE,
            "mem_limit_mb": 200, "local_epochs": LOCAL_EPOCHS,
            "num_features": num_features, "data_source": "real_edge_iiotset_partition",
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