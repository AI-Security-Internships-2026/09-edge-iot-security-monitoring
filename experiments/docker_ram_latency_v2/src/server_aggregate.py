"""
Server-side aggregation timing. Runs AFTER all clients finish (a separate
short-lived container/service in docker-compose). Reads each client's
per-round artifact JSONs from the shared results volume and times:

  - HE_FULL / HE_PARTIAL / HE_PARTIAL_ZKP: homomorphic aggregate + decrypt
    (he_aggregation.aggregate_encrypted_param_lists / he_local equivalents)
  - HE_PARTIAL_ZKP only: zkp.verify_head_norm_proof (per client) +
    zkp.mad_threshold_head_norms (cross-client MAD threshold)
  - ADAPTIVE KRUM (from baseline/dp artifacts' plaintext params):
    krum.adaptive_multi_krum() timing

IMPORTANT SCOPE NOTE ON KRUM/ZKP-THRESHOLD TIMING
---------------------------------------------------
This test suite runs 2 real Docker clients (matching the old He-Full/
He-Partial/pure_dp ablations' own 2-client setup). Adaptive Krum's
distance computation is O(n^2 * d) in the number of clients n, and the
project's main FL pipeline runs with NUM_CLIENTS=10 -- a 2-client Krum
timing would not be representative of real per-round aggregation cost.

Per the explicit scope agreed for this rerun (RAM/timing only, not
accuracy), this script SYNTHESISES additional "clients" by jittering
copies of the 2 real, correctly-shaped trained parameter vectors
(small i.i.d. Gaussian noise added per synthetic copy) up to
SYNTHETIC_N_CLIENTS (default 10, matching the main pipeline's
NUM_CLIENTS). This keeps the model dimensionality and the client COUNT
realistic -- the two things that actually drive Krum's/the MAD-threshold's
runtime -- without claiming any accuracy properties for the synthetic
copies. This is flagged here, in the output JSON, and should be flagged
again in any write-up: Krum/ZKP-threshold TIMING numbers from this
script describe the algorithm's cost at a realistic (n, d), NOT a
detection-rate or accuracy result of any kind.
"""

import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mem_profiler import RamSampler, StageTimer, write_json  # noqa: E402

OUT_DIR = os.environ["OUT_DIR"]
MODE = os.environ["MODE"]
NUM_REAL_CLIENTS = int(os.environ.get("NUM_REAL_CLIENTS", 2))
SYNTHETIC_N_CLIENTS = int(os.environ.get("SYNTHETIC_N_CLIENTS", 10))
NUM_BYZANTINE_ASSUMED = int(os.environ.get("NUM_BYZANTINE_ASSUMED", 2))
JITTER_STD_FRACTION = float(os.environ.get("JITTER_STD_FRACTION", 0.05))
HE_POLY_DEGREE = int(os.environ.get("HE_POLY_DEGREE", 4096))
ROUNDS = int(os.environ.get("ROUNDS", 3))


def load_artifacts(round_idx):
    arts = []
    for cid in range(NUM_REAL_CLIENTS):
        path = os.path.join(OUT_DIR, f"client_{cid}_round_{round_idx}_artifact.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing artifact: {path}")
        with open(path) as f:
            arts.append(json.load(f))
    return arts


def run_he_aggregation():
    import he_aggregation as he
    import he_local

    per_round = []
    with RamSampler(interval_s=0.1) as sampler:
        for rnd in range(1, ROUNDS + 1):
            arts = load_artifacts(rnd)
            weights = [1] * len(arts)  # equal weighting for a timing-only benchmark
            timer = StageTimer()

            if MODE == "he_full":
                he_context = he.create_server_context(poly_modulus_degree=HE_POLY_DEGREE)
                enc_list = [a["enc"] for a in arts]
                with timer.stage("aggregate"):
                    agg = he.aggregate_encrypted_param_lists(enc_list, weights, he_context)
                with timer.stage("decrypt"):
                    _ = he.decrypt_param_list(agg)

            else:  # he_partial / he_partial_zkp
                he_context = he.create_server_context(poly_modulus_degree=HE_POLY_DEGREE)
                accepted = [a["enc"] for a in arts]
                # "bulk" was converted to plain nested lists on the client
                # side before JSON serialization (see client_runner.py's
                # matching fix) -- convert back to real numpy arrays here,
                # since he_local.aggregate_encrypted()'s plaintext
                # weighted-average arithmetic (array * float, .astype())
                # requires actual ndarrays, not lists.
                for c in accepted:
                    c["bulk"] = [np.array(b, dtype=np.float32) for b in c["bulk"]]
                with timer.stage("aggregate"):
                    agg = he_local.aggregate_encrypted(accepted, weights, he_context)
                with timer.stage("decrypt"):
                    _ = he_local.decrypt_params(agg)

                if MODE == "he_partial_zkp":
                    import zkp
                    with timer.stage("zkp_verify"):
                        verified_norms = []
                        for a in arts:
                            ok, reason = zkp.verify_head_norm_proof(
                                a["proof"], a["enc"]["sensitive_enc"]["chunks"]
                            )
                            if not ok:
                                raise RuntimeError(f"ZKP verification failed: {reason}")
                            verified_norms.append(a["proof"]["norm"])

                    # Synthesize extra norms (jittered around the real ones)
                    # so the MAD threshold runs at a realistic client count
                    # -- see module docstring's scope note.
                    synth_norms = _synthesize_norms(verified_norms, SYNTHETIC_N_CLIENTS)
                    with timer.stage("zkp_mad_threshold"):
                        kept, dropped, diag = zkp.mad_threshold_head_norms(synth_norms, k=2.5)

            record = {"round": rnd, **{k: round(v, 5) for k, v in timer.durations.items()}}
            per_round.append(record)
            print(f"[server] HE round {rnd}/{ROUNDS}: {record}")

    ram_summary = sampler.summary()
    return {"per_round": per_round, "ram_peak_mb": ram_summary["peak_mb"],
            "ram_avg_mb": ram_summary["avg_mb"]}


def _synthesize_norms(real_norms, n_total):
    """Jitter the real norms up to n_total entries, documented assumption
    -- see module docstring."""
    rng = np.random.default_rng(42)
    out = list(real_norms)
    i = 0
    while len(out) < n_total:
        base = real_norms[i % len(real_norms)]
        jitter = rng.normal(0, max(abs(base) * JITTER_STD_FRACTION, 1e-6))
        out.append(base + jitter)
        i += 1
    return out[:n_total]


def _synthesize_param_lists(real_param_lists, n_total):
    """Jitter copies of real plaintext parameter lists up to n_total
    'clients' -- keeps model dimensionality (d) and target client count
    (n) realistic for Krum's O(n^2 * d) distance computation, without
    claiming anything about accuracy/detection for the synthetic copies.
    See module docstring's scope note."""
    rng = np.random.default_rng(42)
    out = [[np.array(layer, dtype=np.float32) for layer in pl] for pl in real_param_lists]
    i = 0
    while len(out) < n_total:
        base = real_param_lists[i % len(real_param_lists)]
        jittered = []
        for layer in base:
            arr = np.array(layer, dtype=np.float32)
            std = max(float(np.std(arr)) * JITTER_STD_FRACTION, 1e-8)
            jittered.append(arr + rng.normal(0, std, arr.shape).astype(np.float32))
        out.append(jittered)
        i += 1
    return out[:n_total]


def run_krum_timing():
    """Times adaptive_multi_krum() at a realistic (n=SYNTHETIC_N_CLIENTS,
    d=model_size) using plaintext params saved by baseline/dp-mode
    clients -- see module docstring's scope note on why synthesis is
    used here."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from krum import adaptive_multi_krum

    per_round = []
    with RamSampler(interval_s=0.1) as sampler:
        for rnd in range(1, ROUNDS + 1):
            arts = load_artifacts(rnd)
            real_param_lists = [a["params"] for a in arts]
            synth_params = _synthesize_param_lists(real_param_lists, SYNTHETIC_N_CLIENTS)
            weights = [1] * SYNTHETIC_N_CLIENTS

            timer = StageTimer()
            with timer.stage("adaptive_krum"):
                _, kept, diag = adaptive_multi_krum(
                    synth_params, weights,
                    num_byzantine=NUM_BYZANTINE_ASSUMED, k=2.5, method="mad",
                    return_diagnostics=True,
                )
            record = {
                "round": rnd,
                "adaptive_krum_time_s": round(timer.durations.get("adaptive_krum", 0.0), 5),
                "n_clients_synthetic": SYNTHETIC_N_CLIENTS,
                "n_clients_real": len(arts),
                "num_kept": len(kept),
                "num_dropped": diag["num_dropped"],
            }
            per_round.append(record)
            print(f"[server] Krum round {rnd}/{ROUNDS}: {record}")

    ram_summary = sampler.summary()
    return {"per_round": per_round, "ram_peak_mb": ram_summary["peak_mb"],
            "ram_avg_mb": ram_summary["avg_mb"],
            "note": ("Timing at n=SYNTHETIC_N_CLIENTS via jittered synthetic "
                      "copies of real per-client trained params -- see "
                      "module docstring. NOT an accuracy/detection result.")}


def main():
    result = {"mode": MODE, "config": {
        "num_real_clients": NUM_REAL_CLIENTS,
        "synthetic_n_clients": SYNTHETIC_N_CLIENTS,
        "rounds": ROUNDS,
        "he_poly_degree": HE_POLY_DEGREE if MODE.startswith("he_") else None,
    }}

    if MODE.startswith("he_"):
        result["he_aggregation"] = run_he_aggregation()
    elif MODE in ("baseline", "dp"):
        result["krum_timing"] = run_krum_timing()
    else:
        raise ValueError(f"Unknown MODE={MODE!r} for server_aggregate.py")

    write_json(os.path.join(OUT_DIR, f"server_{MODE}_results.json"), result)
    print(f"[server] DONE, mode={MODE}")


if __name__ == "__main__":
    main()
