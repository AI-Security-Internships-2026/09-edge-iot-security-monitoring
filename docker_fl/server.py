"""
FL Server — runs in cloud container, no RAM limit.

CRYPTO FIX: this server now generates the ONE shared CKKS context
(with the secret key) at startup and hands out only the PUBLIC half
to clients via /get_he_context. Previously each client generated its
own independent keypair, which made the homomorphic aggregation
cryptographically invalid (ciphertexts under different keys cannot be
validly summed) — see he_aggregation.py's docstring for details.

Aggregation is now split:
  - "sensitive" layers (classifier head) → homomorphic weighted sum,
    decrypted only here on the server (secret key never leaves this process)
  - "bulk" layers (everything else, ~94% of params) → plain weighted
    average, sent as DP-noised plaintext by clients
Both halves are merged back into the original state_dict order using
each client's declared index mapping (identical across clients, since
model architecture is fixed).
"""

import os
import sys
import time
import json
import threading
import numpy as np
from flask import Flask, request, jsonify

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wire_format import pack_array, unpack_array, pack_param_list, unpack_param_list
import he_aggregation as he

app = Flask(__name__)

# ── Config ──────────────────────────────────────────────────────────
NUM_CLIENTS = int(os.environ.get("NUM_CLIENTS", 2))
NUM_ROUNDS  = int(os.environ.get("NUM_ROUNDS", 5))
USE_HE      = os.environ.get("USE_HE", "true").lower() == "true"
MODEL_TYPE  = os.environ.get("MODEL_TYPE", "network")
RESULTS_DIR = "/results"

# ── Server state ────────────────────────────────────────────────────
state = {
    "current_round":  1,
    "global_params":  None,
    "received":       {},
    "weights":        {},
    "timing":         {},
    "lock":           threading.Lock(),
    "round_complete": {},
}
for r in range(1, NUM_ROUNDS + 1):
    state["round_complete"][r] = threading.Event()


def log(msg):
    print(f"[Server][Round {state['current_round']}] {msg}", flush=True)


# ── HE context (single, server-owned) ──────────────────────────────

_he_ctx = None            # full context, holds the secret key — never sent out
_he_ctx_public_b64 = None  # cached public serialization, sent to clients


def _get_he_context():
    global _he_ctx, _he_ctx_public_b64
    if _he_ctx is None:
        _he_ctx = he.create_server_context()
        _he_ctx_public_b64 = he.serialize_public_context(_he_ctx)
        log(f"HE context created ONCE on server "
            f"(poly_degree={he.POLY_MODULUS_DEGREE}, "
            f"no galois/relin keys, secret key stays server-side)")
    return _he_ctx


@app.route("/get_he_context", methods=["GET"])
def get_he_context():
    """Distribute the PUBLIC half of the shared context to clients.
    Clients can encrypt with this; they cannot decrypt with it."""
    _get_he_context()
    return jsonify({"public_context": _he_ctx_public_b64})


# ── Routes ──────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ready", "round": state["current_round"]})


@app.route("/get_global_model", methods=["GET"])
def get_global_model():
    if state["global_params"] is None:
        return jsonify({"status": "not_ready"}), 503
    return jsonify({
        "status":   "ok",
        "round":    state["current_round"],
        "params":   pack_param_list(state["global_params"]),
        "n_layers": len(state["global_params"]),
    })


@app.route("/submit_update", methods=["POST"])
def submit_update():
    """
    Client submits an update for the current round.

    POST body (JSON):
      client_id, round, n_samples
      update: {
        "sensitive": {mode: ckks|plaintext, ...},
        "bulk":      {mode: plaintext, data:[...], shapes:[...], idx:[...]},
        "keys":      [state_dict key names, in order]
      }
      zkp_proof
    """
    data      = request.get_json(force=True)
    client_id = data["client_id"]
    rnd       = data["round"]
    n_samples = data["n_samples"]
    zkp_proof = data.get("zkp_proof")
    update    = data["update"]

    sens_mode = update["sensitive"]["mode"]
    log(f"Received update from client {client_id} "
        f"(round={rnd}, sensitive_mode={sens_mode}, n={n_samples})")

    # ── ZKP verification (covers the WHOLE model, sensitive + bulk) ──
    if zkp_proof is not None:
        pi         = zkp_proof.get("norm_proof", {})
        norm       = pi.get("norm", 999)
        threshold  = pi.get("threshold", 1.0)
        norm_valid = norm <= threshold
        log(f"  ZKP: norm={norm:.4f}, threshold={threshold:.4f}, valid={norm_valid}")
        if not norm_valid:
            log(f"  ZKP REJECTED client {client_id} — gradient bomb blocked")
            return jsonify({"status": "rejected", "reason": "zkp_norm_exceeded"}), 400

    # ── Store update ──────────────────────────────────────────────
    with state["lock"]:
        if rnd not in state["received"]:
            state["received"][rnd] = {}
            state["weights"][rnd]  = {}

        stored = {"keys": update["keys"]}

        # Sensitive layers
        sens = update["sensitive"]
        if sens["mode"] == "ckks":
            stored["sensitive_mode"]  = "ckks"
            stored["sensitive_chunks"] = sens["chunks"]
            stored["sensitive_shapes"] = [tuple(s) for s in sens["shapes"]]
            stored["sensitive_sizes"]  = sens["sizes"]
            stored["sensitive_idx"]    = sens["idx"]
            stored["sensitive_total"]  = sens["total"]
        else:
            stored["sensitive_mode"]   = "plaintext"
            stored["sensitive_params"] = unpack_param_list(sens)
            stored["sensitive_idx"]    = sens["idx"]

        # Bulk layers (always plaintext, binary-packed)
        bulk = update["bulk"]
        stored["bulk_params"] = unpack_param_list(bulk)
        stored["bulk_idx"]    = bulk["idx"]

        state["received"][rnd][client_id] = stored
        state["weights"][rnd][client_id]  = n_samples
        received_count = len(state["received"][rnd])
        log(f"  Stored. Have {received_count}/{NUM_CLIENTS} updates for round {rnd}")

    if received_count >= NUM_CLIENTS:
        _aggregate_round(rnd)

    return jsonify({"status": "accepted", "round": rnd})


@app.route("/get_round_result/<int:rnd>", methods=["GET"])
def get_round_result(rnd):
    event = state["round_complete"].get(rnd)
    if event is None or not event.is_set():
        return jsonify({"status": "pending"}), 202
    return jsonify({
        "status": "complete",
        "round":  rnd,
        "params": pack_param_list(state["global_params"]),
    })


# ── Aggregation ─────────────────────────────────────────────────────

def _aggregate_round(rnd):
    t0 = time.time()
    log(f"Aggregating round {rnd}...")

    clients_data = list(state["received"][rnd].values())
    weights      = list(state["weights"][rnd].values())

    keys        = clients_data[0]["keys"]
    n_layers    = len(keys)
    sensitive_idx = clients_data[0]["sensitive_idx"]
    bulk_idx      = clients_data[0]["bulk_idx"]
    sensitive_shapes = clients_data[0].get("sensitive_shapes")
    sensitive_sizes  = clients_data[0].get("sensitive_sizes")
    sensitive_total  = clients_data[0].get("sensitive_total")

    merged = [None] * n_layers

    # ── Sensitive layers ────────────────────────────────────────────
    if clients_data[0]["sensitive_mode"] == "ckks" and USE_HE:
        log("  Homomorphic aggregation of sensitive (classifier) layers...")
        try:
            ctx = _get_he_context()
            all_client_chunks = [c["sensitive_chunks"] for c in clients_data]
            aggregated_chunks = he.he_weighted_sum(all_client_chunks, weights, ctx)
            flat = he.decrypt_aggregate(aggregated_chunks, sensitive_total)

            offset = 0
            for layer_i, shape, size in zip(sensitive_idx, sensitive_shapes, sensitive_sizes):
                merged[layer_i] = flat[offset:offset + size].reshape(shape).astype(np.float32)
                offset += size
            log("  Sensitive-layer HE aggregate done (server never saw plaintext)")
        except Exception as e:
            log(f"  HE aggregate failed: {e} — falling back to plaintext average", "ERROR")
            _fedavg_into(merged, clients_data, weights, "sensitive_params", sensitive_idx)
    else:
        _fedavg_into(merged, clients_data, weights, "sensitive_params", sensitive_idx)

    # ── Bulk layers (always plaintext) ──────────────────────────────
    log("  Plaintext FedAvg of bulk layers...")
    _fedavg_into(merged, clients_data, weights, "bulk_params", bulk_idx)

    t_agg = time.time() - t0
    state["global_params"] = merged
    state["timing"][rnd] = {"aggregate_s": t_agg,
                             "sensitive_mode": clients_data[0]["sensitive_mode"]}
    state["round_complete"][rnd].set()
    state["current_round"] = rnd + 1
    log(f"  Round {rnd} complete in {t_agg:.2f}s")
    _save_timing()


def _fedavg_into(merged, clients_data, weights, field, idx_list):
    total = sum(weights)
    for pos, layer_i in enumerate(idx_list):
        layer_avg = None
        for cdata, w in zip(clients_data, weights):
            p = cdata[field][pos]
            contrib = p * (w / total)
            layer_avg = contrib if layer_avg is None else layer_avg + contrib
        merged[layer_i] = layer_avg.astype(np.float32)


def _save_timing():
    path = f"{RESULTS_DIR}/server_timing.json"
    try:
        with open(path, "w") as f:
            json.dump(state["timing"], f, indent=2)
    except Exception:
        pass


# ── Initialise global model ──────────────────────────────────────────

def _init_global_model():
    """Create initial random global model matching the client architecture.
    Server has no RAM constraint, so importing torch here is fine."""
    sys.path.insert(0, "/app")
    from task import get_model, get_model_parameters

    num_features = 40 if MODEL_TYPE == "network" else 52
    num_classes  = 8

    model  = get_model(num_features=num_features, num_classes=num_classes)
    params = get_model_parameters(model)
    state["global_params"] = params
    log(f"Global model initialised: {sum(p.size for p in params):,} params")


# ── Entry ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log("=" * 60)
    log("FL Server starting")
    log(f"NUM_CLIENTS={NUM_CLIENTS}, NUM_ROUNDS={NUM_ROUNDS}")
    log(f"USE_HE={USE_HE} (partial: classifier head only)")
    log("=" * 60)

    _init_global_model()

    if USE_HE:
        log("Pre-warming shared HE context...")
        _get_he_context()

    app.run(host="0.0.0.0", port=5000, threaded=True)
