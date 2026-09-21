"""
Client-side RAM/timing test runner. Runs INSIDE a resource-constrained
Docker container. Deliberately does NOT import pandas/sklearn/data_loader
-- only reads a pre-built .npz partition (see offline/build_docker_partitions.py),
matching model_defs.py's documented constraint for this kind of subprocess.

MODES (set via env var MODE):
  baseline        -- local training only, no defence mechanism. Control
                      condition; also the source of plaintext params used
                      by server_aggregate.py's Krum timing.
  he_full         -- local training + FULL-model CKKS encryption
                      (he_aggregation.encrypt_param_list on every layer).
  he_partial      -- local training + PARTIAL CKKS encryption
                      (classifier head only, via he_local.split_sensitive_bulk
                      + he_local.encrypt_params).
  he_partial_zkp  -- he_partial, PLUS the ciphertext-bound head-norm proof
                      (he_local.encrypt_params_with_norm_guard) timed as its
                      own separate stage -- isolates the ZKP guard's added
                      cost on top of partial HE, matching how USE_ZKP /
                      USE_HE_KRUM_HYBRID actually use it together in main.py.
  dp              -- local training wrapped in Opacus DP-SGD (dp_safe=True
                      architecture: GroupNorm + DPLSTM), matching the old
                      pure_dp Docker test's config (target_epsilon, delta,
                      max_grad_norm=1.0, dp_batch_size).

ENV VARS
--------
  CLIENT_ID        int, required (0, 1, ...)
  MODE              one of the above, required
  MODEL_TYPE        "network" or "application", required
  PARTITION_DIR     path to the mounted partitions dir (client_<id>.npz +
                     manifest.json), required
  OUT_DIR           path to write result JSON + artifacts, required
  EPOCHS            local epochs per round, default 2
  ROUNDS            number of simulated rounds, default 3
  LR                learning rate, default 0.001
  BATCH_SIZE        default 256 (non-DP) -- ignored in dp mode
  HE_POLY_DEGREE    default 4096 (matches the historical Docker-suite
                     CKKS config, NOT he_local's 8192/128-bit "local
                     research" default -- see he_aggregation.py's own
                     module docstring for why 4096 was chosen for this
                     constrained-client path)
  DP_EPSILON        default 3.0 (matches old pure_dp Docker test)
  DP_DELTA          default 1e-5
  DP_MAX_GRAD_NORM  default 1.0 (matches old pure_dp Docker test;
                     NOTE this differs from main.py's main-pipeline
                     value of 1.5 -- intentional, replicating the old
                     Docker test's own documented config, not the main
                     FL loop's)
  DP_BATCH_SIZE     default 512
  SERVER_PORT       default 8080 -- server_daemon.py's listen port
  SEND_TIMEOUT_S    default 60 -- network send timeout; a failed/timed-
                     out send is logged and skipped, never crashes the run

EVERY round records ALL of: train_time_s (dp mode also splits out
dp_setup_time_s separately), the mode-specific mechanism time
(he_encrypt_time_s / zkp_proof_time_s where applicable), serialize_time_s,
communication_send_time_s, payload_bytes, and round_wall_time_s -- i.e.
training, encryption, DP overhead, ZKP, AND communication are all
captured on every round, for every mode. See server_aggregate.py for
the corresponding server-side aggregate/decrypt/Krum/ZKP-verify timings.
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tud

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_defs import get_model, get_model_parameters, get_model_parameter_keys  # noqa: E402
from mem_profiler import (  # noqa: E402
    RamSampler, StageTimer, write_json,
    read_cgroup_memory_limit_mb, read_cgroup_cpu_limit,
)

MODE = os.environ["MODE"]
CLIENT_ID = int(os.environ["CLIENT_ID"])
MODEL_TYPE = os.environ["MODEL_TYPE"]
PARTITION_DIR = os.environ["PARTITION_DIR"]
OUT_DIR = os.environ["OUT_DIR"]

EPOCHS = int(os.environ.get("EPOCHS", 2))
ROUNDS = int(os.environ.get("ROUNDS", 3))
LR = float(os.environ.get("LR", 0.001))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 256))
HE_POLY_DEGREE = int(os.environ.get("HE_POLY_DEGREE", 4096))
DP_EPSILON = float(os.environ.get("DP_EPSILON", 3.0))
DP_DELTA = float(os.environ.get("DP_DELTA", 1e-5))
DP_MAX_GRAD_NORM = float(os.environ.get("DP_MAX_GRAD_NORM", 1.0))
DP_BATCH_SIZE = int(os.environ.get("DP_BATCH_SIZE", 512))
SERVER_PORT = int(os.environ.get("SERVER_PORT", 8080))
SEND_TIMEOUT_S = float(os.environ.get("SEND_TIMEOUT_S", 60))

VALID_MODES = {"baseline", "he_full", "he_partial", "he_partial_zkp", "dp"}
if MODE not in VALID_MODES:
    raise ValueError(f"MODE={MODE!r} must be one of {VALID_MODES}")

DP_SAFE = (MODE == "dp")


def send_artifact_over_network(artifact_obj, round_idx):
    """
    Sends this round's artifact (ciphertext dict for he_* modes, plain
    param list for baseline/dp) to the server_daemon service over real
    Docker-network HTTP, so "communication" is an actual measured
    network cost, not a shared-volume file write. Returns a dict with:
      serialize_time_s -- json.dumps() cost for this payload
      send_time_s       -- client-perceived round trip (request sent ->
                            response received), i.e. the real network
                            transfer + server receive/ack cost
      payload_bytes     -- size of the serialized payload actually sent
    Failures are caught and reported rather than crashing the run --
    communication timing is a nice-to-have measurement, not something
    that should take down an otherwise-successful training/encryption
    run if the daemon isn't reachable for some reason.
    """
    import urllib.request
    import urllib.error

    t0 = time.time()
    body = json.dumps(artifact_obj, default=str).encode("utf-8")
    serialize_time_s = time.time() - t0

    url = f"http://server:{SERVER_PORT}/submit?client_id={CLIENT_ID}&round={round_idx}"
    req = urllib.request.Request(url, data=body, method="POST",
                                  headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=SEND_TIMEOUT_S) as resp:
            resp.read()
        send_time_s = time.time() - t0
        ok = True
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        send_time_s = time.time() - t0
        ok = False
        print(f"[client {CLIENT_ID}] WARNING: communication send failed "
              f"(round {round_idx}): {e} -- continuing without this "
              f"round's communication timing.")

    return {
        "serialize_time_s": round(serialize_time_s, 5),
        "send_time_s": round(send_time_s, 5),
        "payload_bytes": len(body),
        "communication_ok": ok,
    }


def load_partition():
    with open(os.path.join(PARTITION_DIR, "manifest.json")) as f:
        manifest = json.load(f)
    data = np.load(os.path.join(PARTITION_DIR, f"client_{CLIENT_ID}.npz"))
    return data["X_train"], data["y_train"], data["X_test"], data["y_test"], manifest


def plain_train(model, X_train, y_train, epochs, lr):
    """Plain (non-DP) local training -- CrossEntropyLoss, Adam, StepLR,
    grad clip 1.0, batch_size=BATCH_SIZE. This is a timing/RAM harness,
    not an accuracy run, so class-weighted FocalLoss (which needs
    data_loader's live class counts) is intentionally not used here --
    training cost is driven by architecture + data volume, not the loss
    function's per-class weights."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.95)
    criterion = nn.CrossEntropyLoss()

    X = torch.FloatTensor(X_train)
    y = torch.LongTensor(y_train)
    loader = tud.DataLoader(tud.TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=True)

    for _ in range(epochs):
        for X_b, y_b in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
    return model


def dp_train(model, X_train, y_train, epochs, timer):
    """Opacus DP-SGD training -- mirrors main.py's pattern (PrivacyEngine,
    make_private_with_epsilon), matching the old pure_dp Docker test's
    own config values (target_epsilon=3.0, max_grad_norm=1.0,
    dp_batch_size=512) rather than the main pipeline's (max_grad_norm=1.5).

    Timed as TWO separate stages on the given StageTimer, since
    "DP-SGD overhead" is really two different costs worth seeing
    separately:
      dp_setup  -- PrivacyEngine.make_private_with_epsilon(): noise
                   multiplier search/calibration + wrapping the model/
                   optimizer/loader for per-sample gradients. A one-time
                   per-round cost, not proportional to epochs.
      dp_train  -- the actual per-sample-gradient training loop (the
                   part that's slower than plain training per batch).

    Returns (model, achieved_epsilon, noise_multiplier).
    """
    from opacus import PrivacyEngine

    criterion = nn.CrossEntropyLoss()
    X = torch.FloatTensor(X_train)
    y = torch.LongTensor(y_train)
    loader = tud.DataLoader(tud.TensorDataset(X, y), batch_size=DP_BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    privacy_engine = PrivacyEngine(accountant="rdp")

    with timer.stage("dp_setup"):
        model, optimizer, loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            target_epsilon=DP_EPSILON,
            target_delta=DP_DELTA,
            epochs=epochs,
            max_grad_norm=DP_MAX_GRAD_NORM,
        )

    with timer.stage("dp_train"):
        model.train()
        for _ in range(epochs):
            for X_b, y_b in loader:
                optimizer.zero_grad()
                loss = criterion(model(X_b), y_b)
                loss.backward()
                optimizer.step()

    achieved_epsilon = privacy_engine.get_epsilon(delta=DP_DELTA)
    noise_multiplier = getattr(optimizer, "noise_multiplier", None)

    # make_private_with_epsilon() wraps `model` in a GradSampleModule and
    # attaches forward/backward hooks DIRECTLY onto its submodules
    # (Conv1d, GroupNorm, DPLSTM) for per-sample gradient computation.
    # Unwrapping via `._module` does NOT remove those hooks -- they stay
    # attached. Since this harness reuses the same model object across
    # ROUNDS (not a fresh model each round), the next round's
    # make_private_with_epsilon() call tries to add a second set of hooks
    # on top of the still-attached first set, and Opacus raises
    # "Trying to add hooks twice to the same model". Explicitly removing
    # them here is what makes safe reuse across rounds possible.
    model.remove_hooks()
    real_model = model._module if hasattr(model, "_module") else model
    return real_model, achieved_epsilon, noise_multiplier


def run():
    X_train, y_train, X_test, y_test, manifest = load_partition()
    num_features = manifest["num_features"]
    num_classes = manifest["num_classes"]

    print(f"[client {CLIENT_ID}] mode={MODE} model={MODEL_TYPE} "
          f"features={num_features} classes={num_classes} "
          f"train_rows={len(y_train):,}")

    real_mem_limit_mb = read_cgroup_memory_limit_mb()
    real_cpu_limit = read_cgroup_cpu_limit()
    print(f"[client {CLIENT_ID}] REAL enforced limits (read from cgroup, "
          f"not a hardcoded label): mem={real_mem_limit_mb} MB  cpu={real_cpu_limit} cores")

    model = get_model(num_features=num_features, num_classes=num_classes, dp_safe=DP_SAFE)
    keys = get_model_parameter_keys(model)
    total_params = sum(p.numel() for p in model.parameters())

    round_records = []
    dp_epsilon_achieved = None
    dp_noise_multiplier = None
    prev_round_params = get_model_parameters(model)  # round-1 "starting" params

    with RamSampler(interval_s=0.25) as sampler:
        for rnd in range(1, ROUNDS + 1):
            timer = StageTimer()
            t_round0 = time.time()

            # ── Stage 1: local training ─────────────────────────────
            if MODE == "dp":
                model, dp_epsilon_achieved, dp_noise_multiplier = dp_train(
                    model, X_train, y_train, EPOCHS, timer
                )
                record = {
                    "round": rnd,
                    "dp_setup_time_s": round(timer.durations.get("dp_setup", 0.0), 4),
                    "train_time_s": round(timer.durations.get("dp_train", 0.0), 4),
                }
            else:
                with timer.stage("train"):
                    model = plain_train(model, X_train, y_train, EPOCHS, LR)
                record = {
                    "round": rnd,
                    "train_time_s": round(timer.durations.get("train", 0.0), 4),
                }

            trained_params = get_model_parameters(model)

            # ── Stage 2: mode-specific mechanism (encryption / ZKP / DP metadata) ──
            artifact_obj = None

            if MODE == "he_full":
                import he_aggregation as he
                he_context = he.create_server_context(poly_modulus_degree=HE_POLY_DEGREE)
                with timer.stage("he_encrypt_full"):
                    enc = he.encrypt_param_list(
                        trained_params, he_context, chunk_size=HE_POLY_DEGREE // 2
                    )
                record["he_encrypt_time_s"] = round(timer.durations.get("he_encrypt_full", 0.0), 4)
                record["n_chunks"] = enc["n_chunks"]
                record["pct_encrypted"] = 100.0
                artifact_obj = {"mode": "he_full", "enc": enc}

            elif MODE in ("he_partial", "he_partial_zkp"):
                import he_local
                import he_aggregation as he
                he_context = he.create_server_context(poly_modulus_degree=HE_POLY_DEGREE)

                if MODE == "he_partial":
                    with timer.stage("he_encrypt_partial"):
                        enc = he_local.encrypt_params(
                            trained_params, keys, he_context, HE_POLY_DEGREE
                        )
                    record["he_encrypt_time_s"] = round(timer.durations.get("he_encrypt_partial", 0.0), 4)
                    artifact_obj = {"mode": "he_partial", "enc": enc}
                else:
                    # he_partial_zkp: measure encryption and the norm-guard
                    # proof as two SEPARATE stages, so the guard's added
                    # cost on top of partial HE is isolated. The delta is
                    # computed against the PREVIOUS round's trained params
                    # (round 1 against the freshly-initialised model, which
                    # is what "global params this client started the round
                    # with" means in a single-client harness -- there's no
                    # real FedAvg global model here since this is a per-
                    # client timing test, not a full FL loop).
                    global_params_before = prev_round_params
                    with timer.stage("he_encrypt_partial"):
                        enc = he_local.encrypt_params(
                            trained_params, keys, he_context, HE_POLY_DEGREE
                        )
                    with timer.stage("zkp_head_norm_proof"):
                        sensitive_idx = enc["sensitive_idx"]
                        global_sensitive = [global_params_before[i] for i in sensitive_idx]
                        trained_sensitive = [trained_params[i] for i in sensitive_idx]
                        delta_flat = np.concatenate([
                            (t - g).flatten() for t, g in zip(trained_sensitive, global_sensitive)
                        ]).astype(np.float64)
                        import zkp
                        proof = zkp.generate_head_norm_proof(delta_flat, enc["sensitive_enc"]["chunks"])
                        # proof["salt"] is raw bytes (os.urandom(32)) --
                        # not JSON-serializable, and would silently
                        # become a garbled str() repr on round-trip like
                        # the bulk-array bug above. verify_head_norm_proof()
                        # never actually reads salt (confirmed in zkp.py --
                        # kept only for interface symmetry with Part 1), so
                        # this doesn't affect correctness either way, but
                        # hex-encoding it keeps the artifact genuinely
                        # round-trippable instead of silently lossy.
                        proof["salt"] = proof["salt"].hex()
                    record["he_encrypt_time_s"] = round(timer.durations.get("he_encrypt_partial", 0.0), 4)
                    record["zkp_proof_time_s"] = round(timer.durations.get("zkp_head_norm_proof", 0.0), 4)
                    record["zkp_claimed_norm"] = proof["norm"]
                    artifact_obj = {"mode": "he_partial_zkp", "enc": enc, "proof": proof}

                record["n_chunks"] = enc["n_chunks"]
                record["pct_encrypted"] = enc["pct_encrypted"]

                # CRITICAL FIX: enc["bulk"] is a list of RAW numpy arrays
                # (the plaintext, non-encrypted ~94% of the model).
                # json.dumps(..., default=str) cannot serialize ndarrays
                # directly and falls back to str(array) -- which numpy
                # SILENTLY TRUNCATES for arrays over 1000 elements
                # (summarized with "..."), destroying almost all the
                # actual data while still "succeeding" with no error.
                # Convert to plain nested lists BEFORE this artifact is
                # written to disk or sent over the network, so the full
                # data survives the round trip intact.
                enc["bulk"] = [b.tolist() for b in enc["bulk"]]

            elif MODE == "dp":
                record["dp_epsilon_target"] = DP_EPSILON
                record["dp_epsilon_achieved"] = dp_epsilon_achieved
                record["dp_noise_multiplier"] = dp_noise_multiplier
                artifact_obj = {"mode": "dp", "params": [p.tolist() for p in trained_params]}

            else:  # baseline
                artifact_obj = {"mode": "baseline", "params": [p.tolist() for p in trained_params]}

            # ── Stage 3: write artifact to disk (used by server_aggregate.py's
            # aggregation timing) AND send it over the real Docker network to
            # server_daemon.py (used for communication timing) ────────────
            _save_artifact(rnd, artifact_obj)
            with timer.stage("communication"):
                comm = send_artifact_over_network(artifact_obj, rnd)
            record["serialize_time_s"] = comm["serialize_time_s"]
            record["communication_send_time_s"] = comm["send_time_s"]
            record["payload_bytes"] = comm["payload_bytes"]
            record["communication_ok"] = comm["communication_ok"]

            record["round_wall_time_s"] = round(time.time() - t_round0, 4)
            round_records.append(record)
            prev_round_params = trained_params
            print(f"[client {CLIENT_ID}] round {rnd}/{ROUNDS} done: {record}")

    ram_summary = sampler.summary()

    result = {
        "config": {
            "client_id": CLIENT_ID,
            "mode": MODE,
            "model_type": MODEL_TYPE,
            "num_features": num_features,
            "num_classes": num_classes,
            "total_params": int(total_params),
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
            "epochs_per_round": EPOCHS,
            "rounds": ROUNDS,
            "batch_size": BATCH_SIZE if MODE != "dp" else DP_BATCH_SIZE,
            "he_poly_degree": HE_POLY_DEGREE if MODE.startswith("he_") else None,
            "dp_epsilon_target": DP_EPSILON if MODE == "dp" else None,
            "dp_max_grad_norm": DP_MAX_GRAD_NORM if MODE == "dp" else None,
            # REAL, live-read limits -- not the old hardcoded "200" bug.
            # None means "no limit enforced / not running under cgroup
            # constraints (or unreadable)" -- report honestly rather than
            # ever falling back to a guessed number.
            "real_cgroup_mem_limit_mb": real_mem_limit_mb,
            "real_cgroup_cpu_limit_cores": real_cpu_limit,
        },
        "rounds": round_records,
        "ram_peak_mb": ram_summary["peak_mb"],
        "ram_avg_mb": ram_summary["avg_mb"],
        "ram_n_samples": ram_summary["n_samples"],
    }

    write_json(os.path.join(OUT_DIR, f"client_{CLIENT_ID}_results.json"), result)
    print(f"[client {CLIENT_ID}] DONE. peak_ram={ram_summary['peak_mb']}MB "
          f"avg_ram={ram_summary['avg_mb']}MB")


def _save_artifact(round_idx, obj):
    path = os.path.join(OUT_DIR, f"client_{CLIENT_ID}_round_{round_idx}_artifact.json")
    write_json(path, obj)


if __name__ == "__main__":
    run()
