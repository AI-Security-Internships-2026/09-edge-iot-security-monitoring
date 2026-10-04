#!/usr/bin/env python3
"""docker_bench_client.py -- E8 client driver (training time, serialisation, send time, RAM, payload).

Uses the project's own code: model_defs.get_model (dp_safe=True for the DP mode, as main.py does) and
data_loader.load_partition_network. If data_loader cannot be imported (or --data-source synthetic) it falls back to
random data of the same shape and WRITES data_source="synthetic" into the result JSON: timing/RAM/payload are then
valid for that shape but not for the real data distribution.

Client-side training is identical for fedavg / adaptive_krum / calibrated_krum (aggregation is server-side only).
dp_calibrated_krum trains with Opacus DP-SGD (one persistent PrivacyEngine for all rounds, RDP accountant, batch 512,
as in dp_persistent_client_state.py). The benchmark does not send a global model back, so each client keeps training
its own model for the 3 rounds (the same for every mode). FedProx's proximal pull is not applied.

Usage: python docker_bench_client.py --config configs/adaptive_krum_1.0vcpu.json --client-id 0 --server-host server"""
import argparse, math, os, time
import numpy as np
from bench_common import (RamSampler, timer, write_json, load_config, read_cgroup_limits, read_cgroup_mem_peak_mb,
                          pack_params, send_msg, HARNESS_DIR)
import torch, torch.utils.data as tud

try:
    from model_defs import get_model, get_model_parameters
except ImportError:
    from task import get_model, get_model_parameters


def shannon_entropy(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def load_data(args, cfg):
    """Returns (X_tr, y_tr, X_te, y_te, source). Real path mirrors main.py: load_partition(i, NUM_CLIENTS, seed=, alpha=)."""
    n_classes, n_feat = cfg["num_classes"], cfg["num_features"]
    train_rows, test_rows = args.train_rows or cfg.get("train_rows"), args.test_rows or cfg.get("test_rows")
    if args.data_source in ("auto", "real"):
        try:
            from data_loader import load_partition_network
            X_tr, y_tr, X_te, y_te = load_partition_network(args.client_id, args.num_clients, seed=args.seed, alpha=args.alpha)
            rng = np.random.default_rng(args.seed + args.client_id)
            def sub(X, y, n):          # proportional per-class subsample so class mix is preserved
                if not n or n >= len(y): return X, y
                idx = np.concatenate([rng.choice(np.where(y == c)[0], max(1, int(round(n * (y == c).mean()))), replace=False)
                                      for c in np.unique(y)])
                return X[idx], y[idx]
            X_tr, y_tr = sub(X_tr, y_tr, train_rows); X_te, y_te = sub(X_te, y_te, test_rows)
            return X_tr.astype(np.float32), y_tr.astype(np.int64), X_te, y_te, "real"
        except Exception as e:
            if args.data_source == "real": raise
            print(f"[client] WARNING: real data not available ({type(e).__name__}: {e}); using SYNTHETIC data", flush=True)
    rng = np.random.default_rng(args.seed + args.client_id)
    n_tr, n_te = int(train_rows or 40000), int(test_rows or 5000)
    X_tr = rng.normal(size=(n_tr, n_feat)).astype(np.float32); y_tr = rng.integers(0, n_classes, n_tr).astype(np.int64)
    X_te = rng.normal(size=(n_te, n_feat)).astype(np.float32); y_te = rng.integers(0, n_classes, n_te).astype(np.int64)
    return X_tr, y_tr, X_te, y_te, "synthetic"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--client-id", type=int, required=True)
    ap.add_argument("--server-host", default=os.environ.get("SERVER_HOST", "server"))
    ap.add_argument("--server-port", type=int, default=int(os.environ.get("SERVER_PORT", "9000")))
    ap.add_argument("--out-dir", default="/results")
    ap.add_argument("--num-clients", type=int, default=2); ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-rows", type=int, default=None, help="per-client train rows; set to the legacy value (see README)")
    ap.add_argument("--test-rows", type=int, default=None)
    ap.add_argument("--data-source", choices=["auto", "real", "synthetic"], default="auto")
    args = ap.parse_args()
    cfg = load_config(args.config); mode = cfg["mode"]; use_dp = (mode == "dp_calibrated_krum")

    mem_lim, cpu_lim = read_cgroup_limits()
    cpu_cores = cpu_lim or cfg["real_cgroup_cpu_limit_cores"]
    torch.set_num_threads(max(1, math.ceil(cpu_cores)))       # do not oversubscribe a throttled container
    torch.manual_seed(args.seed + args.client_id)
    device = torch.device("cpu")

    X_tr, y_tr, X_te, y_te, source = load_data(args, cfg)
    n_feat, n_classes = X_tr.shape[1], cfg["num_classes"]
    class_entropy = shannon_entropy(np.bincount(y_tr, minlength=n_classes).astype(float))
    model = get_model(num_features=n_feat, num_classes=n_classes, dp_safe=use_dp).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if cfg.get("total_params") and n_params != cfg["total_params"]:
        print(f"[client] WARNING: model has {n_params} trainable params, config expects {cfg['total_params']}", flush=True)

    loader = tud.DataLoader(tud.TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)), batch_size=cfg["batch_size"], shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("learning_rate", 1e-3))
    engine = noise_mult = None
    if use_dp:
        from opacus import PrivacyEngine
        engine = PrivacyEngine(accountant="rdp")
        model, optimizer, loader = engine.make_private_with_epsilon(
            module=model, optimizer=optimizer, data_loader=loader, target_epsilon=cfg["dp_epsilon_target"], target_delta=1e-5,
            epochs=cfg["rounds"] * cfg["epochs_per_round"], max_grad_norm=cfg["dp_max_grad_norm"])
        noise_mult = float(optimizer.noise_multiplier)
    real_model = model._module if hasattr(model, "_module") else model       # unwrap Opacus GradSampleModule
    criterion = torch.nn.CrossEntropyLoss()

    ram = RamSampler(); ram.start()
    rounds_out = []
    for r in range(1, cfg["rounds"] + 1):
        t_round = time.perf_counter()
        with timer() as t_train:
            model.train()
            for _ in range(cfg["epochs_per_round"]):
                for xb, yb in loader:
                    optimizer.zero_grad(); criterion(model(xb.to(device)), yb.to(device)).backward(); optimizer.step()
        with timer() as t_ser:
            payload = pack_params(get_model_parameters(real_model))
        header = {"client_id": args.client_id, "round": r, "n_samples": int(len(y_tr)), "class_entropy": class_entropy,
                  "noise_multiplier": noise_mult}
        send_s, ok = send_msg(args.server_host, args.server_port, header, payload)
        rounds_out.append({"round": r, "train_time_s": round(t_train.elapsed, 4), "serialize_time_s": round(t_ser.elapsed, 5),
                           "communication_send_time_s": round(send_s, 5), "payload_bytes": len(payload), "communication_ok": ok,
                           "round_wall_time_s": round(time.perf_counter() - t_round, 4)})
        print(f"[client {args.client_id}] round {r}: train={t_train.elapsed:.2f}s ser={t_ser.elapsed:.3f}s send={send_s:.3f}s "
              f"payload={len(payload)}B ok={ok}", flush=True)
    ram.stop()
    out = {"config": {"client_id": args.client_id, "mode": mode, "model_type": "network", "data_source": source,
                      "num_features": n_feat, "num_classes": n_classes, "total_params": n_params, "train_rows": int(len(y_tr)),
                      "test_rows": int(len(y_te)), "epochs_per_round": cfg["epochs_per_round"], "rounds": cfg["rounds"],
                      "batch_size": cfg["batch_size"], "torch_threads": torch.get_num_threads(), "he_poly_degree": None,
                      "dp_epsilon_target": cfg.get("dp_epsilon_target"), "dp_max_grad_norm": cfg.get("dp_max_grad_norm"),
                      "dp_noise_multiplier": noise_mult,
                      "dp_achieved_epsilon": engine.get_epsilon(delta=1e-5) if engine else None,
                      "real_cgroup_mem_limit_mb": mem_lim or cfg["real_cgroup_mem_limit_mb"], "real_cgroup_cpu_limit_cores": cpu_cores},
           "rounds": rounds_out, "cgroup_mem_peak_mb": read_cgroup_mem_peak_mb(), **ram.summary()}
    write_json(os.path.join(args.out_dir, f"client_{args.client_id}_results.json"), out)

if __name__ == "__main__":
    main()
