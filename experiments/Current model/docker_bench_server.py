#!/usr/bin/env python3
"""docker_bench_server.py -- E8 server driver. Imports the REAL aggregators from defences/krum.py (numpy only, no torch).

Per round, once both real clients' updates have arrived, aggregation is timed at n=10 clients (the 2 real updates + 8
Gaussian-jittered copies), using the same call signatures and arguments main.py uses (num_byzantine=2, k=2.5, MAD,
min_keep_fraction=0.5, return_diagnostics=True; calibrated: public median honest noise multiplier, hetero fit JSON,
prior-round honest std carried forward). Timing only -- not an accuracy/detection result.

Modes: fedavg | adaptive_krum | calibrated_krum | dp_calibrated_krum
Usage: python docker_bench_server.py --config configs/adaptive_krum_1.0vcpu.json --port 9000 --out-dir /results"""
import argparse, json, os, socket, statistics, time
import numpy as np
from bench_common import (RamSampler, timer, write_json, load_config, read_cgroup_limits, read_cgroup_mem_peak_mb,
                          recv_msg, unpack_params, HARNESS_DIR)

try:
    from defences.krum import adaptive_multi_krum, calibrated_adaptive_multi_krum, fedavg, public_noise_multiplier_map
except ImportError:
    from krum import adaptive_multi_krum, calibrated_adaptive_multi_krum, fedavg, public_noise_multiplier_map

SYNTHETIC_N_CLIENTS = 10
NUM_BYZANTINE = 2            # main.py default (clients 1,2); N-f-2 = 6 neighbours at n=10
K, METHOD, MIN_KEEP = 2.5, "mad", 0.5   # main.py: ADAPTIVE_KRUM_K default (E3/E5 manifests), "mad", 0.5
JITTER_STD = 0.01            # relative to each array's std; the synthetic clients only need to be distinct points


def synthesize(real, n_total, rng):
    """real: list of dicts {params, n_samples, class_entropy, noise_multiplier}. Returns n_total client dicts."""
    out = list(real)
    i = 0
    while len(out) < n_total:
        b = real[i % len(real)]
        params = [a + rng.normal(0.0, JITTER_STD * (a.std() or 1.0), a.shape).astype(a.dtype) if np.issubdtype(a.dtype, np.floating) else a.copy()
                  for a in b["params"]]
        out.append({"params": params, "n_samples": int(b["n_samples"] * rng.uniform(0.9, 1.1)),
                    "class_entropy": float(b["class_entropy"] + rng.normal(0, 0.02)), "noise_multiplier": None})
        i += 1
    return out[:n_total]


def run_round_timing(mode, real, config, hetero_coeffs, prior_std, repeats, rng):
    clients = synthesize(real, SYNTHETIC_N_CLIENTS, rng)
    params = [c["params"] for c in clients]
    weights = [c["n_samples"] for c in clients]
    new_std, kept = prior_std, None

    def call(prior):
        if mode == "fedavg":
            fedavg(params, weights); return None, None, prior
        if mode == "adaptive_krum":
            _, sel, diag = adaptive_multi_krum(params, weights, num_byzantine=NUM_BYZANTINE, k=K, method=METHOD,
                                               min_keep_fraction=MIN_KEEP, return_diagnostics=True)
            return None, sel, prior
        use_dp = (mode == "dp_calibrated_krum")
        sigma_by = {pos: c["noise_multiplier"] for pos, c in enumerate(clients[:len(real)]) if c["noise_multiplier"] is not None}
        pub = public_noise_multiplier_map(list(range(len(clients))), sigma_by) if (use_dp and sigma_by) else {}
        meta = {pos: {"n_samples": c["n_samples"], "class_entropy": c["class_entropy"],
                      "noise_multiplier": pub.get(pos) if use_dp else None,
                      "epsilon": config.get("dp_epsilon_target") if use_dp else None} for pos, c in enumerate(clients)}
        _, sel, diag = calibrated_adaptive_multi_krum(
            params, weights, meta, num_byzantine=NUM_BYZANTINE, k=K, method=METHOD, min_keep_fraction=MIN_KEEP,
            baseline_honest_std_from_prior_round=prior, dp_max_grad_norm=config.get("dp_max_grad_norm") or 1.0,
            alpha_dirichlet=config.get("alpha_dirichlet", 0.7), hetero_fit_coeffs=hetero_coeffs,
            use_dp_calibration=use_dp, use_hetero_calibration=True, return_diagnostics=True)
        return None, sel, diag["new_baseline_honest_std"]

    call(prior_std)                                       # warm-up (not recorded), so import/alloc cost is excluded
    times = []
    for _ in range(repeats):
        with timer() as t:
            _, kept, new_std = call(prior_std)
        times.append(t.elapsed)
    return {"mean_s": statistics.mean(times), "all_calls_s": [round(x, 5) for x in times], "kept": kept, "new_std": new_std}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--port", type=int, default=int(os.environ.get("SERVER_PORT", "9000")))
    ap.add_argument("--num-clients", type=int, default=2)
    ap.add_argument("--out-dir", default="/results")
    ap.add_argument("--repeats", type=int, default=5, help="timed aggregation calls per round (after one warm-up)")
    ap.add_argument("--hetero-fit-json", default=os.path.join(HARNESS_DIR, "hetero_fit_coeffs.json"))
    args = ap.parse_args()
    cfg = load_config(args.config); mode = cfg["mode"]
    hetero = json.load(open(args.hetero_fit_json)) if mode in ("calibrated_krum", "dp_calibrated_krum") else None
    rng = np.random.default_rng(0)

    ram = RamSampler(); ram.start()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port)); srv.listen(8)
    expected = args.num_clients * cfg["rounds"]
    print(f"[server] mode={mode} listening on :{args.port}, expecting {expected} submissions", flush=True)

    subs, by_round, timing_rounds, prior_std = [], {}, [], None
    wall0 = time.perf_counter()
    while len(subs) < expected:
        conn, _ = srv.accept()
        with conn:
            t0 = time.perf_counter()
            header, payload = recv_msg(conn)
            recv_time = time.perf_counter() - t0
        r, cid = int(header["round"]), int(header["client_id"])
        subs.append({"client_id": cid, "round": r, "recv_time_s": round(recv_time, 5), "payload_bytes": len(payload),
                     "wall_time_s": round(time.perf_counter() - wall0, 3)})
        by_round.setdefault(r, {})[cid] = {"params": unpack_params(payload), "n_samples": header["n_samples"],
                                           "class_entropy": header["class_entropy"], "noise_multiplier": header.get("noise_multiplier")}
        print(f"[server] got client {cid} round {r}: {len(payload)} B in {recv_time:.3f}s", flush=True)
        if len(by_round[r]) == args.num_clients and r not in {x["round"] for x in timing_rounds}:
            real = [by_round[r][c] for c in sorted(by_round[r])]
            res = run_round_timing(mode, real, cfg, hetero, prior_std, args.repeats, rng)
            prior_std = res["new_std"]
            kept = res["kept"]
            timing_rounds.append({"round": r, f"{mode}_time_s": round(res["mean_s"], 5), "all_calls_s": res["all_calls_s"],
                                  "n_clients_synthetic": SYNTHETIC_N_CLIENTS, "n_clients_real": args.num_clients,
                                  "num_kept": None if kept is None else len(kept),
                                  "num_dropped": None if kept is None else SYNTHETIC_N_CLIENTS - len(kept)})
            print(f"[server] round {r} aggregation ({mode}): {res['mean_s']*1e3:.2f} ms (mean of {args.repeats})", flush=True)
    timing_rounds.sort(key=lambda x: x["round"])
    ram.stop(); rs = ram.summary()
    mem_lim, cpu_lim = read_cgroup_limits()
    write_json(os.path.join(args.out_dir, "server_communication_summary.json"), {
        "expected_total": expected, "received_total": len(subs), "submissions": subs,
        "avg_recv_time_s": round(sum(s["recv_time_s"] for s in subs) / len(subs), 5),
        "total_bytes_received": sum(s["payload_bytes"] for s in subs)})
    write_json(os.path.join(args.out_dir, f"server_{mode}_results.json"), {
        "mode": mode,
        "config": {"num_real_clients": args.num_clients, "synthetic_n_clients": SYNTHETIC_N_CLIENTS, "rounds": cfg["rounds"],
                   "num_byzantine": NUM_BYZANTINE, "k": K, "min_keep_fraction": MIN_KEEP, "timed_repeats_per_round": args.repeats,
                   "hetero_fit_json": os.path.basename(args.hetero_fit_json) if hetero else None, "he_poly_degree": None,
                   "real_cgroup_mem_limit_mb": mem_lim or cfg.get("real_cgroup_mem_limit_mb"),
                   "real_cgroup_cpu_limit_cores": cpu_lim or cfg.get("real_cgroup_cpu_limit_cores")},
        "krum_timing": {"per_round": timing_rounds, "ram_peak_mb": rs["ram_peak_mb"], "ram_avg_mb": rs["ram_avg_mb"],
                        "cgroup_mem_peak_mb": read_cgroup_mem_peak_mb(),
                        "note": "Timing at n=10 clients (2 real + 8 jittered copies); NOT an accuracy/detection result."}})

if __name__ == "__main__":
    main()
