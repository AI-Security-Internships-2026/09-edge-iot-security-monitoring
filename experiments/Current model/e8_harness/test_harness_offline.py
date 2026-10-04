#!/usr/bin/env python3
"""Offline test of the SERVER side (no torch, no Docker, no dataset): starts docker_bench_server.py in every mode against the
real defences/krum.py and feeds it fake-client messages built with the same wire protocol and a model-shaped parameter list.
Checks that the server accepts the protocol, calls the real aggregators with valid arguments, and writes the expected JSON.
Usage: python test_harness_offline.py   (needs numpy + psutil; PROJECT_ROOT must contain defences/krum.py)"""
import json, os, subprocess, sys, tempfile, time
import numpy as np
from bench_common import pack_params, send_msg, HARNESS_DIR

def fake_params(rng):
    f = lambda *s: rng.normal(0, 0.1, s).astype(np.float32)
    bn = lambda c: [np.ones(c, np.float32), np.zeros(c, np.float32), f(c), np.abs(f(c)) + 1, np.array(5, dtype=np.int64)]
    return ([f(64, 1, 3), f(64)] + bn(64) + [f(128, 64, 3), f(128)] + bn(128) +
            [f(256, 128), f(256, 64), f(256), f(256), f(64, 64), f(64), f(8, 64), f(8)])

def main():
    port0, rng = 9100, np.random.default_rng(1); failures = []
    for i, mode in enumerate(["fedavg", "adaptive_krum", "calibrated_krum", "dp_calibrated_krum"]):
        out = tempfile.mkdtemp(); port = port0 + i
        cfg = os.path.join(HARNESS_DIR, "configs", f"{mode}_1.0vcpu.json")
        srv = subprocess.Popen([sys.executable, os.path.join(HARNESS_DIR, "docker_bench_server.py"), "--config", cfg,
                                "--port", str(port), "--out-dir", out, "--repeats", "3"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        time.sleep(1.5)
        for r in range(1, 4):
            for cid in (0, 1):
                hdr = {"client_id": cid, "round": r, "n_samples": 40000 + 1000 * cid, "class_entropy": 1.6 + 0.1 * cid,
                       "noise_multiplier": (1.2 + 0.3 * cid) if mode == "dp_calibrated_krum" else None}
                s, ok = send_msg("127.0.0.1", port, hdr, pack_params(fake_params(rng)), connect_timeout_s=10)
                assert ok, f"{mode}: send failed"
        log, _ = srv.communicate(timeout=120)
        if srv.returncode != 0: failures.append((mode, log[-800:])); continue
        res = json.load(open(os.path.join(out, f"server_{mode}_results.json")))
        comm = json.load(open(os.path.join(out, "server_communication_summary.json")))
        rounds = res["krum_timing"]["per_round"]
        ok = (len(rounds) == 3 and comm["received_total"] == 6 and all(f"{mode}_time_s" in x for x in rounds)
              and {s["client_id"] for s in comm["submissions"]} == {0, 1})
        print(f"{mode:<20} rounds={len(rounds)} agg_ms={[round(x[f'{mode}_time_s']*1e3,1) for x in rounds]} "
              f"kept={[x['num_kept'] for x in rounds]} payload={comm['submissions'][0]['payload_bytes']}B  {'OK' if ok else 'FAIL'}")
        if not ok: failures.append((mode, "unexpected output"))
    print("ALL SERVER MODES PASSED" if not failures else f"FAILURES: {failures}")
    sys.exit(1 if failures else 0)

if __name__ == "__main__":
    main()
