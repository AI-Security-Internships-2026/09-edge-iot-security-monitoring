#!/usr/bin/env python3
"""NO-DOCKER SMOKE TEST: runs server + 2 clients as local processes (tiny data, 1 epoch) to confirm the pipeline works
on your machine before the real Docker run. Results are UNTHROTTLED and must NOT be used for Table 5.
Usage: python e8_harness/run_e8_local.py [mode]     (mode default: calibrated_krum; needs torch, opacus for dp mode)"""
import json, os, subprocess, sys, tempfile, time
H = os.path.dirname(os.path.abspath(__file__)); mode = sys.argv[1] if len(sys.argv) > 1 else "calibrated_krum"
cfg = json.load(open(os.path.join(H, "configs", f"{mode}_1.0vcpu.json"))); cfg["epochs_per_round"] = 1
d = tempfile.mkdtemp(); cp = os.path.join(d, "cfg.json"); json.dump(cfg, open(cp, "w")); port = "9055"
srv = subprocess.Popen([sys.executable, os.path.join(H, "docker_bench_server.py"), "--config", cp, "--port", port, "--out-dir", d, "--repeats", "2"])
time.sleep(2)
cl = [subprocess.Popen([sys.executable, os.path.join(H, "docker_bench_client.py"), "--config", cp, "--client-id", str(i), "--server-host", "127.0.0.1",
                        "--server-port", port, "--out-dir", d, "--train-rows", "2000", "--test-rows", "500"]) for i in (0, 1)]
rc = [p.wait() for p in cl] + [srv.wait()]
print("exit codes:", rc, "| outputs:", sorted(os.listdir(d)))
c0 = json.load(open(os.path.join(d, "client_0_results.json")))
print("data_source:", c0["config"]["data_source"], "| params:", c0["config"]["total_params"], "| payload_bytes:", c0["rounds"][0]["payload_bytes"])
sys.exit(0 if all(r == 0 for r in rc) else 1)
