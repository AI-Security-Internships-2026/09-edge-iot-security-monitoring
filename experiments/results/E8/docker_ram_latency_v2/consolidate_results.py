"""
Consolidates every results/<model>_<mode>_<profile>/ run into one
summary JSON + a printed table -- run this after run_suite.sh finishes.

Usage: python consolidate_results.py [results_dir]
"""
import glob
import json
import os
import sys

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "results"


def load_run(run_dir):
    tag = os.path.basename(run_dir)
    clients = []
    for p in sorted(glob.glob(os.path.join(run_dir, "client_*_results.json"))):
        with open(p) as f:
            clients.append(json.load(f))

    server = None
    for p in glob.glob(os.path.join(run_dir, "server_*_results.json")):
        with open(p) as f:
            server = json.load(f)

    return {"tag": tag, "clients": clients, "server": server}


def summarize(run):
    if not run["clients"]:
        return None
    mode = run["clients"][0]["config"]["mode"]
    model = run["clients"][0]["config"]["model_type"]

    avg_train_time = sum(
        r["train_time_s"] for c in run["clients"] for r in c["rounds"]
    ) / sum(len(c["rounds"]) for c in run["clients"])

    all_rounds = [r for c in run["clients"] for r in c["rounds"]]
    avg_serialize = sum(r.get("serialize_time_s", 0) for r in all_rounds) / max(1, len(all_rounds))
    avg_send = sum(r.get("communication_send_time_s", 0) for r in all_rounds) / max(1, len(all_rounds))
    avg_payload_bytes = sum(r.get("payload_bytes", 0) for r in all_rounds) / max(1, len(all_rounds))
    comm_failures = sum(1 for r in all_rounds if not r.get("communication_ok", True))

    dp_setup_rounds = [r for r in all_rounds if "dp_setup_time_s" in r]
    avg_dp_setup = (sum(r["dp_setup_time_s"] for r in dp_setup_rounds) / len(dp_setup_rounds)
                     if dp_setup_rounds else None)

    real_mem_limits = {c["config"]["client_id"]: c["config"]["real_cgroup_mem_limit_mb"]
                        for c in run["clients"]}
    real_cpu_limits = {c["config"]["client_id"]: c["config"]["real_cgroup_cpu_limit_cores"]
                        for c in run["clients"]}
    peak_ram = {c["config"]["client_id"]: c["ram_peak_mb"] for c in run["clients"]}

    summary = {
        "tag": run["tag"],
        "mode": mode,
        "model_type": model,
        "total_params": run["clients"][0]["config"]["total_params"],
        "num_features": run["clients"][0]["config"]["num_features"],
        "avg_train_time_s_per_round": round(avg_train_time, 4),
        "avg_dp_setup_time_s_per_round": round(avg_dp_setup, 4) if avg_dp_setup is not None else None,
        "avg_serialize_time_s": round(avg_serialize, 5),
        "avg_communication_send_time_s": round(avg_send, 5),
        "avg_payload_bytes": round(avg_payload_bytes, 1),
        "communication_failures": comm_failures,
        "real_cgroup_mem_limit_mb": real_mem_limits,
        "real_cgroup_cpu_limit_cores": real_cpu_limits,
        "client_peak_ram_mb": peak_ram,
    }

    comm_summary_path = os.path.join("results", run["tag"], "server_communication_summary.json")
    if os.path.exists(comm_summary_path):
        with open(comm_summary_path) as f:
            comm = json.load(f)
        summary["server_avg_recv_time_s"] = comm.get("avg_recv_time_s")
        summary["server_total_bytes_received"] = comm.get("total_bytes_received")

    if mode.startswith("he_") and run["server"]:
        he = run["server"].get("he_aggregation", {})
        avg_agg = sum(r.get("aggregate", 0) for r in he.get("per_round", [])) / max(
            1, len(he.get("per_round", [])))
        avg_dec = sum(r.get("decrypt", 0) for r in he.get("per_round", [])) / max(
            1, len(he.get("per_round", [])))
        summary["server_avg_aggregate_s"] = round(avg_agg, 5)
        summary["server_avg_decrypt_s"] = round(avg_dec, 5)
        summary["server_peak_ram_mb"] = he.get("ram_peak_mb")
        if mode == "he_partial_zkp":
            avg_zkp = sum(r.get("zkp_verify", 0) + r.get("zkp_mad_threshold", 0)
                          for r in he.get("per_round", [])) / max(1, len(he.get("per_round", [])))
            summary["server_avg_zkp_verify_plus_threshold_s"] = round(avg_zkp, 5)
        avg_enc = sum(
            r["he_encrypt_time_s"] for c in run["clients"] for r in c["rounds"]
        ) / sum(len(c["rounds"]) for c in run["clients"])
        summary["client_avg_he_encrypt_s"] = round(avg_enc, 5)
        if mode == "he_partial_zkp":
            avg_proof = sum(
                r["zkp_proof_time_s"] for c in run["clients"] for r in c["rounds"]
            ) / sum(len(c["rounds"]) for c in run["clients"])
            summary["client_avg_zkp_proof_s"] = round(avg_proof, 5)

    if mode in ("baseline", "dp") and run["server"]:
        kt = run["server"].get("krum_timing", {})
        avg_krum = sum(r["adaptive_krum_time_s"] for r in kt.get("per_round", [])) / max(
            1, len(kt.get("per_round", [])))
        summary["server_avg_adaptive_krum_s"] = round(avg_krum, 5)
        summary["server_peak_ram_mb"] = kt.get("ram_peak_mb")
        summary["krum_note"] = kt.get("note")

    if mode == "dp":
        dp_epsilons = [r.get("dp_epsilon_achieved") for c in run["clients"] for r in c["rounds"]]
        summary["dp_epsilon_achieved_per_round"] = dp_epsilons

    return summary


def main():
    run_dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "*")))
    summaries = []
    for d in run_dirs:
        if not os.path.isdir(d):
            continue
        run = load_run(d)
        s = summarize(run)
        if s:
            summaries.append(s)

    out_path = os.path.join(RESULTS_DIR, "CONSOLIDATED_SUMMARY.json")
    with open(out_path, "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"\n{'tag':<35} {'train_s/rnd':>12} {'send_s':>9} {'payload_B':>10} {'client_peak_MB':>16} {'notes'}")
    for s in summaries:
        peak = s["client_peak_ram_mb"]
        peak_str = "/".join(str(v) for v in peak.values())
        extra = ""
        if "server_avg_aggregate_s" in s:
            extra += f" agg={s['server_avg_aggregate_s']}s dec={s['server_avg_decrypt_s']}s"
        if "server_avg_zkp_verify_plus_threshold_s" in s:
            extra += f" zkp_srv={s['server_avg_zkp_verify_plus_threshold_s']}s"
        if "server_avg_adaptive_krum_s" in s:
            extra += f" krum={s['server_avg_adaptive_krum_s']}s"
        if s.get("avg_dp_setup_time_s_per_round") is not None:
            extra += f" dp_setup={s['avg_dp_setup_time_s_per_round']}s"
        if s["communication_failures"]:
            extra += f" [{s['communication_failures']} comm failures!]"
        print(f"{s['tag']:<35} {s['avg_train_time_s_per_round']:>12} "
              f"{s['avg_communication_send_time_s']:>9} {s['avg_payload_bytes']:>10.0f} "
              f"{peak_str:>16} {extra}")

    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
