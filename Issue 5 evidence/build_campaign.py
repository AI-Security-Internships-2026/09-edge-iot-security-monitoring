#!/usr/bin/env python3
"""Build EXP1_campaign.json + completeness report + results index from an extracted results tree.
Usage: python3 -I build_campaign.py <results_root> <out_dir>
Never edits results. Every expected cell is listed, including incomplete/not-run ones."""
import os, re, sys, json, csv, hashlib, glob, itertools, collections

ROOT, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
SEEDS5 = [42, 123, 456, 789, 2024]; SEEDS3 = [42, 123, 456]

# ---- flat index of every file (basename -> [paths]) ----
files = collections.defaultdict(list)
for dp, _, fs in os.walk(ROOT):
    if dp.endswith("/E6") or "/E6" in dp: pass
    for f in fs:
        files[f].append(os.path.join(dp, f))

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def nlines(p):
    with open(p, errors="ignore") as fh: return sum(1 for _ in fh)

def find(name):
    return files.get(name, [None])[0]

def rel(p): return os.path.relpath(p, ROOT) if p else None

def check(paths_required):
    """paths_required: {label: path or None}. returns (problems, evidence)"""
    probs, ev = [], {}
    for lab, p in paths_required.items():
        if p is None: probs.append(f"missing {lab}"); continue
        if p.endswith(".csv") and nlines(p) <= 1: probs.append(f"header-only {lab}"); continue
        if os.path.getsize(p) == 0: probs.append(f"empty {lab}"); continue
        ev[lab] = {"path": rel(p), "sha256": sha(p)}
    return probs, ev

def prov(manifest_path):
    if not manifest_path: return {"manifest": None, "provenance": "NO_MANIFEST"}
    m = json.load(open(manifest_path))
    keys = ["dataset","model_type","git_sha","git_dirty","split_hash","alpha_dirichlet","dp_full_run_target_epsilon",
            "use_dp","dp_accountant","aggregator","attack_type","num_byzantine","adaptive_krum_k","prox_mu",
            "bounded_tau_override","bounded_margin","minmax_dev_type","hetero_fit_coeffs_path","num_rounds","seed"]
    d = {k: m.get(k) for k in keys if k in m}
    miss = [k for k in ("git_sha","split_hash") if not m.get(k)]
    return {"manifest": rel(manifest_path), "provenance": "COMPLETE" if not miss else "PARTIAL:" + ",".join(miss),
            "git_dirty": m.get("git_dirty"), "fields": d}

cells = []
def add(exp, cid, params, stem_files, optional=None, manifest=None, note=None):
    probs, ev = check(stem_files)
    pv = prov(manifest)
    complete = not probs
    c = {"cell_id": cid, "experiment": exp, "params": params,
         "action": "REUSE" if complete else ("RERUN" if any(not p.startswith("missing") or True for p in probs) and ev else "NEW_RUN"),
         "result_complete": complete, "problems": probs, "source_files": ev, "provenance": pv}
    if not complete and not ev: c["action"] = "NEW_RUN"
    if note: c["note"] = note
    cells.append(c)

# ---------------- E1 ----------------
E1D = next((d for d in glob.glob(f"{ROOT}/E1/*/*") if os.path.isdir(d)), None)
e1conds = ["fedavg","fedprox_mu0","fedprox_mu0.005","fedprox_mu0.02","fedprox_mu0.05","fedprox_mu0.1",
           "fedprox_dpsafe_arch_no_dp","centralized_cnnlstm","centralized_mlp","centralized_rf","centralized_xgboost"]
for model in ("network","application"):
    for cond in e1conds:
        for s in SEEDS5:
            st = f"{model}_task4_{cond}_seed{s}"
            add("E1", f"E1/{st}", {"model": model, "condition": cond, "seed": s, "dataset": "edge_iiotset"},
                {"FINAL_TEST": find(f"results_{st}_FINAL_TEST.csv"), "FINAL_VALIDATION": find(f"results_{st}_FINAL_VALIDATION.csv")})
# ---------------- E2 ----------------
KRUM_FAM = {"multi_krum","adaptive_krum","calibrated_krum"}
for agg in ["fedavg","multi_krum","median","trimmed_mean","adaptive_krum","calibrated_krum"]:
    for atk in ["sign_flip","gaussian","minmax","minsum","bounded_directional"]:
        for s in SEEDS3:
            st = f"network_e2_{agg}_{atk}_seed{s}"
            req = {"FINAL_TEST": find(f"results_{st}_FINAL_TEST.csv"), "FINAL_VALIDATION": find(f"results_{st}_FINAL_VALIDATION.csv"),
                   "per_round_results": find(f"results_{st}.csv"), "manifest": find(f"experiment_config_{st}.json")}
            if agg in KRUM_FAM: req["per_client_scores"] = find(f"per_client_krum_scores_{st}.csv")
            add("E2", f"E2/{st}", {"aggregator": agg, "attack": atk, "seed": s, "alpha": 0.7, "dp": False, "f": 2}, req, manifest=req["manifest"])
# ---------------- E3 ----------------
for agg in ("adaptive","calibrated"):
    for a in (10, 1, 0.7, 0.3, 0.1):
        for s in SEEDS5:
            st = f"network_e3_{agg}_a{a}_seed{s}_seed{s}"
            req = {"FINAL_TEST": find(f"results_{st}_FINAL_TEST.csv"), "FINAL_VALIDATION": find(f"results_{st}_FINAL_VALIDATION.csv"),
                   "per_client_scores": find(f"per_client_krum_scores_{st}.csv"), "manifest": find(f"experiment_config_{st}.json")}
            add("E3", f"E3/{st}", {"aggregator": agg+"_krum", "alpha": a, "seed": s, "attack": "minmax", "dp": False}, req, manifest=req["manifest"])
# ---------------- E4 (seed 42 only in upload) ----------------
for agg in ("adaptive","calibrated"):
    for eps in (0.2, 0.5, 1, 3, 5, 10, 15):
        for s in SEEDS3:
            tag = f"E4_{agg}_krum_eps{eps}_a0.7_minmax_seed{s}"
            pre = f"{tag}__"
            full = f"network_{tag}_seed{s}"
            req = {"FINAL_TEST": find(f"{pre}results_{full}_FINAL_TEST.csv"), "FINAL_VALIDATION": find(f"{pre}results_{full}_FINAL_VALIDATION.csv"),
                   "per_client_scores": find(f"{pre}per_client_krum_scores_{full}.csv"), "dp_final_epsilon": find(f"{pre}dp_final_epsilon_{full}.csv")}
            note = None
            if s != 42: note = "seeds 123/456 not run (no VM access) -- descoping requested from supervisor"
            c0 = len(cells)
            add("E4", f"E4/{tag}", {"aggregator": agg+"_krum", "requested_epsilon": eps, "seed": s, "alpha": 0.7, "attack": "minmax", "dp": True}, req, note=note)
            c = cells[-1]
            dpf = req["dp_final_epsilon"]
            if dpf and nlines(dpf) > 1:
                vals = [float(r["final_total_epsilon"]) for r in csv.DictReader(open(dpf))]
                c["params"]["achieved_final_total_epsilon_max"] = max(vals); c["params"]["achieved_final_total_epsilon_min"] = min(vals)
            c["provenance"] = {"manifest": None, "provenance": "NO_MANIFEST (recover from VM)"}
# ---------------- E4b ----------------
for arm in ("A_bn_nodp","B_gn_nodp","C_gn_dp_eps5"):
    for s in SEEDS5:
        st = f"network_e4b_{arm}_seed{s}"
        mp = find(f"experiment_config_{st}.json")
        req = {"FINAL_TEST": find(f"results_{st}_FINAL_TEST.csv"), "FINAL_VALIDATION": find(f"results_{st}_FINAL_VALIDATION.csv"),
               "per_client_scores": find(f"per_client_krum_scores_{st}.csv"), "manifest": mp}
        if arm.startswith("C"): req["dp_final_epsilon"] = find(f"dp_final_epsilon_{st}.csv")
        add("E4b", f"E4b/{st}", {"arm": arm, "seed": s, "dataset": "edge_iiotset"}, req, manifest=mp)
# ---------------- E5 ----------------
for k in ("2.0","2.5","3.0","3.5","4.0"):
    for agg in ("plain","calibrated"):
        for s in SEEDS5:
            st = f"network_task5_madk_{agg}_k{k}_seed{s}"
            add("E5b", f"E5b/{st}", {"mad_k": float(k), "aggregator": "adaptive_krum" if agg=="plain" else "calibrated_krum", "seed": s, "f": 2},
                {"FINAL_TEST": find(f"results_{st}_FINAL_TEST.csv"), "FINAL_VALIDATION": find(f"results_{st}_FINAL_VALIDATION.csv"),
                 "per_client_scores": find(f"per_client_krum_scores_{st}.csv")})
for f in (1, 2, 3):
    for agg in ("plain","calibrated"):
        for s in SEEDS5:
            st = f"network_task5_fsweep_{agg}_f{f}_seed{s}"
            add("E5a", f"E5a/{st}", {"f": f, "aggregator": "adaptive_krum" if agg=="plain" else "calibrated_krum", "seed": s, "mad_k": 3.5},
                {"FINAL_TEST": find(f"results_{st}_FINAL_TEST.csv"), "FINAL_VALIDATION": find(f"results_{st}_FINAL_VALIDATION.csv"),
                 "per_client_scores": find(f"per_client_krum_scores_{st}.csv")})
# ---------------- E6 ----------------
for var in ("e6_full","e6_dponly","e6_heteroonly","e6_off"):
    for s in SEEDS5:
        st = f"network_{var}_seed{s}"
        add("E6", f"E6/{st}", {"variant": var, "seed": s, "alpha": 0.3, "epsilon": 5, "attack": "bounded_directional"},
            {"FINAL_TEST": find(f"results_{st}_FINAL_TEST.csv"), "per_client_scores": find(f"per_client_krum_scores_{st}.csv"),
             "dp_final_epsilon": find(f"dp_final_epsilon_{st}.json")},
            note="Raw files not in upload; only E6_Results_Analysis.md (claims 20 runs, git f5ff72f2). Recover from VM.")
# ---------------- E7 ----------------
for m in ("fedavg","adaptive_krum","calibrated_krum"):
    for a in (1, 0.3):
        for e in (1, 5, 10):
            for atk in ("gaussian","minmax"):
                for s in SEEDS3:
                    cells.append({"cell_id": f"E7/{m}_a{a}_eps{e}_{atk}_seed{s}", "experiment": "E7",
                        "params": {"dataset": "ciciot2023", "aggregator": m, "alpha": a, "epsilon": e, "attack": atk, "seed": s},
                        "action": "NEW_RUN", "result_complete": False, "problems": ["not run"], "source_files": {},
                        "provenance": {"manifest": None, "provenance": "N/A"},
                        "note": "Not run: no compute access; descoping requested from supervisor"})
# ---------------- E8 ----------------
e8root = next((d for d in glob.glob(f"{ROOT}/E8/*/*") if os.path.isdir(d)), None)
for cfg in ("fedavg","adaptive_krum","calibrated_krum","dp_calibrated_krum"):
    for cpu in ("1.0vcpu","0.5vcpu"):
        d = os.path.join(e8root, f"{cfg}_{cpu}") if e8root else ""
        sr = os.path.join(d, f"server_{cfg}_results.json"); cs = os.path.join(d, "server_communication_summary.json")
        add("E8", f"E8/{cfg}_{cpu}", {"config": cfg, "cpu": cpu},
            {"server_results": sr if os.path.exists(sr) else None, "comm_summary": cs if os.path.exists(cs) else None,
             "client0": os.path.join(d,"client_0_results.json") if os.path.exists(os.path.join(d,"client_0_results.json")) else None})

# ---------------- outputs ----------------
summary = collections.defaultdict(lambda: collections.Counter())
for c in cells:
    summary[c["experiment"]][("COMPLETE" if c["result_complete"] else "INCOMPLETE/MISSING")] += 1
    summary[c["experiment"]]["prov:" + c["provenance"]["provenance"].split(":")[0]] += 1
camp = {"experiment": "EXP1", "issue": 24, "generated_from": "uploaded results.zip (post-hoc record; runs were launched via CLI args/shell scripts)",
        "note": "Post-hoc campaign record. action uses issue vocabulary; result_complete/problems state what actually exists. Nothing excluded.",
        "n_cells": len(cells), "summary": {k: dict(v) for k, v in summary.items()}, "cells": cells}
json.dump(camp, open(os.path.join(OUT, "EXP1_campaign.json"), "w"), indent=1)
with open(os.path.join(OUT, "completeness_report.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["cell_id","experiment","action","result_complete","provenance","git_dirty","problems","note"])
    for c in cells: w.writerow([c["cell_id"],c["experiment"],c["action"],c["result_complete"],c["provenance"]["provenance"],c["provenance"].get("git_dirty",""),"; ".join(c["problems"]),c.get("note","")])
with open(os.path.join(OUT, "RESULTS_INDEX.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["cell_id","label","path","sha256"])
    for c in cells:
        for lab, e in c["source_files"].items(): w.writerow([c["cell_id"], lab, e["path"], e["sha256"]])
print(f"{'exp':5} {'cells':>5} {'complete':>8} {'incomplete':>10}")
for k in sorted(summary):
    comp = summary[k]["COMPLETE"]; inc = summary[k]["INCOMPLETE/MISSING"]; print(f"{k:5} {comp+inc:5} {comp:8} {inc:10}   " + ", ".join(f"{a}={b}" for a,b in summary[k].items() if a.startswith('prov')))
