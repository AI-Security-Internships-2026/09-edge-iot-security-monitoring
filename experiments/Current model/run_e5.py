#!/usr/bin/env python3
"""
E5 campaign driver (Issue 5 / EXP1, Task 1 + Task 3 E5a/E5b + Task 6).

  python scripts/run_e5.py plan       # reuse audit -> rewrites cells[] in E5_campaign.json
  python scripts/run_e5.py run        # launches only NEW_RUN / RERUN cells
  python scripts/run_e5.py summarize  # tables + paired stats from VALID cells only

Design points
  * E5a (f in {1,2,3} at k=default) and E5b (k in {...} at f=default) share the
    (f=2,k=2.5) cell, so it is run once and used by both -> 7 unique configs
    per aggregator per seed.
  * REUSE is decided by CONTENT (manifest fields), not by tag, so valid runs
    from Issues #20-#23 in `scan_dirs` are picked up automatically.
  * main.py writes into its own cwd with tag-derived names; after a run the
    files are moved into <results_root>/<seed>/<tag>/ (E1-E8 layout).
"""
import argparse, csv, glob, json, math, os, re, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CFG_PATH_DEFAULT = "experiments/configs/E5_campaign.json"
RARE_HINTS = ["xss", "ransom", "mitm", "fingerprint", "vulnerab", "scanner"]


# ----------------------------------------------------------------- config / cells
def load_cfg(p):
    return json.load(open(p))

def fmt_k(k):
    return str(k).replace(".", "p")

def unique_cells(cfg):
    """One dict per unique (aggregator, f, k, seed); 'roles' says which sub-experiment uses it."""
    df, dk = cfg["default_f"], cfg["default_k"]
    cells = {}
    def add(agg, f, k, seed, role):
        key = (agg, f, k, seed)
        c = cells.setdefault(key, dict(aggregator=agg, f=f, k=k, seed=seed, roles=[]))
        if role not in c["roles"]:
            c["roles"].append(role)
    for seed in cfg["seeds"]:
        for agg in cfg["aggregators"]:
            for f in cfg["f_sweep"]:
                add(agg, f, dk, seed, "E5a")
            for k in cfg["k_sweep"]:
                add(agg, df, k, seed, "E5b")
    out = []
    for c in cells.values():
        c["tag"] = f"E5_f{c['f']}_k{fmt_k(c['k'])}_{c['aggregator']}"
        out.append(c)
    return sorted(out, key=lambda c: (c["seed"], c["aggregator"], c["f"], c["k"]))

def mode_for(cond, agg):
    if cond["dp"]:
        return "calibrated_krum_dp_sweep" if agg == "calibrated_krum" else "krum_dp_sweep"
    return "krum_baseline"

def build_cmd(cfg, cell, main_py):
    cond = cfg["condition"]
    byz = ",".join(str(i) for i in range(1, cell["f"] + 1))   # 1-indexed for --byzantine
    cmd = [sys.executable, main_py, cond["model_type"],
           "--dataset", cond["dataset"],
           "--ablation-mode", mode_for(cond, cell["aggregator"]),
           "--aggregator", cell["aggregator"],
           "--attack-type", cond["attack_type"],
           "--alpha", str(cond["alpha"]),
           "--byzantine", byz,
           "--krum-k", str(cell["k"]),
           "--seed", str(cell["seed"]),
           "--rounds", str(cond["rounds"]),
           "--tag", cell["tag"]]
    if cond["dp"]:
        cmd += ["--epsilon", str(cond["epsilon"])]
    for k, v in (cond.get("attack_args") or {}).items():
        if v is None or v is False:
            continue
        flag = "--" + k.replace("_", "-")
        cmd += [flag] if v is True else [flag, str(v)]
    if cell["aggregator"] == "calibrated_krum" and cond.get("hetero_fit_coeffs_json"):
        cmd += ["--hetero-fit-coeffs-json", os.path.abspath(cond["hetero_fit_coeffs_json"])]
    # bounded_directional needs the live-guard-independent online tau; see caveat in README notes
    return cmd

def expected_manifest(cfg, cell):
    c = cfg["condition"]
    e = {"model_type": c["model_type"], "dataset": c["dataset"],
         "alpha_dirichlet": c["alpha"], "attack_type": c["attack_type"],
         "num_byzantine": cell["f"], "byzantine_clients": list(range(cell["f"])),
         "adaptive_krum_k": cell["k"], "aggregator": cell["aggregator"],
         "use_dp": bool(c["dp"]), "num_rounds": c["rounds"],
         "byzantine_attack": True, "sanity_check": False}
    if c["dp"]:
        e["dp_full_run_target_epsilon"] = c["epsilon"]
    for k, v in (c.get("attack_args") or {}).items():
        if v is not None and k in ("minmax_dev_type", "minmax_search_iters", "minmax_gamma_init",
                                   "bounded_tau_override", "bounded_margin", "bounded_direction"):
            e[k if k != "bounded_tau" else "bounded_tau_override"] = v
    if "bounded_tau" in (c.get("attack_args") or {}):
        e["bounded_tau_override"] = c["attack_args"]["bounded_tau"]
    if cell["aggregator"] == "calibrated_krum":
        e.update(use_dp_calibration=True, use_hetero_calibration=True,
                 hetero_fit_coeffs_active=True)
    return e

def _eq(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(a - b) < 1e-9
    return a == b


# ----------------------------------------------------------------- validity audit
def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def _fin(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False

def audit_run(manifest_path, cfg):
    """Return (ok, problems, info). Implements Task-1's reuse conditions as far as they are checkable from files."""
    mp = Path(manifest_path)
    stem = mp.name[len("experiment_config_"):-len(".json")]
    d = mp.parent
    problems, info = [], {"stem": stem, "dir": str(d)}
    m = json.load(open(mp))
    info["manifest"] = m
    if not m.get("split_hash"):
        problems.append("split_hash missing/None (crashed run or pre-Issue-5-Task-6 manifest)")
    if not m.get("git_sha"):
        problems.append("git_sha missing")
    if m.get("git_dirty"):
        info["warn_git_dirty"] = True
    ft = d / f"results_{stem}_FINAL_TEST.csv"
    if not ft.exists():
        problems.append("FINAL_TEST csv missing")
    else:
        r = _rows(ft)
        if not r or not _fin(r[0].get("test_f1_macro")):
            problems.append("FINAL_TEST csv header-only or non-finite macro-F1")
        else:
            info["test_row"] = r[0]
    pc = d / f"per_client_krum_scores_{stem}.csv"
    if not pc.exists():
        problems.append("per_client_krum_scores csv missing")
    else:
        r = _rows(pc)
        rounds = {x["round_id"] for x in r}
        if not r:
            problems.append("per_client_krum_scores header-only")
        elif len(rounds) < m.get("num_rounds", 0):
            problems.append(f"per-client log covers {len(rounds)}/{m.get('num_rounds')} rounds")
        info["per_client_rows"] = r
    if m.get("use_dp"):
        dpf = d / f"dp_final_epsilon_{stem}.csv"
        if not dpf.exists():
            problems.append("dp_final_epsilon csv missing (not post-Issue-#21 accounting?)")
        else:
            r = _rows(dpf)
            vals = [float(x["final_total_epsilon"]) for x in r if _fin(x.get("final_total_epsilon"))]
            if not vals:
                problems.append("no finite final_total_epsilon")
            else:
                info["final_total_epsilon"] = vals
        if m.get("dp_accountant") != "rdp" or "dp_full_run_target_epsilon" not in m:
            problems.append("manifest lacks post-PRV1 dp fields")
    sm = re.search(r"_seed(\d+)$", stem)
    info["seed"] = int(sm.group(1)) if sm else None
    return (not problems), problems, info

def index_manifests(cfg):
    roots = [cfg["results_root"]] + list(cfg.get("scan_dirs") or [])
    seen, out = set(), []
    for r in roots:
        for p in glob.glob(os.path.join(r, "**", "experiment_config_*_seed*.json"), recursive=True):
            rp = os.path.realpath(p)
            if rp not in seen:
                seen.add(rp); out.append(p)
    return out

def classify(cfg):
    cells = unique_cells(cfg)
    manifests = index_manifests(cfg)
    loaded = []
    for p in manifests:
        try:
            m = json.load(open(p))
            sm = re.search(r"_seed(\d+)\.json$", p)
            loaded.append((p, m, int(sm.group(1)) if sm else None))
        except Exception:
            continue
    for c in cells:
        exp = expected_manifest(cfg, c)
        cands = [(p, m) for p, m, s in loaded
                 if s == c["seed"] and all(k in m and _eq(m[k], v) for k, v in exp.items())]
        valid, bad = [], []
        for p, m in cands:
            ok, probs, info = audit_run(p, cfg)
            (valid if ok else bad).append((p, probs))
        if valid:
            valid.sort(key=lambda t: os.path.getmtime(t[0]))
            c["action"], c["source"], c["problems"] = "REUSE", valid[-1][0], []
        elif bad:
            c["action"], c["source"], c["problems"] = "RERUN", bad[-1][0], bad[-1][1]
        else:
            c["action"], c["source"], c["problems"] = "NEW_RUN", None, []
    return cells


# ----------------------------------------------------------------- commands
def cmd_plan(cfg, path):
    cells = classify(cfg)
    for c in cells:
        c["dir"] = os.path.join(cfg["results_root"], str(c["seed"]), c["tag"])
    cfg["cells"] = cells
    json.dump(cfg, open(path, "w"), indent=2)
    from collections import Counter
    print(Counter(c["action"] for c in cells))
    print("\nExperiment cell | Existing | Valid? | Action")
    for c in cells:
        ex = "Yes" if c["source"] else "No"
        va = "Yes" if c["action"] == "REUSE" else ("Incomplete/mismatch" if c["action"] == "RERUN" else "-")
        print(f"{'+'.join(c['roles'])}, f={c['f']}, k={c['k']}, seed{c['seed']}, {c['aggregator']} | {ex} | {va} | {c['action']}"
              + (f"  ({'; '.join(c['problems'])})" if c["problems"] else ""))
    print(f"\nWrote {path}")

def run_one(cfg, cell, main_py):
    main_dir = os.path.dirname(os.path.abspath(main_py))
    cmd = build_cmd(cfg, cell, os.path.abspath(main_py))
    dest = Path(cfg["results_root"]) / str(cell["seed"]) / cell["tag"]
    dest.mkdir(parents=True, exist_ok=True)
    log = dest / "stdout.log"
    with open(log, "w") as lf:
        lf.write(" ".join(cmd) + "\n\n"); lf.flush()
        rc = subprocess.call(cmd, cwd=main_dir, stdout=lf, stderr=subprocess.STDOUT)
    stem = f"{cfg['condition']['model_type']}_{cell['tag']}_seed{cell['seed']}"
    for p in glob.glob(os.path.join(main_dir, f"*_{stem}[._]*")):
        shutil.move(p, dest / os.path.basename(p))
    return cell["tag"], cell["seed"], rc

def cmd_run(cfg, path, args):
    if not cfg.get("frozen") and not args.allow_unfrozen:
        sys.exit("REFUSING TO RUN: config 'frozen' is false. Freeze attack/condition on VALIDATION first "
                 "(or pass --allow-unfrozen for a pilot; pilot results must not be reported).")
    cond = cfg["condition"]
    if cond["attack_type"] == "bounded_directional" and not args.allow_unfrozen \
            and cond["attack_args"].get("bounded_tau") is None:
        sys.exit("REFUSING TO RUN: bounded_directional needs a frozen numeric bounded_tau "
                 "(derive with scripts/freeze_bounded_tau.py from a VALIDATION-only pilot).")
    if cfg["condition"]["attack_type"] in ("minmax", "minsum") and cfg["condition"]["attack_args"].get("minmax_gamma_init") is None:
        print("NOTE: minmax_gamma_init is null -> main.py auto-picks 5x coalition spread; "
              "record the value you froze if you tuned it (check_attack_difficulty.py).")
    cells = classify(cfg)
    if args.shard:
        si, sn = (int(x) for x in args.shard.split("/"))
        assert 0 <= si < sn, "--shard must be i/n with 0 <= i < n"
        # shard on the position in the FULL, stable cell list (not on the todo list) so that
        # sessions started at different times never overlap or skip cells
        cells = [c for i, c in enumerate(cells) if i % sn == si]
        print(f"[shard {si}/{sn}] owns {len(cells)} cells")
    todo = [c for c in cells if c["action"] in ("NEW_RUN", "RERUN")]
    print(f"{len(todo)} runs to launch, {len(cells)-len(todo)} reused.")
    if args.dry_run:
        for c in todo:
            print(" ".join(build_cmd(cfg, c, cfg["main_py"])))
        return
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = [ex.submit(run_one, cfg, c, cfg["main_py"]) for c in todo]
        for f in futs:
            tag, seed, rc = f.result()
            print(f"  {'OK ' if rc == 0 else 'FAIL'} {tag} seed{seed} rc={rc}")
    if args.shard:
        print("Shard finished. Run `python scripts/run_e5.py plan` once ALL shards are done to re-audit.")
    else:
        cmd_plan(cfg, path)   # re-audit so cells[] reflects what actually validated

# ----------------------------------------------------------------- summarize
def detection(rows):
    tp = sum(r["classification"] == "TP" for r in rows); fn = sum(r["classification"] == "FN" for r in rows)
    fp = sum(r["classification"] == "FP" for r in rows); tn = sum(r["classification"] == "TN" for r in rows)
    return (tp / (tp + fn) if tp + fn else float("nan"),
            fp / (fp + tn) if fp + tn else float("nan"))

def rare_cols(test_row, prefix):
    return [k for k in test_row if k.startswith(prefix) and any(h in k.lower() for h in RARE_HINTS)]

def mean_finite(vals):
    v = [float(x) for x in vals if _fin(x)]
    return sum(v) / len(v) if v else float("nan")

def cmd_summarize(cfg, args):
    import numpy as np
    from scipy import stats
    cells = [c for c in classify(cfg) if c["action"] == "REUSE"]
    if not cells:
        sys.exit("No valid cells found. Run `plan`/`run` first.")
    recs = []
    for c in cells:
        ok, _, info = audit_run(c["source"], cfg)
        tpr, fpr = detection(info["per_client_rows"])
        t = info["test_row"]
        rc_a, rc_r = rare_cols(t, "test_aucpr_"), rare_cols(t, "test_recall_")
        recs.append(dict(
            aggregator=c["aggregator"], f=c["f"], k=c["k"], seed=c["seed"], roles="+".join(c["roles"]),
            byz_tpr=tpr, honest_fpr=fpr, macro_f1=float(t["test_f1_macro"]),
            rare_aucpr=mean_finite([t[x] for x in rc_a]), rare_recall=mean_finite([t[x] for x in rc_r]),
            rare_classes_used=";".join(x.replace("test_aucpr_", "") for x in rc_a),
            final_total_epsilon_max=max(info["final_total_epsilon"]) if info.get("final_total_epsilon") else "",
            git_sha=info["manifest"].get("git_sha"), split_hash=info["manifest"].get("split_hash"),
            source=c["source"]))
    out = Path(cfg["results_root"]); out.mkdir(parents=True, exist_ok=True)
    keys = list(recs[0].keys())
    with open(out / "E5_per_run.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(recs)

    metrics = ["honest_fpr", "byz_tpr", "macro_f1", "rare_aucpr"]
    def ci95(x):
        x = np.asarray(x, float); n = len(x)
        if n < 2: return float("nan")
        return float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / math.sqrt(n))
    tables, tests = [], []
    for sweep, var, vals, fixed_key, fixed_val in (
            ("E5a", "f", cfg["f_sweep"], "k", cfg["default_k"]),
            ("E5b", "k", cfg["k_sweep"], "f", cfg["default_f"])):
        for v in vals:
            sel = {a: sorted([r for r in recs if r["aggregator"] == a and r[var] == v and r[fixed_key] == fixed_val],
                             key=lambda r: r["seed"]) for a in cfg["aggregators"]}
            row = dict(sweep=sweep, **{var: v})
            for a, rs in sel.items():
                for m in metrics:
                    x = [r[m] for r in rs if _fin(r[m])]
                    row[f"{a}_{m}_mean"] = float(np.mean(x)) if x else float("nan")
                    row[f"{a}_{m}_sd"] = float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")
                    row[f"{a}_{m}_ci95"] = ci95(x) if x else float("nan")
                row[f"{a}_n"] = len(rs)
            tables.append(row)
            A, B = "adaptive_krum", "calibrated_krum"
            common = sorted(set(r["seed"] for r in sel.get(A, [])) & set(r["seed"] for r in sel.get(B, [])))
            for m in metrics:
                a = [next(r[m] for r in sel[A] if r["seed"] == s) for s in common]
                b = [next(r[m] for r in sel[B] if r["seed"] == s) for s in common]
                pairs = [(x, y) for x, y in zip(a, b) if _fin(x) and _fin(y)]
                if len(pairs) < 3:
                    tests.append(dict(sweep=sweep, **{var: v}, metric=m, n_pairs=len(pairs), note="too few paired seeds"))
                    continue
                d = np.array([y - x for x, y in pairs])          # calibrated - adaptive
                sd = d.std(ddof=1)
                if sd == 0:
                    t_p, dz = (1.0 if d.mean() == 0 else 0.0), (0.0 if d.mean() == 0 else float("inf"))
                else:
                    t_p = float(stats.ttest_1samp(d, 0.0).pvalue); dz = float(d.mean() / sd)
                try:
                    w_p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
                except ValueError:
                    w_p = float("nan")
                tests.append(dict(sweep=sweep, **{var: v}, metric=m, n_pairs=len(d),
                                  mean_delta_cal_minus_adapt=float(d.mean()), ci95_delta=ci95(d),
                                  cohens_dz=dz, p_paired_t=t_p, p_wilcoxon=w_p))
    # Holm correction within each metric family
    for m in metrics:
        fam = [t for t in tests if t["metric"] == m and "p_paired_t" in t]
        order = sorted(range(len(fam)), key=lambda i: fam[i]["p_paired_t"])
        prev, n = 0.0, len(fam)
        for rank, i in enumerate(order):
            adj = min(1.0, (n - rank) * fam[i]["p_paired_t"]); prev = max(prev, adj)
            fam[i]["p_holm"] = prev
            fam[i]["significant_holm_0.05"] = bool(prev < 0.05)
    def dump(name, rows):
        ks = list(dict.fromkeys(k for r in rows for k in r))
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, ks); w.writeheader(); w.writerows(rows)
    dump("E5_summary_table.csv", tables); dump("E5_paired_tests.csv", tests)
    write_param_sensitivity(cfg, recs, ci95)
    print(f"Wrote {out}/E5_per_run.csv, E5_summary_table.csv, E5_paired_tests.csv "
          f"({len(recs)} valid runs; tests are seed-paired, n<=5, Holm within metric; NS is reported, not hidden).")
    print("Sweep points with missing pairs:",
          [f"{t['sweep']} {t.get('f', t.get('k'))} {t['metric']}" for t in tests if "note" in t] or "none")


def write_param_sensitivity(cfg, recs, ci95):
    """Issue #23 Task 5 Table A (MAD-k) and Table B (Byzantine f), 5-seed mean +/- 95% CI, with seed counts."""
    import numpy as np
    d = Path(cfg.get("param_sensitivity_dir", "experiments/results/param_sensitivity")); d.mkdir(parents=True, exist_ok=True)
    A, B = "calibrated_krum", "adaptive_krum"
    def pick(agg, seeds, **kw):
        return {r["seed"]: r for r in recs if r["aggregator"] == agg and r["seed"] in seeds
                and all(r[k] == v for k, v in kw.items())}
    def ms(vals):
        v = [x for x in vals if _fin(x)]
        return (float(np.mean(v)) if v else float("nan"), ci95(v) if len(v) > 1 else float("nan"), len(v))
    rowsA, rowsB = [], []
    for k in cfg["k_sweep"]:
        a, b = pick(A, {r["seed"] for r in recs}, f=cfg["default_f"], k=k), pick(B, {r["seed"] for r in recs}, f=cfg["default_f"], k=k)
        s_ = sorted(set(a) & set(b))
        cf, pf = ms([a[i]["honest_fpr"] for i in s_]), ms([b[i]["honest_fpr"] for i in s_])
        dl = ms([b[i]["honest_fpr"] - a[i]["honest_fpr"] for i in s_])   # positive = Calibrated better
        rowsA.append({"k": k, "n_paired_seeds": len(s_), "calibrated_honest_fpr_mean": cf[0], "calibrated_honest_fpr_ci95": cf[1],
                      "plain_adaptive_honest_fpr_mean": pf[0], "plain_adaptive_honest_fpr_ci95": pf[1],
                      "delta_fpr_mean": dl[0], "delta_fpr_ci95": dl[1],
                      "delta_fpr_ge_0": bool(dl[0] >= 0) if dl[2] else None})
    for f in cfg["f_sweep"]:
        a, b = pick(A, {r["seed"] for r in recs}, f=f, k=cfg["default_k"]), pick(B, {r["seed"] for r in recs}, f=f, k=cfg["default_k"])
        s_ = sorted(set(a) & set(b))
        r_ = {}
        for nm, m, src in (("calibrated_byz_tpr", "byz_tpr", a), ("plain_adaptive_byz_tpr", "byz_tpr", b),
                           ("calibrated_honest_fpr", "honest_fpr", a), ("plain_adaptive_honest_fpr", "honest_fpr", b)):
            mu, ci, n = ms([src[i][m] for i in s_]); r_[nm + "_mean"], r_[nm + "_ci95"] = mu, ci
        gap = r_["plain_adaptive_byz_tpr_mean"] - r_["calibrated_byz_tpr_mean"]
        rowsB.append({"f": f, "n_paired_seeds": len(s_), **r_,
                      "honest_fpr_dominance": bool(r_["calibrated_honest_fpr_mean"] < r_["plain_adaptive_honest_fpr_mean"]) if s_ else None,
                      "tpr_gap_pp_plain_minus_cal": 100 * gap,
                      "tpr_within_5pp": bool(abs(gap) <= 0.05) if (s_ and f in (1, 2)) else None})
    for name, rows in (("table_A_mad_k.csv", rowsA), ("table_B_byzantine_f.csv", rowsB)):
        with open(d / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"Wrote {d}/table_A_mad_k.csv, table_B_byzantine_f.csv. "
          "Requirement flags are REPORTED, never enforced by altering parameters; violations stay in the table.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["plan", "run", "summarize"])
    ap.add_argument("--config", default=CFG_PATH_DEFAULT)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shard", default=None, help="i/n: run only cells with index %% n == i (0-based)")
    ap.add_argument("--allow-unfrozen", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    {"plan": lambda: cmd_plan(cfg, a.config), "run": lambda: cmd_run(cfg, a.config, a),
     "summarize": lambda: cmd_summarize(cfg, a)}[a.action]()

if __name__ == "__main__":
    main()
