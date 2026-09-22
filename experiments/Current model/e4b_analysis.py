#!/usr/bin/env python3
"""E4b -- DP architecture confound: verifier + paired deltas (network model).

Arms   A = BatchNorm+LSTM, no DP | B = GroupNorm+DPLSTM, no DP | C = GroupNorm+DPLSTM+DP-SGD
Deltas architecture-only = B - A ; DP-noise = C - B ; total = C - A   (paired by seed)

Usage:
  python e4b_analysis.py --root experiments/results/E4b [--legacy-b-dir path/to/archswap_results]
Outputs (in --out, default e4b_out/):
  e4b_verification.csv   per-cell validity checks (missing / header-only / manifest mismatches)
  e4b_arm_summary.csv    mean, sample SD, 95% t-CI per arm/metric
  e4b_deltas.csv         paired deltas, 95% CI, Cohen's d_z, p, Holm-adjusted p, honest NS flag
Only TEST metrics are reported; nothing here selects or tunes anything.
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

SEEDS = [42, 123, 456, 789, 2024]
MODEL = "network"
RARE = ["Ransomware", "MITM", "Vulnerability_scanner"]
ARM_TAGS = {  # search order per arm; the legacy B tag is the already-finished archswap run
    "A": ["e4b_A_bn_nodp"],
    "B": ["e4b_B_gn_nodp", "task4_fedprox_dpsafe_arch_no_dp"],
    "C": None,  # filled from --eps
}
EXPECT = {  # manifest expectations
    "A": dict(use_dp=False, dp_safe=False),
    "B": dict(use_dp=False, dp_safe=True),
    "C": dict(use_dp=True,  dp_safe=True),
}


def find(root_dirs, name):
    for r in root_dirs:
        hits = list(Path(r).rglob(name))
        if hits:
            return hits[0]
    return None


def locate(arm, seed, roots, eps):
    tags = ARM_TAGS[arm] or [f"e4b_C_gn_dp_eps{eps:g}"]
    for tag in tags:
        stem = f"{MODEL}_{tag}_seed{seed}"
        t = find(roots, f"results_{stem}_FINAL_TEST.csv")
        if t:
            return dict(tag=tag, stem=stem, test=t,
                        val=find(roots, f"results_{stem}_FINAL_VALIDATION.csv"),
                        manifest=find(roots, f"experiment_config_{stem}.json"),
                        dpjson=find(roots, f"dp_final_epsilon_{stem}.json"))
    return None


def read_one_row(p):
    if p is None or not Path(p).exists() or Path(p).stat().st_size == 0:
        return None
    df = pd.read_csv(p)
    return df.iloc[0] if len(df) else None  # header-only -> None


def verify(arm, seed, loc, mu, alpha, eps, rounds):
    rows, ok = [], True
    def add(check, passed, detail=""):
        nonlocal ok
        rows.append(dict(arm=arm, seed=seed, check=check, passed=passed, detail=detail))
        ok = ok and (passed is not False)  # None = warning, not failure
    if loc is None:
        add("run_present", False, "no FINAL_TEST file found -> NEW_RUN/RERUN")
        return rows, False
    t = read_one_row(loc["test"]); v = read_one_row(loc["val"])
    add("test_csv_has_data_row", t is not None, str(loc["test"]))
    add("validation_csv_has_data_row", v is not None)
    if t is not None:
        add("num_rounds", int(t["num_rounds"]) == rounds, f"num_rounds={t['num_rounds']}")
        add("ablation_mode_baseline_or_pure_dp", True, str(t.get("ablation_mode")))
    m = None
    if loc["manifest"]:
        m = json.load(open(loc["manifest"]))
        add("manifest_prox_mu", abs(float(m.get("prox_mu", -1)) - mu) < 1e-12, f"prox_mu={m.get('prox_mu')}")
        add("manifest_alpha", abs(float(m.get("alpha_dirichlet", -1)) - alpha) < 1e-12, f"alpha={m.get('alpha_dirichlet')}")
        add("manifest_no_attack", m.get("byzantine_attack") is False, f"byzantine_attack={m.get('byzantine_attack')}")
        add("manifest_use_dp", m.get("use_dp") == EXPECT[arm]["use_dp"], f"use_dp={m.get('use_dp')}")
        add("manifest_dp_safe_arch", m.get("dp_safe") == EXPECT[arm]["dp_safe"], f"dp_safe={m.get('dp_safe')}")
        add("manifest_dataset_edge_iiotset", m.get("dataset") == "edge_iiotset", f"dataset={m.get('dataset')}")
        add("manifest_git_clean", (m.get("git_dirty") is False) or None, f"git_sha={m.get('git_sha')} dirty={m.get('git_dirty')}")
        add("manifest_split_hash", bool(m.get("split_hash")) or False, f"split_hash={m.get('split_hash')}")
        if arm == "C":
            add("manifest_eps_target", abs(float(m.get("dp_full_run_target_epsilon", -1)) - eps) < 1e-9,
                f"target={m.get('dp_full_run_target_epsilon')}")
    else:
        add("manifest_present", None, "no experiment_config JSON (legacy run): mu/alpha/dp_safe/git SHA UNVERIFIED")
    if arm == "C":
        if loc["dpjson"]:
            d = json.load(open(loc["dpjson"]))
            e = d.get("final_total_epsilon_by_client", {})
            add("dp_all_10_clients_have_final_epsilon", len(e) == 10, f"n_clients={len(e)}")
            if e:
                vals = list(e.values())
                add("dp_final_epsilon_not_above_target", max(vals) <= eps * 1.05,
                    f"final_total_epsilon min/mean/max = {min(vals):.3f}/{np.mean(vals):.3f}/{max(vals):.3f} (target {eps})")
        else:
            add("dp_final_epsilon_json_present", False, "missing dp_final_epsilon_*.json")
    return rows, ok


def metric_frame(arm_results):
    recs = []
    for (arm, seed), loc in arm_results.items():
        t = read_one_row(loc["test"]) if loc else None
        if t is None:
            continue
        r = dict(arm=arm, seed=seed, accuracy=t["test_accuracy"], macro_f1=t["test_f1_macro"])
        for c in RARE:
            r[f"{c}_f1"] = t[f"test_f1_{c}"]; r[f"{c}_recall"] = t[f"test_recall_{c}"]; r[f"{c}_aucpr"] = t[f"test_aucpr_{c}"]
        for k in ("f1", "recall", "aucpr"):
            r[f"rare_mean_{k}"] = np.mean([r[f"{c}_{k}"] for c in RARE])
        recs.append(r)
    return pd.DataFrame(recs)


def ci95(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2: return (np.nan, np.nan)
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / math.sqrt(n)
    return x.mean() - h, x.mean() + h


def holm(p):
    p = np.asarray(p, float); idx = np.argsort(p); out = np.empty_like(p); run = 0.0
    for rank, i in enumerate(idx):
        run = max(run, (len(p) - rank) * p[i]); out[i] = min(1.0, run)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/results/E4b")
    ap.add_argument("--legacy-b-dir", default=None, help="folder holding the already-finished arm-B files")
    ap.add_argument("--out", default="e4b_out")
    ap.add_argument("--mu", type=float, default=0.005)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--eps", type=float, default=5.0)
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--strict", action="store_true", help="exclude cells failing verification from stats")
    a = ap.parse_args()
    roots = [a.root] + ([a.legacy_b_dir] if a.legacy_b_dir else [])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    locs, vrows, valid = {}, [], {}
    for arm in "ABC":
        for s in a.seeds:
            loc = locate(arm, s, roots, a.eps); locs[(arm, s)] = loc
            rows, ok = verify(arm, s, loc, a.mu, a.alpha, a.eps, a.rounds)
            vrows += rows; valid[(arm, s)] = ok
    vdf = pd.DataFrame(vrows); vdf.to_csv(out / "e4b_verification.csv", index=False)
    bad = vdf[vdf.passed == False]
    print(f"\n== Verification: {len(bad)} failed checks, {int((vdf.passed.isna()).sum())} warnings ==")
    for _, r in bad.iterrows(): print(f"  FAIL  arm {r.arm} seed {r.seed}: {r.check}  {r.detail}")
    for _, r in vdf[vdf.passed.isna()].iterrows(): print(f"  WARN  arm {r.arm} seed {r.seed}: {r.check}  {r.detail}")

    used = {k: v for k, v in locs.items() if v and (valid[k] or not a.strict)}
    mf = metric_frame(used)
    if mf.empty: print("No usable runs."); sys.exit(1)
    metrics = [c for c in mf.columns if c not in ("arm", "seed")]

    srows = []
    for arm, g in mf.groupby("arm"):
        for m in metrics:
            lo, hi = ci95(g[m]); srows.append(dict(arm=arm, metric=m, n=len(g), mean=g[m].mean(),
                                                    sd=g[m].std(ddof=1) if len(g) > 1 else np.nan, ci95_lo=lo, ci95_hi=hi))
    sdf = pd.DataFrame(srows); sdf.to_csv(out / "e4b_arm_summary.csv", index=False)

    comps = [("architecture_only (B-A)", "B", "A"), ("dp_noise (C-B)", "C", "B"), ("total (C-A)", "C", "A")]
    drows = []
    for name, x, y in comps:
        gx = mf[mf.arm == x].set_index("seed"); gy = mf[mf.arm == y].set_index("seed")
        seeds = sorted(set(gx.index) & set(gy.index)); block = []
        for m in metrics:
            if len(seeds) < 2:
                block.append(dict(comparison=name, metric=m, n_pairs=len(seeds))); continue
            d = (gx.loc[seeds, m] - gy.loc[seeds, m]).values
            lo, hi = ci95(d); sd = d.std(ddof=1)
            t, p = stats.ttest_1samp(d, 0.0) if sd > 0 else (np.nan, np.nan)
            block.append(dict(comparison=name, metric=m, n_pairs=len(seeds), mean_delta=d.mean(), sd_delta=sd,
                              ci95_lo=lo, ci95_hi=hi, cohen_dz=(d.mean() / sd if sd > 0 else np.nan), p_raw=p))
        ps = [b.get("p_raw", np.nan) for b in block]
        ok = [i for i, p in enumerate(ps) if not np.isnan(p)]
        if ok:
            adj = holm([ps[i] for i in ok])
            for j, i in enumerate(ok): block[i]["p_holm_within_comparison"] = adj[j]
        for b in block:
            ph = b.get("p_holm_within_comparison", np.nan)
            b["significant_at_0.05_holm"] = (bool(ph < 0.05) if not np.isnan(ph) else None)
            b["ci_excludes_zero"] = (bool(b["ci95_lo"] > 0 or b["ci95_hi"] < 0) if "ci95_lo" in b and not np.isnan(b["ci95_lo"]) else None)
        drows += block
    ddf = pd.DataFrame(drows); ddf.to_csv(out / "e4b_deltas.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n== Arm summary (TEST; mean +/- SD, n) ==")
    for m in ["accuracy", "macro_f1", "rare_mean_f1", "rare_mean_recall", "rare_mean_aucpr"]:
        print(f"  {m:18s}", "  ".join(f"{r.arm}: {r['mean']:.4f}+/-{r['sd']:.4f} (n={r.n})"
              for _, r in sdf[sdf.metric == m].iterrows()))
    print("\n== Paired deltas ==")
    show = ddf[ddf.metric.isin(["macro_f1", "rare_mean_f1", "rare_mean_recall", "rare_mean_aucpr"])]
    print(show[["comparison", "metric", "n_pairs", "mean_delta", "ci95_lo", "ci95_hi", "cohen_dz", "p_raw",
                "p_holm_within_comparison"]].round(4).to_string(index=False))
    print(f"\nFiles written to {out}/. Small n: 'NS' means not detectable at this n, not 'no effect'.")

if __name__ == "__main__":
    main()
