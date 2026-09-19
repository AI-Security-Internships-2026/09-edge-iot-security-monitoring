#!/usr/bin/env python3
"""
E6 (Table 4) summary: Honest FPR, Byzantine TPR, Macro-F1, rare-class AUC-PR
for the four calibration variants, with mean / sample SD / 95% t-CI and
seed-paired comparisons (paired t-test, Cohen's dz, Holm correction).

Reads (from --dir) the files main.py wrote for tags e6_full, e6_dponly,
e6_heteroonly, e6_off and seeds 42,123,456,789,2024:
  per_client_krum_scores_network_<tag>_seed<S>.csv
  results_network_<tag>_seed<S>_FINAL_TEST.csv
  dp_final_epsilon_network_<tag>_seed<S>.json
Honest FPR = FP/(FP+TN); Byzantine TPR = TP/(TP+FN), pooled over all rounds
of a run. Missing/empty files are reported and the run is excluded (never
silently imputed). n=5 seeds is low power: non-significant results are
reported as NS, not hidden.
"""
import argparse, csv, json, os, sys
import numpy as np
from scipy import stats

VARIANTS = [("e6_full", "Full calibration"), ("e6_dponly", "DP calibration only"),
            ("e6_heteroonly", "Heterogeneity calibration only"),
            ("e6_off", "Off (plain Adaptive Krum)")]
SEEDS = [42, 123, 456, 789, 2024]
RARE = ["Ransomware", "Vulnerability_scanner", "MITM"]     # network-model rare classes
METRICS = ["honest_fpr", "byz_tpr", "macro_f1", "rare_aucpr", "rare_f1"]


def run_metrics(d, tag, seed, tree=False):
    base = f"network_{tag}_seed{seed}"
    if tree:                       # experiments/results/E6/<seed>/<tag>/
        d = os.path.join(d, str(seed), tag)
    p_kr = os.path.join(d, f"per_client_krum_scores_{base}.csv")
    p_te = os.path.join(d, f"results_{base}_FINAL_TEST.csv")
    p_dp = os.path.join(d, f"dp_final_epsilon_{base}.json")
    for p in (p_kr, p_te, p_dp):
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            return None, f"missing/empty {os.path.basename(p)}"
    c = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    with open(p_kr, newline="") as f:
        for r in csv.DictReader(f):
            if r["classification"] in c:
                c[r["classification"]] += 1
    if c["FP"] + c["TN"] == 0 or c["TP"] + c["FN"] == 0:
        return None, "no classified rows"
    with open(p_te, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        return None, "FINAL_TEST not exactly one data row (header-only/incomplete)"
    t = rows[0]
    eps = list(json.load(open(p_dp))["final_total_epsilon_by_client"].values())
    return {
        "honest_fpr": c["FP"] / (c["FP"] + c["TN"]),
        "byz_tpr": c["TP"] / (c["TP"] + c["FN"]),
        "macro_f1": float(t["test_f1_macro"]),
        "rare_aucpr": float(np.mean([float(t[f"test_aucpr_{k}"]) for k in RARE])),
        "rare_f1": float(np.mean([float(t[f"test_f1_{k}"]) for k in RARE])),
        "final_total_epsilon_mean": float(np.mean(eps)),
        "final_total_epsilon_max": float(np.max(eps)),
    }, None


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i]); adj[i] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--out-prefix", default="E6")
    ap.add_argument("--tree", action="store_true",
                    help="--dir is experiments/results/E6 laid out as <seed>/<tag>/")
    a = ap.parse_args()

    data, problems = {}, []
    for tag, _ in VARIANTS:
        for s in SEEDS:
            m, err = run_metrics(a.dir, tag, s, a.tree)
            if m is None:
                problems.append(f"{tag} seed{s}: {err}")
            else:
                data[(tag, s)] = m
    if problems:
        print("INCOMPLETE RUNS (excluded):"); [print("  ", p) for p in problems]

    with open(f"{a.out_prefix}_per_run.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "seed"] + METRICS + ["final_total_epsilon_mean", "final_total_epsilon_max"])
        for (tag, s), m in sorted(data.items()):
            w.writerow([tag, s] + [m[k] for k in METRICS] +
                       [m["final_total_epsilon_mean"], m["final_total_epsilon_max"]])

    print("\nTable 4 -- mean +/- SD [95% CI], n seeds")
    rows_out = []
    for tag, label in VARIANTS:
        seeds = [s for s in SEEDS if (tag, s) in data]
        line = [f"{label:<32} n={len(seeds)}"]
        rec = {"variant": tag, "n": len(seeds)}
        for k in METRICS:
            v = np.array([data[(tag, s)][k] for s in seeds])
            if len(v) >= 2:
                sd = v.std(ddof=1); h = stats.t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v))
                line.append(f"{k}={v.mean():.4f}+/-{sd:.4f}[{v.mean()-h:.4f},{v.mean()+h:.4f}]")
                rec.update({f"{k}_mean": v.mean(), f"{k}_sd": sd, f"{k}_ci_lo": v.mean() - h, f"{k}_ci_hi": v.mean() + h})
        rows_out.append(rec); print("  " + "  ".join(line))
    keys = sorted({k for r in rows_out for k in r})
    with open(f"{a.out_prefix}_table4.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows_out)

    pairs = [("e6_full", "e6_off"), ("e6_full", "e6_dponly"), ("e6_full", "e6_heteroonly"),
             ("e6_dponly", "e6_off"), ("e6_heteroonly", "e6_off"), ("e6_dponly", "e6_heteroonly")]
    print("\nSeed-paired comparisons (A - B), Holm-corrected within each metric:")
    comp = []
    for k in METRICS:
        res = []
        for A, B in pairs:
            common = [s for s in SEEDS if (A, s) in data and (B, s) in data]
            if len(common) < 3:
                res.append((A, B, len(common), np.nan, np.nan, np.nan, np.nan)); continue
            d = np.array([data[(A, s)][k] - data[(B, s)][k] for s in common])
            if np.allclose(d, 0):
                res.append((A, B, len(common), 0.0, 0.0, 1.0, np.nan)); continue
            tt = stats.ttest_1samp(d, 0.0)
            dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan
            res.append((A, B, len(common), d.mean(), dz, tt.pvalue, np.nan))
        ps = np.array([r[5] for r in res]); ok = ~np.isnan(ps)
        adj = np.full(len(ps), np.nan); adj[ok] = holm(ps[ok])
        for r, pa in zip(res, adj):
            sig = "NS" if (np.isnan(pa) or pa >= 0.05) else "*"
            print(f"  {k:<11} {r[0]:<14} - {r[1]:<14} n={r[2]} diff={r[3]:+.4f} dz={r[4]:+.2f} p={r[5]:.4f} p_holm={pa:.4f} {sig}")
            comp.append([k, r[0], r[1], r[2], r[3], r[4], r[5], pa, sig])
    with open(f"{a.out_prefix}_paired.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric", "A", "B", "n", "mean_diff", "cohens_dz", "p", "p_holm", "sig"]); w.writerows(comp)

    bad = [(k, m["final_total_epsilon_max"]) for k, m in data.items() if m["final_total_epsilon_max"] > 5.0 * 1.02]
    if bad:
        print("\nWARNING: final_total_epsilon exceeds target 5 by >2% in:", bad)
    print(f"\nWrote {a.out_prefix}_per_run.csv, {a.out_prefix}_table4.csv, {a.out_prefix}_paired.csv")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
