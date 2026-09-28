#!/usr/bin/env python3
"""
build_e3_table3_figure3.py

Reads the 200 collected files from the E3 campaign (per_client_krum_scores
+ results_*_FINAL_TEST, one set per (aggregator, alpha, seed)) and produces:

  - table3_heterogeneity_dp_sensitivity.csv
      Issue 5's Table 3: Honest FPR, Byzantine TPR, Macro-F1, rare-class
      AUC-PR, mean +/- 95% CI (paired t, df=n_seeds-1) per
      (aggregator, alpha) cell, plus the raw per-seed values for
      traceability (Task 6: every value must trace to its run).
  - figure3_honest_fpr_vs_alpha.png
      Issue 5's Figure 3: Honest-Client False Positive Rejection Rate
      vs Dirichlet alpha, both aggregators, with 95% CI error bars.
  - figure_bonus_macrof1_vs_alpha.png
      Not one of the ticket's named figures, but included because
      Macro-F1 showed a far more monotonic, lower-variance trend
      across alpha than TPR/FPR in this campaign's data -- useful as
      a supporting panel even though it's not Figure 3 itself.

Usage:
    python build_e3_table3_figure3.py /path/to/e3_collected_outputs
"""

import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHAS = ["10", "1", "0.7", "0.3", "0.1"]
ALPHA_X = [10, 1, 0.7, 0.3, 0.1]  # numeric, for plotting on a log axis
SEEDS = [42, 123, 456, 789, 2024]
AGGS = ["adaptive", "calibrated"]
AGG_LABEL = {"adaptive": "Adaptive Krum", "calibrated": "Calibrated Krum"}
RARE_CLASSES = ["Ransomware", "MITM", "Vulnerability_scanner"]  # network-domain rare classes
T_CRIT_DF4 = 2.776  # t(0.975, df=4), n=5 seeds


def tag(agg, a, s):
    return f"e3_{agg}_a{a}_seed{s}"


def tpr_fpr(indir, agg, a, s):
    fn = os.path.join(indir, f"per_client_krum_scores_network_{tag(agg,a,s)}_seed{s}.csv")
    rows = list(csv.DictReader(open(fn)))
    byz = [r for r in rows if r["ground_truth_client_label"] == "byzantine"]
    hon = [r for r in rows if r["ground_truth_client_label"] == "honest"]
    tp = sum(1 for r in byz if r["classification"] == "TP")
    fp = sum(1 for r in hon if r["classification"] == "FP")
    tpr = (tp / len(byz)) * 100 if byz else float("nan")
    fpr = (fp / len(hon)) * 100 if hon else float("nan")
    return tpr, fpr


def test_metrics(indir, agg, a, s):
    fn = os.path.join(indir, f"results_network_{tag(agg,a,s)}_seed{s}_FINAL_TEST.csv")
    row = list(csv.DictReader(open(fn)))[0]
    macro_f1 = float(row["test_f1_macro"])
    rare_aucpr = float(np.mean([float(row[f"test_aucpr_{c}"]) for c in RARE_CLASSES]))
    return macro_f1, rare_aucpr


def mean_ci95(vals):
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    m = float(vals.mean())
    sd = float(vals.std(ddof=1)) if n > 1 else 0.0
    half = T_CRIT_DF4 * sd / np.sqrt(n) if n > 1 else 0.0
    return m, sd, half


def collect(indir):
    results = {}
    for agg in AGGS:
        for a in ALPHAS:
            tprs, fprs, f1s, aucprs = [], [], [], []
            for s in SEEDS:
                tpr, fpr = tpr_fpr(indir, agg, a, s)
                f1, aucpr = test_metrics(indir, agg, a, s)
                tprs.append(tpr); fprs.append(fpr); f1s.append(f1); aucprs.append(aucpr)
            results[(agg, a)] = dict(tpr=tprs, fpr=fprs, f1=f1s, aucpr=aucprs)
    return results


def write_table3(results, out_csv):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "alpha", "aggregator",
            "tpr_mean", "tpr_ci95_halfwidth",
            "honest_fpr_mean", "honest_fpr_ci95_halfwidth",
            "macro_f1_mean", "macro_f1_ci95_halfwidth",
            "rare_class_aucpr_mean", "rare_class_aucpr_ci95_halfwidth",
            "n_seeds",
        ] + [f"tpr_seed{s}" for s in SEEDS]
          + [f"honest_fpr_seed{s}" for s in SEEDS]
          + [f"macro_f1_seed{s}" for s in SEEDS]
          + [f"rare_class_aucpr_seed{s}" for s in SEEDS])
        for a in ALPHAS:
            for agg in AGGS:
                r = results[(agg, a)]
                tpr_m, _, tpr_ci = mean_ci95(r["tpr"])
                fpr_m, _, fpr_ci = mean_ci95(r["fpr"])
                f1_m, _, f1_ci = mean_ci95(r["f1"])
                ap_m, _, ap_ci = mean_ci95(r["aucpr"])
                w.writerow([
                    a, AGG_LABEL[agg],
                    round(tpr_m, 2), round(tpr_ci, 2),
                    round(fpr_m, 2), round(fpr_ci, 2),
                    round(f1_m, 4), round(f1_ci, 4),
                    round(ap_m, 4), round(ap_ci, 4),
                    len(SEEDS),
                ] + [round(v, 2) for v in r["tpr"]]
                  + [round(v, 2) for v in r["fpr"]]
                  + [round(v, 4) for v in r["f1"]]
                  + [round(v, 4) for v in r["aucpr"]])
    print(f"Wrote {out_csv}")


def plot_metric(results, metric_key, ylabel, title, out_png, pct=True):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"adaptive": "#1f77b4", "calibrated": "#d62728"}
    offset = {"adaptive": -0.03, "calibrated": 0.03}  # tiny x-jitter so error bars don't overlap
    for agg in AGGS:
        means, cis = [], []
        for a in ALPHAS:
            m, _, ci = mean_ci95(results[(agg, a)][metric_key])
            means.append(m)
            cis.append(ci)
        x = np.array(ALPHA_X) * (1 + offset[agg])
        ax.errorbar(x, means, yerr=cis, marker="o", capsize=4,
                    label=AGG_LABEL[agg], color=colors[agg], linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xticks(ALPHA_X)
    ax.set_xticklabels([str(a) for a in ALPHA_X])
    ax.set_xlabel("Dirichlet alpha (non-IID heterogeneity)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    if pct:
        ax.set_ylim(-5, 105)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_png}")


def main():
    indir = sys.argv[1] if len(sys.argv) > 1 else "e3_collected_outputs"
    results = collect(indir)

    write_table3(results, "table3_heterogeneity_dp_sensitivity.csv")
    plot_metric(results, "fpr", "Honest-Client False Positive Rate (%)",
                "Figure 3: Honest-Client FPR vs Dirichlet alpha",
                "figure3_honest_fpr_vs_alpha.png")
    plot_metric(results, "macro_f1" if False else "f1", "Macro-F1 (test)",
                "Macro-F1 vs Dirichlet alpha (supporting panel, not Figure 3)",
                "figure_bonus_macrof1_vs_alpha.png", pct=False)

    # Console summary: is TPR's per-seed variance really that much
    # worse than Macro-F1's, quantified rather than eyeballed.
    print("\nCoefficient of variation (sd/mean) per metric, pooled across all "
          "10 (agg,alpha) cells -- higher = noisier / less reliable signal:")
    for label, key in [("TPR", "tpr"), ("Honest FPR", "fpr"),
                        ("Macro-F1", "f1"), ("Rare-class AUC-PR", "aucpr")]:
        cvs = []
        for agg in AGGS:
            for a in ALPHAS:
                vals = np.asarray(results[(agg, a)][key], dtype=float)
                if vals.mean() != 0:
                    cvs.append(vals.std(ddof=1) / abs(vals.mean()))
        print(f"  {label:>18}: mean CV = {np.mean(cvs):.2f}")


if __name__ == "__main__":
    main()
