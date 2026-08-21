#!/usr/bin/env python3
"""
plot_epsilon_sweep.py
======================
The open item flagged since v4 of the master doc. Isolates the single
question Experiment 1 exists to answer — does DP noise erode Krum's
Byzantine-detection separation as epsilon shrinks? — into one table and
one chart per model, now that the 1-increment sweep gives 16 points
(0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15) instead of
Experiment 1's original 3 (3, 9, 15).

Reads sweep_summary.csv (produced by build_sweep_table.py — run that
FIRST). Does not re-parse the raw results_*.csv files itself, so it
stays in sync with however build_sweep_table.py defines "mean over the
run" for detection rate / score ratio.

Outputs (per model, network + application):
  - krum_vs_epsilon_{model}.csv    — epsilon, detection_rate, score_ratio,
                                       best_f1_macro, dp_epsilon_achieved,
                                       one row per epsilon, sorted ascending
  - krum_vs_epsilon_{model}.png    — two-panel chart: detection rate (top,
                                       0-100%) and score ratio (bottom) vs
                                       epsilon, with Experiment 1's original
                                       three anchor points (3, 9, 15) marked
                                       distinctly from the new fill-in points

Also prints a plain-text verdict: whether detection rate or score ratio
show ANY epsilon-dependence across the full grid, since that's the
actual research question, not just a plot to eyeball.

Usage:
    python3 build_sweep_table.py      # must run first
    python3 plot_epsilon_sweep.py
    python3 plot_epsilon_sweep.py --summary sweep_summary.csv --out .
"""

import argparse
import csv
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")  # headless — no display on the DGX
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


ORIGINAL_EXPERIMENT_1_EPSILONS = {3.0, 9.0, 15.0}


def load_summary(path):
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run build_sweep_table.py first.", file=sys.stderr)
        sys.exit(1)
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def to_float(v):
    try:
        if v in (None, "", "N/A"):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def build_model_table(rows, model_type):
    filtered = [r for r in rows if r.get("model_type") == model_type]
    table = []
    for r in filtered:
        eps = to_float(r.get("dp_epsilon_target"))
        if eps is None:
            continue
        table.append({
            "epsilon_target": eps,
            "epsilon_achieved": to_float(r.get("dp_epsilon_achieved_mean")),
            "krum_detection_rate": to_float(r.get("krum_detection_rate_mean")),
            "krum_score_ratio": to_float(r.get("krum_score_ratio_mean")),
            "best_f1_macro": to_float(r.get("best_f1_macro")),
            "best_accuracy": to_float(r.get("best_accuracy")),
            "is_original_anchor": eps in ORIGINAL_EXPERIMENT_1_EPSILONS,
            "tag": r.get("tag"),
        })
    table.sort(key=lambda r: r["epsilon_target"])
    return table


def write_table_csv(table, out_path):
    fields = ["epsilon_target", "epsilon_achieved", "krum_detection_rate",
              "krum_score_ratio", "best_f1_macro", "best_accuracy",
              "is_original_anchor", "tag"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in table:
            writer.writerow(r)


def print_verdict(model_type, table):
    ratios = [(r["epsilon_target"], r["krum_score_ratio"]) for r in table
              if r["krum_score_ratio"] is not None]
    rates = [(r["epsilon_target"], r["krum_detection_rate"]) for r in table
             if r["krum_detection_rate"] is not None]

    print(f"\n--- {model_type.upper()} ---")
    if not ratios and not rates:
        print("  No Krum diagnostic data found for this model (check that "
              "USE_ADAPTIVE_KRUM was on and BYZANTINE clients existed for "
              "these runs).")
        return

    if rates:
        min_rate = min(v for _, v in rates)
        max_rate = max(v for _, v in rates)
        if min_rate == max_rate == 1.0:
            print(f"  Detection rate: FLAT at 100% across all {len(rates)} "
                  f"epsilon points (eps={[e for e, _ in rates]}). No "
                  f"epsilon-dependence — matches Experiment 1's original "
                  f"finding, now confirmed at higher resolution.")
        elif min_rate == max_rate:
            print(f"  Detection rate: FLAT at {min_rate:.0%} across all "
                  f"epsilon points — no dependence, but not 100%.")
        else:
            print(f"  Detection rate: VARIES ({min_rate:.0%} to {max_rate:.0%}) "
                  f"across the epsilon grid — NEW finding vs. Experiment 1, "
                  f"which saw 100% at every one of its 3 points. Check which "
                  f"epsilon value(s) show degraded detection.")
            for eps, rate in sorted(rates):
                if rate < max_rate:
                    print(f"    -> eps={eps}: detection={rate:.0%}")

    if ratios:
        vals = [v for _, v in ratios]
        spread_pct = (max(vals) - min(vals)) / min(vals) * 100 if min(vals) else float("nan")
        print(f"  Score ratio range: {min(vals):.2f} - {max(vals):.2f} "
              f"({spread_pct:.1f}% spread across the full eps grid; "
              f"Experiment 1's original 3-point spread was ~1.7-2.2%)")


def make_chart(model_type, table, out_path):
    if not _MPL_AVAILABLE:
        print(f"  [skip chart] matplotlib not installed — "
              f"pip install matplotlib --break-system-packages", file=sys.stderr)
        return

    eps = [r["epsilon_target"] for r in table]
    detection = [r["krum_detection_rate"] for r in table]
    ratio = [r["krum_score_ratio"] for r in table]
    is_anchor = [r["is_original_anchor"] for r in table]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # Split points into "new" vs "Experiment 1 original anchor" for
    # distinct markers — makes it visually obvious which 3 points were
    # already trusted going in.
    new_eps  = [e for e, a in zip(eps, is_anchor) if not a]
    new_det  = [d for d, a in zip(detection, is_anchor) if not a]
    new_rat  = [r for r, a in zip(ratio, is_anchor) if not a]
    anc_eps  = [e for e, a in zip(eps, is_anchor) if a]
    anc_det  = [d for d, a in zip(detection, is_anchor) if a]
    anc_rat  = [r for r, a in zip(ratio, is_anchor) if a]

    ax1.plot(eps, detection, "-", color="gray", alpha=0.5, zorder=1)
    ax1.scatter(new_eps, new_det, color="tab:blue", label="New (this sweep)", zorder=2)
    ax1.scatter(anc_eps, anc_det, color="tab:red", marker="D",
                label="Experiment 1 anchor (3, 9, 15)", zorder=3)
    ax1.set_ylabel("Krum detection rate")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title(f"Krum Byzantine detection vs. DP epsilon — {model_type} model")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    ax2.plot(eps, ratio, "-", color="gray", alpha=0.5, zorder=1)
    ax2.scatter(new_eps, new_rat, color="tab:blue", zorder=2)
    ax2.scatter(anc_eps, anc_rat, color="tab:red", marker="D", zorder=3)
    ax2.set_xlabel("Target DP epsilon")
    ax2.set_ylabel("krum_score_ratio\n(byzantine_mean / honest_mean)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="sweep_summary.csv")
    parser.add_argument("--out", default=".")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = load_summary(args.summary)

    for model_type in ("network", "application"):
        table = build_model_table(rows, model_type)
        if not table:
            print(f"[skip] no rows found for model_type={model_type}", file=sys.stderr)
            continue

        csv_out = os.path.join(args.out, f"krum_vs_epsilon_{model_type}.csv")
        write_table_csv(table, csv_out)
        print(f"Wrote {csv_out} ({len(table)} epsilon points)")

        png_out = os.path.join(args.out, f"krum_vs_epsilon_{model_type}.png")
        make_chart(model_type, table, png_out)

        print_verdict(model_type, table)

    print("\nDone. If both models show a flat 100% detection rate and a "
          "<5% score-ratio spread across the full grid, that reconfirms "
          "Experiment 1's negative result at 1-increment resolution — "
          "still a real, citable finding, not a failed sweep.")


if __name__ == "__main__":
    main()
