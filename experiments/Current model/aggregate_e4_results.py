#!/usr/bin/env python3
"""
aggregate_e4_results.py -- Issue 5, E4/Table 3/Figure 4

Walks an --output-root produced by run_e4_campaign.py (one subdirectory
per cell, named by cell_tag()), and for each completed cell reads:

  * dp_final_epsilon_*.json         -> achieved final_total_epsilon
                                        (per client; this script reports
                                        the mean and the per-client spread)
  * per_client_krum_scores_*.csv    -> Honest FPR / Byzantine TPR, via
                                        the 'classification' column
                                        main.py already computes
                                        (TP/FP/TN/FN per round per client)
  * results_*_FINAL_TEST.csv        -> Macro-F1 and per-class
                                        F1/Recall/AUC-PR on the TEST
                                        holdout (rare-class columns
                                        pulled out separately below)

and writes one consolidated CSV row per cell.

CONVENTION FLAGGED EXPLICITLY (not asserted elsewhere in the codebase
as uploaded -- cross-check against your own scripts/analysis_paper.py
if Task 4's version already defines this differently, and use that
instead):

  Honest FPR / Byzantine TPR are computed by pooling every
  (round, client) row across the ENTIRE run (all NUM_ROUNDS rounds),
  not just the final round or a late-round window. This is reported
  as *_all_rounds below. A *_last_25pct_rounds variant is also
  computed (rows from the last quarter of rounds only), since
  early-round detection performance before the model/aggregator has
  stabilized can understate steady-state robustness. Report whichever
  convention your Table 2/3 already uses elsewhere for consistency;
  don't silently mix conventions across tables in the same paper.

Usage
-----
    python aggregate_e4_results.py --output-root /path/to/experiments/results/E4 \\
        --out-csv /path/to/experiments/results/E4/table3_e4_summary.csv
"""
import argparse
import csv
import glob
import json
import statistics
from pathlib import Path

# Exact spellings as they appear in data_loader.py's ALL_CLASSES (and
# therefore in results_*_FINAL_TEST.csv's column names) -- confirmed
# against a real run's output header, not guessed. Case/underscore
# variants matter here: the codebase spells it "Vulnerability_scanner"
# (lowercase 's'), not "Vulnerability_Scanner".
RARE_CLASSES = ["XSS", "Ransomware", "MITM", "Fingerprinting",
                 "Vulnerability_scanner"]


def _read_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def compute_fpr_tpr(krum_csv_path, last_frac=None):
    rows = _read_csv_rows(krum_csv_path)
    if not rows:
        return None
    if last_frac is not None:
        round_ids = sorted({int(r["round_id"]) for r in rows})
        cutoff = round_ids[int(len(round_ids) * (1 - last_frac))] if round_ids else 0
        rows = [r for r in rows if int(r["round_id"]) >= cutoff]

    tp = fp = tn = fn = 0
    for r in rows:
        c = r["classification"]
        if c == "TP":
            tp += 1
        elif c == "FP":
            fp += 1
        elif c == "TN":
            tn += 1
        elif c == "FN":
            fn += 1

    honest_fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    byzantine_tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    return {"honest_fpr": honest_fpr, "byzantine_tpr": byzantine_tpr,
            "n_rows": len(rows), "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def read_achieved_epsilon(eps_json_path):
    with open(eps_json_path) as f:
        data = json.load(f)
    # Confirmed real schema (from an actual end-to-end run of main.py,
    # not guessed): a flat dict of run-level scalars (model_type, seed,
    # ablation_mode, num_rounds, dp_full_run_target_epsilon, dp_delta,
    # dp_total_epochs_per_client) PLUS "final_total_epsilon_by_client"
    # -- a nested {client_idx_str: epsilon_float} dict, which is what we
    # actually want. Do NOT fall back to "any numeric top-level value"
    # -- dp_full_run_target_epsilon/seed/num_rounds/dp_delta are also
    # numeric and would silently corrupt the mean if picked up instead.
    per_client = data.get("final_total_epsilon_by_client")
    if not isinstance(per_client, dict) or not per_client:
        # Fallback ONLY for an unrecognized schema variant: look for
        # any OTHER key whose name contains "epsilon" and whose value
        # is a non-empty dict of numbers (still schema-aware, not a
        # blind numeric-value scan).
        for k, v in data.items():
            if k == "dp_full_run_target_epsilon":
                continue
            if "epsilon" in k.lower() and isinstance(v, dict):
                inner = [x for x in v.values() if isinstance(x, (int, float))]
                if inner:
                    per_client = v
                    break
        if not isinstance(per_client, dict) or not per_client:
            return None

    values = [v for v in per_client.values() if isinstance(v, (int, float))]
    if not values:
        return None
    return {
        "achieved_epsilon_mean": statistics.mean(values),
        "achieved_epsilon_min": min(values),
        "achieved_epsilon_max": max(values),
        "n_clients": len(values),
        "requested_epsilon_from_manifest": data.get("dp_full_run_target_epsilon"),
    }


def read_final_test(test_csv_path):
    rows = _read_csv_rows(test_csv_path)
    if not rows:
        return None
    row = rows[0]  # one row per run, per main.py's FINAL_TEST_CSV writer
    out = {
        "test_accuracy": float(row.get("test_accuracy", "nan")),
        "test_f1_macro": float(row.get("test_f1_macro", "nan")),
    }
    for key, val in row.items():
        for rc in RARE_CLASSES:
            rc_key = rc.replace(" ", "_")
            if key in (f"test_f1_{rc}", f"test_recall_{rc}", f"test_aucpr_{rc}",
                       f"test_f1_{rc_key}", f"test_recall_{rc_key}", f"test_aucpr_{rc_key}"):
                try:
                    out[key] = float(val)
                except ValueError:
                    out[key] = None
    return out


def parse_tag(tag):
    # cell_tag() format: E4_<aggregator>_eps<eps>_a<alpha>_<attack>_seed<seed>
    parts = tag.split("_")
    out = {"tag": tag}
    try:
        agg_end = parts.index([p for p in parts if p.startswith("eps")][0])
        out["aggregator"] = "_".join(parts[1:agg_end])
        for p in parts[agg_end:]:
            if p.startswith("eps"):
                out["requested_epsilon"] = float(p[3:])
            elif p.startswith("a") and p[1:].replace(".", "", 1).isdigit():
                out["alpha"] = float(p[1:])
            elif p.startswith("seed"):
                out["seed"] = int(p[4:])
            else:
                out.setdefault("attack_type", p)
    except (IndexError, ValueError):
        pass
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-root", required=True)
    p.add_argument("--out-csv", required=True)
    args = p.parse_args()

    output_root = Path(args.output_root)
    run_dirs = [d for d in output_root.iterdir() if d.is_dir()]
    print(f"Scanning {len(run_dirs)} run directories under {output_root}")

    all_rows = []
    n_incomplete = 0
    for run_dir in sorted(run_dirs):
        eps_json_candidates = list(run_dir.glob("dp_final_epsilon_*.json"))
        krum_csv_candidates = list(run_dir.glob("per_client_krum_scores_*.csv"))
        test_csv_candidates = list(run_dir.glob("results_*_FINAL_TEST.csv"))

        if not (eps_json_candidates and krum_csv_candidates and test_csv_candidates):
            print(f"  [INCOMPLETE] {run_dir.name} -- missing one or more "
                  f"required output files, skipped (not counted as a result).")
            n_incomplete += 1
            continue

        row = parse_tag(run_dir.name)

        eps_info = read_achieved_epsilon(eps_json_candidates[0])
        if eps_info:
            row.update(eps_info)

        fpr_tpr_all = compute_fpr_tpr(krum_csv_candidates[0])
        if fpr_tpr_all:
            row["honest_fpr_all_rounds"] = fpr_tpr_all["honest_fpr"]
            row["byzantine_tpr_all_rounds"] = fpr_tpr_all["byzantine_tpr"]

        fpr_tpr_late = compute_fpr_tpr(krum_csv_candidates[0], last_frac=0.25)
        if fpr_tpr_late:
            row["honest_fpr_last_25pct_rounds"] = fpr_tpr_late["honest_fpr"]
            row["byzantine_tpr_last_25pct_rounds"] = fpr_tpr_late["byzantine_tpr"]

        test_info = read_final_test(test_csv_candidates[0])
        if test_info:
            row.update(test_info)

        all_rows.append(row)

    if not all_rows:
        print("No complete cells found -- nothing written.")
        return

    fieldnames = sorted({k for row in all_rows for k in row.keys()},
                         key=lambda k: (k != "tag", k != "aggregator",
                                        k != "requested_epsilon", k != "seed", k))
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print(f"Wrote {len(all_rows)} complete cells to {args.out_csv} "
          f"({n_incomplete} incomplete cells skipped).")


if __name__ == "__main__":
    main()
