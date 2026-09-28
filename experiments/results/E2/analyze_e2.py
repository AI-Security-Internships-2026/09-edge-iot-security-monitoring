#!/usr/bin/env python3
import glob
import json
import os
import re
import numpy as np
import pandas as pd

DATA_DIR = "e2_results_pull"
AGGREGATORS = ["fedavg", "multi_krum", "median", "trimmed_mean", "adaptive_krum", "calibrated_krum"]
ATTACKS = ["sign_flip", "gaussian", "minmax", "minsum", "bounded_directional"]
SEEDS = [42, 123, 456]
BYZANTINE_CLIENTS = {1, 2}
RARE_CLASSES = ["Ransomware", "Vulnerability_scanner", "MITM"]  # low-support attack classes in this task

# --- Parse filenames robustly against attack names that themselves
# contain underscores (sign_flip, bounded_directional) ---
def parse_tag(agg, attack, seed):
    return f"e2_{agg}_{attack}_seed{seed}"

rows = []
for agg in AGGREGATORS:
    for attack in ATTACKS:
        for seed in SEEDS:
            tag = parse_tag(agg, attack, seed)
            final_test = os.path.join(DATA_DIR, f"results_network_{tag}_FINAL_TEST.csv")
            plain = os.path.join(DATA_DIR, f"results_network_{tag}.csv")
            if not os.path.exists(final_test):
                print(f"MISSING: {final_test}")
                continue

            ft = pd.read_csv(final_test)
            assert len(ft) == 1, f"expected 1 row in {final_test}, got {len(ft)}"
            ft = ft.iloc[0]

            row = {
                "aggregator": agg,
                "attack": attack,
                "seed": seed,
                "test_accuracy": ft["test_accuracy"],
                "macro_f1": ft["test_f1_macro"],
                "test_loss": ft["test_loss"],
            }
            for c in RARE_CLASSES:
                row[f"f1_{c}"] = ft.get(f"test_f1_{c}", np.nan)
                row[f"recall_{c}"] = ft.get(f"test_recall_{c}", np.nan)
                row[f"aucpr_{c}"] = ft.get(f"test_aucpr_{c}", np.nan)

            # Byzantine TPR / Honest FPR -- averaged across all 25 rounds
            # of the PER-CLIENT rows (excluding the MEAN summary row).
            # krum_selected==1 means "kept/trusted this round"; for
            # fedavg/median/trimmed_mean these aggregators never reject
            # anyone, so krum_selected is always 1 for all clients and
            # krum_detected_byzantine is always 0 -- TPR/FPR read 0%/0%
            # by construction for those three, exactly as flagged when
            # the campaign was built.
            byz_tpr, honest_fpr = np.nan, np.nan
            # krum_selected / krum_detected_byzantine are hardcoded 0
            # placeholders (confirmed against raw output) for any
            # aggregator that never runs a Krum-style selection step --
            # fedavg/median/trimmed_mean. Computing "1 - selected_rate"
            # for those would spuriously read as 100% honest-FPR ("all
            # honest clients rejected"), which is FALSE: these
            # aggregators never reject anyone, they just have no
            # selection mechanism to log in the first place. Only
            # compute real TPR/FPR for aggregators that actually make
            # an accept/reject decision.
            if agg in ("multi_krum", "adaptive_krum", "calibrated_krum") and os.path.exists(plain):
                pr = pd.read_csv(plain)
                pr = pr[pr["client"] != "MEAN"].copy()
                pr["client"] = pr["client"].astype(int)
                byz_rows = pr[pr["client"].isin(BYZANTINE_CLIENTS)]
                honest_rows = pr[~pr["client"].isin(BYZANTINE_CLIENTS)]
                if len(byz_rows):
                    byz_tpr = byz_rows["krum_detected_byzantine"].mean()
                if len(honest_rows):
                    honest_fpr = 1.0 - honest_rows["krum_selected"].mean()
            row["byzantine_tpr"] = byz_tpr
            row["honest_fpr"] = honest_fpr

            rows.append(row)

df = pd.DataFrame(rows)
print(f"Parsed {len(df)}/90 cells.\n")

# ---------------------------------------------------------------
# Table 2: mean +/- SD across 3 seeds, per (aggregator, attack)
# ---------------------------------------------------------------
agg_funcs = {
    "macro_f1": ["mean", "std"],
    "test_accuracy": ["mean", "std"],
    "byzantine_tpr": ["mean", "std"],
    "honest_fpr": ["mean", "std"],
}
for c in RARE_CLASSES:
    agg_funcs[f"recall_{c}"] = ["mean", "std"]
    agg_funcs[f"aucpr_{c}"] = ["mean", "std"]

table2 = df.groupby(["aggregator", "attack"]).agg(agg_funcs)
table2.columns = ["_".join(c) for c in table2.columns]
table2 = table2.reset_index()

# order aggregators/attacks sensibly rather than alphabetically
table2["aggregator"] = pd.Categorical(table2["aggregator"], categories=AGGREGATORS, ordered=True)
table2["attack"] = pd.Categorical(table2["attack"], categories=ATTACKS, ordered=True)
table2 = table2.sort_values(["aggregator", "attack"]).reset_index(drop=True)

table2.to_csv("table2_full.csv", index=False)
df.to_csv("e2_all_cells_raw.csv", index=False)

# ---------------------------------------------------------------
# Compact printable summary
# ---------------------------------------------------------------
print("=" * 100)
print("TABLE 2 SUMMARY (mean +/- SD across seeds 42/123/456)")
print("=" * 100)
for agg in AGGREGATORS:
    print(f"\n--- {agg} ---")
    sub = table2[table2["aggregator"] == agg]
    for _, r in sub.iterrows():
        print(f"  {r['attack']:20s}  MacroF1={r['macro_f1_mean']:.4f}+/-{r['macro_f1_std']:.4f}  "
              f"Acc={r['test_accuracy_mean']:.4f}  "
              f"ByzTPR={r['byzantine_tpr_mean']*100:5.1f}%  "
              f"HonestFPR={r['honest_fpr_mean']*100:5.1f}%")

# ---------------------------------------------------------------
# Flag divergence (any cell with macro_f1 < 0.15 -- well below the
# ~0.40-0.71 range everything else lands in, per what we've seen)
# ---------------------------------------------------------------
print("\n" + "=" * 100)
print("DIVERGED / DEGENERATE CELLS (macro_f1 < 0.15)")
print("=" * 100)
diverged = df[df["macro_f1"] < 0.15]
if len(diverged):
    print(diverged[["aggregator", "attack", "seed", "macro_f1", "test_accuracy"]].to_string(index=False))
else:
    print("None.")

print("\nWrote table2_full.csv and e2_all_cells_raw.csv")
