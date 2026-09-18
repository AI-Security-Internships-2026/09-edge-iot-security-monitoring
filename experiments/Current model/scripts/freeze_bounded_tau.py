#!/usr/bin/env python3
"""
Freeze --bounded-tau for E6 (fix #2) from a VALIDATION-ONLY pilot.

Reads honest_head_norms_<tag>.csv (written by main.py; classifier-head
DELTA L2 norm per honest client per round), applies the SAME rule the
attacker/guard uses (median + k*MAD, defences.byzantine.
estimate_norm_guard_tau) per round, and freezes one scalar tau as the
median over rounds >= --skip-rounds (early rounds have atypically large
deltas). No TEST metric is read, so the parameter is frozen before any
final TEST evaluation.

Pilot recipe (same alpha/epsilon/aggregator family as E6; seed NOT in
the evaluation set; a non-stealth attack so the trajectory is ~clean):
    python main.py network --ablation-mode krum_dp_sweep \
      --aggregator adaptive_krum --alpha 0.3 --epsilon 5 --seed 7 \
      --attack-type sign_flip --log-honest-head-norms --tag e6_pilot
    python scripts/freeze_bounded_tau.py \
      --csv-glob 'honest_head_norms_network_e6_pilot_seed*.csv' \
      --k 2.5 --out experiments/configs/E6_frozen_bounded_tau.json
Use the SAME k as --krum-k in the E6 runs (default adaptive_krum_k from
hyperparams.json). Commit the JSON before launching E6.
"""
import argparse, csv, glob, json, os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from defences.byzantine import estimate_norm_guard_tau  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-glob", required=True)
    ap.add_argument("--k", type=float, required=True)
    ap.add_argument("--skip-rounds", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    files = sorted(glob.glob(a.csv_glob))
    if not files:
        sys.exit(f"No files match {a.csv_glob!r}")
    per_round = defaultdict(list)
    for fpath in files:
        with open(fpath, newline="") as f:
            for row in csv.DictReader(f):
                per_round[(fpath, int(row["round_id"]))].append(
                    float(row["head_delta_l2"]))
    taus = []
    for (fpath, rnd), norms in sorted(per_round.items()):
        if rnd > a.skip_rounds and len(norms) >= 3:
            taus.append(estimate_norm_guard_tau(norms, k=a.k))
    if not taus:
        sys.exit("No usable rounds after --skip-rounds.")
    tau = float(np.median(taus))
    all_norms = np.concatenate([v for (_, r), v in per_round.items()
                                if r > a.skip_rounds])
    out = {
        "bounded_tau": round(tau, 6),
        "k": a.k, "skip_rounds": a.skip_rounds,
        "n_rounds_used": len(taus),
        "per_round_tau_min": float(np.min(taus)),
        "per_round_tau_max": float(np.max(taus)),
        "honest_head_norm_median": float(np.median(all_norms)),
        "honest_head_norm_max": float(np.max(all_norms)),
        "source_files": files,
        "note": "Frozen from validation-only pilot; no TEST metric used.",
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"\n--bounded-tau {out['bounded_tau']}")


if __name__ == "__main__":
    main()
