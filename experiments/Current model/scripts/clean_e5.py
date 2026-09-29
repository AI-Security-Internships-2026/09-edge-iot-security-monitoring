#!/usr/bin/env python3
"""
Delete E5 run artefacts (checkpoints, json, csv, jsonl, logs). DRY RUN by default.

  python scripts/clean_e5.py --main-dir "experiments/Current model"          # list only
  python scripts/clean_e5.py --main-dir "experiments/Current model" --yes    # delete

Matches ONLY files whose name contains the E5 tag pattern this campaign creates
(..._E5_f<F>_k<K>_<aggregator>_seed<S>...), in --main-dir (leftovers from crashed
runs) and everything under experiments/results/E5. Optionally the derived
param_sensitivity tables with --include-tables.

It never touches: experiments/configs/E5_campaign.json, hetero_fit_coeffs.json,
hyperparams.json, or any run from Issues #20-#23 (those don't carry the E5 tag).
WARNING: if your E5 'scan_dirs' REUSE cells came from runs already inside
experiments/results/E5, deleting them means they will be NEW_RUN again.
"""
import argparse, re, shutil
from pathlib import Path

DIR_PAT = re.compile(r"^E5_f\d+_k\d+p\d+_(adaptive_krum|calibrated_krum)$")
PAT = re.compile(r"_E5_f\d+_k\d+p\d+_(adaptive_krum|calibrated_krum)_seed\d+")
EXTS = {".npz", ".json", ".csv", ".jsonl", ".log"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-dir", default=".", help="folder main.py runs in")
    ap.add_argument("--results-root", default="experiments/results/E5")
    ap.add_argument("--include-tables", action="store_true",
                    help="also delete experiments/results/param_sensitivity/table_[AB]_*.csv")
    ap.add_argument("--yes", action="store_true", help="actually delete")
    a = ap.parse_args()

    hits = [p for p in Path(a.main_dir).glob("*") if p.is_file() and p.suffix in EXTS and PAT.search(p.name)]
    root = Path(a.results_root)
    run_dirs = [d for d in root.rglob("*") if d.is_dir() and DIR_PAT.match(d.name)] if root.exists() else []
    extra = []
    if root.exists():
        extra = [p for p in root.glob("E5_*.csv")]
    if a.include_tables:
        extra += list(Path("experiments/results/param_sensitivity").glob("table_[AB]_*.csv"))

    total = 0
    for p in hits + extra:
        total += p.stat().st_size
    print(f"{len(hits)} files in {a.main_dir}, {len(run_dirs)} run folders under {root}, {len(extra)} summary files")
    for p in hits[:200] + extra: print("  ", p)
    for d in run_dirs[:200]: print("   [dir]", d)
    if not a.yes:
        print("\nDRY RUN - nothing deleted. Re-run with --yes to delete.")
        return
    for p in hits + extra: p.unlink()
    for d in sorted(run_dirs, key=lambda x: -len(x.parts)):
        if d.exists(): shutil.rmtree(d)
    print("Deleted.")

if __name__ == "__main__":
    main()
