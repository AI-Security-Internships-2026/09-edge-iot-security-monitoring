#!/usr/bin/env python3
"""Applies the E7 'fedavg_attack' ablation-mode patch to the main.py in the
current directory (or the path given as argv[1]). Idempotent: running it twice
is safe. Refuses to modify the file if the expected anchor text is not found
exactly once (e.g. upstream changed that block) -- in that case it prints what
to do instead of guessing."""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "main.py"
src = open(path, encoding="utf-8").read()

if '"fedavg_attack"' in src and 'elif ABLATION_MODE == "fedavg_attack":' in src:
    print(f"{path}: already patched, nothing to do.")
    sys.exit(0)

old_choices = '"calibrated_krum_dp_sweep"],'
new_choices = '"calibrated_krum_dp_sweep", "fedavg_attack"],'

anchor = 'elif ABLATION_MODE == "krum_baseline":'
new_block = '''elif ABLATION_MODE == "fedavg_attack":
    # E7 (Issue 5): undefended FedAvg baseline UNDER Byzantine attack --
    # mirrors "baseline" but with the attack switched ON, and mirrors
    # "krum_baseline"'s attack-on shape but WITHOUT Adaptive Krum. No existing
    # mode gave plain FedAvg + attack + no DP/Krum/HE.
    USE_KRUM = USE_ADAPTIVE_KRUM = USE_HE = USE_HE_KRUM_HYBRID = USE_NORM_GUARD = False
    USE_DP = False
    USE_BYZANTINE_ATTACK = True
    BYZANTINE_HEAD_ONLY = False
    AGGREGATOR = _args.aggregator if _args.aggregator is not None else "fedavg"

'''

n_choices = src.count(old_choices)
n_anchor = src.count(anchor)
if n_choices != 1 or n_anchor != 1:
    print(f"ABORTED, {path} NOT modified: expected exactly 1 match for each anchor, "
          f"found choices-list={n_choices}, krum_baseline-block={n_anchor}.\n"
          f"Upstream probably changed these lines. Apply the two edits by hand:\n"
          f"  1) add \"fedavg_attack\" to the --ablation-mode choices list\n"
          f"  2) insert the fedavg_attack elif block just before "
          f"'elif ABLATION_MODE == \"krum_baseline\":'")
    sys.exit(1)

src = src.replace(old_choices, new_choices, 1)
src = src.replace(anchor, new_block + anchor, 1)
open(path, "w", encoding="utf-8").write(src)
print(f"{path}: patched OK.")
