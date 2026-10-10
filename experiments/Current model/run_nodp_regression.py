#!/usr/bin/env python3
"""PRV1 acceptance item 4 -- USE_DP=False regression (2 rounds).

Run from 'experiments/Current model/' (same folder as main.py):
    python run_nodp_regression.py

Runs main.py with --ablation-mode baseline (hardcodes USE_DP=False) for 2 rounds,
while counting every PrivacyEngine construction in-process. Then checks:
  1. PrivacyEngine constructed 0 times
  2. dp_persistent_client_state was never imported
  3. no dp_final_epsilon_* output was written for this run
  4. the no-DP model (dp_safe=False) is BatchNorm1d + standard nn.LSTM (no GroupNorm / DPLSTM)
Does not modify any project file.
"""
import glob, json, os, runpy, sys
import torch.nn as nn

TAG = "prv1_nodp_regression"
SEED = 42
ENGINES_BUILT = 0

# Count PrivacyEngine constructions BEFORE main.py runs.
try:
    from opacus import PrivacyEngine
    _orig_init = PrivacyEngine.__init__

    def _counting_init(self, *a, **k):
        global ENGINES_BUILT
        ENGINES_BUILT += 1
        return _orig_init(self, *a, **k)

    PrivacyEngine.__init__ = _counting_init
    OPACUS = True
except ImportError:
    OPACUS = False

sys.argv = ["main.py", "network", "--ablation-mode", "baseline",
            "--rounds", "2", "--tag", TAG, "--seed", str(SEED)]
runpy.run_path("main.py", run_name="__main__")

print("\n" + "=" * 60 + "\n  PRV1 USE_DP=False REGRESSION CHECKS\n" + "=" * 60)
ok = True

print(f"[1] PrivacyEngine constructions during run: {ENGINES_BUILT}"
      f"{'' if OPACUS else ' (opacus not installed -> trivially 0)'}")
ok &= ENGINES_BUILT == 0

imported = "dp_persistent_client_state" in sys.modules
print(f"[2] dp_persistent_client_state imported: {imported}")
ok &= not imported

leaked = glob.glob(f"dp_final_epsilon_*{TAG}*")
print(f"[3] dp_final_epsilon files written: {leaked or 'none'}")
ok &= not leaked

cfg = f"experiment_config_network_{TAG}_seed{SEED}.json"
if os.path.exists(cfg):
    c = json.load(open(cfg))
    print(f"    {cfg}: ablation_mode={c.get('ablation_mode')}  use_dp={c.get('use_dp', '(key absent)')}")

try:
    from task import get_model
    def kinds(dp_safe):
        m = get_model(num_features=40, num_classes=8, dp_safe=dp_safe)
        return sorted({type(x).__name__ for x in m.modules()})
    nodp, dp = kinds(False), kinds(True)
    print(f"[4] dp_safe=False layers: {nodp}")
    print(f"    dp_safe=True  layers: {dp}   (for contrast)")
    arch_ok = ("BatchNorm1d" in nodp and "LSTM" in nodp
               and "GroupNorm" not in nodp and "DPLSTM" not in nodp)
    print(f"    no-DP arch is BatchNorm1d + standard LSTM: {arch_ok}")
    ok &= arch_ok
except Exception as e:
    print(f"[4] could not build model for layer check ({e!r}); "
          f"run: findstr /n \"GroupNorm DPLSTM\" model_defs.py  and paste it")
    ok = False

print("\nRESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
