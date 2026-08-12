# Run: application_he_krum_hybrid_v1

**Experiment:** Experiment 2 — HE vs. Krum, head-only classifier-head attack
**Model:** application
**Date:** captured in this session; see `config.json` for full flag state
**Status:** Complete (25/25 rounds), with two mid-run crash/resume events (see below)

## What this run tested

Does adaptive Krum's plaintext-distance scoring detect a Byzantine client that:
1. Trains normally on the full model locally (so its "bulk" backbone update looks like a
   real honest contribution), then
2. Overwrites *only* the classifier-head layers (`classifier.*` state_dict keys — the
   ~5.8% of parameters that get CKKS-encrypted before reaching the server) with a
   sign-flipped, scaled poison?

Krum only ever sees the plaintext "bulk" slice in the hybrid aggregation branch — it is
structurally blind to whatever the classifier head contains, encrypted or not.

## Result

**0% detection, all 25 rounds.** Full round-by-round data in `results_reconstructed.csv`,
summary in `metrics.json`.

## Exact steps to reproduce

1. **Code state.** This run used a `main.py` with the following fixes applied, relative to
   the version originally shipped for Experiment 2 (see `config.json` →
   `code_state.fixes_present_in_this_run` for the full list). The most important one:
   `classifier_head_flip_attack()`'s call site must run full local `train()` first and
   flip only `classifier.*` keys of the *trained* result — an earlier version of this
   call site skipped training entirely, which produces a different (and uninteresting)
   100%-detection result for an unrelated reason. **If re-running this from a fresh copy
   of the repo, confirm this fix is present in `_train_one_client()`'s Byzantine branch
   before trusting the result.**
2. **Flags** (already the defaults in the `main.py` used here — see `config.json` for the
   complete flag dump):
   ```python
   USE_KRUM = False
   USE_ADAPTIVE_KRUM = False
   USE_HE = False
   USE_HE_KRUM_HYBRID = True
   USE_DP = False
   USE_ZKP = False
   BYZANTINE_HEAD_ONLY = True
   ```
3. **Delete any stale checkpoint first** if a prior run under different flags exists in
   this directory — `checkpoint_application.npz`, `checkpoint_application_best.npz`,
   `checkpoint_application_progress.json`. This run's own log includes the exact warning
   `main.py` prints about this on every resume.
4. **Run:**
   ```bash
   python main.py application
   ```
   (No `--tag` was used for this run — see `config.json` → `note_no_tag_used`. **Use
   `--tag he_krum_hybrid_app` on any repeat** to avoid overwriting these files.)
5. **Dependencies:** requires `tenseal` installed in the active environment (`pip install
   tenseal`) — see `config.json` → `environment` for platform-specific notes; this
   particular run was on local Windows/CPU, not the DGX.
