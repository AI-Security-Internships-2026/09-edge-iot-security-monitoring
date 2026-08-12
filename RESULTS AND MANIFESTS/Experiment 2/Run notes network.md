# Run: network_he_krum_hybrid_v1

**Experiment:** Experiment 2 — HE vs. Krum, head-only classifier-head attack
**Model:** network
**Companion run:** `application_he_krum_hybrid_v1` (same session, same code state)
**Status:** Complete (25/25 rounds), one mid-run crash/resume event

## What this run tested

Same question as the application-model run: does adaptive Krum's plaintext-slice scoring
detect a Byzantine client that trains normally, then poisons only the classifier head
before the head gets CKKS-encrypted? Run here as the cross-model confirmation of that
run's finding, using this model's own attack scale (5.0, vs. application's 2.0 — see
`byzantine.py`'s scale guidance).

## Result: confirms the application-model finding, plus two new observations

1. **0% detection, all 25 rounds** — same qualitative result as application. See
   `metrics.json` → `krum_detection_rate_note`.
2. **NEW: three honest clients (4, 5, 10) were excluded by Krum in literally every round**,
   while the two actual Byzantine clients were never once flagged. This connects directly
   to a previously-documented, unresolved anomaly in the master doc (Condition 5's "Client
   4 persistently excluded 25/25 rounds," hypothesized as Dirichlet non-IID skew, and
   explicitly noted as NOT recurring in Experiment 1). It has now recurred here, and grown
   from 1 to 3 persistently-excluded clients. See `metrics.json` →
   `persistent_exclusion_finding` for the full writeup — this is important enough that it
   should probably get its own paragraph if this experiment goes into a report, not just a
   footnote.
3. **NEW: Vulnerability_scanner F1 = 0.0000 in every single round**, down from 0.7198 in
   the locked clean baseline. Overall accuracy still reaches 90%+, which would look healthy
   on a dashboard that only tracks aggregate accuracy — this is a concrete example of why
   that's not sufficient monitoring when an undetected poisoning attack is active.

## Exact steps to reproduce

Identical to `application_he_krum_hybrid_v1`'s reproduction steps.

Same code-state requirement applies: `classifier_head_flip_attack()`'s call site must
train the full model locally first, then flip only `classifier.*` keys of the trained
result — confirm this before trusting a re-run of this result.


## Still outstanding

- [ ] `per_client_audit.py` (an existing open item from the master doc, now more urgent
      given this run's finding) has not been run against clients 4/5/10's Dirichlet
      partitions to confirm or rule out the non-IID-skew hypothesis for the persistent
      exclusion.
