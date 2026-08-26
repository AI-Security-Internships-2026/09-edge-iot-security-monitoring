# FL-IDS Experiment 2 — Combined Manifest (Layer 2 Mitigated Runs)

Note: the two earlier unmitigated HE-Krum-hybrid runs (0% detection, no Layer 2 guard)
have been removed from this manifest at the user's request. Only the two runs using the
Layer 2 ciphertext-bound head-norm guard (ZKP-style commitment) remain below.

---

## Top-Level Index (manifest.json content)

```json
{
  "runs": [
    {
      "run_id": "network_he_krum_hybrid_norm_guard_v1",
      "model_type": "network",
      "defence": "adaptive_krum + Layer 2 ciphertext-bound head-norm guard (ZKP-style commitment, HE-hybrid)",
      "attack": {"active": true, "clients": [0, 1], "scale": 5.0, "type": "classifier_head_flip_attack"},
      "results_csv_path": "models/network_he_krum_hybrid_norm_guard_v1/results_reconstructed.csv",
      "best_round": 25,
      "best_f1_macro": 0.8282,
      "krum_detection_rate": 1.0,
      "device": "cuda",
      "git_commit": null,
      "status": "complete_with_anomalies",
      "status_note": "SSH disconnect around round 13, resumed cleanly. Rounds 9-13 per-round metrics not captured in console log."
    },
    {
      "run_id": "application_he_krum_hybrid_norm_guard_v1",
      "model_type": "application",
      "defence": "adaptive_krum + Layer 2 ciphertext-bound head-norm guard (ZKP-style commitment, HE-hybrid)",
      "attack": {"active": true, "clients": [0, 1], "scale": 2.0, "type": "classifier_head_flip_attack"},
      "results_csv_path": "models/application_he_krum_hybrid_norm_guard_v1/results_reconstructed.csv",
      "best_round": 24,
      "best_f1_macro": 0.7286,
      "krum_detection_rate": 1.0,
      "device": "cuda",
      "git_commit": null,
      "status": "complete_clean"
    }
  ],
  "manifest_schema_version": 1,
  "manifest_note": "Indexes the two Layer 2 (ZKP-style ciphertext-bound head-norm guard) mitigated runs only. Both show 100% detection and best F1-Macro at or near their model's clean baseline."
}
```

---

## Cross-Run Comparison

| Run | Model | Detection | Best F1-Macro | Best round |
|---|---|---|---|---|
| `network_he_krum_hybrid_norm_guard_v1` | network | **100%** | **0.8282** | 25 (best round of the run) |
| `application_he_krum_hybrid_norm_guard_v1` | application | **100%** | **0.7286** | 24 |

Clean (no-attack) baselines for reference: application round 20 = 0.7293, network round 22 = 0.8289.
Both mitigated runs land within ~0.001 of their model's clean baseline — the Layer 2 guard
doesn't just reduce damage, it appears to fully neutralize this attack's effect on final
model quality.

**Round-25 instability:** the network run's round 25 was its best round (no late-round
drop). The application run's round 25 DID drop (0.7286 at round 24 → 0.6867 at round 25) —
the same recurring pattern documented elsewhere in this project. This suggests the
network run's clean finish was likely single-run luck rather than a systematic effect of
the mitigation. **Use best-round numbers, not final-round, for both models.**

---

## Run 1: `network_he_krum_hybrid_norm_guard_v1` (MITIGATED, 100% detection)

### config.json
```json
{
  "run_id": "network_he_krum_hybrid_norm_guard_v1",
  "model_type": "network", "attack_scale": 5.0, "byzantine_clients": [0, 1],
  "use_he_krum_hybrid": true, "use_head_norm_guard": true,
  "head_norm_guard_k": 2.5, "head_norm_guard_min_keep_fraction": 0.5,
  "device": "cuda", "environment": "DGX Spark, flids_dev container",
  "anomalies": ["SSH disconnect right at round 13's header; resumed cleanly at round 14 with correct running best (0.8071); rounds 9-13 per-round metrics missing from console log"]
}
```
### metrics.json (key findings)
- **100% detection, every captured round.** `[Head-norm guard] rejected_ids=[0,1]` fires
  before Krum runs, every time — Krum never faces the attack.
- Best F1-Macro 0.8282 (round 25, this run's best round — no late-round collapse this time).
- Vulnerability_scanner F1 restored to 0.7864 (final), matching/exceeding the clean
  baseline's 0.7198 — the unmitigated run had this class permanently dead at F1=0.0.
- Krum's usual honest-client exclusion pattern is unaffected by the guard — the same
  narrowed candidate pool (post-guard, 8 clients) still has 2 further honest exclusions
  each round (clients 4, 10), consistent with a non-IID Dirichlet-partition explanation
  rather than a bug — see the exclusion-tuning discussion elsewhere in this session
  (`--krum-k` and `ADAPTIVE_KRUM_HYBRID_ASSUMED_F` added to main.py to address this).
### RUN_NOTES.md (summary)
First working confirmation of the Layer 2 mitigation. Data gap for rounds 9-13 (console
only, real CSV on disk should be complete). Krum's inner score-print uses LOCAL positions
post-filter, not original client IDs — traced by hand, confirmed correct, but a real
readability landmine for anyone skimming the log quickly (not yet fixed in code).

---

## Run 2: `application_he_krum_hybrid_norm_guard_v1` (MITIGATED, 100% detection)

### config.json
```json
{
  "run_id": "application_he_krum_hybrid_norm_guard_v1",
  "model_type": "application", "attack_scale": 2.0, "byzantine_clients": [0, 1],
  "use_he_krum_hybrid": true, "use_head_norm_guard": true,
  "head_norm_guard_k": 2.5, "head_norm_guard_min_keep_fraction": 0.5,
  "device": "cuda", "environment": "DGX Spark, flids_dev container",
  "anomalies": ["None -- clean run, 25/25 rounds, no crashes"]
}
```
### metrics.json (key findings)
- **100% detection, every single round, zero exceptions.** Cleanest run of the project.
- Best F1-Macro 0.7286 (round 24) — matches clean baseline (0.7293) almost exactly.
- Massive per-class recovery vs. the earlier (now-removed-from-manifest) unmitigated run:
  Normal 0.00→0.94, Backdoor 0.00→0.95, XSS 0.00→0.45, Password 0.00→0.43,
  Fingerprinting 0.00→0.80 (round-24 values).
- Round 25 dropped from round 24 (0.7286→0.6867) — the recurring late-round instability
  pattern, present here despite the mitigation.
- Rounds 3-4: norm guard rejected a THIRD client (id 8, honest) alongside 0,1 — a one-off,
  self-corrected by round 5 onward. Worth watching for on repeat runs.
### RUN_NOTES.md (summary)
Completes the mitigated pair (application + network). No anomalies to troubleshoot.
`honest_mean_score` stays essentially flat (~125,945-125,961) across all 25 rounds —
same curious near-frozen-score pattern observed in the network run too; likely an
artifact of squared-Euclidean distance being dominated by large, slow-moving parameter
magnitudes rather than genuine model evolution, not yet investigated further.

---

## Still Outstanding

- [ ] Real (non-reconstructed) `results_*.csv` / `checkpoint_*_best.npz` files for both
      runs — everything above is from console-log reconstruction.
- [ ] Fill the rounds 9-13 gap in the network run from the real CSV on disk.
- [ ] `git_commit` is `null` for both — populate once this code state is committed.
- [ ] Fix `krum.py`'s inner score-table print to use real client IDs.
- [ ] No repeat-seed confirmation for either run.
- [ ] Test whether `--krum-k` / `ADAPTIVE_KRUM_HYBRID_ASSUMED_F` reduce the honest-client
      exclusion without reopening the detection gap — not yet run.

---

## Run 3: `network_he_krum_hybrid_norm_guard_byz4_10_v1` (byzantine=4,10, MITIGATED)

Attack targets the two clients with the largest+smallest-adjacent partitions
(94,918 and 66,672 samples). k=3.5, assumed_f=1 (post-fix defaults).

- **100% detection, all 25 rounds** — `rejected_ids=[3,9]` (0-indexed), every round.
- Best F1-Macro: **0.8360 (round 20)** — highest of any network run to date, edging out
  both the clean baseline (0.8289) and the default-Byzantine mitigated run (0.8282).
- **But volatile:** repeated crash/recover cycles from round 19 onward (0.836→0.536→0.829→
  0.620→0.782→0.591). Mechanistic cause identified: clients 4+10 together hold ~73% of the
  network model's entire Vulnerability_scanner class (21,167 / ~40,090 samples). Excluding
  them every round for Byzantine-robustness also removes over half that class's real signal
  every round — a genuine robustness/utility tension, not a bug.
- Krum's extra (non-attacker) exclusion is only ONE client this round (client 5, smallest
  partition, 16,620 samples) — fewer than the default-Byzantine run's two, consistent with
  the k=3.5/assumed_f=1 tuning working as intended, though not a fully controlled comparison
  (attacker set also changed).
- Source: console log only. Real CSV/checkpoint not yet folded in.

---

## Run 4: `application_he_krum_hybrid_norm_guard_byz4_10_v1` (byzantine=4,10, MITIGATED)

- **100% detection, all 25 rounds** — `rejected_ids=[3,9]`, every round.
- Best F1-Macro: **0.6954 (round 23)**. No sharp round-25 crash this time (0.6954→0.6824→
  0.6914) — another data point that the late-round instability isn't deterministic.
- Krum's extra exclusion: TWO honest clients, symmetric by partition size — client 6
  (largest, 38,868 samples) and client 7 (smallest, 15,152 samples). Confirms the same
  "both extremes get flagged" mechanism seen on the network model.
- Source: console log only. Real CSV/checkpoint not yet folded in.

---

## Run 5: `network_he_krum_hybrid_norm_guard_byz2_7_v1` (byzantine=2,7, MITIGATED)

Attack targets two clients with UNREMARKABLE partition sizes (45,294 and 20,032 samples,
neither extreme) — the key control condition for testing whether Krum's extra exclusions
are attack-related or purely data-related.

- **100% detection, all 25 rounds** — `rejected_ids=[1,6]`, every round.
- **Krum's extra exclusion is STILL clients 4 and 10 (indices 3, 9)** — the same two
  clients flagged in every other network-model run regardless of which clients are
  actually attacking. Strong confirmation the exclusion is a property of clients 4/10's
  DATA, fully decoupled from the attack scenario.
- **Anomaly at round 24 (single occurrence, not seen in any other run):** a THIRD client
  (index 9, previously always kept) was norm-guard-rejected for that one round only,
  changing Krum's whole score structure (n dropped to 7, honest_mean_score dropped from
  ~2.18M to ~1.66M). F1-Macro dropped 0.8096->0.6603 then fully recovered by round 25
  (0.7706). Reads as a one-off local-training fluctuation; worth checking on a repeat run
  before treating as meaningful.
- Best F1-Macro: 0.8237 (round 21) — consistent with the other network mitigated runs
  (0.8282, 0.8360), reinforcing that detection/utility hold up across different attacked
  clients as long as those clients aren't also the naturally-high-variance ones.
- Source: console log only. Real CSV/checkpoint not yet folded in.

---

## Cross-Byzantine-Configuration Summary

| Attacked clients | Model | Extra Krum exclusions (beyond attackers) | Best F1-Macro |
|---|---|---|---|
| 1,2 (default) | network | 4, 10 | 0.8282 |
| 1,2 (default) | application | 6, 7 | 0.7286 |
| 4,10 (extreme data) | network | 5 | 0.8360 |
| 4,10 (extreme data) | application | 6, 7 | 0.6954 |
| 2,7 (moderate data) | network | 4, 10 | 0.8237 |

**The pattern that stands out:** clients 4 and 10 (network) and clients 6 and 7
(application) appear as Krum's "extra" exclusions across MULTIPLE different attack
configurations, including ones that don't target them at all. This is the clearest
available evidence that Krum's bulk-slice exclusions are driven by partition
size/composition, not by anything related to the actual attack -- consistent across
every mitigated run collected so far.
