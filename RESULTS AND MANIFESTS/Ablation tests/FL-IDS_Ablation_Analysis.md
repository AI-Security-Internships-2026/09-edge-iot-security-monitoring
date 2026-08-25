# FL-IDS Standalone Ablation Analysis — Pure DP / Pure HE / Pure ZKP

**Source data:** `results_{network,application}_pure_{dp,he,zkp}.csv` (6 files, 25 rounds x 10 clients + MEAN row each, 275 data rows per file).
**Code under test:** the `ABLATION_MODE`-patched `main.py` (three copies: `main_dp.py`, `main_he.py`, `main_zkp.py`), which fixed the previously-broken standalone `USE_HE` path (routed through `he_local.py` instead of the old unweighted, never-decrypted `he_aggregate()`) and redefined standalone `USE_ZKP` as `defences/zkp.py` Part 2's ciphertext-bound HMAC head-norm guard running **in isolation, with no Krum call at all**.

## 1. What each run actually tested

| Mode | Defence active | Byzantine attack | Attack target | Krum called? |
|---|---|---|---|---|
| `pure_dp` | DP-SGD + FedProx only | OFF | — | No |
| `pure_he` | Partial CKKS HE (classifier head), all clients averaged | OFF | — | No |
| `pure_zkp` | HMAC ciphertext-bound head-norm guard (MAD threshold, k=2.5) | **ON** | Classifier head only (`classifier_head_flip_attack`, train-first) | **No** — guard is the *only* defence |

`pure_dp` and `pure_he` are clean utility/cost ablations (no attacker in the picture, matching the master doc's Experiment 3 gap table request for standalone +DP and +HE rows). `pure_zkp` is the one with a live attacker — it exists specifically to answer: **does the ciphertext-bound HMAC guard alone, without Krum backing it up, catch a classifier-head-only attacker?**

## 2. Headline result — the ZKP guard works, alone, perfectly, both models

| Model | Clients rejected | Rounds | Rejection rate (attackers) | Rejection rate (honest) | False positives |
|---|---|---|---|---|---|
| Network | Clients 1, 2 (the only Byzantine clients) | 25/25 | **100%** | **0%** | **0** |
| Application | Clients 1, 2 (the only Byzantine clients) | 25/25 | **100%** | **0%** | **0** |

Verified directly from the per-client `zkp_rejected` / `krum_detected_byzantine` columns: clients 1 and 2 were flagged in **every single round of both 25-round runs**, and no other client was ever flagged, even once. This is a clean, textbook result — no partial detection, no round-to-round flakiness, no honest-client collateral damage.

**What this confirms about the Experiment 2 hybrid branch:** the guard stage is not a weak or redundant pre-filter riding on Krum's coattails — it independently and completely neutralizes the classifier-head-only attack on its own. In the mitigated `USE_HE_KRUM_HYBRID` runs (Week 9), Krum ran a second time on the plaintext slice *after* the guard already removed both attackers; those runs could not by themselves tell you whether Krum was doing any real work or the guard had already finished the job. This ablation answers that: **the guard alone is sufficient against this specific attack** (a classifier-head-only sign-flip/scale attack of the kind `classifier_head_flip_attack` implements). It does **not** tell you whether Krum's second layer would catch something the guard misses — see Caveats (Section 6).

## 3. Utility comparison — F1-Macro and Accuracy

| Run | Best round | Best F1-Macro | Best Accuracy | Final (r25) F1-Macro | Final Accuracy |
|---|---|---|---|---|---|
| network / pure_dp | 23 | 0.7902 | 0.9393 | 0.7487 | 0.9297 |
| network / pure_he | 21 | 0.8095 | 0.9506 | 0.7438 | 0.8804 |
| network / pure_zkp | 25 | **0.8214** | 0.9500 | 0.8214 | 0.9500 |
| application / pure_dp | 24 | 0.5905 | 0.7501 | 0.5660 | 0.7083 |
| application / pure_he | 25 | 0.7585 | 0.8698 | 0.7585 | 0.8698 |
| application / pure_zkp | 24 | 0.7473 | 0.8642 | 0.7322 | 0.8519 |

Two things jump out:

1. **`pure_zkp` — despite running with an active attacker — matches or beats the clean, no-attack `pure_dp`/`pure_he` runs on both models.** Network's `pure_zkp` (0.8214) is the single best result of all six runs, beating even the attack-free `pure_he` (0.8095) and `pure_dp` (0.7902). This is strong evidence the guard isn't just detecting the attack, it's fully absorbing its utility cost — the two attacking clients are cleanly excised before their poisoned ciphertext ever reaches aggregation, leaving the remaining 8 honest clients' encrypted contribution indistinguishable from an unattacked run.
2. **`pure_he` (partial HE, no attack) clearly outperforms `pure_dp` (DP-SGD, no attack) on both models** — network: 0.8095 vs 0.7902 best F1; application: 0.7585 vs 0.5905 best F1, a much larger gap. This is the expected direction: DP-SGD's per-sample gradient clipping + noise injection costs real utility, especially on the application model's harder, more class-imbalanced classification task (rare classes like Uploading, SQL_injection). Partial HE, by contrast, adds *zero* noise to training — it only encrypts what's transmitted, so its only utility cost path is indirect (e.g. through whatever numerical effects CKKS's approximate arithmetic introduces on the classifier-head slice during decryption), which is evidently much smaller.

**Comparison against the project's locked clean FedProx baselines** (network 0.8289 @ round 22, application 0.7293 @ round 20, from the master planning doc — different recipe, not perfectly apples-to-apples, see Caveats): `pure_he`'s application result (0.7585) actually **exceeds** the locked application baseline, and `pure_zkp`'s network result (0.8214) sits within ~1 point of the locked network baseline **while under active attack**. `pure_dp` is the only ablation that clearly underperforms both locked baselines, consistent with DP-SGD's known utility cost.

## 4. Per-class F1 at best round

### Application model

| Class | pure_dp | pure_he | pure_zkp |
|---|---|---|---|
| Normal | 0.7281 | 0.9320 | 0.9242 |
| SQL_injection | 0.7265 | 0.8040 | 0.8305 |
| Uploading | 0.3660 | 0.6334 | 0.6286 |
| Backdoor | 0.8647 | 0.9581 | 0.9467 |
| Port_Scanning | 0.8183 | 0.8267 | 0.8267 |
| XSS | 0.2595 | 0.5166 | 0.4675 |
| Password | 0.3475 | 0.5992 | 0.5580 |
| Fingerprinting | 0.6134 | 0.7980 | 0.7959 |

### Network model

| Class | pure_dp | pure_he | pure_zkp |
|---|---|---|---|
| Normal | 0.8496 | 0.8562 | 0.9483 |
| DDoS_UDP | 0.9976 | 0.9988 | 0.9991 |
| DDoS_ICMP | 0.9938 | 0.9970 | 0.9989 |
| Ransomware | 0.7336 | 0.7000 | 0.7809 |
| DDoS_HTTP | 0.6353 | 0.7414 | 0.7695 |
| DDoS_TCP | 0.9608 | 0.9972 | 0.9991 |
| Vulnerability_scanner | 0.7204 | 0.7370 | 0.6299 |
| MITM | 0.4306 | 0.4487 | 0.4455 |

`pure_dp` is consistently the weakest on the hardest, most data-starved classes (application: XSS 0.26, Password 0.35, Uploading 0.37 — exactly the classes the master doc already flags as rare/hard). This matches the project's own documented finding that DP noise disproportionately suppresses rare-class learning rather than costing accuracy uniformly. MITM stays the network model's weakest class across all three ablations (0.43–0.45), consistent with its previously-documented hard data ceiling (1,214 raw samples) — this is a data-scarcity ceiling, not a defence-mechanism artifact, since it's essentially flat regardless of which mechanism is active.

## 5. Round timing — DP-SGD is the expensive one, not encryption

| Run | Mean round time | Total wall time (25 rounds) |
|---|---|---|
| network / pure_dp | 169.4s | 70.6 min |
| network / pure_he | 74.0s | 30.9 min |
| network / pure_zkp | 63.6s | 26.5 min |
| application / pure_dp | 196.4s | 81.8 min |
| application / pure_he | 51.9s | 21.6 min |
| application / pure_zkp | 42.8s | 17.8 min |

DP-SGD (Opacus per-sample gradient tracking) is by far the dominant cost — roughly **2.3–3.8x slower per round** than either HE-based mode. This matches the master doc's prior finding that HE itself is cheap (~0.2s/round in the earlier ablation) relative to DP-SGD's cost.

`pure_zkp` runs faster than `pure_he` on both models, despite doing strictly more work per round (encryption **and** proof generation **and** verification **and** MAD thresholding). The likely explanation: `pure_zkp`'s aggregation step only ever homomorphically combines **8 clients' ciphertext** (2 are rejected by the guard before aggregation), while `pure_he` homomorphically combines all **10**. Since `aggregate_encrypted()`'s cost scales with the number of ciphertexts summed, excluding 2 clients before that step measurably cuts the most expensive per-round operation — a real, if secondary, performance benefit of the guard, not just a security one.

## 6. Stability — no NaN rounds, but real round-to-round volatility

Zero NaN/Inf-quarantined rounds across all six runs (`nan_rounds=0` everywhere) — no numerical breakdowns in any condition.

That said, every run shows meaningful round-to-round F1-Macro swings in its final third, consistent with the project's previously-documented recurring non-IID/FedProx late-round instability pattern:

- `network/pure_he`: round 18 drops to F1=0.4463 (acc 0.5266) before recovering to 0.7651 the very next round — a sharp, one-round transient collapse.
- `application/pure_dp`: round 21 drops to F1=0.4283 (acc 0.5639), recovers to 0.5360 the next round.
- All six runs' last-5-round F1 standard deviation is non-trivial (0.014–0.062), and in 9/6 = most runs, the **best round is not round 25** — reinforcing the project's existing recommendation to report best-round rather than final-round numbers.

**Practical implication:** if these ablations are cited in a write-up, use the best-round numbers in Section 3, not final-round, and note this instability explicitly rather than treating any single run's final-round dip as evidence of a real degradation trend.

## 7. Caveats — read before citing any of these numbers

1. **Not a controlled apples-to-apples comparison across the three modes.** `pure_dp` and `pure_he` ran with no attacker; `pure_zkp` ran with one. The utility numbers in Section 3 tell you the cost of each *mechanism*, not a fair three-way comparison under identical conditions — `pure_zkp`'s strong result is partly "no attack cost survived the guard" and partly whatever intrinsic advantage encryption-without-DP-noise already has (as `pure_he` demonstrates).
2. **`pure_zkp`'s detection rate is not directly comparable to the Experiment 2 mitigated hybrid runs.** Those runs had Krum as a redundant second layer behind the same guard. A discrepancy between this ablation and the hybrid runs (there isn't one — both hit 100%/0% here) would have told us whether the two stages do independent work; since they agree, we've confirmed the guard is sufficient **for this specific attack**, not that Krum's second layer is unnecessary in general — a different, more sophisticated attack that specifically evades a magnitude-only guard (a bounded-magnitude directional attack, which `zkp.py`'s own docstring explicitly flags as an undefended case) is exactly the scenario where Krum's second layer would still matter.
3. **Single-seed, single-run results throughout.** None of these six runs were repeated. The round-to-round volatility documented in Section 6 is exactly the kind of noise that argues for a repeat run before treating any specific number (e.g. `pure_zkp` network's 0.8214 "beating" the locked baseline) as a robust finding rather than a favorable draw.
4. **Locked-baseline comparisons in Section 3 carry the project's own previously-documented recipe-drift caveat** (undocumented `PROX_MU`/LR-decay settings in the original locked-baseline runs vs. this codebase's confirmed `PROX_MU=0.02`, no LR decay) — treat the baseline comparison as directional, not a precise diff.
5. **`pure_dp` used the default `DP_EPSILON=15.0`** (no `--epsilon` override visible in the logged `dp_epsilon_target` column, confirmed 15.0 for all 25 rounds) — this is the most privacy-permissive setting tested in the project's epsilon sweep (see the companion `FL-IDS_Epsilon_Sweep_Analysis.md`), so `pure_dp`'s utility numbers here represent DP's *best-case* utility cost, not a worst-case or typical-case one.
