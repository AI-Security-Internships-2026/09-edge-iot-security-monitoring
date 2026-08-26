# FL-IDS DP-Epsilon Sweep Analysis — 16-Point Extended Sweep

**Source data:** a `build_table`-style summary (one row per condition, not raw per-round CSVs) covering **16 epsilon values per model** (0.5, 1, 2, 3, ..., 15) x 2 models = 32 conditions, adaptive Krum active throughout (k=2.5), Byzantine attack ON (clients 1, 2) in every condition, 25 rounds each, all 32 completed with 0 NaN rounds.

**Important scope note, stated up front:** this is a **different, much finer-grained sweep** than the 3-point ε∈{3,9,15} sweep documented as "Experiment 1" in the project's master planning doc. The F1-Macro values at the three overlapping epsilon points (3, 9, 15) are close to but **not identical** to the originally-documented Experiment 1 numbers (e.g. this sweep's network ε=15 best F1 is 0.8044 vs. the master doc's 0.8068; application ε=15 is 0.6523 here vs. 0.6067 there). This is consistent with the master doc's own changelog note that any epsilon-sweep run **before** the sign-flip attack fix (revision 20, which corrected `sign_flip_attack` to train-first-then-negate) is not directly comparable to a later run — this sweep is very likely a re-run collected after that fix and other later corrections (Gaussian defaults, DP-safe FedProx, etc.), given the close-but-not-identical values and the 100% detection rate holding throughout. Treat this sweep as the more current, more granular dataset, and treat the original 3-point Experiment 1 numbers as superseded by it where they overlap — but this is an inference from the data, not confirmed against a run log, so state it as an assumption if cited.

## 1. Full sweep table

### Network model

| ε (target) | ε (achieved) | Noise multiplier | Best round | Best F1-Macro | Best Acc | Final F1-Macro | Krum score ratio |
|---|---|---|---|---|---|---|---|
| 0.5 | 0.4942 | 2.3413 | 25 | 0.6235 | 0.8826 | 0.6235 | 307.13 |
| 1.0 | 0.9964 | 1.4380 | 25 | 0.6650 | 0.8691 | 0.6650 | 308.43 |
| 2.0 | 1.9948 | 1.0129 | 25 | 0.7381 | 0.9068 | 0.7381 | 311.93 |
| 3.0 | 2.9954 | 0.8592 | 13 | 0.7339 | 0.8929 | 0.7066 | 313.16 |
| 4.0 | 3.9962 | 0.7711 | 23 | 0.7660 | 0.9359 | 0.7606 | 312.77 |
| 5.0 | 4.9958 | 0.7112 | 25 | 0.7819 | 0.9198 | 0.7819 | 311.86 |
| 6.0 | 5.9982 | 0.6666 | 24 | 0.7227 | 0.9190 | 0.6923 | 314.26 |
| 7.0 | 6.9968 | 0.6316 | 25 | 0.7748 | 0.9347 | 0.7748 | 314.47 |
| 8.0 | 7.9980 | 0.6030 | 23 | 0.7714 | 0.9244 | 0.7661 | 314.65 |
| 9.0 | 8.9975 | 0.5789 | 21 | 0.7837 | 0.9333 | 0.7525 | 316.48 |
| 10.0 | 9.9964 | 0.5584 | 24 | 0.7881 | 0.9360 | 0.7668 | 313.53 |
| 11.0 | 10.9981 | 0.5404 | 25 | 0.7588 | 0.9304 | 0.7588 | 316.77 |
| 12.0 | 11.9977 | 0.5246 | 16 | 0.7466 | 0.9095 | 0.6782 | 313.19 |
| 13.0 | 12.9989 | 0.5105 | 25 | 0.7995 | 0.9527 | 0.7995 | 316.05 |
| 14.0 | 13.9993 | 0.4977 | 24 | **0.8068** | 0.9475 | 0.8000 | 316.73 |
| 15.0 | 14.9995 | 0.4862 | 24 | 0.8044 | 0.9436 | 0.7933 | 316.90 |

### Application model

| ε (target) | ε (achieved) | Noise multiplier | Best round | Best F1-Macro | Best Acc | Final F1-Macro | Krum score ratio |
|---|---|---|---|---|---|---|---|
| 0.5 | 0.4949 | 2.6953 | 25 | 0.4541 | 0.6752 | 0.4541 | 100.79 |
| 1.0 | 0.9949 | 1.5985 | 24 | 0.4484 | 0.6593 | 0.3744 | 101.93 |
| 2.0 | 1.9944 | 1.0933 | 23 | 0.5605 | 0.7101 | 0.5574 | 102.07 |
| 3.0 | 2.9931 | 0.9183 | 25 | 0.5139 | 0.6315 | 0.5139 | 102.96 |
| 4.0 | 3.9948 | 0.8192 | 25 | 0.5620 | 0.7184 | 0.5620 | 103.43 |
| 5.0 | 4.9950 | 0.7526 | 25 | 0.6134 | 0.7675 | 0.6134 | 103.79 |
| 6.0 | 5.9954 | 0.7033 | 21 | 0.5515 | 0.6869 | 0.5133 | 103.30 |
| 7.0 | 6.9935 | 0.6647 | 21 | 0.5834 | 0.7454 | 0.5177 | 103.70 |
| 8.0 | 7.9959 | 0.6333 | 25 | 0.6460 | 0.7758 | 0.6460 | 104.15 |
| 9.0 | 8.9962 | 0.6070 | 23 | 0.6140 | 0.7626 | 0.6065 | 104.27 |
| 10.0 | 9.9960 | 0.5846 | 25 | 0.5862 | 0.7038 | 0.5862 | 104.56 |
| 11.0 | 10.9951 | 0.5652 | 25 | 0.5805 | 0.7243 | 0.5805 | 105.31 |
| 12.0 | 11.9945 | 0.5480 | 24 | 0.6467 | 0.8026 | 0.6260 | 105.85 |
| 13.0 | 12.9937 | 0.5326 | 24 | 0.6199 | 0.7703 | 0.6058 | 104.56 |
| 14.0 | 13.9948 | 0.5185 | 23 | **0.6669** | 0.8125 | 0.6595 | 105.58 |
| 15.0 | 14.9940 | 0.5063 | 25 | 0.6523 | 0.8114 | 0.6523 | 106.24 |

## 2. Headline result 1 — Krum robustness under DP noise, reconfirmed at 5x the resolution

**Detection rate is 100% at every single one of the 32 conditions** — every epsilon from 0.5 (the most noise tested anywhere in this project) up to 15.0, on both models. The original Experiment 1 only tested this at 3 points (ε=3, 9, 15); this sweep extends the same finding down to ε=0.5, an order of magnitude more noise than Experiment 1's most aggressive tested point, and it still holds.

The `krum_score_ratio` (mean Byzantine score / mean honest score) trend also reconfirms Experiment 1's original finding, now with much more data to support it:

| Model | ε=0.5 ratio | ε=15 ratio | Change | Correlation(ε, ratio) |
|---|---|---|---|---|
| Network | 307.13 | 316.90 | +3.18% | 0.833 |
| Application | 100.79 | 106.24 | +5.41% | 0.945 |

The ratio does trend upward with epsilon (strong correlation), but the *magnitude* of that trend — 3–5% total swing across the entire tested range — is small next to the sheer size of the ratio itself (300x and 100x separation between attacker and honest scores respectively). DP noise at these levels is nowhere close to large enough to meaningfully close that gap. This is the same conclusion Experiment 1 already reached, now confirmed across 16 points instead of 3, including territory (ε<3) Experiment 1 never actually tested.

## 3. Headline result 2 — the epsilon-utility relationship is real but noisy, not clean

Correlation between epsilon and best F1-Macro is positive and moderate-to-strong on both models (network: 0.758, application: 0.839) — more DP budget generally does help, as expected. But this relationship is **not monotonic**:

- **Network:** 6 of 15 epsilon-to-epsilon steps show a *decrease* in best F1-Macro despite epsilon increasing.
- **Application:** 8 of 15 steps show a decrease — over half.

This matters for how the "ε=9 might be a local sweet spot" hypothesis from the master doc's original 3-point Experiment 1 should be read. That hypothesis was based on ε=9 beating ε=15 at just 3 tested points. **This finer sweep does not support ε=9 specifically as a sweet spot** — the actual best-performing epsilon in this sweep is **ε=14 on both models** (network 0.8068, application 0.6669), and the F1-Macro curve is visibly bumpy across the whole range rather than having one clean local maximum. The original ε=9 anomaly looks, in retrospect, more like an instance of this pervasive round-to-round/condition-to-condition noise than a specific, reproducible sweet spot — which is exactly the outcome the master doc's own "needs repeat-seed confirmation" caveat anticipated.

**Worst epsilon on both models: ε=0.5** (network 0.6235, application 0.4541) — unambiguous, expected, not in question.

## 4. DP calibration — good, but not quite as tight as previously claimed

| Model | Worst calibration point | Achieved vs. target | Error |
|---|---|---|---|
| Network | ε=0.5 | 0.4942 vs 0.5 | 1.160% |
| Application | ε=0.5 | 0.4949 vs 0.5 | 1.020% |

The master doc's original Experiment 1 write-up claimed achieved epsilon was "within 0.25% of target in every case." Across this wider 16-point sweep, the **maximum** calibration error is closer to **1.0–1.2%**, concentrated specifically at the lowest tested epsilon (0.5) on both models — every other point stays comfortably under 0.25–0.5%. This isn't a contradiction of the earlier claim so much as a boundary effect the original 3-point sweep (which never tested below ε=3) simply never encountered: Opacus's noise-multiplier calibration search appears to lose a small amount of precision at very low target epsilon, where the required noise multiplier is largest (2.34–2.70 at ε=0.5, vs. 0.49–0.51 at ε=15). Worth stating this precisely (1.16% max, not "within 0.25% always") if this sweep is cited in place of the original 3-point numbers.

## 5. Best-round vs. final-round — reinforces the project's existing recommendation

| Model | Conditions peaking before round 25 | Mean best-final F1 gap | Largest single gap |
|---|---|---|---|
| Network | 9 of 16 | 0.0129 | 0.0684 (at ε=12) |
| Application | 8 of 16 | 0.0144 | 0.0740 (at ε=1) |

More than half the conditions on both models peaked before the final round — consistent with the master doc's documented recurring late-round instability pattern, now observed at scale across 32 independent runs rather than a handful of isolated occurrences. This is a strong, high-volume confirmation that **best-round reporting, not final-round, should be the default convention** for any future sweep or ablation in this project — using final-round numbers would understate true achievable utility by a non-trivial margin (up to 0.074 F1-Macro in the worst case here) in roughly half of all conditions.

## 6. Recommendations if this sweep is used in a write-up

1. **State explicitly that this supersedes the original 3-point Experiment 1 sweep** where they overlap (ε=3, 9, 15) — cite this dataset's numbers, not the master doc's original ones, and flag the likely reason for the difference (post-sign-flip-fix re-run) as an inference, not a confirmed fact, unless you can verify it against the actual run's `experiment_config_*.json` / code version.
2. **Retract or soften the "ε=9 sweet spot" claim** from the original Experiment 1 write-up — this finer sweep shows the epsilon-utility curve is bumpy throughout, with ε=14 (not ε=9) as the actual empirical best on both models, and no single point standing out as a clean, isolated local maximum.
3. **Report calibration accuracy as "≤1.16% across the full ε∈[0.5,15] range, tightening to <0.5% for ε≥1"** rather than a single "within 0.25%" figure — the earlier claim was accurate for the narrower range it was tested on but doesn't generalize to ε=0.5.
4. **Lead with the extended, 16-point version of Headline Result 1** (Krum detection ε-invariant down to ε=0.5, not just ε≥3) — this is a strictly stronger, more citable version of the original finding, at no extra cost since the data already exists.
5. **Use best-round numbers as the default** for any table or chart built from this sweep, given how frequently (17/32 conditions) the final round underperforms the run's actual peak.
6. **This sweep still doesn't resolve the single-seed-noise question** — the non-monotonicity in Section 3 is consistent with either (a) genuine epsilon-dependent volatility in this training setup, or (b) ordinary single-seed variance that would smooth out under repeated runs. A repeat-seed version of even a handful of these 32 conditions (e.g. the ε=9, ε=12, ε=14 points, where the local swings are largest) would be the highest-value next step for turning this from "a real, individually-observed pattern" into "a statistically supported one."
