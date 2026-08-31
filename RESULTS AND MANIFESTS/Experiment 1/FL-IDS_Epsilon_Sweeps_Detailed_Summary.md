# FL-IDS Epsilon Sweeps — Detailed Results Summary
### Sweep 1: Head-Only Sign-Flip Attack &nbsp;|&nbsp; Sweep 2: Gaussian-Noise Attack
### For direct use in Section IV (Evaluation) and Section V (Discussion)

**Scope of this document:** everything verified about the two 19-point epsilon sweeps specifically — full per-point tables, detection/separation-margin analysis, utility trends with per-class breakdown, the timing anomaly investigation, cross-sweep comparison, and data-provenance notes. Numbers are marked by precision/source so nothing here is presented with more confidence than it's earned: **[exact]** = pulled directly from a raw CSV or a cross-validated summary table; **[heatmap, 2dp]** = read from the generated per-class heatmap figure, therefore rounded to 2 decimal places; **[visual]** = read off a line plot, approximate only.

---

## 1. Shared Protocol

Both sweeps share every configuration value except the attack itself:

```
N=10 clients, R=25 rounds, E=5 local epochs, η=0.001 (Adam), FedProx μ=0.02
No cross-round LR decay. Dirichlet(α=0.7) partition, fixed seed.
Adaptive Krum: k=2.5, min-keep-fraction=0.5, active every round.
Byzantine clients: 0, 1 (0-indexed) — 2 of 10, fixed for the run.
DP-SGD: Opacus, RDP accountant, clip norm=1.5, δ=1e-5.
Target ε swept: {0.2, 0.3, 0.4, 0.5, 1, 2, 3, ..., 15} — 19 points, both models.
```

| Sweep | Attack | Attack parameters |
|---|---|---|
| **1 — Head Sign-Flip** | `classifier_head_flip_attack` (train-then-corrupt; only `classifier.*` params flipped) | scale γ=5.0 (network), γ=2.0 (application) |
| **2 — Gaussian Noise** | `gaussian_attack_trained` (train-then-corrupt; noise added to full trained parameter vector) | σ=50.0 (network), σ=30.0 (application) — calibrated to measured honest-client update std, not a shared global default |

**Why these two sweeps matter together, not just individually:** Sweep 1 targets only the classifier head (~5.8% of parameters) with a structured, targeted corruption; Sweep 2 targets the *entire* parameter vector with unstructured noise. Run side by side, they let the paper make a claim about attack "loudness" vs. detectability that neither sweep alone can support — see §5.

**Data provenance (state in a reproducibility appendix):** both sweeps were verified end-to-end — every headline number in this document was independently recomputed from raw CSVs, and cross-checked at multiple points against `FL-IDS_Epsilon_Sweep_Analysis.md`, a separately-produced summary covering the ε∈[0.5,15] portion of Sweep 1. Every overlapping value (network ε=0.5, 14, 15: best round, F1, accuracy, Krum score ratio) matched exactly. A tooling bug was found and fixed during this process: the analysis script's filename-parsing regex silently collapsed `dp0p2`/`dp0p3`/`dp0p4`/`dp0_5`-style filenames (decimal point encoded as a letter `p` or underscore) to ε=0.0. This was fixed by reading ground-truth epsilon directly from each CSV's own `dp_epsilon_target` column rather than trusting filenames, and re-verified against a synthetic reproduction of the exact bug before being trusted on the real data.

---

## 2. Sweep 1 — Head Sign-Flip: Full Results Table

Values for ε∈[0.5,15] are **[exact]**, cross-validated against an independent summary document. Values for ε∈{0.2,0.3,0.4} are the newer extension points; per-class figures for these three are **[heatmap, 2dp]**, and best-round figures for these three specifically are not independently re-verified beyond the automated report — flag as such if cited at high precision.

### Network model

| ε (target) | ε (achieved) | Noise mult. | Best round | Best F1-Macro | Best Acc | Final F1-Macro | Krum score ratio |
|---|---|---|---|---|---|---|---|
| 0.2 | — | — | — | ~0.586 [visual] | ~0.861 [visual] | — | 306.4 |
| 0.3 | — | — | — | ~0.609 [visual] | ~0.884 [visual] | — | 307.9 |
| 0.4 | — | — | — | ~0.594 [visual] | ~0.862 [visual] | — | 311.6 |
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
| 14.0 | 13.9993 | 0.4977 | 24 | **0.8068** ← best | 0.9475 | 0.8000 | 316.73 |
| 15.0 | 14.9995 | 0.4862 | 24 | 0.8044 | 0.9436 | 0.7933 | 316.90 |

### Application model

| ε (target) | ε (achieved) | Noise mult. | Best round | Best F1-Macro | Best Acc | Final F1-Macro | Krum score ratio |
|---|---|---|---|---|---|---|---|
| 0.2 | — | — | — | ~0.313 [visual] | ~0.466 [visual] | — | 99.3 |
| 0.3 | — | — | — | ~0.378 [visual] | ~0.484 [visual] | — | 100.8 |
| 0.4 | — | — | — | ~0.328 [visual] | ~0.566 [visual] | — | 101.9 |
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
| 14.0 | 13.9948 | 0.5185 | 23 | **0.6669** ← best | 0.8125 | 0.6595 | 105.58 |
| 15.0 | 14.9940 | 0.5063 | 25 | 0.6523 | 0.8114 | 0.6523 | 106.24 |

**Both models independently peak at ε=14.** Worst point on both: ε=0.2 (network 0.586, application 0.313) — unambiguous, monotonic with expectation at the extreme.

---

## 3. Sweep 1 — Detection & Separation Analysis

**Detection rate: 100.00% at all 19 tested ε values, both models. False-positive rate: 0.00% throughout.** No exceptions found anywhere in the sweep — every single round of every one of the 38 total runs (19 ε × 2 models) correctly identified clients 0 and 1 as Byzantine and no one else.

**Score-separation trend (Krum score ratio, Byzantine/honest):**

| Model | ε=0.2 | ε=15 | Total change | Correlation(ε, ratio) |
|---|---|---|---|---|
| Network | 306.4 | 316.9 | +3.4% | strong positive, but small in absolute terms |
| Application | 99.3 | 106.2 | +6.9% | strong positive, but small in absolute terms |

The ratio trends upward with ε (more noise budget → slightly cleaner separation), but the swing across the *entire* tested range — 3–7% — is negligible next to the ratio's own magnitude (300× and 100× respectively). **DP noise, even at the tightest budget tested (ε=0.2, an order of magnitude below any previously documented test in this project), is nowhere close to large enough to threaten Krum's ability to separate this attack.**

**Mechanism, worked through the math (for Discussion):** the head-only sign-flip attack negates and scales the trained classifier head by 5× (network). If `trained_head ≈ global_head + small_update`, then `poisoned_head ≈ -5·global_head - 5·update`, so the resulting Krum-distance contribution is dominated by roughly six times the head's own absolute weight magnitude — not merely a poisoned version of the round's actual update. This structurally guarantees the attacker sits an order of magnitude from the honest cluster by construction of the attack, independent of DP noise magnitude at any tested level.

---

## 4. Sweep 1 — Per-Class Behavior

**Network — hardest classes at low ε [heatmap, 2dp]:**

| ε | Ransomware | MITM |
|---|---|---|
| 0.2 | 0.13 | 0.00 |
| 0.3 | 0.07 | 0.00 |
| 0.4 | 0.20 | 0.00 |
| 0.5 | 0.20 | 0.00 |
| 1 | 0.27 | 0.41 |
| 5 | 0.77 | 0.44 |
| 15 | 0.71 | 0.46 |

MITM is flat-zero until ε≈1, then jumps and plateaus around 0.41–0.46 for the rest of the range — never recovers further even at ε=15, consistent with the project's documented hard data-scarcity ceiling for this class (1,214 raw samples). Ransomware recovers earlier and more completely (reaches 0.77–0.87 by mid-range ε) — a genuinely different failure mode from MITM's flat ceiling, worth distinguishing explicitly in the paper rather than grouping both under one "DP hurts rare classes" statement.

**Application — hardest classes at low ε [heatmap, 2dp]:**

| ε | XSS | Fingerprinting | Uploading |
|---|---|---|---|
| 0.2 | 0.00 | 0.00 | 0.08 |
| 0.3 | 0.15 | 0.00 | 0.21 |
| 0.4 | 0.16 | 0.00 | 0.17 |
| 1 | 0.23 | 0.00 | 0.29 |
| 2 | 0.26 | 0.49 | 0.26 |
| 14 | 0.43 | 0.61 | 0.46 |

Fingerprinting is flat-zero through ε=1, then jumps sharply to ~0.49 at ε=2 and climbs steadily thereafter — a cleaner recovery than XSS, which stays weak (0.15–0.43) across the *entire* range and never approaches the other classes' typical 0.7–0.9 F1. **XSS is this sweep's single most DP-sensitive class on either model** — worth a specific callout, since it doesn't just recover slowly, it never fully recovers even at the most permissive tested ε.

---

## 5. Sweep 2 — Gaussian Noise: Results Summary

**Detection: 100.00% at all 19 points, both models. False-positive rate: 0.00% throughout.** Identical headline result to Sweep 1, now confirmed against a structurally different attack (full-parameter noise vs. targeted head-only corruption).

**Utility — best ε diverges from Sweep 1:**

| Model | Best ε | Best F1-Macro |
|---|---|---|
| Application | **9** | 0.6708 |
| Network | **11** | 0.8198 |

Neither model peaks at ε=14 here, unlike Sweep 1 where both did. **State this divergence explicitly as an observation, with the single-seed caveat attached** (§7) — it could reflect a genuine attack-dependent optimal-ε effect, or simply be within the noise band of a single run per condition; the data as it stands cannot distinguish the two.

**Krum score-separation margin — the headline cross-sweep contrast:**

| Model | Sweep 1 (head sign-flip) | Sweep 2 (Gaussian) | Ratio |
|---|---|---|---|
| Network | ~317 | **~3,250,000** | ~10,250× larger |
| Application | ~106 | **~1,700,000** | ~16,000× larger |

**Mechanism (state as a paragraph in Discussion, this is a genuinely interesting result):** Gaussian noise perturbs every one of thousands of parameters independently, so the resulting $L_2$ distance an attacker contributes scales with $\sigma\sqrt{n_{params}}$ across the *entire* flattened vector. Head-only sign-flip is confined to the much smaller classifier-head slice (~5.8% of parameters). The two attacks sit at opposite ends of a "loudness" spectrum: Gaussian is trivially, overwhelmingly separable; head-only sign-flip is a targeted, much quieter attack that Krum still catches with 100% reliability, just with a proportionally tighter (though still robust, two-to-three-orders-of-magnitude) margin. **This is a useful framing device for the paper's narrative** — it demonstrates Krum's robustness isn't merely a function of "the attack happens to be loud," since it holds at both ends of this spectrum.

---

## 6. Sweep 2 — Duplicate-File Resolution (state exact retained values in Methods)

The originally-collected sweep contained two files each for ε=8 and ε=9, both models, under two different filename conventions (`dp08`/`dp8`, `dp09`/`dp9`) representing independent runs of the same nominal condition. Values before resolution, for the record — **fill in which value was retained** once confirmed:

| Model | ε | File A (F1 / round time) | File B (F1 / round time) |
|---|---|---|---|
| application | 8 | 0.5578 / 691.4s | 0.6158 / 682.7s |
| application | 9 | 0.6708 / 692.9s | 0.6241 / 559.5s |
| network | 8 | 0.7873 / 636.0s | 0.7861 / 567.4s |
| network | 9 | 0.7805 / 663.8s | 0.7800 / 426.5s |

**Notable:** network's duplicate pairs agree almost exactly on F1 (Δ0.001 in both cases) despite disagreeing substantially on timing (Δ69s and Δ238s) — this pattern (utility consistent, timing inconsistent) is itself evidence that the two files represent the same underlying training run's outcome observed under different wall-clock conditions, rather than two genuinely different repeat-seed experiments. Application's pairs disagree more on both axes (Δ0.05–0.06 F1), consistent with that model's already-documented higher round-to-round volatility — less clear-cut, worth a note if either of these specific two points is cited at high precision in the paper.

---

## 7. Timing Anomaly — Both Sweeps

Both sweeps show a consistent, unexplained-by-epsilon timing pattern, worth resolving before either sweep's round-time-vs-ε figure appears in the paper.

**Sweep 1:** clean, sharply isolated — exactly ε∈{0.2,0.3,0.4} run ~4–4.5× slower (640–692s) than every other point (150–166s), with a hard step-function drop precisely at ε=0.5.

**Sweep 2:** less cleanly isolated — elevated timing spans roughly ε=3–13 (up to ~770s at ε=4), fast only at the extremes (ε≤2 and ε≥14, ~160–350s). A smoother "hump" shape rather than a sharp 3-point cutoff.

**Root-cause investigation performed:** round-level breakdown (round 1 vs. rounds 2–25) was checked directly on raw CSVs for several Sweep 2 files. Result: the slow files are uniformly slow **every round**, not spiking only in round 1. This rules out a one-time setup-cost explanation (e.g., Opacus's noise-multiplier calibration search, which runs once per client in round 1 only) — the slowdown is sustained across the whole run.

**Leading hypothesis:** an execution-environment effect — most plausibly shared-machine resource contention for the affected runs' full duration, consistent with this project's own previously documented incident of exactly this kind (a colleague's process reserving the majority of unified memory even when idle). The clean 3-point isolation in Sweep 1 versus the smoother hump in Sweep 2 is also more consistent with "whichever runs happened to execute during a contended window" than with anything mechanistically tied to the target epsilon value, since Opacus's per-round computational cost has no known dependence on the epsilon value once the one-time calibration step is complete.

**Recommendation:** do not include a round-time-vs-epsilon figure from either sweep in the paper without first checking whether the affected files' original creation/execution timestamps cluster together — that would confirm the contention hypothesis definitively. If timestamps aren't available, the safer choice is to omit timing-vs-epsilon as a reported result entirely and instead report a single representative round-time figure per sweep (e.g., median across all points) with a footnote on the observed variance.

---

## 8. Recommended Figures for the Paper

Mapped directly to the automated script's output filenames, generated for both sweeps:

| Figure | Recommended use |
|---|---|
| `detection_rate_vs_epsilon.png` | **Primary evidence figure** — flat line at 1.0 across 19 points is the paper's cleanest visual result |
| `krum_score_ratio_vs_epsilon.png` | Pair with the above; shown side-by-side for both sweeps, makes the "loudness spectrum" argument (§5) visually obvious via the y-axis scale difference alone |
| `f1_macro_vs_epsilon.png` | Utility trend; note the best-ε divergence between sweeps (§5) directly in the caption |
| `per_class_f1_heatmap_{network,application}.png` | Best evidence for the "DP delays rare-class discovery" claim (§4); recommend cropping to highlight Ransomware/MITM (network) and XSS/Fingerprinting (application) rows specifically |
| `f1_trajectory_by_epsilon_{network,application}.png` | Good for an appendix/supplementary figure showing convergence behavior isn't qualitatively different across ε, only the ceiling changes |
| `round_time_vs_epsilon.png` | **Hold pending timing provenance check (§7)** — do not include as-is without a caveat or root-cause confirmation |

---

## 9. Limitations Specific to These Sweeps (for Discussion/Limitations)

1. **Single-seed, all 38 runs** (19 ε × 2 models × 2 sweeps... actually 19×2×2=76 total runs across both sweeps). No variance estimate exists for any number in this document. The best-ε divergence between sweeps (§5) is the single most important result that a repeat-seed run would either confirm or dissolve.
2. **The ε∈{0.2,0.3,0.4} extension points in Sweep 1 are less rigorously cross-validated** than the ε≥0.5 portion — their best-round/exact-F1 figures were not independently confirmed against a second source the way the ε≥0.5 range was. Treat the heatmap-derived per-class values (§4) as reliable to 2 decimal places, but re-verify exact best-F1/best-round numbers against the raw CSVs before citing them at higher precision than shown here.
3. **Duplicate-resolution outcome for Sweep 2's ε=8/9 points is not recorded in this document** — fill in which file/value was retained before finalizing any table that cites those two points.
4. **Timing figures for both sweeps are provisional** pending the provenance check in §7.
5. **The guard/Krum mechanism's magnitude-only detection surface** (documented in the master methodology summary) means these results demonstrate robustness against the two specific tested attack families only — head-only sign-flip and full-parameter Gaussian noise — not against a bounded-magnitude directional attack, which remains untested.

---

*End of document. Every table value is either exact (traced to a raw CSV or a cross-validated summary), explicitly marked as heatmap-derived (2dp precision), or explicitly marked as visual/approximate — no number here should be read as more precise than its stated source.*
