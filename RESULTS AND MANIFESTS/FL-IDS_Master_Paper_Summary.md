# FL-IDS: Master Methodology & Results Summary
### Consolidated for paper drafting — Sections III (Proposed Method) and IV (Evaluation)

**Purpose of this document:** everything verified, computed, and cross-checked across the full analysis pass on this project — the single-mechanism ablations, the Krum-necessity comparison, and both 19-point epsilon sweeps — organized the way it would appear in the paper, with every number traced back to a source file rather than reconstructed from memory. Where a number is still open (unresolved duplicate, unconfirmed timing cause), it is flagged as such rather than silently rounded into something clean.

---

# PART A — METHODOLOGY (for Section III)

## A.1 System Overview

A horizontal federated learning setup with **N=10** edge/IoT gateway clients and one aggregation server, training two independently-scoped intrusion-detection models over **R=25** communication rounds on a non-IID (Dirichlet, α=0.7, fixed seed) partition of Edge-IIoTset. Each round: local training → optional attack (Byzantine clients only) → optional privacy mechanism (DP-SGD and/or partial HE and/or norm-bound guard) → server-side aggregation (plain FedProx average, or adaptive Krum, or both) → new global model.

## A.2 Dual-Model Architecture

| Model | Features | Classes |
|---|---|---|
| Network-layer | 39 | Normal, DDoS_UDP, DDoS_ICMP, Ransomware, DDoS_HTTP, DDoS_TCP, Vulnerability_scanner, MITM |
| Application-layer | 90 | Normal, SQL_injection, Uploading, Backdoor, Port_Scanning, XSS, Password, Fingerprinting |

Identical CNN+LSTM topology for both (Conv1D→Norm→ReLU→MaxPool ×2 → LSTM → FC head), differing only in input width and which normalization/recurrent layer variant is active (see A.5).

## A.3 Federated Training Recipe (fixed across every condition below)

```
N=10 clients, R=25 rounds, E=5 local epochs, η=0.001 (Adam)
FedProx μ = 0.02          (confirmed via configs and user confirmation — NOT 0.1)
Cross-round LR decay: NONE (confirmed — a documented dead code path, never invoked)
Within-round schedule: StepLR(step_size=3, γ=0.95), reset every round
Dirichlet partition: α=0.7, fixed seed → identical client splits across every condition
```

**Reproducibility note (state in paper):** the client data partition is deterministic (seeded), but model weight initialization and DP-SGD's Gaussian noise draws are **not** seeded anywhere in the current codebase. Exact bit-for-bit reproduction of any single run is not currently possible without adding explicit `torch.manual_seed()` calls — worth stating as a limitation rather than implying full reproducibility.

## A.4 Threat Model

Static set of f ∈ {1,2} Byzantine clients (0-indexed clients 0,1 throughout), unknown to the server a priori. Every attack variant used in this project trains the Byzantine client normally first, then corrupts the *result* — the literature-standard "train-then-corrupt" formulation (Blanchard et al. / RSA / SpectralKrum / FedSV), not a replay-of-the-untouched-global-model attack.

| Attack | Formula | Where used |
|---|---|---|
| **Full-model sign-flip** | $w_i' = -\gamma \cdot w_i^{trained}$, γ=5.0 (network) / 2.0 (application) | Krum-only, No-Defence conditions |
| **Head-only sign-flip (stealthy)** | Same, applied only to `classifier.*` parameters — backbone submitted clean | Pure-guard condition; probes the partial-HE blind spot |
| **Gaussian noise** | $w_i' = w_i^{trained} + \mathcal{N}(0,\sigma^2 I)$, σ=50 (network) / 30 (application), calibrated to measured honest-update std | Gaussian-noise epsilon sweep |

## A.5 Defense Mechanisms

**Local DP (DP-SGD).** Opacus `PrivacyEngine`, RDP accountant, per-sample gradient clip $C=1.5$, $\delta=10^{-5}$. Noise multiplier solved once per client (round 1) via `make_private_with_epsilon`, cached and reused for all subsequent rounds. **Architecture coupling — important for the paper:** `dp_safe` is derived directly from `USE_DP` in the codebase (a single switch, not independent). Any DP-active condition uses `GroupNorm`+`DPLSTM` instead of the standard `BatchNorm`+`LSTM` — meaning DP-active conditions differ from DP-inactive ones by *both* noise injection *and* architecture. This should be stated explicitly wherever a DP-vs-non-DP utility comparison is drawn; the two costs are not currently separable from existing runs.

**Adaptive Multi-Krum.** MAD-threshold variant of Krum: for each accepted client, a score is the sum of squared distances to its $n-f-2$ nearest neighbors; clients whose score exceeds $\text{median}(s) + k \cdot 1.4826 \cdot \text{MAD}(s)$ are excluded, subject to a 0.5 min-keep-fraction floor. $k=2.5$ default.

**Partial Homomorphic Encryption (CKKS).** Only the classifier head (~5.8% of total parameters, confirmed identical ratio on both models despite the large feature-count difference — the head's absolute size is architecture-fixed) is CKKS-encrypted; the backbone (~94%) is transmitted plaintext. Single server-generated public context (fixes an earlier per-client-keypair bug that broke homomorphic summation validity). Parameters used in every HE-touching condition analyzed here: $n=8192$, coefficient chain $[60,40,40,60]$, scale $2^{40}$ — **TenSEAL's standard 128-bit security configuration**, not the separate RAM-constrained Docker path's $n=4096$/64-bit configuration described elsewhere in the project. State 128-bit explicitly for these results.

**Norm-Bound Commitment Guard** (not a real ZKP — stated honestly as such). Each client computes the $L_2$ norm of its classifier-head *delta* (trained head − round-starting global head), HMAC-signs it bound to a hash of the submitted ciphertext bytes, and the server applies the same MAD-threshold formula as Krum to the vector of verified norms — the only quantity an encrypted slice reveals without decryption. **Known, structural scope limitation:** magnitude-only; cannot detect a bounded-magnitude, direction-only corruption. State this explicitly, do not generalize detection results beyond the tested attack family.

## A.6 Single-Mechanism Ablation Conditions

Five conditions, all sharing the recipe in A.3 exactly (confirmed $\mu=0.02$, no LR decay for **every** condition including Krum-only/No-Defence, per direct confirmation):

| Condition | DP | HE | Guard | Krum | Attack |
|---|---|---|---|---|---|
| Krum-only | – | – | – | ✓ | Full-model sign-flip |
| No-Defence | – | – | – | – | Full-model sign-flip |
| Pure-DP | ✓ | – | – | – | – |
| Pure-HE | – | ✓ | – | – | – |
| Pure-guard | – | ✓ | ✓ | – | Head-only sign-flip |

**Krum-only vs. No-Defence** is the paper's cleanest isolated demonstration of adaptive Krum's necessity — identical everything except whether the robust aggregator runs. **Recommended as the opening result of the Evaluation section**, before DP or HE are introduced.

**Architecture note for Krum-only:** since `USE_DP=False`, this condition runs on the standard (non-DP-safe) architecture — architecture-matched to Pure-HE/Pure-guard, **not** to any point in the DP epsilon sweep (all of which use the DP-safe substitution). Do not read Krum-only as the literal $\varepsilon\to\infty$ endpoint of the DP sweep.

## A.7 Epsilon Sweep Protocol (two sweeps)

Both sweeps: adaptive Krum active throughout (k=2.5), 2/10 clients Byzantine (0,1), 25 rounds, both models, target $\varepsilon \in \{0.2, 0.3, 0.4, 0.5, 1, 2, ..., 15\}$ — **19 points per model**, the densest grid produced in this project to date (extends the earlier documented 13-point $\{0.5,...,14\}$ grid down to 0.2 and up to 15).

- **Sweep 1 — Head Sign-Flip:** attack = head-only sign-flip (the stealthy, classifier-head-only variant).
- **Sweep 2 — Gaussian Noise:** attack = full-parameter Gaussian noise, σ calibrated per model (A.4).

**Data provenance note for the paper's reproducibility statement:** both sweeps were independently verified end-to-end against raw CSVs and cross-validated against a separately-generated summary document (`FL-IDS_Epsilon_Sweep_Analysis.md`) at multiple overlapping points (network ε=0.5, 14, 15) — every checked value matched exactly. A filename-parsing tool bug (ambiguous `dp0p2`/`dp0_5`-style decimal encoding silently collapsing to ε=0.0) was found and fixed by reading the ground-truth epsilon directly from each CSV's own `dp_epsilon_target` column rather than trusting filenames — worth a one-line methodological note if this tooling detail is relevant to a reproducibility appendix. The Gaussian sweep originally had duplicate files at ε=8 and ε=9 for both models (two independent runs under different filename conventions for the same nominal condition); one file per duplicated point has since been removed by the author, leaving a clean 19-file, 19-point grid per model for both sweeps.

---

# PART B — RESULTS (for Section IV)

## B.1 Krum Necessity: Krum-only vs. No-Defence

The single most dramatic, cleanest-attributable result in this project.

| Condition | Application (best F1 / acc) | Network (best F1 / acc) |
|---|---|---|
| Krum-only | 0.7979 / 0.8913 (round 24) | 0.8651 / 0.9856 (round 25) |
| **No-Defence** | **0.063 / 0.396 — flat, every round, no learning at all** | **~0.045 peak — oscillates, never converges** |

**Mechanism (state in Discussion):** absent a robust aggregator, the two attackers' 5×-scaled, sign-flipped updates enter the plain FedProx weighted average directly every round. The proximal term (Eq. for FedProx) then pulls every *honest* client's next local update back toward this already-corrupted global model — propagating the corruption to clients that behaved correctly. A per-client breakdown at round 10 (network, No-Defence) confirms this: **every one of the 10 clients**, not just the two attackers, sits at 0.001–0.17 local accuracy, versus 0.85–0.98 for every honest client at the same round under Krum-only.

**Data-quality note:** two of the four raw source files (`Krum_Baseline_Network.csv` round 12; `Krum_No_Defence_Application.csv` round 10) contained a duplicated block of per-client rows from an apparent checkpoint-resume artifact — the aggregate MEAN row was unaffected in both cases, so the headline numbers above are unaffected, but any per-client table built from those two specific rounds should be deduplicated first.

## B.2 Pure-DP / Pure-HE / Pure-guard — Component-Level Utility & Detection

| Run | Best round | Best F1-Macro | Best Accuracy | Mean round time |
|---|---|---|---|---|
| network / Pure-DP | 23 | 0.7902 | 0.9393 | 169.4s |
| network / Pure-HE | 21 | 0.8095 | 0.9506 | 74.0s |
| network / Pure-guard | 25 | **0.8214** ← best of all six | 0.9500 | 63.6s |
| application / Pure-DP | 24 | 0.5905 | 0.7501 | 196.4s |
| application / Pure-HE | 25 | 0.7585 | 0.8698 | 51.9s |
| application / Pure-guard | 24 | 0.7473 | 0.8642 | 42.8s |

**Detection (Pure-guard only, live attacker):** 100% detection, 0% false positives, both models, all 25 rounds — clients 0,1 flagged every round, no one else ever flagged. Verified independently against both the CSV `zkp_rejected` column and the console log's own diagnostic line.

**Pure-DP is the slowest condition by a wide margin** (2.3–3.8× the HE-based runs) — mechanistically explained by Opacus's per-sample gradient hooks plus the non-fused `DPLSTM` replacement for cuDNN's standard `LSTM`, both tied to the same `dp_safe=True` architecture substitution (A.5).

**Guard-alone result beats the clean HE baseline while under active attack** (0.8214 vs. 0.8095) — not a paradox: the guard excises both attackers *before* aggregation, so the surviving 8 honest clients' contribution is computationally indistinguishable from an unattacked 8-client run. Combined with single-seed sampling variance, this is a real but not-yet-repeated result (see Limitations).

## B.3 Epsilon Sweep 1 — Head Sign-Flip Attack

**Detection: 100.00% at every one of 19 tested epsilon values (0.2–15), both models, 0% false positives throughout.** This is the strongest version of the "Krum survives DP noise" finding produced in this project — previously validated only down to ε=3 or ε=0.5 in earlier, coarser sweeps; now confirmed down to ε=0.2, an order of magnitude tighter than any prior test.

**Krum score-separation margin:** network 306.4 → 316.9 (+3.4% across the whole range), application 99.3 → 106.2 (+6.9%) — essentially flat. DP noise, even at the tightest tested budget, does not meaningfully compress the honest/Byzantine score gap.

**Utility:** both models independently peak at **ε=14** (network F1=0.8068/acc=0.9475; application F1=0.6669/acc=0.8125) — reconfirms and extends the project's earlier retraction of an "ε=9 sweet spot," now validated on a materially denser grid.

**Per-class pattern:** network's Ransomware (0.13→0.87) and MITM (0.00 until ε≈1, plateaus ~0.41–0.46) are hit hardest at extreme-low ε; application's XSS (0.00 until ε≈14) and Fingerprinting (0.00 until ε≈2) show the same shape — directly reconfirming "DP delays rare-class discovery," now demonstrated down to ε=0.2 (previously the lowest tested point was ε=3).

**Timing anomaly (flag, do not silently report):** ε∈{0.2,0.3,0.4} run ~4–4.5× slower than every other point (640–692s vs. 150–166s), both models, with a clean step-function drop exactly at ε=0.5. Round-level inspection shows this is a **sustained** slowdown (every round elevated, not a one-time round-1 setup spike) — most consistent with an execution-environment effect (shared-machine contention for that run's full duration) rather than anything intrinsic to the epsilon value. **Do not report a round-time-vs-epsilon figure from this sweep without first confirming (e.g. via file creation timestamps) that this reflects environment, not a genuine epsilon-dependent cost.**

## B.4 Epsilon Sweep 2 — Gaussian Noise Attack

**Detection: 100.00% flat, both models, all 19 points, 0% false positives.** Same headline result as Sweep 1, now confirmed against a structurally different (full-parameter, not head-only) attack.

**Krum score-separation margin — dramatically larger than Sweep 1:** network ~3.25M, application ~1.7M, versus ~317 and ~106 respectively for the head-only sign-flip attack. **Mechanistic explanation for the paper:** Gaussian noise perturbs the *entire* flattened parameter vector, so its $L_2$ norm scales with $\sigma \cdot \sqrt{n_{params}}$ across thousands of dimensions, versus sign-flip's perturbation confined to the much smaller classifier-head slice. Frame this as a spectrum: Gaussian is a "loud," trivially-separable attack; head-only sign-flip is a targeted, much quieter attack that Krum still catches perfectly, with a proportionally tighter (but still robust, ~100–300×) margin.

**Utility:** application best at **ε=9** (F1=0.6708); network best at **ε=11** (F1=0.8198) — a genuine divergence from Sweep 1, where both models peaked at ε=14. Worth reporting as an observation (optimal ε may be attack-dependent) but explicitly caveated as single-seed (B.5) rather than a confirmed effect.

**Duplicate-file resolution:** the original sweep contained two files each for ε=8 and ε=9 (both models) under differing filename conventions (`dp08`/`dp8`, `dp09`/`dp9`) for what appears to be the same nominal condition. Values before resolution, for the record:

| Model | ε | File A (F1 / round time) | File B (F1 / round time) |
|---|---|---|---|
| application | 8 | 0.5578 / 691.4s | 0.6158 / 682.7s |
| application | 9 | 0.6708 / 692.9s | 0.6241 / 559.5s |
| network | 8 | 0.7873 / 636.0s | 0.7861 / 567.4s |
| network | 9 | 0.7805 / 663.8s | 0.7800 / 426.5s |

Network's pairs agreed almost exactly on F1 (Δ0.001 both); application's disagreed more (Δ0.05–0.06), consistent with the application model's already-documented higher round-to-round volatility. Timing disagreed more than F1 in every case — mildly favors an environmental (not pure-seed-noise) explanation for the timing spread specifically. **One file per duplicated point has since been removed by the author** — state in the paper's methods which file/value was retained for ε=8 and ε=9 on each model, since this document cannot determine that after the fact.

**Timing:** shows the same qualitative pattern as Sweep 1 (fast at both extremes, elevated through the middle of the range) but less cleanly isolated — elevated timing spans roughly ε=3–13 rather than being confined to 3 points. Same recommendation: do not report round-time-vs-epsilon from this sweep without confirming an environmental vs. epsilon-dependent cause.

## B.5 Cross-Cutting Limitations (state explicitly in Discussion/Limitations)

1. **Single-seed throughout.** Every condition in Parts B.1–B.4 — ablations and both full 19-point sweeps — is a single run. No statistical variance estimate currently exists for any headline number. Highest-value next step: repeat a handful of conditions (e.g. the two sweeps' respective best-ε points, plus Krum-only/No-Defence) under ≥3 seeds before treating any single figure as more than a point estimate.
2. **DP architecture confound.** Every DP-active condition (Pure-DP; every point in both epsilon sweeps once DP is reintroduced, if applicable) uses `GroupNorm`+`DPLSTM` instead of the standard architecture used by every DP-inactive condition. Utility comparisons across this boundary currently conflate noise cost with architecture cost.
3. **Guard's structural blind spot** (A.5) — magnitude-only detection, untested against a bounded-magnitude directional attack. The 100% detection results in this document say nothing about that case.
4. **Timing figures need provenance confirmation** before inclusion — both sweeps show anomalous elevated round-times not attributable to the epsilon value itself by any known mechanism; most likely a shared-machine execution-environment effect, not yet confirmed via timestamps.
5. **Reproducibility:** data partitioning is seeded and reproducible; model initialization and DP-SGD noise are not. State this precisely rather than implying full reproducibility from the fixed partition alone.

---

# PART C — Suggested Section IV (Evaluation) Ordering

Based on the results above, a suggested presentation order for maximum narrative clarity:

1. **Krum necessity** (B.1) — the cleanest, most visually dramatic result; establishes why the rest of the paper's mechanisms matter at all.
2. **Component costs** (B.2) — Pure-DP/HE/guard, establishing what each mechanism costs or protects alone before composing them.
3. **Epsilon sweeps** (B.3, B.4) — the paper's core empirical contribution; present side-by-side (head-only vs. Gaussian) to let the score-separation-margin contrast (317 vs. 3.25M) land as a single clear finding about attack "loudness" vs. detectability.
4. **Limitations** (B.5) as a dedicated subsection or fold into Discussion — every item there is a genuine, traceable open question, not boilerplate.

---

*End of summary. Every numeric claim above was independently recomputed from raw CSVs, cross-validated against at least one independently-generated document where available, and traced to the specific code path producing it — not transcribed from any single source without verification.*
