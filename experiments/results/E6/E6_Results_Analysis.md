# Issue 5 / E6 — Calibration-Component Ablation: Results Analysis

**Condition:** Edge-IIoTset, Network IDS model, α = 0.3, ε = 5 (DP where applicable), bounded-directional attack, seeds {42, 123, 456, 789, 2024}, aggregator k = 3.5.

**Frozen parameters** (fixed before any final-TEST result was observed):
- `bounded_tau = 2.111776`, margin = 0.05 → Byzantine classifier-head delta norm target = 2.06178, verified every round in every run.
- Heterogeneity fit: `experiments/configs/hetero_fit_coeffs_a0.3.json` — 3 runs (seeds 7, 11, 13), α = 0.3, plain Adaptive Krum, DP off. `n_pairs_fit = 2100`, `r_squared = 0.486`.
  - `intercept = 10.8444`, `coef_n_samples_diff = 5.172e-05`, `coef_entropy_diff = -0.3476`.
- All 20 runs recorded git SHA `f5ff72f2` and target ε = 5.0; final composed ε ranged 4.99–5.00 across clients in every run.
- DP calibration metadata uses the public-median-honest-sigma policy (`calibrated_byz_sigma_policy = public_median_honest_sigma`) so Byzantine clients are not distinguishable from honest ones via the DP noise field alone.

**Threat model note:** Byzantine clients in this pipeline do not run DP-SGD (no Opacus engine is built for them), so the bounded-directional attacker's updates carry no DP noise while honest clients' updates do. This is a design property of the codebase, not a per-run artifact.

---

## 1. Table 4 — Calibration Ablation (n = 5 seeds, mean ± SD [95% CI])

| Variant | Honest FPR | Byzantine TPR | Macro-F1 | Rare-class AUC-PR |
|---|---|---|---|---|
| Full calibration | 0.000 ± 0.000 [0.000, 0.000] | 0.000 | 0.759 ± 0.090 [0.648, 0.871] | 0.739 ± 0.148 [0.555, 0.922] |
| DP calibration only | 0.115 ± 0.185 [−0.115, 0.345]* | 0.000 | 0.621 ± 0.255 [0.304, 0.937] | 0.573 ± 0.304 [0.197, 0.950] |
| Heterogeneity calibration only | 0.000 ± 0.000 [0.000, 0.000] | 0.000 | 0.755 ± 0.087 [0.648, 0.863] | 0.719 ± 0.132 [0.555, 0.883] |
| Off (plain Adaptive Krum) | 0.050 ± 0.069 [−0.035, 0.135]* | 0.000 | 0.611 ± 0.223 [0.334, 0.888] | 0.677 ± 0.166 [0.471, 0.882] |

\* Normal-theory CIs are invalid here (they include impossible negative FPR); see §3.

Rare-class metrics averaged over Ransomware, Vulnerability_scanner, MITM.

## 2. Seed-paired significance (Holm-corrected within metric)

All 30 pairwise comparisons across the four metrics were **non-significant** after Holm correction (all `p_holm ≥ 0.947`, most = 1.000). Smallest raw p-value observed: 0.158 (rare-class F1, Full vs. DP-only). With n = 5 paired seeds, the minimum attainable two-sided p-value under an exact sign/rank test is 0.0625, so this design has low power to detect anything short of a very large, seed-consistent effect.

Effect sizes (Cohen's dz) for the headline contrasts:
- Full vs. Off, Macro-F1: dz = +0.73 (diff = +0.148)
- Full vs. DP-only, Macro-F1: dz = +0.53 (diff = +0.138)
- Full vs. Hetero-only, Macro-F1: dz = +0.13 (diff = +0.004) — negligible
- DP-only vs. Off, Macro-F1: dz = +0.13 (diff = +0.010) — negligible

## 3. Per-seed detail (the data the aggregate table is averaging over)

| Seed | Variant | Macro-F1 | Rare AUC-PR | False Positives (of 250 client-rounds) | FP clients |
|---|---|---|---|---|---|
| 42 | Full | 0.8117 | 0.8203 | 0 | — |
| 42 | DP-only | **0.2178** | 0.1526 | **85** | 4(×25), 9(×25), 10(×25), 5(×10) |
| 42 | Hetero-only | 0.8112 | 0.7408 | 0 | — |
| 42 | Off | 0.3240 | 0.5531 | **25** | 10(×25) |
| 123 | Full | 0.7411 | 0.7550 | 0 | — |
| 123 | DP-only | 0.7572 | 0.7725 | 0 | — |
| 123 | Hetero-only | 0.7637 | 0.7795 | 0 | — |
| 123 | Off | 0.7170 | 0.7789 | 0 | — |
| 456 | Full | 0.8016 | 0.8010 | 0 | — |
| 456 | DP-only | 0.8259 | 0.7871 | 0 | — |
| 456 | Hetero-only | 0.8160 | 0.8057 | 0 | — |
| 456 | Off | 0.8057 | 0.8007 | 0 | — |
| 789 | Full | 0.8308 | 0.8360 | 0 | — |
| 789 | DP-only | 0.7841 | 0.8076 | 0 | — |
| 789 | Hetero-only | 0.7807 | 0.7837 | 0 | — |
| 789 | Off | 0.7870 | 0.8024 | 0 | — |
| 2024 | Full | 0.6103 | 0.4803 | 0 | — |
| 2024 | DP-only | 0.5189 | 0.3472 | **30** | 4(×5), 7(×25) |
| 2024 | Hetero-only | 0.6050 | 0.4867 | 0 | — |
| 2024 | Off | 0.4210 | 0.4471 | **25** | 7(×25) |

**Key observation:** every false-positive event, and every large Macro-F1 gap between variants, occurs in seeds **42** and **2024** only. In seeds 123, 456, and 789, all four variants reject nobody and produce nearly identical Macro-F1 (spread ≤ 0.046 within a seed). That spread is the run-to-run noise floor for this pipeline; treat any cross-variant difference smaller than ~0.05 as noise, not signal.

DP-only consistently has more (and more varied) false positives than Off in the two seeds where either rejects anyone — 85 vs. 25 in seed 42, 30 vs. 25 in seed 2024 — and DP-only's Macro-F1 is correspondingly the lowest of all four variants in seed 42.

## 4. Findings

1. **No variant detected the attack.** 0 of 250 Byzantine client-rounds were flagged TP in every one of the 20 runs (0/1000 pooled). Byzantine TPR carries no information in this experiment and should not be interpreted as a null defense result — it reflects that the attack, as frozen, is fully invisible to distance-based Krum scoring under this DP-noise/no-DP-noise asymmetry.
2. **Heterogeneity calibration eliminates the false rejections that plain Adaptive Krum produces.** Full and Hetero-only reject 0/1000 honest client-rounds across all seeds; Off and DP-only reject honest clients in 2/5 seeds (up to 85/250 in one run). The effect is real but seed-concentrated, which is why the paired tests are NS at n = 5.
3. **DP calibration adds no measurable benefit over heterogeneity calibration alone**, and in the two seeds where it matters, it performs worse than doing nothing (Off): more false positives (85 vs. 25 in seed 42; 30 vs. 25 in seed 2024) and lower Macro-F1.
4. **Mechanism for DP-only's false positives:** in seed 42, the four rejected clients (4, 5, 9, 10) were confirmed as the four *lowest*-sigma clients in that run (DP noise multipliers 0.95–1.68, vs. 2.12–2.19 for the four not rejected). DP calibration divides pairwise distance by a variance term that scales with σ², so a low-noise client receives the least discount and appears anomalously distant — the opposite of the intended effect. (Not yet re-confirmed in seed 2024 at time of writing — flagged as an open check, see §6.)
5. **Full ≈ Hetero-only in aggregate (dz = 0.13, diff = 0.004 Macro-F1)**, consistent with the DP term contributing negligibly relative to the heterogeneity term at this scale — DP variance was estimated (back-of-envelope, ~79k params, batch 512, clip 1.0) at ~10²–10³ per client, versus the fitted heterogeneity variance of ~10⁹–10¹¹ across honest pairs. This unit/scale mismatch was not independently re-verified against the final run logs and should be confirmed before stating it as fact in the paper.
6. **Full and Hetero-only reject nobody, ever.** Their good Macro-F1 (~0.76 mean) reflects an aggregator that behaves like FedAvg over all 10 clients (2 Byzantine + 8 honest) under this specific attack — not evidence of successful outlier rejection. E6 has no no-attack baseline, so the actual damage the attack does to the global model cannot be quantified from this experiment alone.

## 5. Caveats / limitations to state explicitly in the paper

- **Threat model:** the bounded-directional attacker trains without DP noise while honest clients use DP-SGD; this asymmetry is a property of the pipeline (Byzantine clients have no persistent DP state), not a tuning choice made for this experiment.
- **Statistical power:** n = 5 seeds cannot achieve significance for an effect present in only 2/5 seeds; report per-seed values alongside the aggregate, and do not claim a significant effect where p_holm ≥ 0.95.
- **CI validity:** the normal-theory 95% CIs reported by the summarizer for Honest FPR are invalid (asymmetric, bounded-at-zero binomial-type data producing negative lower bounds). Prefer per-seed counts or a Wilson/Clopper–Pearson interval on pooled client-rounds (e.g., "0/1000" → upper bound ≈0.3% via rule-of-three) instead of the t-based CI for this metric.
- **Heterogeneity fit scope:** fit only at α = 0.3 from 3 runs (~84 distinct client pairs, R² = 0.49); not valid for other α values used elsewhere in the campaign (E3, E4, E7 need their own fit).
- **DP-mechanism claim (finding 4):** confirmed only for seed 42; the analogous sigma check for seed 2024 was proposed but its output was not captured in this conversation — verify before publishing the mechanistic claim.
- **Scale-mismatch claim (finding 5):** based on an approximate calculation, not a direct read of logged DP/hetero variance terms from the actual run outputs — verify against real per-round diagnostics before stating as a quantitative fact.
- **No attack-free control arm** exists within E6; do not report an "attack success rate" or damage estimate from this data.
- **Provenance:** all 20 runs traced to commit `f5ff72f2` (author metadata on that commit is a placeholder, "Your Name <you@example.com>" — recommend a follow-up commit correcting authorship before the repository is finalized, without amending/rewriting `f5ff72f`, to preserve the SHA already recorded in every manifest).
- **Superseded runs:** an earlier seed-42 batch (4 runs) was produced across three different, uncommitted code states (`93f22f92`, `5d315e4e`, `d595cce9`) and is excluded from this analysis; archived separately for reference, not part of the reported campaign.

## 6. Open items / suggested follow-ups

- [ ] Re-run the low-sigma-rejection check for seed 2024 (client 4 and 7's noise multipliers vs. the other 6 clients) to confirm/refute finding 4 as a general mechanism rather than a seed-42 coincidence.
- [ ] Pull the actual per-round `dp_variance` / `hetero_variance` diagnostic values from run logs (if logged) to replace the approximate scale comparison in finding 5 with measured numbers.
- [ ] Decide and document the reported CI method for Honest FPR (binomial-appropriate, not the default t-interval) before this table goes into the paper.
- [ ] Confirm whether `scripts/analysis_paper.py` needs to be extended to regenerate this table automatically from the archived manifests, per the Issue 5 acceptance criteria (Task 4).

---

*This document was assembled from terminal output pasted into a chat conversation, not from direct inspection of the underlying CSV/JSON files. Numbers above should be treated as a working draft and spot-checked against the source files (`E6_per_run.csv`, `E6_table4.csv`, `E6_paired.csv`, `per_client_krum_scores_*.csv`, `dp_final_epsilon_*.json`, `experiment_config_*.json`) before being cited in the paper.*
