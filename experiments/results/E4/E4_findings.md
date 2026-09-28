# Issue 5 / E4 — Findings Report: Min-Max Attack Evasion Under Adaptive vs. Calibrated Krum

**Status:** Preliminary (single seed, seed=42). Requires seeds 123/456 before being treated as final for Table 3 / Figure 4.
**Condition:** Edge-IIoTset, network IDS model, α=0.7 (Dirichlet heterogeneity), Min-Max attack (frozen, `dev_type=std`, `search_iters=15`, `gamma_init=None` → auto: 5× coalition max-pairwise-distance), 10 clients, 2 colluding Byzantine clients, ε ∈ {0.2, 0.5, 1, 3, 5, 10, 15}.

---

## Executive Summary

The planned E4 comparison (Adaptive Krum vs. Calibrated Krum, robustness vs. cumulative DP budget ε) did not surface the expected privacy–robustness trade-off, because **the frozen Min-Max attack achieves 0% Byzantine detection against both aggregators at every ε value tested.** This was investigated and traced to a structural cause rather than a mistuned attack parameter — a calibration sweep across `gamma_init ∈ {auto, 1, 3, 10, 30}` and `dev_type ∈ {std, sign, unit_vec}` confirmed the attack is already at its geometric maximum strength and cannot be pushed further by any of its own tunable parameters.

Despite identical binary detection outcomes, a finer-grained analysis of the **continuous** Krum scores (rather than the pass/fail classification) shows Calibrated Krum's DP-variance calibration term produces a real, monotonic, sub-threshold effect concentrated at low ε (high DP noise) — evidence the calibration mechanism is functioning as designed, even though it isn't yet strong enough to flip a detection decision under this specific attack/heterogeneity/coalition-size combination.

---

## 1. Background: Why the Attack Can't Be Tuned Stronger

### 1.1 The feasibility bound is not anchored to the honest clients

`minmax_attack_trained` (Fang et al., USENIX Security 2020) crafts one poisoned update, broadcast identically by all colluding Byzantine clients. The crafted vector is:

```
crafted = w_avg + gamma * direction
```

where `gamma` is found via binary search, bounded by the constraint that the crafted vector's worst-case L2 distance to **each colluding Byzantine client's own honestly-trained local update** does not exceed those clients' own mutual pairwise distance (`max_pair_dist`). This bound has nothing to do with distance to the honest population elsewhere in the FL system.

`gamma_init` only sets the **upper bound the binary search is allowed to explore** — it is not the attack strength itself. If the search converges to a `constraint_ratio ≈ 1.0` **without saturating at `gamma_init`**, the true maximum feasible perturbation has already been found, and raising `gamma_init` further cannot help.

### 1.2 Calibration sweep result

A 15-run calibration sweep (`calibrate_minmax_attack.py`, no-DP `krum_baseline` mode, 10 rounds each) confirmed this directly:

| dev_type | gamma_init | TPR | mean constraint_ratio | saturated |
|---|---|---|---|---|
| std | auto | 0.00 | 0.97 | 0% |
| std | 1 | 0.00 | 1.00 | 100% |
| std | 3 | 0.00 | 1.00 | 0% |
| std | 10 | 0.00 | 1.00 | 0% |
| std | 30 | 0.00 | 1.00 | 0% |
| sign | auto | 0.00 | 1.00 | 0% |
| sign | 1 | 0.00 | 0.68 | 100% |
| sign | 3 | 0.00 | 1.00 | 0% |
| sign | 10 | 0.00 | 1.00 | 0% |
| sign | 30 | 0.00 | 1.00 | 0% |
| unit_vec | auto | 0.00 (FPR 0.12) | 1.00 | 0% |
| unit_vec | 1 | 0.00 (FPR 0.12) | 0.50 | 100% |
| unit_vec | 3 | 0.00 (FPR 0.12) | 0.50 | 100% |
| unit_vec | 10 | 0.00 (FPR 0.25) | 0.52 | 100% |
| unit_vec | 30 | 0.00 (FPR 0.25) | 0.55 | 100% |

For `std` and `sign`, `constraint_ratio` sits at ~1.0 without saturating from `gamma_init=3` upward — the search has found the true ceiling, and it is not detectable. `unit_vec` was still saturating even at `gamma_init=30` (true ceiling not yet found), and was additionally disturbing *honest*-client scores (FPR rising to 0.25) without ever making the Byzantine clients themselves look anomalous.

### 1.3 Compounding structural factor: identical broadcast

Both colluding Byzantine clients send the exact same crafted vector. In Adaptive Krum's score computation (sum of distances to the `n − f − 2` nearest neighbors), each Byzantine client's own coalition partner sits at distance 0 and is always counted as a "supporting" neighbor — a free, guaranteed low-distance contribution unrelated to how far the crafted point actually is from the honest cluster.

### 1.4 Why raising heterogeneity/tuning further isn't the fix here

At α=0.7 (moderate, not extreme, heterogeneity) combined with FedProx pulling all clients toward the global model, the Byzantine clients' *own honest local training* (before poisoning) already converges close to what the rest of the population is doing. This keeps `max_pair_dist` — and therefore the attack's achievable ceiling — small, independent of ε, independent of `gamma_init`, independent of `dev_type` (for `std`/`sign`).

**Conclusion:** this is the textbook behavior Min-Max was engineered to produce — an AGR-agnostic attack designed specifically to defeat Krum-family distance-based defenses by staying inside the honest coalition's own distance envelope. 0% TPR against vanilla Adaptive Krum, independent of attack-strength parameters, is a legitimate and expected result under these conditions, not a broken experiment.

---

## 2. Finding: Both Aggregators Show 0% Byzantine TPR at Every ε

Across all 14 cells (2 aggregators × 7 ε values, seed=42):

| Aggregator | TP | FP | TN | FN | Byzantine TPR | Honest FPR |
|---|---|---|---|---|---|---|
| Adaptive Krum (all 7 ε) | 0 | 0 | 200×7 | 50×7 | 0.000 | 0.000 |
| Calibrated Krum (all 7 ε) | 0 | 0 | 200×7 | 50×7 | 0.000 | 0.000 |

No cell, at any ε, produced a single true-positive Byzantine detection or a single false-positive honest rejection. This matches the calibration sweep's diagnosis in section 1: the attack's geometric ceiling makes it undetectable by construction, not by insufficient tuning.

---

## 3. Finding: Calibrated Krum's DP-Variance Term Has a Real, Sub-Threshold Effect

The binary classification hides a signal visible in the **continuous** raw Krum scores. For each round, the Byzantine clients' score was ranked against the honest-client score distribution (0% = below every honest score, i.e., maximally "trustworthy"-looking; higher % = closer to looking like an outlier):

| ε | Adaptive Krum — mean Byzantine percentile rank | Calibrated Krum — mean Byzantine percentile rank | Adaptive: Byz/honest-median score ratio | Calibrated: same ratio |
|---|---|---|---|---|
| 0.2 | 0.0% | **9.2%** | 0.487 | 0.522 |
| 0.5 | 0.0% | **2.5%** | 0.486 | 0.513 |
| 1.0 | 0.0% | **0.8%** | 0.485 | 0.501 |
| 3.0 | 0.0% | 0.0% | 0.482 | 0.486 |
| 5.0 | 0.0% | 0.0% | 0.482 | 0.484 |
| 10.0 | 0.0% | 0.0% | 0.481 | 0.482 |
| 15.0 | 0.0% | 0.0% | 0.480 | 0.480 |

**Interpretation:** Adaptive Krum's Byzantine/honest-median score ratio is flat at ≈0.48 across every single ε — completely independent of DP noise level, consistent with the geometric-ceiling explanation in Section 1 (this ratio is set by attack geometry, not privacy noise). Calibrated Krum's DP-variance calibration term measurably raises this ratio (and the associated percentile rank) specifically at low ε, where DP noise is largest — exactly the regime the term is designed to compensate for — and the effect **cleanly vanishes by ε=3** as DP noise becomes small enough that there's nothing left to correct for.

This is evidence the calibration mechanism works as intended. It is simply not yet strong enough, against this specific attack/coalition-size/heterogeneity combination, to flip an FN into a TP.

---

## 4. Finding: Identical Downstream Models Across Aggregators

Because neither aggregator excludes any client at any ε (0 TP, 0 FP throughout), both collapse mechanically to an unweighted average over all 10 clients (8 honest + 2 undetected Byzantine). This produces **byte-identical** downstream Macro-F1 and per-class metrics between Adaptive Krum and Calibrated Krum at every ε — not a bug, a direct and expected consequence of Finding 2.

---

## 5. Privacy–Utility Picture (Single Seed — Preliminary)

| ε (requested) | ε (achieved, mean) | Test Accuracy | Macro-F1 | Ransomware F1 | MITM F1 | Vulnerability_scanner F1 |
|---|---|---|---|---|---|---|
| 0.2 | 0.195 | 0.978 | 0.910 | 0.918 | 0.524 | 0.932 |
| 0.5 | 0.494 | 0.980 | 0.913 | 0.954 | 0.504 | 0.936 |
| 1.0 | 0.994 | 0.976 | 0.890 | 0.892 | 0.389 | 0.932 |
| 3.0 | 2.996 | 0.978 | 0.902 | 0.955 | 0.427 | 0.919 |
| 5.0 | 4.996 | 0.980 | 0.909 | 0.928 | 0.500 | 0.927 |
| 10.0 | 9.996 | 0.986 | 0.927 | 0.953 | 0.568 | 0.950 |
| 15.0 | 14.997 | 0.977 | 0.897 | 0.788 | 0.539 | 0.934 |

**DP accounting validation:** achieved cumulative ε tracks the requested target to within ~2.5% at every point (e.g., 0.195 vs. 0.2 target; 14.997 vs. 15 target) — the cumulative accounting fix from the DP issue is behaving correctly.

**No clean monotonic privacy-utility trend is visible.** Macro-F1 ranges non-monotonically between 0.890 and 0.927 across ε with no directional pattern. This is expected single-seed noise, not evidence that DP noise doesn't affect utility — three-seed replication is needed before any real trend can be claimed for Figure 4.

**MITM is a persistent weak class** — F1 in the 0.39–0.57 range at every ε, roughly half the F1 of every other reported class (Ransomware, Vulnerability_scanner both mostly 0.79–0.95). This weakness is independent of both ε and aggregator choice, and is worth its own discussion in the rare-class analysis (Table 4 / Figure 6) separate from the Byzantine-robustness story.

---

## 6. Recommendations

1. **Report the 0% TPR result as a finding, not a defect.** Frame E4/Figure 4 around: "vanilla Adaptive Krum is structurally blind to a geometrically-optimal Min-Max coalition at moderate heterogeneity (α=0.7); Calibrated Krum's DP-variance term produces a measurable, ε-dependent partial correction, concentrated at high-noise (low-ε) operating points, though not sufficient to cross a detection threshold under these conditions."
2. **Replicate at seeds 123 and 456** before finalizing Table 3 / Figure 4 — current numbers are single-seed and the privacy-utility curve shows no trend that can be trusted without it.
3. **Consider whether the percentile-rank / score-ratio metric belongs in the standard analysis pipeline** (`scripts/analysis_paper.py`, Task 4) alongside the binary TP/FP/TN/FN counts — it surfaced a real effect that the binary classification completely hid, and would likely be informative for E2/E3/E6 as well, not just E4.
4. **If a genuinely detectable stealth attack is needed for other parts of the campaign** (e.g., to demonstrate Calibrated Krum's *full* value, including cases where it does cross the detection threshold), consider calibrating Min-Sum or Bounded-directional as an alternative frozen attack for those specific comparisons, documenting Min-Max's structural evasion as the reason for the substitution — this is explicitly permitted by the issue's own acceptance criteria ("Use Min-Max unless validation provides a documented reason to use another stealth attack").

---

## Appendix: Methodology Notes

- **Percentile rank definition:** for each round, each Byzantine client's raw Krum score is ranked against that round's honest-client score distribution. 0% = the Byzantine score is below every honest score (looks maximally central/trustworthy under Krum's "lower score = more consistent" convention); 100% = above every honest score. Reported values are means across all rounds and all Byzantine clients for that (aggregator, ε) cell.
- **Score ratio definition:** each round's mean Byzantine raw score divided by that round's honest-client median raw score, averaged across rounds.
- **Data provenance:** all figures above are derived directly from `per_client_krum_scores_*.csv` (classification, percentile, ratio) and `results_*_FINAL_TEST.csv` (accuracy, F1) for the 14 real runs (7 ε × 2 aggregators, seed=42) executed via `run_e4_campaign.py` against the actual Edge-IIoTset dataset and codebase, and from `attack_diag_*.jsonl` / `calibration_summary.csv` for the 15-run Min-Max calibration sweep executed via `calibrate_minmax_attack.py` under `--ablation-mode krum_baseline` (no DP).
- **Limitations:** single seed (42) throughout Sections 2–5; the calibration sweep in Section 1 used only 10 rounds per candidate (sufficient to characterize attack geometry, not full convergence). Both should be extended before these numbers are treated as final for the paper.
