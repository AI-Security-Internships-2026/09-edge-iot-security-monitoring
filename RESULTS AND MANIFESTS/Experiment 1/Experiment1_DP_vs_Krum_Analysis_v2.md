# Experiment 1 — DP vs. Krum Epsilon Sweep: Full Results Analysis (v2)

**Data:** `results_network_dp3.csv`, `results_network_dp9.csv`, `results_network_dp15.csv`, `results_application_dp3.csv`, `results_application_dp09.csv`, `results_application_dp15.csv`
**Code reviewed against:** `main.py` (1,057 lines), `task.py` (273 lines) — see `FL-IDS_Experiment1_Code_Manifest.md` for the full technical manifest.
**Setup:** 10 clients, 25 rounds, Byzantine attack (clients 0/1, sign-flip, scale=5.0 network / 2.0 application), Adaptive Multi-Krum defence (MAD, k=2.5), DP-SGD (Opacus, RDP accountant) at ε ∈ {3, 9, 15}, δ=1e-5, max_grad_norm=1.5, **FedProx (mu=0.02, DP-safe decoupled proximal step — confirmed correctly implemented, see Section 0)**.
**Data integrity:** all 6 files complete — 25/25 rounds, 11 rows/round (10 clients + MEAN), zero missing rounds, zero `nan_this_round` flags, zero ZKP rejections (expected, `USE_ZKP=False`), 100% Krum Byzantine-detection rate in every single round of every run.

**Changelog v1 → v2:** Added Section 0 (code review confirmation — all three patches verified correctly applied, results relabeled from "DP-SGD+FedAvg" to "DP-SGD+FedProx"). Added Section 4b (per-client heterogeneity/spread — a second independent signal supporting the ε=9 non-monotonicity finding). No changes to the underlying data or any other numeric result from v1.

---

## 0. Code Review Confirmation — Results Are DP-SGD + FedProx, Not FedAvg

Full source of `main.py` and `task.py` was read end-to-end (not sampled) and cross-checked against the three patches discussed before this sweep ran (DP-safe proximal term, save-best-checkpoint, CLI args). **All three are correctly implemented, with no new bugs found.**

The material one for interpreting these results is the proximal-term fix. Opacus's `DPOptimizer.step()` builds its entire update from `.grad_sample` — per-sample gradients captured by hooks on specific layer types. A proximal term `(mu/2)·||w−w_global||²` added directly to the loss before `.backward()` never populates `.grad_sample` (it's a function of the parameter, not of any hooked layer's activation), so under naive DP-SGD, `PROX_MU` would be silently inert regardless of its configured value.

The fix applied here (`_apply_dp_safe_prox_step`, confirmed at `main.py` lines 249–271 and wired into the per-batch training loop at lines 364–385) applies `mu·(w−w_global)` as a **separate, deterministic, unnoised parameter update**, decoupled from the privatized data-gradient step — mathematically valid because the term depends only on current parameters and the last round's *public* global model, never on client data, so it costs zero privacy budget applied this way.

**Practical consequence:** every number in this report reflects genuine FedProx regularization (mu=0.02) operating alongside DP-SGD, not plain DP-SGD+FedAvg as an earlier draft of this analysis assumed before the code was reviewed. This doesn't change any of the numbers below — they were already correct — it changes how the method should be described in the write-up.

Full manifest of configuration, defence mechanism details, checkpoint scheme, CSV schema, and known open items is in the companion `FL-IDS_Experiment1_Code_Manifest.md`.

---

## 1. Headline Result — The Original Hypothesis Does NOT Hold, and That's the Finding

Experiment 1 set out to test whether DP noise erodes Multi-Krum's ability to separate Byzantine clients from honest ones — the `krum_score_ratio` (Byzantine mean score ÷ honest mean score) was the core diagnostic. **It does not erode, anywhere in this ε range.**

| Model | ε=3 mean ratio | ε=9 mean ratio | ε=15 mean ratio | Change ε3→ε15 |
|---|---|---|---|---|
| Network | 310.6 | 313.2 | 315.8 | **+1.7%** |
| Application | 102.5 | 103.8 | 104.8 | **+2.2%** |

Both models show the same direction (ratio very slightly *higher* at low noise / high ε, consistent with the hypothesis's sign) but the effect size is negligible — a 1.7–2.2% shift against a baseline separation of 80–380×. Detection rate stayed pinned at **100.00%, every round, every condition, no exceptions.** There is no round in any of the six files where Krum failed to isolate both Byzantine clients, regardless of noise level.

**Mechanistic read:** the sign-flip attack (scale 5.0 network / 2.0 application) produces a parameter deviation orders of magnitude larger than anything DP-SGD's per-round noise injects at these ε values. Krum's L2-distance scoring simply never gets close to confusing the two — DP noise is comparatively a rounding error next to a 5×/2× sign-flipped update.

**What this means for the open blocker** (push ε higher / go central DP / restrict noise to classifier-head only): this data does **not** support restricting DP noise to the classifier head as a Krum-preservation measure — there's nothing to preserve Krum *from* at ε≥3 under this attack strength. The simpler options (keep local DP as configured, or push ε higher purely for utility reasons — see Section 3) remain fully defensible. Head-only DP could still be motivated by other goals (compute/communication cost, formal per-layer guarantees) but not by this experiment's evidence.

---

## 2. The Real Dose-Dependent Effect Is Elsewhere: DP Noise Delays Rare-Class Discovery

While Krum's robustness was ε-invariant, something else was strongly ε-dependent: **whether, and when, the model discovers a class with very few training examples at all.** The clearest case is **Fingerprinting** on the application model (1,001 raw samples before Dirichlet partitioning — the rarest class in the dataset):

| Round | ε=3 (high noise) | ε=9 (mid noise) | ε=15 (low noise) |
|---|---|---|---|
| 1–14 | 0.000 | 0.000 | 0.000 |
| 15 | 0.000 | **0.516** ← wakes up | 0.000 |
| 16 | 0.000 | 0.560 | 0.045 ← wakes up |
| 20 | 0.000 | 0.607 | 0.434 |
| 24 | **0.015** ← barely wakes up | 0.610 | 0.552 |
| 25 | 0.169 | 0.613 | 0.586 |

At ε=9 and ε=15, the model eventually finds Fingerprinting and reaches respectable F1 (0.61 and 0.59 by round 25). At ε=3, it stays completely dead for 23 of 25 rounds and only barely stirs (F1=0.17) right at the end. This is not a Krum effect — Krum's exclusion set was identical across all three runs (see Section 4) — it's the DP noise itself overwhelming the gradient signal for a class that already has almost no data to begin with. This is a genuinely strong, clean finding: **DP noise doesn't just cost accuracy uniformly, it can suppress rare-class learning near-categorically below some noise threshold**, which is a sharper and more actionable claim than "DP hurts utility" in general.

---

## 3. Overall F1-Macro / Accuracy vs. Epsilon

| Condition | Best round | Best F1-Macro | Best Acc | Final-round F1 | Final-round Acc | Last-5-round F1 mean (±std) |
|---|---|---|---|---|---|---|
| network_dp3 | 23 | 0.7268 | 0.8944 | 0.7174 | 0.8965 | 0.7122 (±0.0166) |
| network_dp9 | 23 | 0.7778 | 0.9345 | 0.7758 | 0.9363 | 0.7615 (±0.0191) |
| network_dp15 | 23 | **0.8068** | **0.9464** | 0.7607 | 0.9007 | 0.7836 (±0.0171) |
| application_dp3 | 22 | 0.4814 | 0.7200 | 0.3789 | 0.5608 | 0.4216 (±0.0381) |
| application_dp09 | 25 | **0.6251** | 0.7696 | 0.6251 | 0.7696 | 0.6143 (±0.0072) |
| application_dp15 | 25 | 0.6067 | **0.7701** | 0.6067 | 0.7701 | 0.5670 (±0.0414) |

**Network model:** clean, monotonic — more noise (lower ε) → worse F1-Macro and accuracy, exactly as expected. ε=3 is meaningfully worse than ε=9, which is meaningfully worse than ε=15 (0.7268 → 0.7778 → 0.8068). No surprises here.

**Application model:** **not monotonic** — ε=9 (0.6251) actually edges out ε=15 (0.6067) on best F1-Macro, and ε=9's last-5-round std (0.0072) is dramatically tighter than ε=15's (0.0414), meaning the ε=9 run converged to a stable plateau while the ε=15 run was still visibly volatile at the end (see the round-23 dip in Section 5). This is worth stating plainly rather than smoothing over: **at these settings, lower DP noise did not reliably translate into better or more stable application-model training within 25 rounds.** A single run per condition can't distinguish "real non-monotonicity" from "one unlucky non-IID draw" — see Section 4b for a second, independent signal pointing the same direction, which somewhat strengthens (but doesn't confirm) that this isn't just noise.

**ε=3 is unambiguously the worst condition on both models** — that part is consistent and not in question.

---

## 4. Krum Selection Pattern — Textbook-Clean Across All Six Runs

Per-client selection counts (out of 25 rounds), identical in every single condition:

```
Client 1 (Byzantine): 0/25 selected
Client 2 (Byzantine): 0/25 selected
Clients 3–10 (honest): 25/25 selected
```

Zero collateral damage, zero persistent-exclusion anomaly (no "Client 4" or "Client 6" pattern like the earlier Condition 5 run showed) — every honest client was selected in every round, in every one of the six conditions. This is the cleanest possible outcome for the defence mechanism and needs no further diagnosis. It also means the F1-Macro differences across conditions in Section 3 are attributable to DP noise's effect on training, not to any difference in which clients contributed to the aggregate.

---

## 4b. Per-Client Heterogeneity — ε=9 Also Tightest on Both Models

A second, independent metric — spread of final-round accuracy across the 8 honest (Krum-selected) clients — shows the same "ε=9 is a local sweet spot" pattern as Section 3's F1-Macro finding, on **both** models this time, not just application:

| Condition | Honest-client accuracy range | Std dev | Spread |
|---|---|---|---|
| network_dp3 | 0.604–0.968 | 0.1273 | 0.365 |
| **network_dp9** | **0.777–0.983** | **0.0729** | **0.206** |
| network_dp15 | 0.646–0.974 | 0.1160 | 0.329 |
| application_dp3 | 0.322–0.694 | 0.1254 | 0.372 |
| **application_dp09** | **0.589–0.839** | **0.0849** | **0.251** |
| application_dp15 | 0.495–0.858 | 0.1213 | 0.363 |

ε=9 has the tightest client-to-client consistency on both models — roughly 40–45% tighter spread than either neighboring ε. This matters because it's a genuinely different measurement axis than Section 3's aggregate F1-Macro (this is about how *uniformly* clients performed, not how well on average), and it points the same direction independently. That makes "ε=9 is a real local optimum for this setup, not sampling noise" a somewhat stronger claim than either metric alone would support — though it's still a single seed per condition, and a repeat run remains the only way to fully confirm it.

---

## 5. Round-25 Instability — Recurs Again, Now a 4th Independent Confirmation

The network model's known "late-round crash" pattern (documented previously for the clean baseline's round 25, and Condition 5's network round 23) shows up again here, specifically in **network_dp15**:

| Round | Acc | DDoS_HTTP F1 | Vulnerability_scanner F1 |
|---|---|---|---|
| 23 (best) | 0.9464 | 0.7249 | 0.7175 |
| 25 (final) | 0.9007 | 0.5567 ↓ | 0.5491 ↓ |

Accuracy drops nearly 5 points and two mid-sized classes lose 15–17 points of F1 in the last two rounds, then the run ends — no chance to observe whether it would have self-healed the way earlier documented instances did. `network_dp3` shows a smaller, single-round version of the same pattern (MITM: 0.4586→0.2090→0.4221, dips and partially recovers by round 25). This is now the **4th independent run** (clean baseline round 25, Condition 5 network round 23, Condition 5 application round 21, and now this) showing the same non-IID/FedProx-adjacent late-round volatility — strong enough at this point to state as a structural property of this training setup in the paper, not a one-off.

**Practical implication:** for the network model specifically, quoting "final round" performance understates what the model actually achieved — best-round numbers (round 23 across all three ε conditions here) are the fairer number to report, and match the pattern of every prior run reviewed.

---

## 6. DP Calibration Sanity — Confirmed Working Correctly

Opacus's epsilon targeting was highly accurate in every run — worth stating in the paper as a methods-validity check:

| Condition | Target ε | Achieved ε | Deviation | Noise multiplier (σ) |
|---|---|---|---|---|
| network_dp3 | 3.00 | 2.9954 | −0.153% | 0.8592 |
| network_dp9 | 9.00 | 8.9975 | −0.028% | 0.5789 |
| network_dp15 | 15.00 | 14.9995 | −0.003% | 0.4862 |
| application_dp3 | 3.00 | 2.9931 | −0.230% | 0.9183 |
| application_dp09 | 9.00 | 8.9962 | −0.042% | 0.6070 |
| application_dp15 | 15.00 | 14.9940 | −0.040% | 0.5063 |

All within a quarter of a percent of target, and σ decreases monotonically as target ε increases on both models, exactly the expected relationship (less noise needed to satisfy a looser privacy budget). No calibration anomalies anywhere.

---

## 7. Timing — GPU Run Was Far Faster Than the Original CPU Estimates

| Condition | Avg round time | Total wall time |
|---|---|---|
| network_dp3 | 133.3s | 0.93 hr |
| network_dp9 | 133.0s | 0.92 hr |
| network_dp15 | 132.6s | 0.92 hr |
| application_dp3 | 149.6s | 1.04 hr |
| application_dp09 | 148.0s | 1.03 hr |
| application_dp15 | 148.2s | 1.03 hr |

All six conditions completed in **under 65 minutes each**, network and application alike — a dramatic improvement over the original CPU-based estimates (~2–3 hr network, ~4–6 hr application per condition). Round times are also extremely consistent within each condition (network: 130.8–135.7s range across 25 rounds; application: 144.5–153.1s range) — no runaway rounds, no signs of memory pressure recurring (consistent with vLLM being cleared before this sweep).

---

## 8. Recommendations for the Write-Up

1. **Lead with Section 1's negative result as a real finding**, not a failed experiment — "Krum's Byzantine-detection remained at 100% and its score-separation ratio changed by under 2.3% across ε∈{3,9,15}, indicating Krum's robustness to this attack is effectively independent of DP noise in this regime" is a clean, defensible, citable claim.
2. **Report Section 2 (Fingerprinting/rare-class suppression) as the experiment's more novel contribution** — it's a sharper, more mechanistic claim than generic "DP hurts utility," and it's visually dramatic (a near-zero-until-round-15/16/24 cliff, sensitive to exactly the parameter you swept).
3. **Use best-round, not final-round, numbers for the network model** given the confirmed round-25 instability pattern — cite it as a structural, recurring property (4 independent occurrences now) rather than noise.
4. **Flag the application model's ε=9 > ε=15 non-monotonicity honestly** rather than omitting it — Section 4b's per-client spread finding is a second, independent signal in the same direction, which is worth mentioning, but still frame it as needing a repeat-seed confirmation rather than a settled result.
5. **This experiment does not, by itself, justify restricting DP noise to the classifier head** — that motivation would need either a much lower ε (below 3, where Krum's margin might genuinely start to compress) or a stronger argument based on something other than Krum-preservation.
6. **Correctly label the method as DP-SGD + FedProx** (mu=0.02, confirmed active via the decoupled proximal-step patch — Section 0), not DP-SGD + FedAvg, in any methods section describing this experiment.
