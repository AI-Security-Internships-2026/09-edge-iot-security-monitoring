# Adaptive Multi-Krum + Byzantine Attack — Results Analysis

**Condition:** Byzantine attack (clients 0,1 / labels 1,2; sign-flip) defended by
`adaptive_multi_krum()` (MAD-threshold, k=2.5, min_keep_fraction=0.5), both models,
25 rounds. Log source: `results_application.csv`, `results_network.csv`.

---

## 1. Headline numbers vs. locked clean baselines

| Model | Metric | Locked clean baseline | This run (final/best round) | Δ |
|---|---|---|---|---|
| Application | Accuracy | 0.8504 (round 20) | 0.8913 (round 24, best) | **+0.0409** |
| Application | F1-Macro | 0.7293 (round 20) | 0.7979 (round 24, best) | **+0.0686** |
| Network | Accuracy | 0.9697 (round 22) | 0.9856 (round 25, best) | **+0.0159** |
| Network | F1-Macro | 0.8289 (round 22) | 0.8651 (round 25, best) | **+0.0362** |

Both models beat their locked clean baseline on both metrics, despite this run
having an active Byzantine attack running the whole time. **This is not evidence
that adaptive Krum improves on the baseline** — see Section 5 for why.

Application's round 25 (last row in your CSV, MEAN not logged) was reconstructed
by averaging its 10 client rows: Accuracy 0.8913, F1-Macro 0.7969 — statistically
indistinguishable from round 24, i.e. the model had plateaued by the end of training.

---

## 2. Krum defense performance

- **Detection rate: 100% every round, both models, all 25 rounds.** Clients 1
  and 2 (the configured Byzantine clients) were correctly identified and
  discarded in every single round with zero false negatives.
- **Selection count — Application: perfectly stable.** Exactly 8/10 clients kept,
  every round, no variation. The honest cohort's Krum scores cluster tightly;
  nothing besides the two real attackers ever crosses the MAD threshold. This is
  the ideal adaptive-Krum outcome — no collateral damage.
- **Selection count — Network: mostly stable, one persistent outlier.** 7/10 kept
  in 24 of 25 rounds; 6/10 in round 1 only (an extra discard — client 10 — that
  never recurs, plausibly early-training score noise before the model stabilizes).

### Client 4 (network model): excluded in all 25/25 rounds
Not Byzantine, never flagged as such, but never once selected. Under adaptive
thresholding this is a real statistical signal, not an artifact of a fixed
margin — client 4's Krum score genuinely and persistently exceeds
`median(scores) + k · 1.4826 · MAD(scores)` relative to the rest of the honest
cohort, regardless of round or training state. Strong candidate for a skewed
per-class sample distribution under the Dirichlet(α=0.7) partition.

**Action item:** run the per-client sample audit (`per_client_audit.py`) against
the **network** partitions, focused on client 4 — this mirrors the Client 6
Password/XSS anomaly already documented for the application model, just on the
network side. If confirmed, this client is contributing zero data to the global
network model for the entire run, every run, under this Dirichlet seed.

**Missing telemetry:** the CSV only carries `krum_selected` / `krum_detected_byzantine`
(reused from the fixed-Krum schema). `adaptive_multi_krum()`'s actual
threshold/center/spread values only went to stdout, which wasn't captured. Add
`krum_threshold`, `krum_center`, `krum_spread` columns for future adaptive runs —
turns "client 4 keeps getting dropped" into "client 4's score is 3.2× the
threshold vs. 1.1× for the next-closest honest client," a stronger claim for
the writeup.

---

## 3. Per-class final performance (last-5-round average, rounds 21–25)

**Application** — weakest to strongest:

| Class | F1 |
|---|---|
| XSS | 0.516 |
| Uploading | 0.592 |
| Password | 0.661 |
| SQL_injection | 0.832 |
| Fingerprinting | 0.798 |
| Port_Scanning | 0.910 |
| Backdoor | 0.964 |
| Normal | 0.958 |

**Network** — weakest to strongest:

| Class | F1 |
|---|---|
| MITM | 0.464 |
| Ransomware | 0.569 |
| DDoS_HTTP | 0.599 |
| Vulnerability_scanner | 0.718 |
| Normal | 0.821 |
| DDoS_TCP | 0.937 |
| DDoS_UDP | 0.999 |
| DDoS_ICMP | 0.999 |

XSS/Uploading/Password (application) and MITM (network) remain the known hard
classes — consistent with your documented sample-scarcity ceilings
(MITM: 1,214 raw samples).

---

## 4. Transient collapses (both self-heal within 1–2 rounds)

| Model | Round | F1-Macro drop | Classes hit hardest | Recovered by |
|---|---|---|---|---|
| Network | 23 | 0.825 → 0.626 (**−0.199**) | DDoS_HTTP 0.84→0.10, Vuln_scanner 0.79→0.45, Normal 0.86→0.64 | Round 25 |
| Application | 21 | 0.796 → 0.725 (**−0.071**) | Normal 0.98→0.80, Uploading 0.67→0.45, Password 0.70→0.57 | Round 22 |

Krum's detection rate stayed at 100% through both events — these are not Krum
failures. They match the same non-IID client-sampling instability pattern
already documented for the clean baseline's round-25 crash: a genuine, recurring
FedProx/non-IID limitation, not an attack- or defense-specific artifact. Worth
citing as a structural finding (recurs across three independent runs now: clean
baseline round 25, this network run's round 23, this application run's round 21).

**Caveat on the network's round-25 "best" number:** it's the immediate recovery
peak right after the round-23 crash, not a steady-state value. Rounds 20–22
(pre-crash) average F1-Macro ≈ 0.808 — a more representative "typical" number if
you want one that isn't inflated by a post-crash bounce.

---

## 5. Why accuracy and F1 are higher here than in the locked baseline

The improvement is real in the sense that the numbers are correct, but it should
**not** be attributed to adaptive Krum being beneficial. Three uncontrolled
differences separate this run from the runs that produced the locked baselines,
and any one of them is a more plausible explanation than the defense mechanism:

1. **`PROX_MU` differs.** This run's `main.py` sets `PROX_MU = 0.1`. The locked
   baselines' documentation doesn't pin down their exact value, but the
   `main.py` itself carries an explicit warning that a mu mismatch versus the
   baseline-generating run will show up exactly as an unexplained gap in
   reported metrics — check `experiment_config_*.json` from the baseline runs
   to confirm.
2. **Round-level cosine LR decay is new.** `get_round_lr()` decays the
   client-side learning rate across rounds (`min_lr_frac=0.15`); your own
   locked-baseline documentation states explicitly that **LR decay and EMA
   were NOT used in the runs that produced the 0.7293/0.8289 baselines.** By
   round 20+, clients in this run are taking much smaller gradient steps than
   clients in the baseline run were — a well-known contributor to late-round
   stability and higher final accuracy.
3. **"Best round" vs. baseline's "best round" isn't from the same round-selection
   process.** The baseline was explicitly selected as the best round out of 25
   on both Accuracy and F1-Macro simultaneously. This report does the same
   (round 24 application, round 25 network), so the comparison method is
   consistent — but it means both numbers are each run's ceiling, not typical
   performance, which is worth stating plainly rather than implying "the model
   is just better now."

**What argues against Krum itself being the cause:** if anything, Krum
mechanically *removes information* — client 4 contributed zero updates to the
network model for all 25 rounds, and the two Byzantine clients' data (which,
absent the attack, would have been legitimate FL data) is also gone. A defense
that's actively discarding client data every round improving on a no-attack,
no-discarding baseline is not the expected direction if Krum were the driver.
The training-recipe differences (mu, LR schedule) are the more parsimonious
explanation.

**To actually isolate Krum's effect:** rerun Condition 1 (clean baseline, no
attack, no defense) under this exact `main.py` — same `PROX_MU`, same LR decay,
same EMA — and compare *that* number to this run's, not the old locked baseline.
Right now the comparison has two variables changing at once (defense condition
+ training recipe) and only one (defense condition) is the thing you actually
want to measure.

---

## 6. Recommended next steps, in order

1. Confirm `PROX_MU`/`KRUM_M`-vs-adaptive-`k` values in `experiment_config_*.json`
   for both this run and the original baseline runs, to close the mu-mismatch
   question definitively.
2. Rerun Condition 1 (clean, no attack/defense) under the current `main.py` to
   get a genuinely comparable baseline for this training recipe.
3. Run `per_client_audit.py` against network partitions — Client 4.
4. Add `krum_threshold`/`krum_center`/`krum_spread` to the CSV schema for the
   next adaptive-Krum run so client-exclusion claims have a magnitude, not just
   a count.
5. Once the recipe-matched clean baseline exists, re-quote the "adaptive Krum
   vs. baseline" delta against *that* number instead of the original 0.7293/0.8289.
