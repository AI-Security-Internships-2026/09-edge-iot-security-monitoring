# FL-IDS Experiment 2 Re-Run Manifest — v11 (Supersedes Week 9 Experiment 2 Record)

**Provenance:** 8 result CSVs + 7 experiment-config JSONs, uploaded directly (not console-log-reconstructed), covering a re-run of Experiment 2 (HE+Krum hybrid vs. the encrypted-classifier-head blind spot) after the original Week 9 result files were found corrupted. This document supersedes the Week 9 Experiment 2 section of the v10 master context prompt as the **authoritative** record for this experiment, per your instruction — but every material difference from the Week 9 numbers is logged explicitly below rather than silently overwritten, since single-seed FL runs are expected to drift run-to-run and some deltas here are large enough to need a decision, not just a note.

**Files analyzed (all read directly, row-by-row — no numbers below are reconstructed from filenames or manifests):**

| Run | Model | Byzantine clients (1-idx) | Guard | CSV | Config JSON |
|---|---|---|---|---|---|
| `app_mit_byz1_2` | Application | 1, 2 | ON (mitigated) | ✅ | ✅ |
| `app_mit_byz2_7` | Application | 2, 7 | ON (mitigated) | ✅ | ✅ |
| `app_mit_byz4_10` | Application | 4, 10 | ON (mitigated) | ✅ | ✅ |
| `app_unmit_byz1_2` | Application | 1, 2 | OFF (unmitigated) | ✅ | ✅ |
| `net_mit_byz1_2` | Network | 1, 2 | ON (mitigated) | ✅ | ✅ |
| `net_mit_byz2_7` | Network | 2, 7 | ON (mitigated) | ✅ | ✅ |
| `net_mit_byz4_10` | Network | 4, 10 | ON (mitigated) | ✅ | ✅ |
| `net_unmit_byz1_2` | Network | 1, 2 | OFF (unmitigated) | ✅ | ⚠️ **missing** — no `experiment_config_network_network_exp2_unmitigated_byz1_2.json` was uploaded. Its methodology is inferred by analogy to `app_unmit_byz1_2`'s config (identical flag pattern expected: `use_he_krum_hybrid=true`, `use_head_norm_guard=false`), not confirmed from its own file. **Flag this as an open gap — request the actual JSON before final publication.** |

**Coverage vs. Week 9:** this re-run recovers the one CSV that was permanently missing from Week 9 (`network` default-Byzantine, clients 1/2 — Contradiction #11 in v10), and adds two configurations Week 9 never ran at all: `app_mit_byz2_7` (application model, clients 2/7) and `app_unmit_byz1_2` / `net_unmit_byz1_2` as *explicit, labeled* unmitigated baselines (Week 9's "unmitigated" result was a single default-client run per model, not paired 1:1 with every attacked-client configuration).

**Document status:** this manifest is now **self-contained**. Section 5 below incorporates every load-bearing finding from the 7 Week 9-era legacy files (`Run_notes_network.md`, `RUN_NOTES_Application.md`, `config_Application.json`, `config_network.json`, both `manifest.json` copies, `COMBINED_MANIFEST.md`) that were cross-referenced while producing this revision. Those source files can be deleted — nothing in them is needed going forward that isn't reproduced here.

---

## 0. CODE PROVENANCE NOTE — the pasted `main.py` is the *current* file, not a byte-identical snapshot of what produced these 8 runs

You supplied the current `main.py` after the CSVs/configs. Cross-checking it against the config JSONs those runs actually wrote finds **two hardcoded values in the current file that disagree with what's logged in the run configs**:

| Field | Current `main.py` | Config JSON actually written by the Experiment-2 runs |
|---|---|---|
| `LOCAL_EPOCHS` | `10` (hardcoded, line ~90) | `5` (every one of the 7 uploaded configs) |
| `DP_BATCH_SIZE` | `16` (hardcoded, line ~250) | `512` (every uploaded config — though irrelevant here since `use_dp=false` throughout) |
| `ABLATION_MODE` | `"krum_dp_sweep"` (hardcoded, line ~205) | `"exp2_mitigated"` / `"exp2_unmitigated"` |

Since the config JSON is written directly from these same in-memory variables (`"local_epochs": LOCAL_EPOCHS`, etc.) at the end of each run, **the JSON is ground truth for what actually happened** — these 8 runs genuinely used `LOCAL_EPOCHS=5`. The file you pasted is `main.py` as it currently stands, already edited forward in preparation for the next run (`ABLATION_MODE="krum_dp_sweep"`, i.e. Sweep 2 / the Gaussian-noise epsilon sweep) — `LOCAL_EPOCHS` and `DP_BATCH_SIZE` were bumped as part of that edit, after Experiment 2 finished. **Do not use the pasted file's `LOCAL_EPOCHS=10` or `DP_BATCH_SIZE=16` in the paper's description of Experiment 2** — the aggregation logic, attack routing, and guard/Krum mechanics below are otherwise the actual code path these runs executed (confirmed by matching every other logged config field), so everything else in this methodology section is drawn from it directly.

---

## 1. METHODOLOGY SECTION (for the paper)

### 1.1 Experimental design

Experiment 2 tests the central hypothesis of the project's Layer 2/Layer 3 interaction: that a hybrid HE+Krum aggregator, which runs Byzantine-robust aggregation (Krum) only over the **plaintext** ("bulk") slice of each client's update while aggregating the **encrypted** classifier-head slice separately, has a structural blind spot — an attacker confined entirely to the encrypted head is invisible to Krum, which never sees it. The mitigation under test is a ciphertext-bound HMAC head-norm guard (Layer 2 in the project's three-layer privacy stack) that runs as a pre-filter *before* Krum, gating on the one quantity the encrypted slice still reveals: the L2 norm of the head update, committed and bound to the actual ciphertext bytes.

This re-run tests both arms of that hypothesis directly, per model:
- **Unmitigated arm** (`use_head_norm_guard: false`): HE+Krum hybrid runs with no norm-guard pre-filter — Krum sees only the plaintext bulk slice, exactly the blind-spot condition.
- **Mitigated arm** (`use_head_norm_guard: true`): the same hybrid pipeline, but with the norm guard active as a pre-filter, at three different attacked-client identities per model (clients 1/2, 2/7, 4/10) to test whether the guard's effectiveness is attacker-identity-dependent or general.

### 1.2 Confirmed configuration (from the uploaded JSONs directly — not assumed from prior documentation)

All eight runs share the following core FL recipe, confirmed identical across every config file provided:

```
NUM_ROUNDS = 25, NUM_CLIENTS = 10, LOCAL_EPOCHS = 5, LEARNING_RATE = 0.001
PROX_MU = 0.02 (FedProx)
NUM_BYZANTINE = 2 (paired attacker configurations throughout — no single-attacker or ≥3-attacker condition tested this round)
attack_type = "sign_flip", attack_function = "classifier_head_flip_attack"
byzantine_head_only = true   (i.e. the corrected, train-then-flip-head attack, not the pre-Week-9 no-training bug)
use_krum = false, use_adaptive_krum = false, use_he = false, use_zkp = false
use_he_krum_hybrid = true    (constant across all 8 runs — this is what makes them "Experiment 2" runs)
dp_safe = false, use_dp = false
device = cuda, framework = "custom Python simulation (direct, parallel client training)"
```

Model-specific:
- **Network:** 39 features (matches the project's confirmed-current feature count), `attack_scale = 5.0`.
- **Application:** 90 features (matches confirmed-current feature count), `attack_scale = 2.0`.

Guard/Krum thresholds, mitigated runs only:
```
head_norm_guard_k = 2.5, head_norm_guard_min_keep_fraction = 0.5
adaptive_krum_k = 2.5, adaptive_krum_method = "mad", adaptive_krum_min_keep_fraction = 0.5
adaptive_krum_hybrid_assumed_f = 1
```

**⚠️ Methodology deltas vs. Week 9 — flag these explicitly in the paper's Methodology section, don't silently carry forward the old numbers:**

1. **`he_poly_degree = 8192` in this re-run, not the `4096` used throughout Week 9 and stated as the project's fixed CKKS parameter in every prior revision of the master doc.** This is a real cryptographic-parameter change, not noise — `poly_modulus_degree` directly determines CKKS's security level (v10 states 4096 → 64-bit post-quantum security, explicitly "not 128-bit"). Doubling to 8192 very likely raises that security level (commonly ~109–128-bit territory for CKKS at this degree, though the exact figure depends on the coefficient-modulus chain, which isn't in these config files). **Do not state "64-bit security" for this re-run's numbers — that claim is retired along with `poly_modulus_degree=4096`. Confirm the actual coefficient-modulus chain before stating a new security-level figure in the paper.**
2. **`head_norm_guard_k` and `adaptive_krum_k` are both `2.5` here, not the `3.5` Week 9 reported as its hybrid-specific default.** Either the `k=3.5` change was never actually merged into the `main.py` revision used for this re-run, or it was deliberately reverted. This is worth resolving before writing the Methodology section, since v10 explicitly justified `k=3.5` as "the current, correct hybrid default." **State the actual value used (`k=2.5`) in the paper regardless of which explanation is correct — the CSVs and configs agree unambiguously on 2.5.**
3. **`adaptive_krum_hybrid_assumed_f` now appears as an explicit key in the run-level config JSON** (`= 1`), where v10's Contradiction #9 found it hardcoded with no CLI path to set it. Its presence in the config *output* doesn't by itself prove it's now genuinely tunable — it may simply be a logged/reported value of a still-hardcoded constant. **Don't claim this is now configurable without checking the current `main.py` source directly; only claim it's confirmed to equal 1 in every run here, which is consistent with either explanation.**
4. **`krum_m: 7` / `krum_discards: 3` fields are present in every config JSON despite `use_krum: false`.** These are vestigial fixed-m-Krum fields, consistent with v10's Contradiction #15 finding that `multi_krum()` is missing from the codebase but its call site/config plumbing lingers. Harmless here (the branch never fires), but don't mistake their presence for evidence that fixed-m Krum is active or working.
5. **No DP, no standalone ZKP, no full-model HE in any of the eight runs** — confirmed both by config (`use_dp/use_zkp/use_he = false` throughout) and independently by the data itself (`dp_epsilon_spent`, `dp_epsilon_target`, `dp_noise_multiplier` are `NaN`/`N/A` in literally every row of every file, and `zkp_rejected` is `0` in every row of every file — see §3).

### 1.3 What "mitigated" vs. "unmitigated" means in this pipeline, precisely

Both arms run the identical `USE_HE_KRUM_HYBRID` split — plaintext bulk slice scored by adaptive (MAD-threshold) Krum, encrypted classifier-head slice aggregated separately. The **only** difference is whether the ciphertext-bound HMAC norm guard runs as a pre-filter on the encrypted slice before Krum sees the bulk slice. This is the same architectural contrast v10 documents for `USE_HE_KRUM_HYBRID` vs. `pure_zkp` (Contradiction #12) — except here both arms are the *hybrid* pipeline; the ablation is guard-on vs. guard-off within the same pipeline, not hybrid-vs-standalone-guard. State this distinction explicitly in the Methodology section, since it's easy to conflate with the older `pure_zkp` comparison.

### 1.4 Exact per-round mechanism, traced from `main()`'s `USE_HE_KRUM_HYBRID` branch

This is the level of detail the paper's Methodology section should describe, since it determines what "detection" and "exclusion" actually mean in §2:

**Step 1 — client-side (every client, every round), for a Byzantine client under `BYZANTINE_HEAD_ONLY=True`:**
1. The client trains **normally**, on its own real local data, for `LOCAL_EPOCHS=5` epochs with the standard FedProx loss (`mu=0.02`) — identical to an honest client's training call. This produces a fully legitimate, honestly-trained parameter set.
2. Only *after* training completes, `classifier_head_flip_attack(trained_params, model_state_keys, scale=ATTACK_SCALE)` is applied — this corrupts **only the classifier-head layers**, leaving the backbone/bulk layers exactly as honestly trained. `ATTACK_SCALE` is 5.0 (network) / 2.0 (application).
3. The client then calls `he_local.encrypt_params_with_norm_guard()` (mitigated arm) or `he_local.encrypt_params()` (unmitigated arm), which splits the parameter list via `MODEL_STATE_KEYS` into a `sensitive_enc` (CKKS-encrypted classifier head — corrupted for Byzantine clients) and a `bulk` slice (plaintext backbone — **honestly trained even for Byzantine clients**, since the attack never touches it).

**This is the mechanistic reason the blind-spot hypothesis holds, not just an empirical pattern:** the bulk slice Krum actually scores is, by construction, identical in kind between honest and Byzantine clients (both honestly trained). Krum has no information in the slice it sees that could ever separate the two groups for this attack. The unmitigated arm's 0.00% detection rate is not a limitation of Krum's threshold tuning — it is a structural guarantee, given this attack, that plaintext-only Krum returns to zero.

**Step 2 — server-side, mitigated arm only (`USE_HEAD_NORM_GUARD=True`):**
1. For every accepted client, `zkp.verify_head_norm_proof(proof, chunks)` checks the HMAC commitment binds correctly to the actual encrypted-head ciphertext chunks submitted. A failure here (`PROOF_MISSING` or a binding mismatch) rejects the client immediately, before any threshold logic runs.
2. The L2 norms of all *verified* clients' encrypted-head deltas are passed to `zkp.mad_threshold_head_norms(verified_norms, k=2.5, min_keep_fraction=0.5)` — a MAD-based outlier filter: `threshold = median(norms) + k · 1.4826 · MAD(norms)`, with a floor guaranteeing at least 50% of verified clients are kept regardless of how many exceed the threshold.
3. Clients failing either check (bad proof or above-threshold norm) are added to `norm_guard_rejected_ids` and **removed from consideration for both Krum and the encrypted-slice aggregate** — a guard rejection is a full exclusion, not just a Krum-input filter.

**Step 3 — server-side, both arms — plaintext-bulk Krum:**
1. `adaptive_multi_krum()` runs on the `bulk` slice of whichever clients survived Step 2 (all of them, in the unmitigated arm), with `num_byzantine=ADAPTIVE_KRUM_HYBRID_ASSUMED_F`. This value is **hardcoded to `min(1, NUM_BYZANTINE)`, which equals `1`** in every run here (`NUM_BYZANTINE=2` throughout) — i.e., Krum is told to assume only **one** attacker is present, despite two actually being active. Given Step 1's mechanistic point, this mismatch is unlikely to matter for *this* attack (Krum can't distinguish either client regardless of its assumed-f setting), but it is a real, documented configuration detail worth stating precisely rather than glossing over.
2. Krum's own `k=2.5` (same value as the guard's `k`, driven by the same `--krum-k` CLI default — see v10 Contradiction #10) determines its own, independent MAD-threshold exclusion on the bulk-slice scores.

**Step 4 — the two exclusion sources are merged, and the CSV cannot distinguish which caught which client:**
```
krum_discarded_ids = (bulk-Krum's own discards) ∪ (norm_guard_rejected_ids)
krum_detected_byz  = krum_discarded_ids ∩ {the true Byzantine client IDs}
```
Both the per-client `krum_selected`/`krum_detected_byzantine` CSV columns and the `MEAN` row's aggregate detection rate are computed from this **merged** set. `zkp_rejected_this_round` is a *separate* list that is only ever populated inside the standalone `USE_ZKP` branch — it is never touched inside `USE_HE_KRUM_HYBRID`, which is exactly why `zkp_rejected` reads `0` in every row of every Experiment-2 CSV (confirmed in §2.6): the guard's rejections are real, they are just filed under `krum_selected`/`krum_discarded_ids`, not the `zkp_rejected` column.

**Consequence for interpreting §2's "100.00% detection" figures:** given Step 1's mechanistic argument (bulk-slice Krum cannot see this attack at all) and the unmitigated arm's directly-observed 0.00% detection with the identical Krum step active, **the 100% detection credit in the mitigated runs is almost certainly coming entirely from the head-norm guard, not from Krum** — Krum in this pipeline is very likely contributing zero independent detections for a head-only attack, functioning here purely as the plaintext-slice aggregator for whichever clients the guard already let through. This cannot be proven definitively from the CSV alone (the merged discard set hides which stage acted), but it follows directly from the attack's construction and is strongly consistent with the unmitigated control. **Recommend for the paper:** state this as the mechanistically-expected explanation, and log it as an open item that the CSV export should separate `norm_guard_rejected_ids` from `krum_discarded_ids` as two distinct columns so this can be confirmed directly in a future run rather than inferred.

### 1.5 Does the `he_poly_degree` (4096→8192) or `k` (2.5 vs. Week 9's claimed 3.5) change affect the results?

**`he_poly_degree`: no material effect on F1/accuracy; real effect on security level and cost. Traced from the code, not assumed:**

`HE_POLY_DEGREE` is passed only into `he_local.create_ckks_context()`, `he_local.encrypt_params()` / `encrypt_params_with_norm_guard()`, and `he_local.aggregate_encrypted()` / `decrypt_params()` — i.e., it governs the CKKS ciphertext structure (slot count ≈ `poly_degree/2`, ciphertext size, coefficient-modulus chain), not anything in the training, attack, or Krum-scoring path. CKKS is an *approximate* homomorphic scheme, but its rounding error at any reasonable scale/modulus setting is on the order of `2^-40` — utterly negligible next to the classifier head's actual parameter magnitudes or the model's training noise. Provided `he_local.py`'s parameter presets for 8192 are internally consistent (correct coefficient-modulus chain for that degree — not verifiable from `main.py` alone, since `he_local.py` wasn't included in this upload), doubling the degree should be **numerically transparent to the decrypted, aggregated result** — it does not change what gets aggregated or how well the model trains.

What it *does* change:
- **Security level, upward** — a larger `poly_modulus_degree` at a comparable or larger coefficient-modulus budget raises CKKS's bit-security. v10 documented `poly_degree=4096` as "64-bit, not 128-bit — state honestly." **That 64-bit figure no longer applies to this re-run's numbers** — 8192 is very likely materially more secure (commonly cited in the ~109–128-bit range for CKKS at that degree, depending on the exact modulus chain), but the true figure needs confirming against `he_local.py`'s actual parameter set before restating it in the paper. This is a genuine, positive change to the crypto configuration, not a bug — but it must be re-stated, not silently left at the old "64-bit" claim.
- **Slot count / chunking** — roughly double the slots per ciphertext at 8192 vs. 4096, meaning each client's classifier head packs into **fewer, larger ciphertext chunks** this run than an equivalent 4096-degree run would need. This affects the *granularity* of the HMAC guard's per-chunk binding (`verify_head_norm_proof(proof, chunks)` operates over however many chunks the head was packed into) but not the final aggregate value.
- **Latency/compute cost** — larger ciphertexts cost more to encrypt, transmit, and homomorphically sum. Consistent with the project's own Docker RAM/Latency findings (v10): HE cost scales with ciphertext size/parameter count, not with model accuracy. Expect this re-run's HE-stage timing (folded into the per-round `round_time_s` figures in §2.1/§2.6) to run somewhat slower than an equivalent 4096-degree run would have, all else equal.

**Net: the `he_poly_degree` change is very unlikely to explain any of the F1/accuracy deltas logged in §3 — those are far more likely single-seed variance. It is a real, separate finding for the paper's cryptographic-parameters subsection (stronger security than previously documented), not a confound for the accuracy results.**

**⚠️ UPDATE — the exact coefficient-modulus chain is now confirmed, resolving the open question above.** The Week 9-era `config_Application.json`/`config_network.json` files (uploaded separately, covering the *original* unmitigated Windows/CPU runs, not this re-run) both log `"he_coeff_mod_bit_sizes": [60, 40, 40, 60]` and `"he_global_scale": 1099511627776` (`= 2^40`) at `poly_degree = 8192`. This exactly matches the parameter set `main.py`'s own comment describes as `he_local.py`'s fixed "standard, non-RAM-constrained" default (`n=8192, [60,40,40,60], scale=2**40`) — i.e., this is not a per-run choice, it is `he_local.py`'s one hardcoded CKKS parameterization, so it applies to this re-run's 8192-degree runs too, not just the older Windows runs that happened to log it explicitly. `[60,40,40,60]` at `n=8192` is a standard, widely-cited CKKS parameterization commonly rated at **~128-bit security** in TenSEAL/Microsoft SEAL's own security estimator — a real, citable upgrade from the `poly_degree=4096` configuration's documented 64-bit level. **State ~128-bit security (not "64-bit," and not "unconfirmed") for every run in this re-run's dataset, and update the project's Layer 3 documentation accordingly** — this was the single largest open cryptographic question from the prior manifest revision and is now closed.

**`k` (both `head_norm_guard_k` and `adaptive_krum_k`, sharing the `--krum-k` CLI default of 2.5 here): yes, this plausibly explains part of what changed, and matters directly for both detection and accuracy.**

`k` is the MAD-multiplier in `threshold = median(scores) + k · 1.4826 · MAD(scores)`, used independently by the guard (on head norms) and by Krum (on bulk-slice scores). Its effect is mechanical and two-sided:

- **Smaller `k` → stricter threshold → more clients flagged as outliers.** Given §1.4's finding that detection in this experiment is realistically carried entirely by the guard, the guard's `k` is the single parameter most directly responsible for whether an attacker's head-norm is judged anomalous. At `k=2.5` (this re-run), detection is 100% and reliable — but a stricter threshold also has no way to distinguish "this client's head-norm is high because it's an attacker" from "this client's head-norm is naturally high because of its data distribution" (e.g., clients 4/10's oversized, `Vulnerability_scanner`-heavy partitions plausibly producing larger, but entirely legitimate, gradient/head-norm magnitudes). **This is a direct, mechanistic explanation for §2.3's finding that this re-run's collateral-exclusion rate (30–40% of the fleet per round) is worse than Week 9's documented ~10%:** if Week 9's hybrid-specific default really was `k=3.5` (more permissive) and this re-run used the older, stricter `k=2.5`, a stricter threshold would be expected to catch more legitimate variance as false positives, exactly matching the pattern observed. This turns §3's "unresolved k discrepancy" into a plausible root cause for §2.3's severity finding, not just two unrelated open items — **recommend the paper connect these two findings explicitly, and recommend re-running at least one configuration at `k=3.5` to test this directly before finalizing the collateral-exclusion numbers.**
- **Krum's own `k`**, in this specific pipeline, contributes a second, independent source of possible collateral exclusion on the bulk slice — separate from, and additive to, the guard's exclusions (§1.4, Step 4's discard-set union). A smaller `k` here also excludes more bulk-slice clients, independent of the head-norm guard's behavior.
- **Effect on accuracy/F1:** indirect but real — every client excluded (by either mechanism) removes that client's data from that round's aggregate. Since clients 4/10 (network) hold a disproportionate share of `Vulnerability_scanner` samples (per v10's `print_data_split()` finding), their frequent exclusion under a stricter `k` plausibly explains why `Vulnerability_scanner` F1 plateaus around 0.71–0.78 in the mitigated network runs rather than tracking closer to the other well-represented classes (0.94–0.99) — the class's main data source keeps getting dropped from the round. A larger `k` would keep these clients in more rounds, which should be expected to help that specific class's F1, at the cost of *possibly* letting a genuine attacker's more-marginal deviations through undetected — a real, worth-stating trade-off for the paper's discussion section.

---

## 2. RESULTS SECTION (for the paper)

### 2.1 Summary table (all 8 runs, computed directly from per-round per-class F1 columns)

F1-Macro is computed here as the unweighted mean of the 8 per-class F1 columns on each round's `MEAN` row — the same method used throughout the project's prior documentation.

| Run | Model | Byz. clients | Best round | Best F1-Macro | Round 24 | Round 25 | Δ(25−24) | Detection rate | Clients kept/round | F1 std (25 rounds) | Mean round time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `app_mit_byz1_2` | Application | 1,2 | 24 | **0.7527** | 0.7527 | 0.7280 | −0.0247 | **100%** | 5/10 | 0.190 | 57.8 |
| `app_mit_byz2_7` | Application | 2,7 | 23 | **0.7085** | 0.6981 | 0.6860 | −0.0121 | **100%** | 6/10 | 0.164 | 57.0 |
| `app_mit_byz4_10` | Application | 4,10 | 24 | **0.7017** | 0.7017 | 0.5208 | −0.1809 | **100%** | 5–6/10 | 0.177 | 58.1 |
| `app_unmit_byz1_2` | Application | 1,2 | 25 | **0.1327** | 0.1127 | 0.1327 | +0.0200 | **0%** | 7/10 | 0.035 | 36.9 |
| `net_mit_byz1_2` | Network | 1,2 | 25 | **0.8387** | 0.7046 | 0.8387 | +0.1342 | **100%** | 6/10 | 0.132 | 59.8 |
| `net_mit_byz2_7` | Network | 2,7 | 24 | **0.8346** | 0.8346 | 0.6840 | −0.1505 | **100%** | 5/10 | 0.137 | 59.1 |
| `net_mit_byz4_10` | Network | 4,10 | 17 | **0.8203** | 0.6540 | 0.7393 | +0.0853 | **100%** | 6/10 | 0.161 | 78.5 |
| `net_unmit_byz1_2` | Network | 1,2 | 15 | **0.6853** | 0.4983 | 0.4812 | −0.0170 | **0%** | 8/10 | 0.149 | 60.9 |

### 2.2 Headline result — the blind spot reproduces exactly, the mitigation reproduces exactly

**Unmitigated arm confirms the Week 9 finding at essentially the same magnitude.** Detection rate is **0.00%** in every round of both unmitigated runs (`krum_detected_byzantine == 0` on every one of the 25 `MEAN` rows, both files) — the attackers are never once flagged. This is not "low detection," it is total, structural blindness, consistent with v10's framing that Krum, seeing only the plaintext bulk slice, has no signal at all about an attack confined to the encrypted head.

The damage pattern matches the specific class-level signature Week 9 reported:
- **Application, unmitigated:** `Normal` F1 is `0.0000` in 24 of 25 rounds (a single transient recovery to 0.455 at round 2, collapsing again from round 3 onward) — matching Week 9's "`Normal` F1 sat at 0.0000 every round" almost exactly.
- **Network, unmitigated:** `Vulnerability_scanner` F1 is `0.0000` in 24 of 25 rounds (round 1 shows a transient 0.154, then permanent collapse) — the same class Week 9 flagged as destroyed, reproduced here independently.
- Best F1-Macro collapses to **0.1327 (application)** and **0.6853 (network)** — the application number is nearly identical to Week 9's reported 0.13; the network number is meaningfully lower than Week 9's reported ~0.72 (see §2.5, deltas).

**Mitigated arm recovers to near-baseline in all six configurations, with 100.00% Byzantine detection in every round of every run, regardless of attacker identity.** This is the strongest result in this dataset: the guard's effectiveness does not depend on *which* two clients are attacking — clients 1/2, 2/7, and 4/10 are all caught with equal, perfect reliability. Best F1-Macro across the six mitigated runs ranges 0.7017–0.8387, comparable to the project's locked clean baselines (0.7293 application / 0.8289 network) — i.e., the mitigation absorbs essentially all of the attack's utility cost, on both models, at every tested attacker identity.

### 2.3 New finding — the honest-client exclusion anomaly is confirmed structural, attacker-identity-independent, and larger than Week 9 characterized it

This re-run gives per-client `krum_selected` values for every round of every run, which lets exclusion identity be read directly rather than inferred from aggregate counts (the ambiguity v10's Contradiction #13 flagged). The pattern is unambiguous and, in the network model's case, **stronger and more persistent than Week 9's data showed**:

**Network model — clients 4 and 10 are excluded in literally every round (25/25) of every configuration in which they are not themselves the attackers:**

| Run | Honest clients excluded every round (25/25) |
|---|---|
| `net_mit_byz1_2` (attackers 1,2) | **4, 10** |
| `net_mit_byz2_7` (attackers 2,7) | **4, 5, 10** |
| `net_mit_byz4_10` (attackers 4,10) | **5, 7** (i.e., once 4/10 are themselves attacking, the exclusion pattern shifts to different clients — see below) |
| `net_unmit_byz1_2` (attackers 1,2, **guard OFF**) | **4, 10** — *identical exclusion pair, with the guard disabled* |

The last row is the most important new evidence: clients 4 and 10 are excluded by the **plaintext-bulk Krum step alone**, with the norm guard turned off entirely. This directly confirms v10's Contradiction #7 root-cause hypothesis (clients 4/10 hold an oversized partition and a disproportionate share of the `Vulnerability_scanner` class under Dirichlet(0.7) partitioning) — the exclusion is a Krum-on-imbalanced-data artifact, not an interaction with the norm guard or the attack. It is present with or without the guard.

**Application model — client 6 is excluded in every round of every configuration in which it is not itself an attacker; clients 3 and 7 show a similar but attacker-identity-dependent pattern:**

| Run | Honest clients excluded every round (25/25) | Transient extra exclusion |
|---|---|---|
| `app_mit_byz1_2` (attackers 1,2) | **3, 6, 7** | — |
| `app_mit_byz2_7` (attackers 2,7) | **3, 6** | — |
| `app_mit_byz4_10` (attackers 4,10) | **6, 7** | **client 5, round 13 only** (`krum_selected` count drops 6→5 for exactly one round, confirmed via the per-client column: client 5 reads `krum_selected=0` only at round 13, `1` in all other 24 rounds) |
| `app_unmit_byz1_2` (attackers 1,2, **guard OFF**) | **3, 6, 7** — identical set to the guard-ON version of the same attacker pair |

Client 6's presence in all three mitigated configurations, and the guard-off/guard-on identity match for the 1,2-attacker case, together confirm the same conclusion as the network model: this is a partition-composition effect that the norm guard neither causes nor fixes — it is inherited entirely from the underlying adaptive-Krum bulk-slice step.

**This has a direct, quantifiable cost.** "Clients kept per round" in §2.1 ranges from 5–6 out of 10 for mitigated application runs (i.e., 30–40% of the fleet excluded every single round, only 20% of which are actual attackers) and 5–6 out of 10 for mitigated network runs (also 30–40%, half of it collateral). This is a materially larger collateral-exclusion rate than v10 characterized Week 9's version of this anomaly as ("clients 4/5/10" — 3 of 10, i.e., 10% collateral on top of the 2 attackers). **This re-run's data supports revising the anomaly's severity upward in the paper: the exclusion cost is roughly 1.5–2 extra honest clients per round, consistently, not an occasional 1-client edge case.**

**Recommendation for the paper:** report this as a first-class limitation of the plaintext-bulk-Krum design, not a footnote. The guard successfully solves the problem it targets (encrypted-head attacks), but it operates downstream of an already partition-sensitive Krum step whose collateral cost is now confirmed, with per-client identity evidence, across 7 of 8 runs in this dataset.

### 2.4 Round-25 instability — recurs again, in both directions

Consistent with v10's characterization of round-25 instability as structural to this FedProx/non-IID setup (5 prior documented occurrences), this dataset shows a round 24→25 swing in **6 of 8 runs**, split between drops and recoveries:

- **Drops:** `app_mit_byz1_2` (−0.025), `app_mit_byz2_7` (−0.012), `app_mit_byz4_10` (−0.181, the largest swing in this dataset — driven substantially by `Fingerprinting` F1, already low at 0.272 on round 24, collapsing further), `net_mit_byz2_7` (−0.151), `net_unmit_byz1_2` (−0.017).
- **Recoveries:** `net_mit_byz1_2` (+0.134 — round 25 is actually this run's *best* round), `net_mit_byz4_10` (+0.085), `app_unmit_byz1_2` (+0.020, though this run is so degraded throughout that a 0.02 swing is within its own noise floor).

Two of the eight best-F1 rounds in this dataset are round 25 itself (`net_mit_byz1_2`), reinforcing v10's point that round-25 instability is bidirectional volatility, not a one-directional "always gets worse" effect — worth stating precisely rather than as a blanket "round 25 is unreliable."

### 2.5 New volatility outlier: `net_mit_byz4_10`

This run's round-to-round F1-Macro swings (mean absolute round-to-round delta = **0.152**, roughly double every other mitigated run's 0.04–0.07 range) make it the most volatile run in the dataset, with 100% detection maintained throughout — volatility here is not a Krum-detection failure. Its trajectory dips sharply at rounds 13 (0.485) and 19 (0.481), each time recovering within 1–2 rounds. This echoes, but at a different round range, the Week 9 finding of a `network_byz4_10` round-19–25 volatility sequence — **the same configuration family shows the same volatility signature in this independent re-run**, which is reasonably strong evidence this is a real property of the byz4_10 attacker-identity configuration on the network model, not a one-off artifact of the original (possibly corrupted) Week 9 files.

This run also has the highest mean round time (78.5s vs. 57–61s for every other run) — worth a footnote checking whether this reflects genuine extra compute (e.g., more guard rejections triggering more re-verification) or GPU contention during that specific run, per v10's documented vLLM-contention incident precedent.

### 2.6 Data integrity checks (performed directly on the raw CSVs)

- **Row structure:** all 8 files are 275 data rows (11 rows/round × 25 rounds: 10 clients + 1 `MEAN` row), confirming `NUM_CLIENTS=10`, `NUM_ROUNDS=25` — consistent with every prior experiment in this project.
- **DP columns:** `dp_epsilon_spent`, `dp_epsilon_target`, `dp_noise_multiplier` are null in every single row of all 8 files — directly confirms `USE_DP=False` from the data itself, not just the config.
- **ZKP-rejection column:** `zkp_rejected` is `0` in every row of all 8 files, including rounds with real exclusions — consistent with v10's documented code behavior (`USE_HE_KRUM_HYBRID` folds norm-guard rejections into `krum_selected`, not the separate `zkp_rejected` list, which is only populated by the standalone `USE_ZKP` branch that isn't active here).
- **NaN guard:** `nan_this_round` is `0` in every populated row of all 8 files — no NaN-collapse events in this dataset.
- **Detection-flag consistency:** per-byzantine-client mean detection rate is exactly `1.0` for both named attacker clients in every mitigated run, and exactly `0.0` for both in every unmitigated run — no partial/inconsistent detection anywhere.

No corrupted rows, truncated files, or structural anomalies were found in any of the 8 uploaded CSVs.

---

## 3. DELTAS VS. WEEK 9 — EXPLICIT LOG (per your "supersede but log deltas" instruction)

| Item | Week 9 (v10 record) | This re-run | Delta / Verdict |
|---|---|---|---|
| Application, unmitigated, best F1-Macro | 0.13 | 0.1327 | Matches closely — reproduces. |
| Network, unmitigated, best F1-Macro | ~0.72 | 0.6853 | **−0.035, a real gap.** Both show the same blind-spot pattern and 0% detection, but the magnitude differs enough to note; likely single-seed variance, possibly compounded by the `poly_modulus_degree` change (§1.2) or other un-logged config drift between the corrupted original run and this one. |
| `krum_score_ratio`, application unmitigated | ~0.38 (flat) | 0.415 (flat, std≈0.0001) | Close but not identical; both show the signature "flat, near-constant ratio" pattern that indicates attackers scoring statistically indistinguishable from honest clients. |
| `krum_score_ratio`, network unmitigated | ~0.237 (flat) | 0.325 (flat, std≈0.00003) | **Larger gap than the application model's.** Same qualitative finding (flat ratio, blind spot), different absolute level — flag as needing explanation if the paper reports this number precisely rather than just the qualitative pattern. |
| Application default (clients 1,2), mitigated, best F1 | 0.7286 @ round 24 | 0.7527 @ round 24 | +0.024, same best round — reproduces well, slightly better recovery this time. |
| Application byz4_10, mitigated, best F1 | 0.6954 @ round 23 | 0.7017 @ round 24 | Close in value, round shifted by one — reproduces well. |
| Network byz4_10, mitigated, best F1 | 0.8360 @ round 20 | 0.8203 @ round 17 | Close in value, round shifted — reproduces well. |
| Network byz2_7, mitigated, best F1 | 0.8237 @ round 21 | 0.8346 @ round 24 | Close in value, round shifted — reproduces well. |
| Network default (clients 1,2), mitigated | **CSV was empty in Week 9 — never verified** | **0.8387 @ round 25 — now recovered** | **Closes the single largest open gap in Week 9's Experiment 2 record** (v10 Contradiction #11's "one run remains fully unverified"). This is the first real data for this configuration. |
| Honest-client collateral exclusion, network | Clients 4/5/10 (Week 9's byz2_7 run; client 5's exclusion described as a **transient one-round anomaly**, round 24 only) | Clients 4/10 in byz1_2 and byz4_10; **4, 5, 10 in byz2_7 — but client 5 is excluded in ALL 25 rounds here, not one** | **Material change in severity.** If Week 9's "one transient round" description was accurate for the original run, this re-run shows a persistently worse version of the same anomaly for the identical configuration (byz2_7). Recommend flagging this explicitly rather than treating the two as consistent. |
| Honest-client collateral exclusion, application | Clients 6/7 (cross-configuration) | Clients 6 (all 3 configs), 7 (2 of 3), 3 (2 of 3) | Consistent with, and adds detail to, the Week 9 finding — client 6 is confirmed as the one truly configuration-independent exclusion; 7 and 3 are more attacker-identity-dependent than Week 9's summary suggested. |
| `head_norm_guard_k` / `adaptive_krum_k` | **Refined, not simply contradicted — see below.** | **2.5 (+ `assumed_f=1`) in every config file in this re-run** | See the dedicated note immediately below the table — the "k=3.5" claim was real, but only for 3 of Week 9's 5 mitigated runs, and this re-run's exact `k`/`assumed_f` pairing doesn't match either of Week 9's two historical configurations cleanly. |
| `he_poly_degree` | 4096 (stated as fixed project-wide parameter, tied to the 64-bit security claim) | **8192** | **New, undocumented change.** Retire the "64-bit security" claim for this re-run's numbers; the true security level at degree=8192 needs to be confirmed against the actual coefficient-modulus chain before restating. |
| Zero-detection blind-spot claim (headline hypothesis) | Confirmed, Week 9 | **Confirmed again, independently, both models** | No change — this is now doubly confirmed. |
| 100% detection under mitigation, all attacker identities | Confirmed for 5 configurations, Week 9 | **Confirmed for 6 configurations, including one new attacker-identity pair (application 2,7) not tested in Week 9** | Strengthens the claim — detection reliability now demonstrated across more of the attacker-identity space. |

---

**⚠️ `k`/`assumed_f` correction, from the Week 9-era `COMBINED_MANIFEST.md` and its `manifest.json` (uploaded separately, not available when the deltas table above was first written):** Week 9's mitigated runs were not uniform. The two **default-attacker (clients 1,2)** runs — the ones this re-run's `app_mit_byz1_2`/`net_mit_byz1_2` most directly correspond to — used `k=2.5` with `assumed_f = NUM_BYZANTINE` (i.e. **2**, explicitly logged as `"NUM_BYZANTINE (pre-tuning defaults)"`). Only the *later* three Week 9 runs (`byz4_10` on both models, `byz2_7` on network) were run after a mid-week tuning change to `k=3.5, assumed_f=1`, specifically to reduce the honest-client collateral-exclusion problem those later runs' authors had already noticed. **This re-run's configs (`k=2.5`, `assumed_f=1` throughout) match neither Week 9 configuration exactly** — it has Week 9's *older* `k` alongside Week 9's *newer* `assumed_f`. This is worth stating precisely rather than as a single contradiction: it looks like the re-run's `main.py` reverted `k` back to 2.5 (or never received the mid-week bump) while keeping the `assumed_f=1` change, which is a specific, checkable claim about which commit/edit state produced this data — not just "the config disagrees with the log."

**This also resolves part of §2.3's severity question.** Week 9's *own* `k=2.5/assumed_f=2` runs (the directly comparable ones, `app_mit_byz1_2`/`net_mit_byz1_2`) are the right baseline for comparison, not the `k=3.5` runs — and even Week 9's `k=2.5` config only reported **2 extra honest exclusions** per model (network: 4,10; application: 6,7), versus this re-run's `k=2.5/assumed_f=1` showing **3 extra exclusions** in the equivalent application default run (3,6,7) and holding at 2 for the network default run (4,10). The `assumed_f` drop from 2→1 is the more likely lever here, not `k` alone — a smaller assumed-Byzantine-count going into Krum's own MAD threshold makes Krum's *own* (not the guard's) bulk-slice exclusion stricter too, independent of the guard-side reasoning in §1.5. **Revise §1.5's recommended follow-up test accordingly: re-run at `assumed_f=2` (matching true `NUM_BYZANTINE`) as well as `k=3.5`, to separate which of the two changes is actually driving the collateral-exclusion difference.**

**One more concrete delta surfaced by direct comparison against these older files, not previously logged:**

| Finding | Week 9 (console-log reconstructed) | This re-run (real CSV) |
|---|---|---|
| Network, **unmitigated**, honest-client exclusions | **3 clients (4, 5, 10)** — per the original `RUN_NOTES_network.md` | **2 clients (4, 10)** — measured directly from `net_unmit_byz1_2`'s per-client `krum_selected` column |
| Network, **byz4_10 mitigated**, extra (non-attacker) exclusions | **1 client (5)** — per `COMBINED_MANIFEST.md`'s Run 3 | **2 clients (5, 7)** — measured directly from `net_mit_byz4_10`'s per-client `krum_selected` column |

Both entries move in the same direction as §2.3's headline finding: this re-run shows *more* collateral exclusion than the corresponding Week 9 console-log record, in both the guard-off and `assumed_f=1` guard-on conditions. Three independent comparisons (this table's two rows, plus §2.3/§3's client-5-in-byz2_7 finding) now all point the same way — this is no longer a single anomaly, it's a consistent pattern across this whole re-run relative to Week 9, and `assumed_f` is the most likely single lever given the config trace above.

---

## 4. OPEN ITEMS FROM THIS RE-RUN

1. **Missing config JSON for `net_unmit_byz1_2`.** Its methodology is inferred by analogy, not confirmed. Request the actual file.
2. **Resolve the `he_poly_degree` discrepancy (4096 → 8192) and the associated security-level claim** before writing the paper's cryptographic-parameters subsection.
3. **Resolve the `k=2.5` vs. Week 9's claimed `k=3.5`** — check which `main.py` revision actually produced this re-run's data.
4. **Explain the network unmitigated best-F1 gap (0.6853 here vs. ~0.72 in Week 9)** and the `krum_score_ratio` level differences (both models) — likely single-seed noise, but worth a repeat-seed run to confirm before treating either number as final.
5. **Decide whether the worsened client-5 exclusion persistence in network byz2_7 (25/25 rounds here vs. 1/25 in Week 9) reflects a real behavioral change or was always the true behavior and Week 9's "transient" framing was itself an artifact of the now-corrupted original files.** This matters for how confidently the paper can describe the exclusion anomaly's severity.
6. **No repeat-seed runs exist for any of these 8 configurations** — every number in this manifest is single-seed, consistent with the project's standing, not-yet-resolved reproducibility gap.
7. **The collateral-exclusion severity finding (§2.3) should be escalated in the paper's limitations discussion** — this re-run's per-client evidence is stronger and more consistent than what was available at Week 9, and shows a materially larger cost (30–40% of the fleet excluded per round) than previously characterized.
8. **Confirm `main.py`'s `LOCAL_EPOCHS`/`DP_BATCH_SIZE` divergence from these runs' logged config (§0) doesn't reflect a deeper, unlogged code drift** — the two fields checked disagree with the config JSONs (config wins, since it's a direct dump of the runtime variables), but this means the pasted file is confirmed *not* byte-identical to what ran Experiment 2. Nothing else checked disagreed, but this is worth a sanity note in the paper's reproducibility section: cite the config JSON's values, never the current source file's hardcoded defaults, for describing these 8 runs.
9. **Test whether `k=3.5` (Week 9's claimed hybrid default) meaningfully reduces the collateral-exclusion rate found in §2.3, as §1.5 mechanistically predicts it should.** This is now a concrete, testable follow-up rather than an open question — re-run at least one configuration (e.g. `net_mit_byz2_7`, which shows the worst collateral exclusion — clients 4, 5, 10 every round) at `--krum-k 3.5` and compare.
10. **Split `norm_guard_rejected_ids` and Krum's own `krum_discarded_ids` into two separate CSV columns**, rather than the current merged `krum_discarded_ids`/`krum_selected` fields — this would let §1.4's "detection is almost certainly coming entirely from the guard, not Krum" claim be confirmed directly from data instead of inferred from the attack's construction and the unmitigated control.
11. ~~Resolve the `he_poly_degree`→security-level open question~~ → **Resolved (§1.5 update): `[60,40,40,60]`/`scale=2^40` confirmed, ~128-bit security.**
12. **Re-run at least one configuration with `ADAPTIVE_KRUM_HYBRID_ASSUMED_F=2` (matching true `NUM_BYZANTINE`), separately from the `k=3.5` test in item #9** — the Week 9 cross-reference above suggests `assumed_f` (2→1) is at least as likely a driver of the worse collateral-exclusion rate as `k` (2.5 vs. 3.5) is, and the two haven't been tested independently in any run seen so far.

---

## 5. WEEK 9 LEGACY DOCUMENTS — TRIAGE (keep / fold in / archive)

You separately uploaded 7 older Week 9-era files (`Run_notes_network.md`, `RUN_NOTES_Application.md`, `config_Application.json`, `config_network.json`, two copies of `manifest.json`, `COMBINED_MANIFEST.md`). These are console-log-reconstructed records from before this project had any real, verified CSVs for Experiment 2 — i.e., exactly the material v10's Contradiction #11 was independently checking when it found "these check out almost exactly, with two small corrections." Here's what each contributes now that real data exists, and what to do with it:

| File | Contains anything not already in this manifest? | Verdict |
|---|---|---|
| `Run_notes_network.md` / `RUN_NOTES_Application.md` (the original **unmitigated**, pre-guard run notes) | Yes — the "3 clients (4,5,10)" unmitigated exclusion claim, now compared against real data in §3's new table above. Also documents the original Windows/CPU/`ProcessPoolExecutor` environment and the `classifier_head_flip_attack` training-order fix, both already covered elsewhere in this project's documentation. | **Archive, don't delete outright.** The exclusion-count discrepancy is now captured in §3; nothing else here is uniquely load-bearing, but these are the original primary source for that finding and worth keeping in a `/legacy` folder rather than deleting, in case the discrepancy needs re-tracing later. |
| `config_Application.json` / `config_network.json` (the unmitigated runs' configs) | **Yes — the CKKS `he_coeff_mod_bit_sizes`/`he_global_scale` values**, which resolved this manifest's §1.5 security-level open item (now folded in above). Also confirms the pre-correction `he_pct_encrypted_measured: 5.8` figure (already known-superseded by v10 Contradiction #21's authoritative 3.6%) and documents an unresolved Python-version ambiguity on the original Windows machine (venv named `.venv311` but pip pointed at `pythoncore-3.14-64`) that was never a blocker for anything downstream. | **Keep — the CKKS parameter values are now cited directly in §1.5.** Low priority otherwise; the Python-version note is a closed/irrelevant environment detail (this project has since moved entirely to the DGX). |
| `manifest.json` (the unmitigated-runs version — `application_he_krum_hybrid_v1`/`network_he_krum_hybrid_v1`, both `krum_detection_rate: 0.0`) | No — every number here (best F1 0.1310/0.7203, 0% detection, BrokenProcessPool crash counts) is superseded by this re-run's real, verified `app_unmit_byz1_2`/`net_unmit_byz1_2` CSVs (§2.2). The crash/resume history is pure environment trivia specific to the old Windows machine, not relevant to the DGX-based re-run. | **Safe to delete or archive.** Nothing here is uniquely informative once the real unmitigated CSVs exist (which they now do, per this manifest). |
| `COMBINED_MANIFEST.md` (the **mitigated** Week 9 runs — 5 configs, console-log reconstructed) | **Yes — this is the single most load-bearing legacy file.** It's the source of: (a) the precise `k`/`assumed_f` evolution timeline used to correct §3 above, (b) the original Cross-Byzantine-Configuration Summary table that first established the clients-4/10-and-6/7 persistent-exclusion pattern (which this re-run's §2.3 now independently reproduces with real per-client data — the two together are a genuine two-generation confirmation of the same finding, worth citing as such in the paper), and (c) the `honest_mean_score`-stays-flat observation about Krum's score behavior, not previously logged anywhere in this manifest and not yet investigated. | **Keep — do not delete.** This is a primary source the paper's limitations/robustness section should cite directly (with the caveat that it's console-log-reconstructed, not raw-CSV-verified, which the doc itself already states plainly). |
| `manifest.json` (the **mitigated**-runs version, backing `COMBINED_MANIFEST.md`) | Same content as `COMBINED_MANIFEST.md`'s embedded JSON blocks, just consolidated into one file across all 5 runs (the Markdown version splits it across the top and the per-run sections). No new information beyond what's already extracted above. | **Redundant with `COMBINED_MANIFEST.md` — keep one, delete the other.** Recommend keeping `COMBINED_MANIFEST.md` (the Markdown version has the narrative/anomaly writeups the bare JSON doesn't) and treating this JSON file as disposable. |

**Bottom line:** delete the unmitigated `manifest.json` (doc 14) freely — it's fully superseded. Archive (don't necessarily delete) the two `RUN_NOTES` files and the two unmitigated `config_*.json` files — low ongoing value, but they're the traceable source for one real delta now logged in §3. **Keep `COMBINED_MANIFEST.md` specifically** — it's still your best citable evidence for the cross-configuration exclusion pattern predating this re-run, and the paper should probably cite it alongside this re-run's data as two independent generations of the same finding, not just supersede it silently. Delete its duplicate `manifest.json`.

### 5.1 Full content carried forward from `COMBINED_MANIFEST.md` (so the source file is no longer needed)

Everything below is reproduced directly from the Week 9 `COMBINED_MANIFEST.md`, so that file can now be deleted along with the rest of the legacy batch without losing anything.

**Cross-Byzantine-Configuration Summary (Week 9, console-log reconstructed — the original source for the persistent-exclusion finding this re-run's §2.3 independently reproduces with real per-client CSV data):**

| Attacked clients | Model | Extra Krum exclusions (beyond attackers) | Best F1-Macro |
|---|---|---|---|
| 1,2 (default) | network | 4, 10 | 0.8282 |
| 1,2 (default) | application | 6, 7 | 0.7286 |
| 4,10 (extreme data) | network | 5 | 0.8360 |
| 4,10 (extreme data) | application | 6, 7 | 0.6954 |
| 2,7 (moderate data) | network | 4, 10 | 0.8237 |

Week 9's own reading of this table: clients 4 and 10 (network) and clients 6 and 7 (application) recur as Krum's "extra" exclusions across multiple different attack configurations, including ones that don't target them at all — read at the time as the clearest available evidence that these exclusions are driven by partition size/composition, not by anything related to the actual attack. **This re-run's §2.3 reproduces the same client identities (network 4/10, application 6) from real per-client CSV data, independently, two generations of runs apart — cite both together in the paper as corroborating evidence, not as one superseding the other.**

**Two additional Week 9 technical observations, not previously carried into this manifest:**

- **`honest_mean_score` (Krum's own diagnostic column) stayed essentially flat across all 25 rounds in both `network_he_krum_hybrid_norm_guard_v1` (~2.18M region) and `application_he_krum_hybrid_norm_guard_v1` (~125,945–125,961) — a narrow enough band that Week 9 flagged it as curious.** The working hypothesis at the time (never investigated further) was that Krum's squared-Euclidean distance metric is dominated by large, slow-moving parameter magnitudes in the bulk slice, so it's largely insensitive to the smaller, round-to-round training deltas that would otherwise be the more interesting signal. **This is a real, still-open methodological question for the paper's limitations/future-work section** — if Krum's score is mostly tracking raw parameter scale rather than genuine behavioral deviation, that has implications for how much of this pipeline's separation between honest and Byzantine clients is really coming from the score's sensitivity vs. just the attack being a large enough outlier to clear a fairly insensitive bar regardless. Worth checking this re-run's own `krum_scores_honest_mean` column (present in every CSV, §2.6) for the same flatness pattern before writing this up as settled.
- **`defences/krum.py`'s inner per-round score-table print uses *local, post-filter* client positions, not original client IDs.** Confirmed correct by hand-tracing in Week 9, but flagged as "a real readability landmine for anyone skimming the log quickly" — i.e., a console-log line showing "client 3" mid-run may not correspond to actual client ID 3 once guard-rejected clients have already been removed from the indexed list that round. **Not a data-correctness bug** (the final CSV's client-ID columns are unaffected, per Week 9's confirmation and this manifest's own §2.6 integrity checks), but a real source of confusion if anyone re-derives findings from raw console logs (as several early Week 9 documents did) rather than the CSV. **Not yet fixed in code as of this revision — flag as a minor but still-open code-quality item if `main.py`'s console output is ever cited directly in the paper.**

---

*Generated from direct analysis of 8 result CSVs (275 rows each, verified) and 7 of 8 expected config JSONs. All F1-Macro, detection-rate, and exclusion-identity figures in this document were computed directly from the raw per-round, per-client data — none are carried forward from prior documentation without independent recomputation from these files.*
