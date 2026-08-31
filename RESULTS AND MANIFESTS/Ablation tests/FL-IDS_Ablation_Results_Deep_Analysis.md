# FL-IDS Standalone Ablation Results — Deep Analysis
### `pure_dp` / `pure_he` / `pure_zkp` — Verification, Mechanism, Caveats, Blockers, Reproduction Logic

**Source data:** `results_{network,application}_pure_{dp,he,zkp}.csv` (6 files, 25 rounds × 10 clients + MEAN row each), `main_{dp,he,zkp}_{network,application}.log` (6 console logs), `experiment_config_network_pure_{dp,he,zkp}.json` (3 configs), cross-checked against `main.py` and `defences/{krum,zkp,he_local,he_aggregation,byzantine}.py`.

**Method:** every number below was recomputed directly from the raw CSVs with pandas (not copied from any prior write-up), then cross-validated against independently-generated console log lines, then traced back to the exact code path in `main.py` that produces it — theory, not just arithmetic.

---

## 1. Experimental design — what each run actually is

| Mode | Defence active | Attack | `dp_safe` (architecture) | HE poly degree | Krum called? |
|---|---|---|---|---|---|
| `pure_dp` | DP-SGD (Opacus) + FedProx only | OFF | **True** — GroupNorm + DPLSTM | N/A | No |
| `pure_he` | Partial CKKS HE (classifier head only) | OFF | False — BatchNorm + LSTM | 8192 (128-bit) | No |
| `pure_zkp` | HMAC ciphertext-bound head-norm guard (MAD, k=2.5) | **ON**, head-only | False — BatchNorm + LSTM | 8192 (128-bit) | No |

`pure_dp` and `pure_he` are clean, attack-free utility/cost ablations. `pure_zkp` is the only one with a live attacker, and exists specifically to test whether the HMAC head-norm guard alone — with no Krum backing it up — catches a classifier-head-only attacker.

**Fixed config across all six runs** (confirmed identical in all three JSON configs): `NUM_CLIENTS=10`, `NUM_ROUNDS=25`, `LOCAL_EPOCHS=5`, `LEARNING_RATE=0.001`, `PROX_MU=0.02`, `NUM_BYZANTINE=2`, `BYZANTINE_CLIENTS=[0,1]`, `ATTACK_SCALE=5.0`, device=`cuda`, `client_pool_workers=1` (GPU sequential-in-process mode, no `ProcessPoolExecutor`).

---

## 2. Section-by-section verified results

### 2.1 — Detection performance (`pure_zkp` only)

Recomputed directly from the `zkp_rejected` column (not read from any prior summary):

| Model | Clients rejected | Rounds flagged | Detection rate | False positive rate |
|---|---|---|---|---|
| Network | 1, 2 (0-indexed 0,1 — both true attackers) | 25/25 | **100.00%** | **0.00%** |
| Application | 1, 2 (0-indexed 0,1 — both true attackers) | 25/25 | **100.00%** | **0.00%** |

No client outside {0,1} was ever flagged, in either model, in any round. Cross-validated against `main_zkp_network.log`, which independently logs `rejected_ids=[0, 1]  detected_byz=[0, 1]` and `Detection rate this round: 100.00%` on every round — two independently-generated artifacts agree exactly, which rules out a CSV-writing bug masquerading as a clean result.

### 2.2 — Full round-by-round F1-Macro / accuracy trajectories

Recomputed for all six runs (F1-Macro = unweighted mean of the 8 per-class F1 columns on each round's MEAN row):

**Network models:**

| Round | pure_dp F1 / Acc | pure_he F1 / Acc | pure_zkp F1 / Acc |
|---|---|---|---|
| 1 | 0.126 / 0.370 | 0.156 / 0.442 | 0.270 / 0.571 |
| 5 | 0.650 / 0.892 | 0.749 / 0.910 | 0.730 / 0.887 |
| 10 | 0.683 / 0.917 | 0.763 / 0.903 | 0.760 / 0.914 |
| 15 | 0.748 / 0.923 | 0.805 / 0.951 | 0.719 / 0.875 |
| 18 | 0.782 / 0.934 | **0.446 / 0.527** ← collapse | 0.762 / 0.904 |
| 19 | 0.787 / 0.936 | 0.765 / 0.903 ← recovers | 0.820 / 0.955 |
| 21 | 0.759 / 0.924 | **0.810 / 0.951** ← best | 0.769 / 0.919 |
| 23 | **0.790 / 0.939** ← best | 0.722 / 0.849 | 0.809 / 0.933 |
| 25 | 0.749 / 0.930 | 0.744 / 0.880 | **0.821 / 0.950** ← best |

**Application models:**

| Round | pure_dp F1 / Acc | pure_he F1 / Acc | pure_zkp F1 / Acc |
|---|---|---|---|
| 1 | 0.081 / 0.409 | 0.129 / 0.428 | 0.106 / 0.419 |
| 2 | 0.085 / 0.294 | 0.129 / 0.428 ← **identical to r1** | 0.115 / 0.421 |
| 5 | 0.172 / 0.294 | 0.411 / 0.637 | 0.441 / 0.690 |
| 10 | 0.302 / 0.531 | 0.624 / 0.794 | 0.660 / 0.792 |
| 15 | 0.436 / 0.587 | 0.678 / 0.845 | 0.539 / 0.647 ← dip |
| 20 | 0.579 / 0.712 | 0.715 / 0.839 | 0.725 / 0.847 |
| 21 | **0.428 / 0.564** ← collapse | 0.754 / 0.865 | 0.731 / 0.852 |
| 24 | **0.591 / 0.750** ← best | 0.701 / 0.821 | **0.747 / 0.864** ← best |
| 25 | 0.566 / 0.708 | **0.759 / 0.870** ← best | 0.732 / 0.852 |

**New observation not in any prior write-up — `application_pure_he` rounds 1 and 2 have byte-identical accuracy and all 8 per-class F1 values** (only `loss` and `round_time_s` differ between the two rows). This means the aggregated model made *zero different classification decisions* between rounds 1 and 2 — the argmax output was frozen even though the underlying logits shifted slightly (loss dropped 1.808→1.732). Checked systematically: this is the **only** such occurrence across all six runs (no other adjacent-round pair in any run matches this closely). Most plausible explanation: an early-training stall where the model is still predicting a narrow majority-class-dominated output (both rounds show F1=0.0 for Backdoor, Port_Scanning, and Fingerprinting — i.e. three of eight classes are being ignored entirely), consistent with the application model's severe class imbalance and its harder 90-feature/8-class problem. **This is not a bug indicator by itself, but is worth a one-line footnote if this run's early-round trajectory is shown in a figure** — a reader could otherwise mistake it for a caching/logging error.

### 2.3 — Best-round summary (matches all figures cited in prior verification passes, recomputed independently here)

| Run | Best round | Best F1-Macro | Best Accuracy |
|---|---|---|---|
| network / pure_dp | 23 | 0.7902 | 0.9393 |
| network / pure_he | 21 | 0.8095 | 0.9506 |
| network / pure_zkp | 25 | **0.8214** ← best of all 6 | 0.9500 |
| application / pure_dp | 24 | 0.5905 | 0.7501 |
| application / pure_he | 25 | 0.7585 | 0.8698 |
| application / pure_zkp | 24 | 0.7473 | 0.8642 |

Ranked across all six: `network_pure_zkp` (0.8214) > `network_pure_he` (0.8095) > `network_pure_dp` (0.7902) > `application_pure_he` (0.7585) > `application_pure_zkp` (0.7473) > `application_pure_dp` (0.5905). The network model consistently outperforms the application model across every mechanism — consistent with the application model's harder, more class-imbalanced 8-class problem (rare classes as low as 1,001 raw samples for SQL_injection) versus the network model's more separable classes.

### 2.4 — Per-class F1 at best round (both models, all three mechanisms)

**Application** (round 24 for pure_dp/pure_zkp, round 25 for pure_he):

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

**Network** (round 23 for pure_dp, round 21 for pure_he, round 25 for pure_zkp):

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

**Pattern confirmed across both tables:** `pure_dp` is consistently the weakest on the hardest, most data-starved classes (application: XSS 0.26, Password 0.35, Uploading 0.37 — exactly the classes flagged elsewhere as rare/hard). MITM stays the network model's weakest class across all three mechanisms (0.43–0.45), essentially flat regardless of which defence is active — consistent with a hard data-scarcity ceiling (1,214 raw samples) rather than a mechanism-specific artifact.

### 2.5 — Timing

| Run | Mean round time | Total wall time (25 rounds) |
|---|---|---|
| network / pure_dp | 169.40s | 70.58 min |
| network / pure_he | 74.05s | 30.85 min |
| network / pure_zkp | 63.59s | 26.50 min |
| application / pure_dp | 196.42s | 81.84 min |
| application / pure_he | 51.91s | 21.63 min |
| application / pure_zkp | 42.81s | 17.84 min |

DP-SGD is 2.29× (network) to 3.78× (application) slower per round than the HE-based modes. `pure_zkp` is consistently faster than `pure_he` on both models, despite doing strictly more cryptographic work (proof generation + verification + MAD thresholding on top of encryption).

### 2.6 — Stability

- Zero NaN/Inf-quarantined rounds across all six runs (`nan_this_round` sums to 0 everywhere).
- Last-5-round (rounds 21–25) F1-Macro standard deviation across the six runs: 0.0174, 0.0353, 0.0369, 0.0620, 0.0248, 0.0138 — every run shows non-trivial late-round volatility.
- **4 of 6 runs** peak before round 25 (`network_pure_dp`@23, `network_pure_he`@21, `application_pure_dp`@24, `application_pure_zkp`@24) — only `network_pure_zkp` and `application_pure_he` peak at the final round. **Best-round reporting, not final-round reporting, should be used in any figure or table for this reason.**
- Two single-round transient collapses independently verified: `network_pure_he` round 18 (F1 0.446, acc 0.527 → recovers to 0.765 at round 19); `application_pure_dp` round 21 (F1 0.428, acc 0.564 → recovers to 0.536 at round 22).

---

## 3. Mechanism — the theory behind each result, traced through the code

### 3.1 `pure_dp` — how the number is actually produced

1. `main.py`'s `_train_one_client` builds a fresh Opacus `PrivacyEngine(accountant="rdp")` per honest client.
2. Round 1 only: `privacy_engine.make_private_with_epsilon(target_epsilon=15.0, target_delta=1e-5, epochs=5, max_grad_norm=1.5)` — Opacus numerically solves for the noise multiplier σ that hits the target (ε,δ) budget.
3. The solved σ is cached (`_noise_multiplier_cache`, keyed on client/epsilon/delta/epochs/batch_size/clip_norm/data-size) and **reused via `make_private(noise_multiplier=σ)` for all 24 remaining rounds** — this is why `dp_noise_multiplier` is exactly `0.484` for every single round in the CSV (verified directly), and `dp_epsilon_spent` is exactly `14.9984` every round (0.011% off target — well inside the documented calibration bound).
4. FedProx's proximal pull is applied via a **separate, non-privatized** step (`_apply_dp_safe_prox_step`) after `optimizer.step()`, because Opacus's `DPOptimizer.step()` only reads `.grad_sample` and a loss-added proximal term would silently be discarded under DP.

**Why `pure_dp` scores lower than `pure_he`/`pure_zkp` — two compounding causes, only one of which the mechanism story usually credits:**

- **Cause 1 (the documented one): noise injection.** DP-SGD's calibrated Gaussian noise genuinely costs utility, especially on rare/hard classes — this is visible directly in the per-class table (XSS 0.26 vs 0.47–0.52 elsewhere).
- **Cause 2 (a real confound, not previously isolated in this dataset): architecture swap.** `main.py` sets `DP_SAFE = USE_DP` globally (line 356), threaded into every model instantiation. Checking the configs directly: `pure_dp` runs with `dp_safe=true` → `nn.GroupNorm` + `opacus.layers.DPLSTM`; `pure_he`/`pure_zkp` run with `dp_safe=false` → standard `nn.BatchNorm1d` + `nn.LSTM`. GroupNorm and DPLSTM are both known to underperform their non-DP counterparts even independent of any noise being injected. **The current three-run design cannot separate "cost of DP noise" from "cost of the DP-safe architecture" — they're bundled into one number.** See Caveats (§4) for what to do about this.

**Why `pure_dp` is the slowest run by a wide margin:** two separate, genuine Opacus costs, both tied to the same `dp_safe=True` branch — (1) per-sample gradient hooks (Opacus retains a gradient *per training example*, not per batch, to enable per-sample clipping — fundamentally more expensive than a standard batched backward pass), and (2) DPLSTM's non-fused, Python-level recurrence replacing cuDNN's fused LSTM kernel, which is a substantial and separate cost from the noise-injection mechanism itself.

### 3.2 `pure_he` — how the number is actually produced

1. `USE_HE=True`, `USE_HE_KRUM_HYBRID=False`, `USE_ZKP=False` → routes every accepted client through `he_local.encrypt_params()` (confirmed via the branch condition — this evaluates to the non-guard path here).
2. `split_sensitive_bulk()` partitions `state_dict()` layers by key prefix: anything starting with `"classifier"` is CKKS-encrypted; everything else (CNN+LSTM backbone) stays plaintext. Logs independently confirm **5.8% of parameters encrypted** for both models — architecture-determined (classifier head is a fixed `Linear(64,64)→Linear(64,8)` regardless of input feature count).
3. `HE_POLY_DEGREE=8192` (confirmed in configs: `"he_poly_degree": 8192`), passed to `he_local.create_ckks_context()`, which uses `HE_COEFF_MOD_BIT_SIZES=[60,40,40,60]` — **this is TenSEAL's standard 128-bit security parameter set for n=8192**, not the RAM-constrained Docker path's 64-bit config (`poly_modulus_degree=4096`). **These two configurations are easy to conflate — the ablation runs analyzed here use 128-bit CKKS security, and this should be stated explicitly and correctly if cited in a write-up** (see Caveats).
4. All 10 clients' encrypted heads are homomorphically summed via `he_weighted_sum()`; bulk layers are plaintext-averaged; `decrypt_params()` merges both halves back into the model's original layer order.

**Why `pure_he` outperforms `pure_dp`:** HE adds *zero* noise to training — only what's transmitted changes, not what's locally computed. CKKS's approximate-arithmetic error is negligible relative to DP's calibrated noise floor. `pure_he` also runs on the undamaged architecture (§3.1's Cause 2) — so this comparison is stacked in `pure_he`'s favor twice, not once, and the gap should not be read as "the cost of encryption is negative/free" so much as "DP-SGD, as currently configured, costs noticeably more than encryption."

### 3.3 `pure_zkp` — how the number is actually produced, traced in full

**Attack mechanism** (`classifier_head_flip_attack`, `byzantine.py`, called at `main.py` ~line 571): the two Byzantine clients (indices 0,1) **train normally on the full model first**, so their backbone contribution looks legitimate to any inspection of the plaintext bulk slice. Only `classifier.*` layers are then overwritten with `-scale × trained_param`, `scale=5.0` (confirmed: `"attack_scale": 5.0`, `"byzantine_head_only": true` in the config). The submitted head is not a subtle perturbation — it's the honestly-trained head negated and scaled 5×.

**Guard mechanism** (`he_local.encrypt_params_with_norm_guard()` → `zkp.generate_head_norm_proof()`):
1. Computes `delta = trained_classifier_head − global_classifier_head_at_round_start` (the round's *delta*, deliberately, not the head's absolute magnitude — this keeps the proof meaningful round-independently, since a large-but-stable head would otherwise always look anomalous).
2. HMAC-SHA256-signs `norm(delta)` bound to a SHA-256 hash of the actual ciphertext bytes submitted — specifically prevents proof/ciphertext substitution after signing.
3. Server-side: `zkp.verify_head_norm_proof()` checks the ciphertext-hash binding, then `zkp.mad_threshold_head_norms()` — **the exact same `median + k·1.4826·MAD` robust-statistic formula as `adaptive_multi_krum()`**, applied to one scalar (the delta norm) instead of a full parameter vector, since that's the only thing an encrypted slice reveals without decryption.

**Why detection is a flat 100.00%, not probabilistic — worked through the math:** if `trained_head ≈ global_head + small_update`, then `poisoned_head = -5·trained_head ≈ -5·global_head − 5·update`, so `delta ≈ -6·global_head − 5·update` — the attacker's delta magnitude is dominated by **six times the head's own absolute weight magnitude**, not merely a poisoned version of the round's actual update. This structurally guarantees the attacker sits an order of magnitude from the honest cluster, by construction of the attack — not a coincidence of this particular run.

**Empirical confirmation, pulled directly from `main_zkp_network.log`'s per-round MAD diagnostics:**

```
Round 1:  center=1.796  spread=0.156  threshold=2.186  → kept 8/10, rejected [0,1]
Round 5:  center=0.398  spread=0.070  threshold=0.572  → kept 8/10, rejected [0,1]
Round 25: center≈0.280  spread≈0.09   threshold≈0.50   → kept 8/10, rejected [0,1]
```

The honest-client median delta-norm shrinks steadily over rounds (1.796 → 0.398 → ~0.280), exactly matching the expected FedProx convergence pattern (per-round local updates shrink as clients converge, even with LR decay disabled) — a real, theory-consistent signal underneath the security result, not noise. Every round, clients 0 and 1 land with wide margin on the wrong side of the threshold, which is *why* detection never shows round-to-round flakiness — the threshold never sits close to a real decision boundary for this specific attack.

**Why the guard-protected utility number ends up beating the clean baseline — not a paradox:** the guard excises clients 0 and 1 **before aggregation** (`survivor_enc = [accepted_params[pos] for pos in survivor_positions]`), so there is no residue of the attack anywhere in the resulting aggregate — the remaining 8 honest clients' homomorphically-summed contribution is computationally indistinguishable from an unattacked 8-client run. Combined with HE's architectural advantage (§3.2) and ordinary single-seed Dirichlet-partition sampling variance, `pure_zkp` besting `pure_he`'s clean run is fully explained by mechanism.

**Why `pure_zkp` is faster than `pure_he` despite more cryptographic work:** `aggregate_encrypted()`'s cost scales with `n_chunks × n_clients` ciphertext operations — the single most expensive step in the pipeline. `pure_he` aggregates 10 clients; `pure_zkp` aggregates only the 8 guard-survivors — 20% fewer ciphertext operations at the most expensive step. Proof generation/verification/MAD-thresholding is cheap numpy/HMAC arithmetic on flattened arrays by comparison — negligible next to CKKS ciphertext arithmetic.

### 3.4 What the guard mathematically cannot answer (independent of this run's result)

`zkp.mad_threshold_head_norms()` only ever inspects **magnitude** (the L2 norm of the delta), never direction. A hypothetical attacker who keeps their delta norm inside the honest cluster's range but corrupts the *direction* of the classifier head (e.g. targeted misclassification between two specific classes, achieved with a bounded-magnitude perturbation) would not be flagged by this mechanism at all — this is a structural blind spot of a magnitude-only guard, not a bug, and this ablation's 100%-detection result says nothing about that case, because `classifier_head_flip_attack` was never designed to probe it. This is worth stating explicitly in any paper claiming this mechanism as a general solution.

---

## 4. Caveats to state explicitly in the paper

1. **Not a controlled three-way comparison.** `pure_dp` and `pure_he` ran attack-free; `pure_zkp` ran with an active attacker. The utility numbers reported here measure the cost of each *mechanism* individually, not a fair three-way comparison under identical conditions. `pure_zkp`'s strong utility result is partly "no attack cost survived the guard" and partly whatever intrinsic advantage HE-without-DP-noise already has.

2. **The `pure_dp` architecture confound (new, not previously documented).** `dp_safe = USE_DP` means `pure_dp` uses GroupNorm+DPLSTM while `pure_he`/`pure_zkp` use BatchNorm+LSTM. The reported utility gap between `pure_dp` and the other two modes conflates "cost of DP noise" with "cost of the DP-compatible architecture." **State this explicitly rather than attributing the entire gap to noise injection.** A clean isolation requires a fourth run: `dp_safe=True, USE_DP=False` (architecture swap, zero noise) to measure the architecture's standalone cost.

3. **CKKS security level — state the correct one for these specific runs.** These `pure_he`/`pure_zkp` runs use `poly_modulus_degree=8192` with `[60,40,40,60]` coefficient moduli — **TenSEAL's standard 128-bit security configuration** — not the RAM-constrained Docker path's `poly_modulus_degree=4096` (64-bit) configuration used elsewhere in the project. If the paper's cryptography section states a security level for the HE layer, make sure it matches whichever configuration actually produced the cited numbers; conflating the two is an easy mistake since both exist in the codebase.

4. **Single-seed, single-run results throughout.** None of the six runs were repeated. The documented round-to-round volatility (last-5-round F1 std 0.014–0.062) is exactly the kind of noise that argues against treating any single number (e.g. `network_pure_zkp`'s 0.8214 "beating" other runs) as a robust finding without a repeat run.

5. **No global random seed is set for model initialization or DP noise.** Confirmed via code trace: `main.py` never calls `torch.manual_seed()`. Only `data_loader.py` seeds two numpy RNGs (`np.random.default_rng(42)` for majority-class capping, and the Dirichlet partition RNG, also seed=42) — **the data split is reproducible, but model weight initialization and Opacus's DP noise are not.** Any reproduction attempt will get the same client data partitions but a different exact trajectory, especially for `pure_dp`. State this precision limit rather than implying exact reproducibility.

6. **Guard tests only one attack shape.** The `pure_zkp` result (100% detection) demonstrates the guard defeats a magnitude-heavy, sign-flip-and-scale attack on the classifier head. It does not demonstrate robustness against a bounded-magnitude, direction-only attack — the guard's own math (magnitude-only MAD threshold) cannot detect that case by construction (§3.4). Do not generalize "the guard solves the HE+Krum blind spot" beyond this specific attack family without a follow-up experiment using a magnitude-bounded attack.

7. **`application_pure_he`'s rounds 1–2 identical-prediction artifact (§2.2).** Not a bug, but worth a one-line footnote if this run's early trajectory appears in a figure, so a reviewer doesn't mistake it for a logging error.

8. **Best-round vs. final-round reporting.** 4 of 6 runs peak before round 25. Any comparison table in the paper should use best-round numbers (as done throughout this analysis) and say so explicitly, rather than defaulting to final-round numbers, which would understate results for 4 of the 6 conditions.

---

## 5. Open blockers / unresolved items directly relevant to this ablation set

- **No architecture-isolated DP run exists** (`dp_safe=True, USE_DP=False`) — needed to separate noise cost from architecture cost per Caveat #2. Not yet run.
- **No repeat-seed runs exist** for any of the six conditions — single-seed results throughout (Caveat #4). A minimum of 3 seeds per condition would be needed before citing exact numbers (e.g. "0.8214") as more than a point estimate.
- **No bounded-magnitude/directional attack variant has been implemented or tested** against the head-norm guard — the guard's one documented blind spot (Caveat #6 / §3.4) remains untested, not just theoretically flagged.
- **`ABLATION_MODE` is not CLI-controllable.** As uploaded, `main.py` line 301 hardcodes `ABLATION_MODE = "krum_dp_sweep"` — none of the six runs analyzed here could have been produced by running the uploaded `main.py` as-is without first hand-editing that line (or maintaining separate copies, as the analysis doc's own methods note states: "three copies: `main_dp.py`, `main_he.py`, `main_zkp.py`"). This is a reproducibility blocker in its own right — see §6 below for the exact workaround.
- **No manifest/config-tracking system links these six runs to their exact code version.** If `main.py` changes again, there is currently no automated way to confirm which commit produced which CSV — only the JSON configs (which capture flag values, not code hashes).

---

## 6. Reproduction logic — exact steps for someone re-running this ablation set

This section describes the precise mechanical steps required to reproduce these six runs from the uploaded code, including the parts that are **not** obvious from `main.py`'s CLI surface alone.

### Step 1 — Environment
- NGC PyTorch container matching the project's documented GPU setup (`nvcr.io/nvidia/pytorch:25.10-py3` per the project's other documentation), or any environment with CUDA available (`_CUDA_AVAILABLE = torch.cuda.is_available()` controls parallelization mode — GPU uses sequential in-process client training, CPU uses a 4-way `ProcessPoolExecutor`; results are not guaranteed numerically identical across the two paths, so match the original hardware if exact reproduction matters).
- Install: `torch`, `opacus` (for `pure_dp`), `tenseal` (for `pure_he`/`pure_zkp`), `pandas`, `scikit-learn`, `numpy`.
- Place the Edge-IIoTset dataset CSV at the path `data_loader.py` expects, and let the preprocessing cache build (or reuse an existing `.npz` cache) — the Dirichlet partition seed (`np.random.default_rng(42)`) is fixed, so the **client data splits will be identical** across reproduction attempts, which is the one part of this pipeline that genuinely is deterministic.

### Step 2 — Select the ablation mode (manual file edit required)
`ABLATION_MODE` is **not** a CLI argument — it's a hardcoded string at line 301 of `main.py`, and as uploaded its value is `"krum_dp_sweep"`, not any of the three modes analyzed in this document. To reproduce:

```python
# In main.py, line 301:
ABLATION_MODE = "pure_dp"    # or "pure_he" or "pure_zkp"
```

Each of the three values sets a distinct, hardcoded flag bundle (verified directly in the code, lines 306–324):

```
pure_dp:  USE_DP=True;  USE_HE=USE_ZKP=USE_KRUM=USE_ADAPTIVE_KRUM=USE_HE_KRUM_HYBRID=False
          USE_BYZANTINE_ATTACK=False; BYZANTINE_HEAD_ONLY=False

pure_he:  USE_HE=True;  USE_DP=USE_ZKP=USE_KRUM=USE_ADAPTIVE_KRUM=USE_HE_KRUM_HYBRID=False
          USE_BYZANTINE_ATTACK=False; BYZANTINE_HEAD_ONLY=False

pure_zkp: USE_ZKP=True; USE_DP=USE_HE=USE_KRUM=USE_ADAPTIVE_KRUM=USE_HE_KRUM_HYBRID=False
          USE_BYZANTINE_ATTACK=True; BYZANTINE_HEAD_ONLY=True
```

No other flags need to be hand-edited — everything else derives from these three (`DP_SAFE = USE_DP`, `HE_POLY_DEGREE = 8192`, `USE_HEAD_NORM_GUARD = True`, etc. are all set automatically once `ABLATION_MODE` is picked).

### Step 3 — Run the CLI command

```bash
python main.py network --tag pure_dp
python main.py application --tag pure_dp

python main.py network --tag pure_he
python main.py application --tag pure_he

python main.py network --tag pure_zkp
python main.py application --tag pure_zkp
```

(with `ABLATION_MODE` hand-edited to the matching value before each pair of runs). The positional `model_type` argument (`network` or `application`) is required; `--tag` controls the output filename suffix (`results_{model_type}_{tag}.csv`) — confirmed this is exactly how the six source CSVs are named. No `--epsilon`, `--byzantine`, `--krum-k`, `--attack-type`, or `--gaussian-std` overrides are needed for these three modes — all relevant values (epsilon=15.0 default, byzantine=[0,1] default, attack_scale=5.0 default, k=2.5 default) are already the defaults baked into the code and confirmed present in all three config JSONs.

### Step 4 — What will NOT reproduce exactly, and why

Per Caveat #5 above, expect **the same client data partitions** (seeded) but **different exact per-round numbers** on any re-run, because:
- Model weight initialization is unseeded (no `torch.manual_seed()` anywhere in `main.py`).
- Opacus's DP-SGD Gaussian noise draws are unseeded — this affects `pure_dp` specifically, every round.
- TenSEAL/CKKS itself does not introduce meaningful result-level randomness (the approximate arithmetic error is deterministic given the same plaintext/context), so `pure_he`/`pure_zkp` should be closer to reproducible than `pure_dp`, modulo model-init randomness common to all three.

**To get exact reproduction**, add explicit seeding before training starts: `torch.manual_seed(SEED)`, `torch.cuda.manual_seed_all(SEED)` (if using GPU), and pin Opacus's internal RNG if the installed Opacus version exposes a seedable generator (varies by version — check `PrivacyEngine`'s constructor in the installed version). This is not currently done anywhere in `main.py` and would need to be added as a new, explicit step for anyone claiming bit-exact reproducibility in a paper.

### Step 5 — Verify the run against this document

After a run completes, an exact-match check should include:
- `dp_noise_multiplier` and `dp_epsilon_spent` should be constant across all 25 rounds for `pure_dp` (confirms the noise-multiplier caching path activated correctly, per §3.1).
- `zkp_rejected` should equal 1 for exactly clients 0 and 1, every round, for `pure_zkp` (confirms the guard is running and the attack is correctly targeted).
- Log output should show `[HE] 5.8% of params encrypted (classifier head)` once near the start of any `pure_he`/`pure_zkp` run — if this percentage differs, the classifier head's parameter count changed (e.g. a model architecture edit), which would invalidate a direct comparison against the numbers in this document.

---

*End of analysis.*
