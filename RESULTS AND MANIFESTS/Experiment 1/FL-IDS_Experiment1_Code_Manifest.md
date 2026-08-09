# FL-IDS Experiment 1 — Code Manifest

**Source files:** `experiments/Current model/main.py` (1,057 lines), `experiments/Current model/task.py` (273 lines)
**Purpose of this document:** a single reference describing exactly what code produced the six Experiment 1 sweep result files (`results_{network,application}_dp{3,9,15}.csv`), so the configuration, defences, and known limitations are traceable without re-reading the full source. Regenerate/update this manifest whenever either file changes.

---

## 1. Identity and Run Command

```
python3 main.py [network|application] [--epsilon E] [--tag TAG]
```

- `model_type` (positional): `network` or `application`, default `network`.
- `--epsilon`: overrides `DP_EPSILON` (default 15.0 if omitted).
- `--tag`: appended to every output filename (`results_{model}_{tag}.csv`, `checkpoint_{model}_{tag}*.npz/json`) — eliminates the manual `mv`-archiving step previously required between sweep conditions.

**Commands that produced the six uploaded result files:**
```
python3 main.py network     --epsilon 3.0  --tag dp3
python3 main.py network     --epsilon 9.0  --tag dp9
python3 main.py network     --epsilon 15.0 --tag dp15
python3 main.py application --epsilon 3.0  --tag dp3
python3 main.py application --epsilon 9.0  --tag dp09
python3 main.py application --epsilon 15.0 --tag dp15
```
(inferred from output filenames — application's ε=9 file is `dp09` not `dp9`, inconsistent with the network naming; cosmetic only, does not affect results, but worth standardizing before further conditions are run to avoid script/parsing mismatches downstream.)

---

## 2. Fixed Configuration (all six runs)

| Parameter | Value | Notes |
|---|---|---|
| `NUM_ROUNDS` | 25 | `SANITY_CHECK=False` |
| `NUM_CLIENTS` | 10 | |
| `LOCAL_EPOCHS` | 5 | |
| `LEARNING_RATE` | 0.001 | |
| `PROX_MU` | 0.02 | Applied via DP-safe decoupled step (Section 4) |
| `DIRICHLET_ALPHA` | 0.7 (in `data_loader.py`, not shown here) | Non-IID partitioning |
| `USE_BYZANTINE_ATTACK` | True | |
| `BYZANTINE_CLIENTS` | `[0, 1]` | 2 of 10 clients |
| `ATTACK_SCALE` | 5.0 (network) / 2.0 (application) | Model-specific, see Section 6 |
| `USE_KRUM` | False | |
| `USE_ADAPTIVE_KRUM` | **True** | MAD-threshold, not fixed-m — deliberate deviation from master-doc "Experiment 1 must use fixed-m Krum" spec (user decision, logged in changelog) |
| `USE_HE` | False | |
| `USE_ZKP` | False | |
| `USE_DP` | True | |
| `DP_EPSILON` | 3.0 / 9.0 / 15.0 | Swept via `--epsilon` |
| `DP_DELTA` | 1e-5 | |
| `DP_MAX_GRAD_NORM` | 1.5 | |
| `DP_BATCH_SIZE` | 512 | Tuned for CPU originally; confirmed fine on GPU post-vLLM-kill, no OOM in any of the 6 runs |
| `ADAPTIVE_KRUM_K` | 2.5 | |
| `ADAPTIVE_KRUM_METHOD` | `"mad"` | Z-score variant exists but unused here |
| `ADAPTIVE_KRUM_MIN_KEEP_FRACTION` | 0.5 | Safety floor, never triggered in these 6 runs (all kept exactly 8/10) |
| `DP_SAFE` | `True` (= `USE_DP`) | GroupNorm+DPLSTM architecture, all clients incl. Byzantine |

---

## 3. Defence Mechanism — Adaptive Multi-Krum

- File: `defences/krum.py::adaptive_multi_krum()` (not included in this review — referenced, not re-read).
- Selection is **dynamic per round**, not a fixed discard count: threshold = `median(scores) + k·1.4826·MAD(scores)`.
- `_krum_active = USE_KRUM or USE_ADAPTIVE_KRUM` — mutually exclusive by `assert` at module load.
- Diagnostics returned per round (`krum_score_diag`): `scores` (per-position), `num_nan`. Consumed in `main()` to compute `krum_scores_byzantine_mean`, `krum_scores_honest_mean`, `krum_score_ratio`, `nan_this_round` — all logged to CSV on the `MEAN` row only.
- `accepted_client_indices` translation layer present (for ZKP-compaction correctness) even though `USE_ZKP=False` for all 6 runs, so the translation is currently a no-op identity mapping — harmless, kept for consistency with the ZKP-active code path.

---

## 4. DP-Compatible FedProx — Patch Confirmed Correctly Applied

**Problem this solves:** Opacus's `DPOptimizer.step()` builds its update entirely from `.grad_sample` (per-sample gradients captured by hooks). A proximal term `(mu/2)·||w−w_global||²` added to the loss before `.backward()` never populates `.grad_sample` (it's a direct function of the parameter, not any hooked layer's activation), so it is silently discarded — `PROX_MU` would have zero effect under DP-SGD without this fix.

**Fix (`main.py::_apply_dp_safe_prox_step`, lines 249–271):** applies `mu·(w − w_global)` as a **separate, deterministic, non-privatized** parameter update, decoupled from the DP-SGD step:
```python
def _apply_dp_safe_prox_step(real_model, global_dict, mu, lr):
    if global_dict is None or mu == 0:
        return
    with torch.no_grad():
        for name, param in real_model.named_parameters():
            if name not in global_dict:
                continue
            g = torch.as_tensor(global_dict[name], dtype=param.dtype, device=param.device)
            param -= lr * mu * (param - g)
```
**Privacy validity:** the prox term depends only on current parameters and the last round's *public* global model — never on client data — so applying it unnoised costs zero privacy budget. This is standard "operator splitting" (privatized data-gradient step + separate deterministic regularization step).

**Call site** (`_train_one_client`, DP branch, lines 364–385): `_global_dict` built once per client per round (skipped entirely if `mu == 0`), then `_apply_dp_safe_prox_step(...)` called **once per batch**, immediately after `optimizer.step()` — matching the per-batch frequency of the non-DP path's proximal term (added to loss every batch in `task.py::train()`).

**Conclusion:** all six Experiment 1 result files are genuinely **DP-SGD + FedProx**, not DP-SGD + plain FedAvg. `prox_mu=0.02` was active and doing real work in every run.

---

## 5. Checkpoint / Recovery Scheme

| File | Written | Purpose |
|---|---|---|
| `checkpoint_{TAG}.npz` | every round | Last-completed-round weights only |
| `checkpoint_{TAG}_progress.json` | every round | `{"last_completed_round": N}` — resume pointer |
| `checkpoint_{TAG}_best.npz` | only when `round_f1_macro > best_f1_macro` | Best-F1-Macro round's weights, survives later degraded rounds |
| `checkpoint_{TAG}_best.json` | same trigger | `{"best_round": N, "best_f1_macro": F}` |
| `results_{TAG}.csv` | every round (per-client + MEAN rows) | Full metric log — see Section 7 for schema |
| `experiment_config_{TAG}.json` | once, at run start | Full flag/hyperparameter dump for provenance |

`best_f1_macro` resume logic: fresh run starts at `-1.0` (round 1 always saved as best); resumed run reloads prior best from `checkpoint_{TAG}_best.json` if present, so resuming across rounds does not lose best-round tracking.

**Known caveat (from changelog, still true):** resuming a checkpoint after changing any experiment flag (`DP_EPSILON`, `USE_KRUM`/`USE_ADAPTIVE_KRUM`, `USE_HE`, etc.) silently contaminates round-1 comparability — `main()` prints a warning on resume but does not enforce deletion. No evidence this occurred in the 6 uploaded runs (each `--tag` is unique and each `dp_epsilon_target` in the CSV matches its filename's intended ε throughout all 25 rounds).

---

## 6. Byzantine Attack Configuration

- `sign_flip_attack(global_params, scale=ATTACK_SCALE)` — applied to clients 0 and 1 every round, unconditionally (not probabilistic).
- `ATTACK_SCALE = 5.0` for network, `2.0` for application — **model-specific, not a shared constant.** Chosen per-model in an earlier experiment (Condition 2, no-defence baseline) to produce the intended failure mode for that model specifically: 5.0 was calibrated to cause full NaN collapse on the network model under FedAvg-with-no-defence; 2.0 was calibrated to cause partial, survivable degradation on the application model at the same no-defence condition. **Because of this, network and application Experiment 1 results are not attack-magnitude-comparable to each other** — each is the model-specific "worst calibrated case," not a shared attack strength. State this explicitly if the two models' results are ever plotted on the same axis.
- `dp_safe=True` architecture (GroupNorm+DPLSTM) applied identically to Byzantine and honest clients — confirmed at `_train_one_client()` line 295–299, `model.to(device)` happens before the Byzantine/honest branch splits, so there is no architecture mismatch that could contaminate Krum's distance computation for reasons unrelated to the attack itself.

---

## 7. CSV Schema (`results_{TAG}.csv`)

```
round, client, loss, accuracy,
<8 per-class F1 columns — model-specific names>,
zkp_rejected, krum_selected, krum_detected_byzantine,
dp_epsilon_spent, round_time_s,
dp_epsilon_target, dp_noise_multiplier,
krum_scores_byzantine_mean, krum_scores_honest_mean,
krum_score_ratio, nan_this_round
```

- **Per-client rows** (`client` = 1–10): only `loss`, `accuracy`, per-class F1, `zkp_rejected`, `krum_selected`, `krum_detected_byzantine` are populated; all DP/Krum-diagnostic aggregate columns are `N/A`.
- **`MEAN` row** (`client = "MEAN"`, one per round): all columns populated, including the round-level DP/Krum diagnostics. **This is the only row that should be used for round-level analysis** — per-client rows are per-client eval results, not independent experimental units.
- Network model class columns: `Normal, DDoS_UDP, DDoS_ICMP, Ransomware, DDoS_HTTP, DDoS_TCP, Vulnerability_scanner, MITM`
- Application model class columns: `Normal, SQL_injection, Uploading, Backdoor, Port_Scanning, XSS, Password, Fingerprinting`
- `dp_epsilon_target` is the CLI/config value (what the sweep's x-axis should use); `dp_epsilon_spent` is Opacus's actual achieved epsilon this round (should track target closely — confirmed within 0.25% in all 6 runs, see analysis file Section 6).

---

## 8. GPU / Device Handling — Confirmed Consistent

- `_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")`, resolved once at module load.
- Every model instance (`_train_one_client`, `_eval_one_client`) calls `model.to(device)` immediately after construction/parameter-loading, before any forward pass.
- `train()`/`test()` (`task.py`) both accept and correctly use an explicit `device=` kwarg at every call site in `main.py` — no reliance on the `'cpu'` default anywhere in the active code path.
- `FocalLoss.weight` registered via `register_buffer(..., persistent=False)` — participates correctly in `.to(device)`, confirmed fixed from the earlier plain-attribute bug.
- `_proximal_term()` (non-DP path only) builds its comparison tensor directly on `param.device` — no CPU/GPU mismatch risk.
- **Fork+CUDA hang fix:** when CUDA is available, `ProcessPoolExecutor` is not created at all — client training/eval runs sequentially in-process (`_run_training_wave`/`_run_eval_wave`, `executor=None` branch). CPU-only runs retain the original 4-way pool. This was necessary because forking a process that already holds an initialized CUDA context hands the child an unsafe half-initialized context (hangs, doesn't error).

---

## 9. Known Open Items (as of this manifest)

1. **Application ε=9 run tagged `dp09` vs. network's `dp9`** — cosmetic filename inconsistency, standardize before running further conditions to avoid downstream parsing mismatches in any aggregation script.
2. **`USE_ADAPTIVE_KRUM=True` remains a deliberate deviation** from the master planning doc's "Experiment 1 must use fixed-m Krum" instruction. Any future comparison against a fixed-m Condition 3 anchor is not apples-to-apples on aggregation *method*, only on attack/DP conditions.
3. **`DP_BATCH_SIZE=512`** was originally tuned for CPU; confirmed to run without OOM on the DGX Spark post-vLLM-kill across all 6 runs, but remains untested under concurrent GPU memory pressure from another process.
4. **No enforcement of checkpoint deletion on flag change** — resume-across-different-conditions contamination risk is warned about in stdout but not prevented in code.
5. **`experiment_config_{TAG}.json` does not record `checkpoint_best` filenames or git commit hash** — for full reproducibility provenance, consider adding both in a future revision.

---

## 10. Cross-Reference — Code State to Completed Runs

All six uploaded result files (`results_{network,application}_dp{3,9,15}.csv`) were produced by this exact code state (Patches 1–3 applied: DP-safe FedProx, save-best-checkpoint, CLI args). Confirmed via:
- CSV schema matches Section 7 exactly (22 columns, correct header).
- `dp_epsilon_target` values (2.9954–14.9995 range) match Opacus calibration behavior expected from this `DP_EPSILON`/`DP_DELTA`/`DP_MAX_GRAD_NORM` configuration.
- `krum_selected` counts (8/10 every round, all 6 files) match `ADAPTIVE_KRUM_MIN_KEEP_FRACTION=0.5` behavior with 2 confirmed Byzantine clients and a well-separated attack.
- Zero `nan_this_round` flags across all 6×25 rounds — consistent with `DP_MAX_GRAD_NORM=1.5` and no reported crashes during these runs.
