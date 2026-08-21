# FL-IDS Complete Master Context Prompt — All Corrections Applied (v5)

**Changelog v4 → v5 (this revision):** Added the next two planned experiments as fully-specified next steps, now that Experiment 1 is complete: **Experiment 2 — HE vs. Krum** (does encrypting the classifier head create a Byzantine blind spot Krum can't see into?) and **Experiment 3 — Privacy Configuration Checkpoint Manifest** (systematic, reloadable checkpoint+config+metrics record for every defence-combination run, replacing the current one-off/manual checkpoint discipline). Experiment 2 surfaces a real, currently-unresolved architectural blocker — `main.py`'s own `assert` currently makes `USE_HE` and `USE_KRUM`/`USE_ADAPTIVE_KRUM` mutually exclusive, so Experiment 2 cannot run until a hybrid aggregation branch is built; this is documented in detail below, including what already exists (`BYZANTINE_HEAD_ONLY`, `classifier_head_flip_attack`) versus what's still missing. **Also restored full Layer 2/Layer 3 detail in the Three-Layer Privacy Stack section**, which had been compressed to "unchanged from v3" placeholders in v4 — the HMAC mechanism/terminology-conflation warning, CKKS parameters (64-bit not 128-bit), RAM/timing figures, and the two known `he_aggregate()` bugs (now blocking for Experiment 2, not just noted) are all back in full.

**Changelog v1 → v2:** Added Experiment 1 — DP vs. Krum epsilon sweep as the immediate next actionable step. This supersedes/absorbs the standalone Condition 3 (Krum-without-DP) run — Condition 3 became the ε=∞ anchor point of the sweep.

**Changelog v2 → v3:** Added `adaptive_multi_krum()` (MAD-threshold-based selection) as **Condition 5**, plus a completed results analysis from its first full run. Surfaced a training-recipe drift hazard (`PROX_MU`, LR decay) between that run and the locked clean baselines, called out in Constraints.

**Changelog v3 → v4 (this revision):** Experiment 1 has been **run to completion** — all 6 of 8 planned conditions (ε ∈ {3, 9, 15} × {network, application}; the two ε=∞ anchor rows were not separately re-run) executed on GPU (NVIDIA DGX Spark, Grace Blackwell GB10) and fully analyzed. Major updates in this revision:
- **Deviation from spec, now permanent and documented:** Experiment 1 actually ran with `USE_ADAPTIVE_KRUM=True` (MAD-threshold), **not** fixed-m Krum as v1–v3 specified. This was a deliberate user decision, logged in the code's own changelog. The "do not substitute adaptive Krum into Experiment 1" constraint from v3 is retracted below and replaced with an honest note — a true fixed-m Experiment 1 sweep, if ever needed for an apples-to-apples Condition 3 comparison, remains an open follow-up, not something already done.
- **New bug found and fixed: PROX_MU was silently inert under DP-SGD.** Opacus's `DPOptimizer.step()` builds its update entirely from `.grad_sample`; a proximal term added to the loss before `.backward()` never populates `.grad_sample` and is silently discarded. Fixed via a separate, deterministic, unnoised parameter update (`_apply_dp_safe_prox_step`), applied after `optimizer.step()` every batch. Privacy-safe: the prox term depends only on current params + last round's public global model, never client data. **All Experiment 1 results are now confirmed genuine DP-SGD+FedProx, not DP-SGD+FedAvg.**
- **`save_best_checkpoint()` — now actually implemented and confirmed working** (code-reviewed, not just planned). Best-F1-Macro round's weights are recoverable separately from the round-by-round overwritten checkpoint.
- **CLI args added** (`--epsilon`, `--tag`) — sweep conditions no longer require manual `main.py` edits or manual `mv`-archiving between runs.
- **GPU migration completed:** training moved from CPU (Windows local + earlier estimates) to an NVIDIA DGX Spark (ARM64/aarch64, Grace Blackwell GB10, unified CPU/GPU memory) via an NGC PyTorch container. Full round times dropped from an estimated 2–6 hours/condition to **under 65 minutes/condition**, confirmed across all 6 completed runs.
- **Experiment 1's actual finding: the original hypothesis does NOT hold.** Krum's Byzantine-detection separation (`krum_score_ratio`) changed by under 2.3% across ε∈{3,9,15} on both models, and detection rate was 100% in every single round of every run — DP noise does not measurably erode Krum's robustness in this range, under this attack strength. This is treated as a real, citable negative result, not a failed experiment.
- **New, more novel finding surfaced instead:** DP noise strongly delays or suppresses rare-class discovery, independent of Krum — clearest on Fingerprinting (application model, rarest class, 1,001 raw samples), which stays at F1=0.000 for 23/25 rounds at ε=3 but reaches ~0.6 by round 25 at ε=9/15.
- Repository path corrected: the actual working directory observed in practice is `experiments/Current model/`, not `experiments/Current tests/` as earlier revisions assumed — updated throughout.

---

## Project Identity

**Student:** Zarawar Khan, BE Electrical Engineering, SEECS NUST, 4.0 CGPA
**Internship:** 12-week AI Security internship at CNIT/PNTLab, TECIP, Scuola Superiore Sant'Anna, Pisa, Italy
**Supervisor:** Rana Abu Bakar
**GitHub org:** AI-Security-Internships-2026
**Repo:** 09-edge-iot-security-monitoring
**Compute:** NVIDIA DGX Spark (Grace Blackwell GB10, ARM64, unified 128GB CPU/GPU memory), accessed via OpenVPN + SSH, training run inside an NGC PyTorch container (`nvcr.io/nvidia/pytorch:25.10-py3` — GB10's sm_121 compute capability requires NGC ≥25.10, older tags do not support it correctly).

**Supervisor's framing:** "End goal is just to make FL more secure and more private." Research contribution orientation — results evaluated against FL literature, not commercial viability.

---

## CRITICAL: Everything Before the Label Bug Fix Is Invalid

**This section must be read first.** A fundamental bug was discovered that invalidates all previously locked baselines and all Byzantine/Krum experiment results. Do not use any F1 numbers from before this fix.

### The LabelEncoder Alphabetical Sorting Bug

**Root cause:** `sklearn.LabelEncoder.fit()` sorts labels **alphabetically**, completely ignoring the intended `ALL_CLASSES` list order. The code had:

```python
_encoder = LabelEncoder()
_encoder.fit(ATTACK_CLASSES)  # ATTACK_CLASSES defines intended order
y = _encoder.transform(df['Attack_type'].values)
```

Because `fit()` sorts alphabetically, the actual encoding was:
```
Alphabetical order:           Intended order (ALL_CLASSES):
0: Backdoor                   0: Normal
1: DDoS_HTTP                  1: DDoS_UDP
2: DDoS_ICMP                  2: DDoS_ICMP
3: DDoS_TCP                   3: Ransomware
4: DDoS_UDP                   4: DDoS_HTTP
5: Fingerprinting              5: SQL_injection
6: MITM                       6: Uploading
7: Normal          ← y==7     7: DDoS_TCP       ← y==7 was SUPPOSED to mean this
8: Password                   8: Backdoor
9: Port_Scanning              9: Vulnerability_scanner
10: Ransomware                10: Port_Scanning
11: SQL_injection              11: XSS
12: Uploading                  12: Password
13: Vulnerability_scanner      13: MITM
14: XSS                       14: Fingerprinting
```

**Concrete damage this caused:**

| What code said | What code meant | What actually happened |
|---|---|---|
| `y == 7` (cap DDoS_TCP to 18%) | DDoS_TCP | Was capping **Normal** — the majority class was Normal all along, not DDoS_TCP |
| `APP_ORIG_IDX = [0, 5, 6, 8, 10, 11, 12, 14]` | Normal, SQL_inj, Upload, Backdoor, Port_Scan, XSS, Password, Fingerprint | Was actually Backdoor, Fingerprint, MITM, Password, Ransomware, SQL_inj, Uploading, XSS |
| `NETWORK_ORIG_IDX = [0, 1, 2, 3, 4, 7, 9, 13]` | Normal, DDoS_UDP, DDoS_ICMP, Ransomware, DDoS_HTTP, DDoS_TCP, Vuln_scanner, MITM | Was actually Backdoor, DDoS_HTTP, DDoS_ICMP, DDoS_TCP, DDoS_UDP, Normal, Port_Scanning, Vulnerability_scanner |

**What the raw class distribution actually was:**

| Class label | Previously believed count | Actual count |
|---|---|---|
| Normal | 24,862 | **1,615,643** (the real majority) |
| DDoS_TCP (raw) | 1,615,643 | **50,062** |
| MITM | 50,110 | 1,214 (was actually Uploading's count) |
| Fingerprinting | 15,915 | 1,001 (was actually SQL_injection's count) |

The entire claim "DDoS_TCP is 72.8%, cap it to 18%" was describing **Normal** the whole time. Real DDoS_TCP was only 50,062 rows.

**All previously reported results are therefore invalid:**
- Old network baseline: F1-Macro 0.839 — INVALID
- Old application baseline: F1-Macro 0.660 — INVALID
- Old Byzantine results (network attack no defence F1=0.012, network krum F1=0.857) — INVALID
- Old application Krum result (F1=0.570) — INVALID
- All ablation results (pure DP, pure HE, he_full/he_partial) — timing/RAM numbers are still valid; F1 numbers are invalid

### The Fix

```python
# REMOVED: LabelEncoder entirely — prevents reintroduction
# ADDED: Manual mapping that respects intended order

ALL_CLASSES = [
    'Normal', 'DDoS_UDP', 'DDoS_ICMP', 'Ransomware', 'DDoS_HTTP',
    'SQL_injection', 'Uploading', 'DDoS_TCP', 'Backdoor',
    'Vulnerability_scanner', 'Port_Scanning', 'XSS', 'Password',
    'MITM', 'Fingerprinting'
]

_class_to_idx = {name: i for i, name in enumerate(ALL_CLASSES)}

def encode_labels(series):
    return series.map(_class_to_idx).values
```

This guarantees Normal=0, DDoS_UDP=1, DDoS_ICMP=2, Ransomware=3, DDoS_HTTP=4, SQL_injection=5, Uploading=6, DDoS_TCP=7, Backdoor=8, Vulnerability_scanner=9, Port_Scanning=10, XSS=11, Password=12, MITM=13, Fingerprinting=14 — exactly the intended order.

**After the fix, the cache was rebuilt from scratch.**

---

## Second Major Discovery: Text Features Being Silently Destroyed

### What Was Happening

The preprocessing applied `pd.to_numeric(errors='coerce')` to all columns after dropping a few explicit ones. Every column that could not be parsed as a number became NaN, then `nan_to_num` converted it to 0.

**Columns that were silently zeroed (still counted in "52 features" but carried zero information):**
- `http.file_data` → contains actual HTTP payload content
- `http.request.uri.query` → contains GET query string
- `http.referer` → HTTP referer header
- `http.request.full_uri` → full request URI
- `dns.qry.name` → DNS query name
- `tcp.payload` → raw TCP payload bytes

### Why This Mattered

These are precisely the features that distinguish application-layer attacks:

| Class | Primary signal | Was it zeroed? |
|---|---|---|
| XSS | `http.request.uri.query` contains `<script`, `onerror=` | YES — zeroed |
| SQL_injection | `http.request.uri.query` contains `UNION`, `SELECT`, `OR 1=1` | YES — zeroed |
| Uploading | `tcp.payload` length, entropy | YES — zeroed (`tcp.payload` also explicitly dropped) |
| Password | `http.file_data`, `tcp.payload` | YES — zeroed/dropped |
| Fingerprinting | `http.request.full_uri`, `dns.qry.name` | YES — zeroed |

The application model was effectively running without its most informative features. The network model was unaffected because it relies on packet-level numeric features (port numbers, packet sizes, timing) not HTTP payload content.

### The Fix: Feature Engineering Before Numeric Coercion

A function `engineer_text_features(df)` runs **before** the `pd.to_numeric()` step, extracts signal from text columns, then drops the raw text columns.

**Engineered features added (application model only):** length/entropy/keyword-flag features from `http.request.uri.query`, `http.file_data`, `tcp.payload` (hex-decoded first), `http.referer`, `dns.qry.name`; plus HTTP method one-hot, User-Agent length/known-attack-tool flag, and coarse URI keyword buckets (xss/sqli/upload/login/admin) — added after `inspect_weak_class_payloads.py` found the real distinguishing signal for XSS/Password/Uploading in this dataset is tool fingerprints in the User-Agent header and request method, not injected exploit syntax.

**Net result on feature count:**
- Application model: **90 features confirmed measured** (per Experiment 1 run logs: "Application model: 90 features after text engineering") — supersedes the earlier "~80-91" estimate range.
- Network model: **39 features confirmed measured** (per Experiment 1 run logs: "Network cache loaded: 532,714 rows, 39 features") — supersedes the earlier "38" figure; the discrepancy is small (±1) and likely reflects a minor change in `VarianceThreshold`'s surviving column count after the METHOD_COL / overlooked-column fixes documented in `data_loader.py`. **Treat 39 as current ground truth for the network model**, confirmed by direct measurement, not the older "38" estimate.

---

## Additional Fixes Applied Alongside Label Fix

### task.py: Stale Hardcoded Count Tables

`task.py` had hardcoded class count tables for the inverse-sqrt weights. These used the old (wrong) class counts and the old (wrong) class indices. After the label fix, these were recalculated using the corrected counts from the rebuilt cache, computed **live** via `get_class_counts_network()`/`get_class_counts_application()` on every run — never hardcoded again.

### Weight Ratio Clipping and Gamma Reduction

After the label fix, the corrected class distribution revealed a much more severe true imbalance than previously understood. Final settings: weight ratio clipped (`MAX_WEIGHT_RATIO=5.0` for application model), FocalLoss gamma reduced to 2.0 for both models, small targeted per-class boosts (1.3–1.5×) for classes still underperforming (Uploading, XSS, Fingerprinting on the application model), applied after the ratio clamp so they're bounded on both ends.

### Parallelisation — Now Device-Dependent

**CPU runs:** `ProcessPoolExecutor` with up to 4 workers, `_train_one_client()` a top-level function (required for pickling), pool created once per run.

**GPU runs (current default, DGX Spark):** **no process pool at all.** Client training/eval runs sequentially in-process. This was a necessary fix, not a simplification — see "Fork+CUDA Hang Fix" below.

---

## Correct Feature Counts — Ground Truth (Updated, Directly Measured)

| Model | Features | How determined |
|---|---|---|
| Network model | **39** | Directly measured from Experiment 1 run logs (`VarianceThreshold(1e-6)` applied to raw numeric features) |
| Application model | **90** | Directly measured from Experiment 1 run logs (52 raw + engineered text features − raw text cols dropped) |

**Any reference to 40, 38, 35, or 52 features for either model is stale.** Any reference to "~80-91" for the application model is superseded by the confirmed 90.

---

## NEW Locked Baselines — These Replace All Previous Baselines

Both baselines were established after: label bug fix + text feature engineering (application model) + weight ratio clipping + gamma reduction, on CPU, before the GPU migration and before Experiment 1.

**LR decay and EMA were NOT used in the runs that produced these baselines. Do not attribute stability to those mechanisms.**

### Application Model Baseline — Round 20 (LOCKED)

| Metric | Value |
|---|---|
| Accuracy | **0.8504** |
| F1-Macro | **0.7293** |

Per-class F1:
```
Normal:          0.8776
SQL_injection:   0.8184
Uploading:       0.6110
Backdoor:        0.9399
Port_Scanning:   0.8268
XSS:             0.4518
Password:        0.5052
Fingerprinting:  0.8035
```

**Checkpoint limitation (historical, now resolved for future runs):** the original `checkpoint_application.npz` only held round 25 weights; round 20's actual parameters were unrecoverable. `save_best_checkpoint()` (confirmed implemented and code-reviewed as of v4) prevents this from recurring in any run going forward, including all of Experiment 1.

**Open anomaly — Client 6:** Password F1 stuck at 0.03-0.08 for Client 6 across all 25 rounds vs 0.35-0.60+ for all other clients. Same pattern on XSS. Hypothesis: Client 6 has very few or zero real Password/XSS samples under Dirichlet α=0.7 partitioning. **Still unresolved as of v4** — `per_client_audit.py` proposed but not yet run against the application-model partitions.

---

### Network Model Baseline — Round 22 (LOCKED)

| Round | Accuracy | F1-Macro |
|---|---|---|
| 21 | 0.9543 | 0.8278 |
| **22** | **0.9697** | **0.8289** |
| 23 | 0.9340 | 0.7970 |
| 24 | 0.9549 | 0.8217 |
| 25 | 0.8090 | 0.6800 |

Per-class F1 at round 22:
```
Normal:                 0.9364
DDoS_UDP:               0.9992
DDoS_ICMP:              0.9996
Ransomware:              0.7387
DDoS_HTTP:               0.7902
DDoS_TCP:                0.9955
Vulnerability_scanner:   0.7198
MITM:                    0.4514
```

**Round 25 crash — now a confirmed recurring structural pattern, not an anomaly.** Originally documented once here; since then this exact "late-round instability" pattern has recurred **three more times independently** — Condition 5's network run (round 23), Condition 5's application run (round 21), and Experiment 1's `network_dp15` run (round 23→25). Four independent occurrences now supports stating this as a structural property of this FedProx/non-IID training setup in the paper, not noise. Root cause remains non-IID client-sampling variance interacting badly with mid-sized classes.

**MITM sample scarcity:** MITM has only 1,214 samples in the corrected distribution. F1 flat around 0.41-0.48 for nearly the entire run — a hard data ceiling, not a modelling problem, confirmed again in every subsequent MITM-involving run including Experiment 1.

---

## Literature Comparison With New Baselines

| System | F1-Macro | Accuracy | Clients | Rounds | Notes |
|---|---|---|---|---|---|
| **This work — Network** | **0.8289** | **0.9697** | 10 | 25 | Correct labels + FedProx |
| **This work — Application** | **0.7293** | **0.8504** | 10 | 25 | Correct labels + text features |
| VARS-FL (2025) | 0.6422 | 0.8185 | 100 | 100 | Best published on same dataset |
| Rashid et al. | ~0.92 acc only | 0.9249 | N/A | N/A | Near-IID, majority-class bias |

Both models now substantially exceed VARS-FL with correct labels, fewer clients, and fewer rounds.

---

## Dataset — Corrected Understanding

**Name:** Edge-IIoTset DNN subset
**File:** `datasets/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv`
**Cache:** now split per model — `dnn_preprocessed_cache_network.npz`, `dnn_preprocessed_cache_application.npz` (loaded independently, `load_and_preprocess(model_type=...)` — fixed to prevent one model's worker process from wastefully loading the other model's cache into memory, see Bugs Fixed list).

**Correct class distribution (after label fix, before per-model class subsetting):**
```
Normal:                1,615,643  (the true majority class, previously misidentified)
DDoS_TCP:                 50,062
DDoS_ICMP:               116,436
Ransomware:               50,062  (same count as corrected DDoS_TCP — coincidence)
DDoS_HTTP:               121,568
SQL_injection:             1,001  (genuinely rare)
Uploading:                 1,214  (genuinely rare)
Backdoor:                 50,153
Vulnerability_scanner:    22,564
Port_Scanning:            10,925
XSS:                      51,203
Password:                 37,634
MITM:                     50,110
Fingerprinting:           15,915
```

**Cap methodology (confirmed via Experiment 1 run logs):** capping is applied to **Normal** at 18% of total post-cap (VARS-FL methodology target, corrected from the original mistaken "cap DDoS_TCP" framing under the LabelEncoder bug) — confirmed in run logs: "Capping Normal to 18% of total... Samples after capping: 736,046, Normal 132,488 (18.00%)". This resolves the v1–v3 open verification item about whether the cap target needed changing after the label fix — **it did, and the fix is in place and confirmed working.**

---

## Dual-Model Architecture — Corrected, Feature Counts Updated

### Model 1: Network-Layer Model

**Features:** **39** (directly measured — see "Correct Feature Counts" above)

**Classes (8), with CORRECTED indices:**

| Class | Corrected original index | Remapped index |
|---|---|---|
| Normal | 0 | 0 |
| DDoS_UDP | 1 | 1 |
| DDoS_ICMP | 2 | 2 |
| Ransomware | 3 | 3 |
| DDoS_HTTP | 4 | 4 |
| DDoS_TCP | 7 | 5 |
| Vulnerability_scanner | 9 | 6 |
| MITM | 13 | 7 |

### Model 2: Application-Layer Model

**Features:** **90** (directly measured — see "Correct Feature Counts" above)

**Classes (8), with CORRECTED indices:**

| Class | Corrected original index | Remapped index |
|---|---|---|
| Normal | 0 | 0 |
| SQL_injection | 5 | 1 |
| Uploading | 6 | 2 |
| Backdoor | 8 | 3 |
| Port_Scanning | 10 | 4 |
| XSS | 11 | 5 |
| Password | 12 | 6 |
| Fingerprinting | 14 | 7 |

---

## CNN-LSTM Model Architecture

```python
class CNN_LSTM(nn.Module):
    def __init__(self, num_features, num_classes=8, dp_safe=False):
        # CNN block 1
        Conv1d(1, 64, kernel_size=3, padding=1)
        GroupNorm(8, 64) if dp_safe else BatchNorm1d(64)
        ReLU → MaxPool1d(2)
        # CNN block 2
        Conv1d(64, 128, kernel_size=3, padding=1)
        GroupNorm(8, 128) if dp_safe else BatchNorm1d(128)
        ReLU → MaxPool1d(2)
        # LSTM
        DPLSTM(input=128, hidden=64) if dp_safe else LSTM(input=128, hidden=64)
        # Classifier head (target of partial HE and head-only Byzantine attack)
        Linear(64→64) → ReLU → Dropout(0.3) → Linear(64→num_classes)
```

**dp_safe=False:** BatchNorm1d + nn.LSTM. All non-DP experiments.
**dp_safe=True:** GroupNorm + DPLSTM. Required by Opacus per-sample gradient tracking. Used for every client (honest and Byzantine alike) in every DP-active run — confirmed code-reviewed, no architecture mismatch risk to Krum's distance computation.

**FocalLoss.weight — GPU device-placement bug found and fixed.** A plain `self.weight = weight` attribute assignment is invisible to `nn.Module`'s `.to(device)` machinery — only registered parameters/buffers move. Fixed via `register_buffer('weight', weight, persistent=False)` — participates correctly in `.to()`/`.cuda()` while staying out of `state_dict()` (not a trainable/checkpointed parameter).

**`train()`/`test()` now accept an explicit `device=` kwarg**, used at every call site in `main.py` (no reliance on the `'cpu'` default anywhere in the active code path) — confirmed via full code review.

**FedProx proximal term (non-DP path):** matched by parameter NAME, built directly on `param.device` (was previously always-CPU regardless of where the model lived — fixed).

**FedProx proximal term (DP path) — new fix, see "DP-Safe FedProx" section below.**

**Save-best-checkpoint — implemented and confirmed:**
```python
def save_best_checkpoint(global_params, round_num, f1_macro):
    np.savez(CHECKPOINT_BEST_PARAMS, *global_params)
    with open(CHECKPOINT_BEST_PROGRESS, "w") as f:
        json.dump({"best_round": round_num, "best_f1_macro": float(f1_macro)}, f)
```
Fires automatically in the round loop whenever `round_f1_macro > best_f1_macro`; resumes correctly from `checkpoint_{TAG}_best.json` if a run is resumed mid-way.

---

## DP-Safe FedProx — New Fix, Confirmed Correctly Applied (v4)

**The problem:** Opacus's `DPOptimizer.step()` builds its entire update from `.grad_sample` (per-sample gradients captured by hooks attached to specific layer types during forward/backward). A proximal term `(mu/2)·||w−w_global||²` added directly to the loss before `.backward()` is a direct function of the *parameter*, not of any hooked layer's activation — it never populates `.grad_sample`, so it is silently discarded at `step()` time. Under DP-SGD, `PROX_MU` would have been fully inert regardless of its configured value, and every DP run (including all of Experiment 1) would have silently been plain DP-SGD+FedAvg, not FedProx, despite `experiment_config_*.json` logging a nonzero `prox_mu` as if it were active.

**The fix (`_apply_dp_safe_prox_step`):**
```python
def _apply_dp_safe_prox_step(real_model, global_dict, mu, lr):
    """
    Applies FedProx's proximal pull as a SEPARATE, non-privatized
    parameter update — not via loss.backward(). Safe: the prox term
    depends only on current params + last round's public global model,
    never on client data, so it costs zero privacy budget applied
    this way (standard operator-splitting: privatized data-gradient
    step + separate deterministic regularization step).
    """
    if global_dict is None or mu == 0:
        return
    with torch.no_grad():
        for name, param in real_model.named_parameters():
            if name not in global_dict:
                continue
            g = torch.as_tensor(global_dict[name], dtype=param.dtype, device=param.device)
            param -= lr * mu * (param - g)
```
Called **once per batch**, immediately after `optimizer.step()`, matching the per-batch frequency of the non-DP path's proximal term. `real_model` is the Opacus-unwrapped model (`model._module if hasattr(model, "_module") else model`), so parameter names line up correctly against `global_dict`.

**Conclusion:** all six completed Experiment 1 runs are confirmed **DP-SGD + FedProx** (mu=0.02), not DP-SGD + FedAvg. Label results accordingly in any write-up.

---

## GPU Migration & Infrastructure (New Section, v4)

### Hardware and Environment

- **NVIDIA DGX Spark**, Grace Blackwell **GB10** superchip, **ARM64/aarch64** (not the usual x86_64) — standard PyPI PyTorch wheels do not have working aarch64+CUDA builds for this chip; must go through NVIDIA's NGC container path.
- **Unified 128GB CPU/GPU memory** (not separate VRAM) — `nvidia-smi`'s table view reports `Memory-Usage: Not Supported` on this hardware; this is a known reporting gap, not an error. `free -h` (system-wide) or `torch.cuda.mem_get_info()` (CUDA-driver-level) are the reliable ways to check actual available memory.
- **NGC container:** `nvcr.io/nvidia/pytorch:25.10-py3` — GB10's sm_121 compute capability requires NGC ≥25.10; older tags (e.g. commonly-referenced 24.08/24.01) do not support it correctly.
- Access: OpenVPN (`arno_external.ovpn`, lab subnet `10.30.7.0/24`) → SSH to the DGX host → `docker exec` into the running `flids_dev` container, which has the repo bind-mounted at `/workspace` (so files persist on the host filesystem, survive container restarts).
- Actual working directory on this machine: **`experiments/Current model/`** — the earlier assumed path `experiments/Current tests/` in v1–v3 of this document does not match what's actually on disk; corrected throughout this revision.

### Fork+CUDA Hang Fix (Critical)

**Symptom:** the GPU sanity-check run hung indefinitely at round 1, 0% CPU and 0% GPU utilization on the worker process.

**Root cause:** `ProcessPoolExecutor`'s worker was being created via Linux's default `'fork'` start method **after** CUDA had already been initialized in the main process (`torch.cuda.is_available()` runs at module import time, before the pool exists). Forking a process that already holds an active CUDA context hands the child a half-initialized, unsafe context — a well-known PyTorch/CUDA footgun that hangs rather than erroring cleanly.

**Fix:** when CUDA is available, no `ProcessPoolExecutor` is created at all. Client training/eval for each round runs via a plain sequential in-process loop (`_run_training_wave()`/`_run_eval_wave()`, `executor=None` branch), calling `_train_one_client()`/`_eval_one_client()` directly. CPU-only runs are unchanged (still the original 4-way pool — fork is safe there since no CUDA context ever exists in the parent process). This is also the practically-correct choice independent of the hang: a single GPU has no benefit from multiple processes each opening their own CUDA context — that causes memory contention and per-process context overhead, not speedup.

### vLLM Memory Contention Incident (resolved, documented for future reference)

A colleague's (Dr. Rana's) vLLM server (`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`, `--gpu-memory-utilization 0.85`) was running on the same shared DGX, reserving ~101GB of the 119GB unified memory pool for the lifetime of the process — even when idle — leaving too little free for training. This produced two distinct symptoms depending on exact timing: an indefinite hang (uninterruptible `D`-state process, waiting on a memory allocation that could never succeed) and, later, a hard `CUDA error: out of memory` on an allocation as small as a few hundred bytes (the `FocalLoss` criterion's `.to(device)` call). Root-caused via `free -h` (system-wide memory, not container-scoped — confirmed via `cat /sys/fs/cgroup/memory.max` returning `max`, i.e. no container memory cap) cross-referenced against `docker stats`/`ps aux --sort=-%mem` on the **host**, not inside any container (vLLM's actual memory footprint was invisible to standard `ps`/`RSS` accounting — CUDA/unified-memory allocations aren't reflected there the way normal heap pages are). Resolved by stopping the vLLM process (with supervisor + Dr. Rana's confirmation) — confirmed via `torch.cuda.mem_get_info()` returning `(119183532032, 128524255232)` (~111GB free) immediately after. **No code-side fix was needed or applicable — this was pure infrastructure contention on a shared research machine, not a bug.**

### CLI Args — `--epsilon`, `--tag`

```
python3 main.py [network|application] [--epsilon E] [--tag TAG]
```
`--epsilon` overrides `DP_EPSILON` (default 15.0). `--tag` suffixes every output filename (`results_{model}_{tag}.csv`, `checkpoint_{model}_{tag}*.npz/json`), eliminating manual `mv`-based archiving between sweep conditions. **Naming inconsistency found and flagged (cosmetic, not yet fixed):** the application ε=9 run was tagged `dp09` while network's equivalent run was tagged `dp9` — harmless for the already-completed sweep but should be standardized (e.g. always `dp{int(epsilon)}` with consistent zero-padding, or none) before running further conditions, to avoid downstream parsing mismatches in any aggregation/plotting script.

### GPU Timing — Confirmed Dramatically Faster Than CPU Estimates

| Condition | Avg round time | Total wall time |
|---|---|---|
| network (any ε) | ~132-135s | ~0.92-0.93 hr |
| application (any ε) | ~144-153s | ~1.03-1.04 hr |

All six completed conditions finished in **under 65 minutes each** — versus the original CPU-based estimates of ~2-3 hr (network) / ~4-6 hr (application) per condition. Round times were also extremely consistent within each condition (no runaway rounds), confirming no recurring memory pressure once vLLM was cleared.

---

## FL Configuration (Confirmed Values, GPU Runs)

```python
MODEL_TYPE      = "network" or "application"
NUM_CLIENTS     = 10
NUM_ROUNDS      = 25
LOCAL_EPOCHS    = 5
LEARNING_RATE   = 0.001
PROX_MU         = 0.02       # confirmed 0.02, NOT the 0.01 originally specified in v1-v3's
                              # "Exact Configuration" — this was a user-confirmed intended
                              # value change; note it if comparing against any older doc text
                              # that still says 0.01.
DIRICHLET_ALPHA = 0.7
BATCH_SIZE      = 512        # DP_BATCH_SIZE specifically; confirmed no OOM issues on GPU
                              # post-vLLM-kill across all 6 completed runs
```

**No LR decay. No EMA.** Confirmed still true for all Experiment 1 runs (`get_round_lr()` present in code but unused/dead — deliberately not called in the active round loop).

**Parallelisation:** CPU path unchanged (4-way `ProcessPoolExecutor`). **GPU path: sequential in-process, no pool** — see "Fork+CUDA Hang Fix" above.

---

## Repository Structure (Corrected Path)

```
09-edge-iot-security-monitoring/
├── experiments/
│   ├── Current model/               ← CORRECTED — actual working directory (was
│   │   │                               mistakenly documented as "Current tests/" in v1-v3)
│   │   ├── main.py                  ← UNIFIED: all flags in one file; now with
│   │   │                               argparse (--epsilon, --tag), DP-safe FedProx,
│   │   │                               save-best-checkpoint, GPU sequential-mode fork fix
│   │   ├── model_defs.py            ← dependency-free model (dp_safe flag)
│   │   ├── task.py                  ← CORRECTED weight tables + gamma; FocalLoss.weight
│   │   │                               now register_buffer'd; train()/test() accept device=
│   │   ├── data_loader.py           ← CORRECTED labels + text features; load_and_preprocess()
│   │   │                               now takes model_type= to avoid loading both caches
│   │   │                               into every worker unnecessarily
│   │   ├── check_features.py        ← confirms actual feature counts
│   │   ├── verify_label_bug.py      ← proof script (historical, keep)
│   │   ├── confusion_matrix.py      ← per-class diagnosis tool — NOT YET RUN on any
│   │   │                               Experiment 1 checkpoint, still an open item
│   │   ├── plot_epsilon_sweep.py    ← STILL NOT WRITTEN — open item, see below
│   │   └── defences/
│   │       ├── __init__.py
│   │       ├── byzantine.py         ← sign_flip, gaussian, zero_gradient, classifier_head_flip
│   │       ├── krum.py              ← multi_krum() (fixed-m, confirmed m propagation fixed);
│   │       │                           adaptive_multi_krum() (MAD-threshold, USED for
│   │       │                           Experiment 1's actual completed runs, Condition 5)
│   │       ├── zkp.py               ← HMAC-SHA256 commitment (canonical, merged)
│   │       ├── local_dp.py          ← apply_local_dp() (canonical, merged)
│   │       └── homomorphic.py       ← DEAD CODE — real HE not here
│   └── Docker tests for RAM and Latency/
│       ├── he_aggregation.py        ← REAL HE implementation
│       ├── he_local.py              ← REAL HE encryption
│       └── ...
├── scripts/
│   └── build_manifest.py            ← auto-generates manifests from CSV+config pairs
├── RESULTS AND MANIFESTS/
└── datasets/
    ├── dnn_preprocessed_cache_network.npz       ← split per model, see Dataset section
    └── dnn_preprocessed_cache_application.npz
```

---

## Three-Layer Privacy Stack

The architecture design and implementation are correct across all three layers. F1 numbers from experiments run before the label fix are invalid (as noted at the top of this document); timing/RAM numbers from those pre-fix ablations remain valid since they don't depend on label correctness.

### Layer 1 — Opacus DP-SGD

**File:** `experiments/Current model/task.py` (training loop), `main.py` (`_train_one_client`'s DP branch)
**Mechanism:** `PrivacyEngine.make_private_with_epsilon()`, per-sample gradient clipping to `max_grad_norm=1.5` (note: increased from the originally documented 1.0 — confirmed 1.5 in the actual Experiment 1 config and code).
**Quantum safety:** Information-theoretic — DP's guarantee holds against any adversary regardless of computational power, unlike the HE layer's guarantee (see Layer 3), which is only as strong as CKKS's underlying lattice-hardness assumptions.
**Known limitation:** Per-round ε only — no cross-round composition accountant tracks cumulative privacy spend across all 25 rounds. This was an explicit "Option A" decision (report the per-round guarantee honestly with this caveat, rather than implement full composition tracking). State this plainly in the paper: the reported ε values (3/9/15) are each round's individual guarantee, not a bound on what an adversary observing all 25 rounds' worth of updates could learn in aggregate.
**Noise-multiplier caching (new, confirmed working):** Opacus's epsilon→sigma calibration search is expensive; the achieved `noise_multiplier` is cached per `(client_idx, epsilon, delta, epochs, batch_size, max_grad_norm, dataset_size)` key after round 1's `make_private_with_epsilon()` call, and reused via `make_private(noise_multiplier=cached)` on subsequent rounds — confirmed not to change per-round epsilon (each round still gets a fresh `PrivacyEngine` and reports its own achieved epsilon), purely a performance optimization.
**DP calibration accuracy — confirmed excellent** across all 6 completed Experiment 1 runs: achieved epsilon within 0.25% of target in every case (e.g. target 3.0 → achieved 2.9931–2.9954; target 15.0 → achieved 14.9940–14.9995).
**Timing:** GPU timing (see GPU Migration section) — ~132-153s/round for all 10 clients combined, dramatically faster than the original ~300s/client/round CPU estimate (that older figure was measured on CPU, pre-GPU-migration).

**Open blocker — now answered by Experiment 1's actual results, not merely "designed to answer it."** The three-way question (push epsilon higher / move to central DP / restrict noise to classifier-head only) is resolved: this experiment's evidence does **not** support restricting DP noise to the classifier head as a Krum-preservation measure, because Krum showed no meaningful degradation across ε∈{3,9,15} under this attack strength to begin with (see Experiment 1's Headline Result 1).

### Layer 2 — HMAC-SHA256 Commitment

**File:** `defences/zkp.py`
**What it actually is:** a norm-bound commitment with an HMAC signature — **not** a full zero-knowledge proof (which would require an actual proving system such as Bulletproofs or a STARK). Naming it "ZKP" in the codebase and config is a simplification carried over from early in the project; do not claim a real ZKP construction in the paper.
**Mechanism:** `C = HMAC-SHA256(key, params_bytes || salt)`; server verifies `||w||_2 ≤ clip_norm + 1.15 · σ · √n_params` before accepting a client's update.
**Critical bug previously fixed:** a `×0.01` scaling error in the threshold formula made validation 100× too strict, rejecting every legitimate update — fixed via the `NOISE_NORM_SAFETY_FACTOR = 1.15 · σ · √n_params` formula now in use.
**Terminology note — do not conflate in the paper:** `main.py`'s `USE_ZKP` flag controls this plain norm check specifically. `defences/zkp.py`'s actual mechanism is the HMAC commitment described above. Both names are in use in different parts of the codebase/docs for the same thing — pick one consistent term for the write-up (recommend "norm-bound commitment" over "ZKP" for accuracy) and use it throughout.
**Not used in Experiment 1 or in any of Experiments 2/3 as currently scoped:** `USE_ZKP=False` throughout, to keep each experiment to the minimum number of interacting mechanisms needed to isolate its specific question (DP↔Krum for Experiment 1, HE↔Krum for Experiment 2). A future experiment characterizing the norm-bound commitment's own interaction with DP/Krum/HE would need its own dedicated sweep, not scoped yet.
**Possible future role (flagged in Experiment 2's discussion above):** if Experiment 2 confirms a Byzantine blind spot in the encrypted classifier-head slice, one concrete mitigation path is extending this layer's norm-bound check to cover the encrypted slice — e.g. a client-side commitment to the ciphertext's norm bound, computed before encryption, verifiable server-side without decryption. Not yet designed in detail; a follow-on question, not a solved problem.

### Layer 3 — Partial CKKS HE

**Source of truth:** `experiments/Docker tests for RAM and Latency/he_aggregation.py`, `he_local.py`. **`defences/homomorphic.py` is dead code** — HE has not been ported into the main `experiments/Current model/` experiment path in the same unified way DP/Krum have; the functional HE implementation lives only in the separate Docker-based RAM/latency-testing tree.
**Parameters:** `poly_modulus_degree=4096` — this yields **64-bit** post-quantum security, **not** the more commonly cited 128-bit figure. State this exact number honestly in the paper; do not round up to the more impressive-sounding 128-bit.
**Scope:** Partial — only the classifier head is encrypted (~5.8% of total parameters for the network model), not the full model. This is precisely the design choice Experiment 2 interrogates: encrypting only a slice creates a slice Krum cannot see into.
**Timing:** ~0.2s/round for HE operations specifically (confirmed from the earlier ablation, CPU-based) — cheap relative to DP-SGD's per-round cost, HE is not the bottleneck in this stack.
**RAM:** ~400MB floor for pure HE; dominated by training's own ~600MB floor when DP is also active (pure-DP ablation) — HE's memory footprint is not the limiting factor either.
**Critical architecture fix already applied:** clients now use a single shared public context distributed from the server. Earlier in the project, each client generated an independent keypair, which silently broke homomorphic addition mathematically (ciphertexts encrypted under different keys cannot be homomorphically combined) — this was caught and fixed before any HE numbers were trusted.
**Known open bugs, now blocking (not just noted) for Experiment 2:** `he_aggregate()` does not decrypt before returning its result, and its averaging is unweighted (plain mean across clients rather than sample-count-weighted, unlike `fedprox_aggregate()`'s weighted average). These were flagged as open issues in earlier revisions when only HE's timing/RAM ablation numbers mattered; now that Experiment 2 needs HE's actual F1-Macro output to be meaningful, both must be fixed first — see Experiment 2's Prerequisites above.
**Not used in Experiment 1:** `USE_HE=False` throughout, to keep that sweep to exactly two interacting mechanisms (DP × Krum).
**Literature pointer for the harder version of this problem:** if a genuinely encrypted-compatible Byzantine-robust aggregation is ever wanted (rather than the plaintext/encrypted split Experiment 2 proposes), see Lancelot (arXiv 2408.06197) and PBFL (COCOON 2024) for encrypted Byzantine-robust alternatives — not implemented here, noted for future reference only.

---

## Multi-Krum Byzantine Defence (Fixed-m)

**File:** `defences/krum.py::multi_krum()`
**Reference:** Blanchard et al., NeurIPS 2017

`m` propagation bug (from earlier revisions) is **confirmed fixed** — call site passes `m=effective_m` explicitly, no silent fallback to the internal `n-f-2` default. **Not the aggregation method actually used in Experiment 1's completed runs** — see the deviation note in the Experiment 1 section below. Still required and kept in the codebase as the Condition 3 comparison point and for any future true fixed-m epsilon sweep.

---

## Adaptive Multi-Krum (MAD-Threshold) — Condition 5, and the Method Actually Used for Experiment 1

**File:** `defences/krum.py::adaptive_multi_krum()`

**Design:** threshold = `median(scores) + k·1.4826·MAD(scores)`, `k=2.5`, `method="mad"`, `min_keep_fraction=0.5` safety floor. Returns `(aggregated_params, selected_indices, diagnostics_dict)` — the diagnostics dict carries `num_dropped`, `threshold`, `center`, `spread`.

**Confirmed via Experiment 1's six completed runs:** selection is textbook-clean — exactly 8/10 clients kept (clients 0/1, the Byzantine pair, discarded) in **every single round of every one of the six runs**, zero collateral damage, zero persistent-exclusion anomaly on any honest client (unlike the earlier Condition 5 run, which found Client 4 (network) persistently excluded 25/25 rounds — that anomaly did **not** recur in the Experiment 1 runs, worth noting as a possible difference between the Condition-5 recipe and Experiment 1's DP-active recipe, though not yet root-caused).

**Historical Condition 5 run results (unchanged from v3, kept for reference)** — see "Condition 5 Results" section further below; that run's headline numbers should still not be directly compared to the locked clean baseline without the documented `PROX_MU`/LR-decay recipe-drift caveat.

---

## Condition 5 Results — Adaptive Krum + Byzantine Attack (Historical, Pre-Experiment-1)

**Setup:** Byzantine attack (clients 0, 1; sign-flip), defended by `adaptive_multi_krum()` (MAD-threshold, `k=2.5`, `min_keep_fraction=0.5`), both models, 25 rounds, **not on GPU, not with the DP-safe FedProx fix** (this run predates both).

### 1. Headline numbers vs. locked clean baselines — read with the caveat in Section 5 below

| Model | Metric | Locked clean baseline | This run (final/best round) | Δ |
|---|---|---|---|---|
| Application | Accuracy | 0.8504 (round 20) | 0.8913 (round 24, best) | **+0.0409** |
| Application | F1-Macro | 0.7293 (round 20) | 0.7979 (round 24, best) | **+0.0686** |
| Network | Accuracy | 0.9697 (round 22) | 0.9856 (round 25, best) | **+0.0159** |
| Network | F1-Macro | 0.8289 (round 22) | 0.8651 (round 25, best) | **+0.0362** |

Both models numerically beat their locked clean baseline on both metrics, **despite this run having an active Byzantine attack running the entire time.** This is *not* evidence that adaptive Krum improves on the clean baseline — see Section 5.

### 2. Krum Defence Performance

- **Detection rate: 100%, every round, both models, all 25 rounds.**
- **Application: perfectly stable**, 8/10 kept every round, zero variation.
- **Network: mostly stable, one real persistent outlier** — Client 4 excluded 25/25 rounds, a genuine statistical signal (score persistently exceeds the MAD threshold), most likely a skewed per-class sample distribution under Dirichlet(α=0.7). **Still an open, unresolved item as of v4** — `per_client_audit.py` proposed but not run against network partitions.

### 3. Per-Class Final Performance (last-5-round average, rounds 21-25)

**Application — weakest to strongest:** XSS 0.516, Uploading 0.592, Password 0.661, SQL_injection 0.832, Fingerprinting 0.798, Port_Scanning 0.910, Backdoor 0.964, Normal 0.958.

**Network — weakest to strongest:** MITM 0.464, Ransomware 0.569, DDoS_HTTP 0.599, Vulnerability_scanner 0.718, Normal 0.821, DDoS_TCP 0.937, DDoS_UDP 0.999, DDoS_ICMP 0.999.

### 4. Transient Collapses (both self-heal within 1-2 rounds)

Network round 23 (F1-Macro 0.825→0.626), application round 21 (0.796→0.725) — both non-Krum-related, matching the recurring non-IID/FedProx instability pattern (now confirmed 4 independent times as of Experiment 1, see Network Baseline section above).

### 5. Why Accuracy/F1 Read Higher Here Than the Locked Baseline — Confound, Not a Real Improvement

Three uncontrolled differences vs. the locked-baseline recipe: `PROX_MU=0.1` (vs. baseline's unknown/undocumented value), active cosine LR decay (vs. baseline's documented "no LR decay"), and "best round" being each run's ceiling not a typical value. **Still not resolved as of v4** — the recipe-matched Condition 1 rerun recommended in v3 has not been done. Do not cite Condition 5's delta over the locked baseline as an adaptive-Krum improvement without first closing this gap.

---

## ⭐ EXPERIMENT 1 — DP vs. KRUM (EPSILON SWEEP) — COMPLETED (v4)

**Status change from v3: this experiment is DONE, not a next step.** All 6 of 8 originally-planned conditions (ε∈{3,9,15} × {network, application}; the two ε=∞/Condition-3 anchor rows were not separately re-run as part of this batch) have been executed, and results fully analyzed. Two companion documents exist and should be treated as the canonical results record:
- `Experiment1_DP_vs_Krum_Analysis_v2.md` — full results analysis
- `FL-IDS_Experiment1_Code_Manifest.md` — full technical manifest of the code that produced these results

### ⚠️ Deviation From Original Spec — Now Permanent, Not a Mistake to Fix

**v1–v3 of this document specified fixed-m Krum (`m=7`) for Experiment 1 and explicitly said "do not substitute `adaptive_multi_krum()` into Experiment 1."** That constraint was **not followed** — the actual code used `USE_ADAPTIVE_KRUM=True` throughout, by deliberate user decision (documented in the code's own changelog: *"USER DECISION: running the epsilon sweep with adaptive (MAD-threshold) Krum instead of fixed-m... this makes the sweep a distinct adaptive-Krum variant, and any comparison against a fixed-m Condition 3 anchor is no longer apples-to-apples"*).

**This is now accepted as the actual experiment that was run, not a bug to retroactively fix.** The v3 "Constraints — Never Violate" line forbidding this substitution is retracted in this revision (see updated Constraints below). **Practical consequence:** if a true fixed-m-Krum epsilon sweep (directly comparable to the standalone Condition 3 fixed-m result) is ever needed for the paper, it does not exist yet and would be a new, separate set of 6 runs — Experiment 1's completed results are an **adaptive-Krum-under-DP** sweep, not a **fixed-m-Krum-under-DP** sweep.

### Configuration (Actually Run)

```
NUM_CLIENTS=10, NUM_ROUNDS=25, LOCAL_EPOCHS=5, LEARNING_RATE=0.001, PROX_MU=0.02
BYZANTINE_CLIENTS=[0,1], ATTACK_SCALE=5.0 (network) / 2.0 (application)
USE_ADAPTIVE_KRUM=True, method="mad", k=2.5, min_keep_fraction=0.5
USE_DP=True, DP_EPSILON∈{3.0, 9.0, 15.0}, DP_DELTA=1e-5, DP_MAX_GRAD_NORM=1.5, DP_BATCH_SIZE=512
USE_ZKP=False, USE_HE=False
DP-safe FedProx active (mu=0.02, decoupled proximal step — see fix above)
Device: CUDA (DGX Spark GB10), sequential in-process client training
```

**Run commands used:**
```
python3 main.py network     --epsilon 3.0  --tag dp3
python3 main.py network     --epsilon 9.0  --tag dp9
python3 main.py network     --epsilon 15.0 --tag dp15
python3 main.py application --epsilon 3.0  --tag dp3
python3 main.py application --epsilon 9.0  --tag dp09   # NOTE: inconsistent tag vs network's "dp9" — cosmetic only
python3 main.py application --epsilon 15.0 --tag dp15
```

### Headline Result 1 — The Original Hypothesis Does NOT Hold (and that's the finding)

The hypothesis (DP noise erodes Krum's Byzantine-detection separation) is **not supported** by this data:

| Model | ε=3 mean `krum_score_ratio` | ε=9 mean | ε=15 mean | Change ε3→ε15 |
|---|---|---|---|---|
| Network | 310.6 | 313.2 | 315.8 | +1.7% |
| Application | 102.5 | 103.8 | 104.8 | +2.2% |

**Detection rate was 100.00% in every single round of every one of the six runs, no exceptions.** The sign-flip attack (scale 5.0/2.0) produces a parameter deviation orders of magnitude larger than anything DP-SGD's per-round noise injects at these ε values — DP noise never gets close to confusing Krum's L2-distance scoring.

**Resolution of the Layer-1 open blocker:** this data does **not** justify restricting DP noise to the classifier head as a Krum-preservation measure — there's nothing to preserve Krum from at ε≥3 under this attack strength. The simpler options (keep local DP as configured, or push ε higher purely for utility) remain fully defensible.

### Headline Result 2 — The Real Dose-Dependent Effect: DP Noise Delays Rare-Class Discovery

While Krum's robustness was ε-invariant, **rare-class learnability was strongly ε-dependent.** Clearest on Fingerprinting (application model, 1,001 raw samples, rarest class):

| Round | ε=3 | ε=9 | ε=15 |
|---|---|---|---|
| 1–14 | 0.000 | 0.000 | 0.000 |
| 15 | 0.000 | **0.516** (wakes up) | 0.000 |
| 16 | 0.000 | 0.560 | 0.045 (wakes up) |
| 25 | 0.169 | 0.613 | 0.586 |

At ε=3, Fingerprinting stays dead for 23/25 rounds and only barely stirs at the very end. At ε=9/15, it wakes up mid-run and reaches ~0.6. **This is not a Krum effect** (Krum's exclusion set was identical across all three ε conditions) — it's DP noise directly overwhelming gradient signal for an already-data-starved class. Positioned as the experiment's more novel, citable contribution: DP doesn't just cost accuracy uniformly, it can suppress rare-class learning near-categorically below some noise threshold.

### Overall F1-Macro / Accuracy vs. Epsilon

| Condition | Best round | Best F1-Macro | Best Acc |
|---|---|---|---|
| network_dp3 | 23 | 0.7268 | 0.8944 |
| network_dp9 | 23 | 0.7778 | 0.9345 |
| network_dp15 | 23 | **0.8068** | **0.9464** |
| application_dp3 | 22 | 0.4814 | 0.7200 |
| application_dp09 | 25 | **0.6251** | 0.7696 |
| application_dp15 | 25 | 0.6067 | **0.7701** |

**Network model:** clean, monotonic — more noise → worse F1/accuracy, as expected.
**Application model: NOT monotonic** — ε=9 beats ε=15 on best F1-Macro, and ε=9's last-5-round std (0.0072) is far tighter than ε=15's (0.0414). Reinforced by a second, independent signal: per-client accuracy spread (heterogeneity) is also tightest at ε=9 on **both** models (network: std 0.073 at ε=9 vs 0.127/0.116 at ε=3/15; application: std 0.085 at ε=9 vs 0.125/0.121 at ε=3/15). Two independent metrics pointing the same direction makes "ε=9 is a genuine local sweet spot" a somewhat stronger claim than either alone — still single-seed, a repeat run would be needed to fully confirm it isn't a specific run's volatility.

**ε=3 is unambiguously the worst condition on both models** — not in question.

### Krum Selection — Textbook-Clean, All Six Runs

```
Client 1 (Byzantine): 0/25 selected, every run
Client 2 (Byzantine): 0/25 selected, every run
Clients 3-10 (honest): 25/25 selected, every run
```
Zero collateral damage, zero persistent-exclusion anomaly anywhere in Experiment 1's six runs (the earlier Condition 5 network run's Client 4 anomaly did not recur here).

### Round-25 Instability — 4th Independent Confirmation

`network_dp15` shows the same late-round instability pattern (accuracy 0.9464→0.9007, DDoS_HTTP F1 0.7249→0.5567, Vulnerability_scanner F1 0.7175→0.5491, rounds 23→25). This is now the 4th independent occurrence (clean baseline round 25, Condition 5 network round 23, Condition 5 application round 21, and this) — solid enough to state as structural in the paper. **Use best-round, not final-round, numbers for the network model.**

### DP Calibration — Confirmed Excellent

All six runs' achieved epsilon within 0.25% of target; noise multiplier decreases monotonically as target epsilon increases on both models, exactly as expected. No calibration anomalies.

### Recommendations for the Write-Up

1. Lead with the negative Krum-robustness result (Section "Headline Result 1") as a real, citable finding, not a failed experiment.
2. Report the Fingerprinting/rare-class-suppression finding (Section "Headline Result 2") as the more novel contribution.
3. Use best-round numbers for the network model given the confirmed round-25 instability.
4. Flag the application model's ε=9 non-monotonicity honestly, citing both the F1-Macro and per-client-spread evidence, but frame as needing repeat-seed confirmation.
5. Correctly label the method throughout as **DP-SGD + FedProx** (mu=0.02, DP-safe decoupled step), not FedAvg.
6. Clearly state that this is an **adaptive-Krum** epsilon sweep, not the originally-planned fixed-m-Krum sweep — see the Deviation note above.
7. This experiment does not, by itself, justify restricting DP noise to the classifier head.

---

## ⭐ EXPERIMENT 2 — HE vs. KRUM — NEXT STEP (v5)

### The Core Tension

Krum's Byzantine-detection depends entirely on computing plaintext Euclidean distances between client parameter vectors. Layer 3's partial CKKS HE encrypts the classifier head (~5.8% of parameters for the network model) before it reaches the server. **These two mechanisms are architecturally incompatible on the encrypted slice specifically:** Krum cannot compute a distance on ciphertext without an expensive homomorphic distance operation that does not exist in this codebase (`he_aggregate()` only does homomorphic *summation*, not distance/comparison — CKKS supports approximate arithmetic well but comparison/distance-under-encryption is a substantially harder primitive, generally requiring bootstrapping or a different scheme entirely).

**The concrete risk this implies:** a Byzantine client could scale or sign-flip *only* the encrypted classifier-head parameters, leaving the plaintext feature-extraction layers (CNN/LSTM) looking completely normal. Krum — which only ever sees the plaintext layers in the current architecture — would have no way to detect this, because the attack is invisible to it by construction, not by chance.

### Hypothesis

If Krum's distance computation only ever operates on the plaintext (non-head) parameter slice, a head-only attack passes through completely undetected — Krum's reported detection rate would be 0% for a head-only attack, in contrast to Experiment 1's 100% detection rate for a full-model sign-flip attack. This would be a genuine, concrete architectural blind spot, not a hypothetical one.

### What Already Exists (Confirmed From Code Review)

Encouragingly, this experiment was clearly anticipated when the codebase was designed — several pieces are already in place:
- **`BYZANTINE_HEAD_ONLY` flag** already defined in `main.py`, with a comment noting it's "Only meaningful when `USE_HE=True` (full model encrypted, subtle attack needed)."
- **`classifier_head_flip_attack()`** already implemented in `defences/byzantine.py`, already imported and wired into `_train_one_client()`'s Byzantine branch: `if client_cfg["use_he"] and client_cfg["byzantine_head_only"]:` → calls `classifier_head_flip_attack(global_params, model_state_keys, scale=...)` instead of the full-model `sign_flip_attack`.
- **`he_aggregate()`** exists and performs homomorphic summation over encrypted classifier-head vectors — the aggregation half of the HE path is functional (with a known, previously-flagged open issue: it does not decrypt before returning, and averaging is unweighted — must be fixed/verified before Experiment 2 produces trustworthy numbers, see Prerequisites below).

### What's Missing — The Actual Blocker

**`main.py`'s own assertion currently forbids this experiment from running at all:**
```python
assert sum([USE_KRUM, USE_ADAPTIVE_KRUM, USE_HE]) <= 1, \
    "USE_KRUM, USE_ADAPTIVE_KRUM, and USE_HE are mutually exclusive aggregation " \
    "branches — pick at most one."
```
`USE_HE=True` and `USE_KRUM`/`USE_ADAPTIVE_KRUM=True` cannot both be set. This has to change before Experiment 2 can produce any data at all — it is not a config flip, it is new aggregation logic.

**Proposed hybrid design (new work required):**
1. Split each client's parameter list into two slices at aggregation time: `classifier_*` keys (the encrypted-under-HE slice) vs. everything else (CNN/LSTM feature-extraction layers, plaintext).
2. **Plaintext slice:** run Krum (fixed-m or adaptive, matching whichever variant is under test) exactly as today, computing distances **only over the plaintext slice** — this is the natural/honest implementation of "Krum literally cannot see the encrypted slice," not a workaround.
3. **Encrypted slice:** aggregate via `he_aggregate()` as today, but **restricted to whichever clients Krum selected from the plaintext-slice scoring** — i.e., Krum's selection decision (made on plaintext evidence only) still determines which clients' encrypted contributions get included, even though Krum never "saw" those encrypted contributions directly. This is the realistic threat model: the server can still exclude a client Krum flagged as Byzantine from the encrypted aggregate too, it just can't have used the encrypted data to make that determination.
4. New flag, e.g. `USE_HE_KRUM_HYBRID`, replacing the current three-way mutual exclusion with: `USE_HE_KRUM_HYBRID` (this experiment) vs. the existing three (unchanged, still mutually exclusive with each other and with the new hybrid mode).
5. **Fix the known `he_aggregate()` issues first** (no decrypt-before-return, unweighted averaging) — flagged as an open issue in earlier revisions of this document, carried forward and now blocking, not just noted.

### Exact Test Protocol (as specified)

1. Repeat the Experiment 1 Byzantine simulation setup (same 10 clients, same 2 Byzantine clients, same attack scale per model), but with `BYZANTINE_HEAD_ONLY=True` instead of the default full-model sign-flip — attacking clients scale/flip **only** the encrypted classifier-head parameters, leaving their plaintext CNN/LSTM layers looking like an honest update.
2. Run the hybrid aggregation (Step 3 above) and log Krum's detection rate exactly as in Experiment 1 (`krum_detected_byzantine`, `krum_score_ratio` computed over the plaintext slice only).
3. **Primary question:** does Krum still flag the attacking clients (because, e.g., their plaintext layers also drift detectably due to how local training interacts with a corrupted head) or does the attack slip through completely (detection rate near 0%)?
4. **If it slips through** (the expected/hypothesized result): this is a second novel, citable finding — a concrete, demonstrated blind spot. Frame it as a real argument for either (a) extending the Layer 2 HMAC commitment scheme to cover the encrypted slice (a norm-bound check *can* be computed differently — e.g., a commitment to the ciphertext's norm bound, computed client-side before encryption, verified server-side without decryption — this is a reasonable follow-on design question, not yet solved here), or (b) explicitly documenting the gap as a scoped limitation of the current privacy architecture rather than silently leaving it unaddressed.
5. **If Krum still catches it** (less expected, but possible if local training's proximal/regularization dynamics cause detectable plaintext-layer drift as a side effect of a corrupted head): this is also worth reporting — it would suggest the plaintext/encrypted split isn't as clean an attack surface as the architecture suggests, which is itself an interesting result worth explaining mechanistically.

### Prerequisites Before Running

1. Fix `he_aggregate()`'s two known issues (decrypt-before-return, weighted not unweighted averaging) — otherwise Experiment 2's F1-Macro numbers are not trustworthy regardless of what the detection-rate finding shows.
2. Implement the hybrid aggregation branch (plaintext-slice Krum + encrypted-slice HE, gated by Krum's plaintext-only selection) described above.
3. Confirm `classifier_head_flip_attack()`'s actual behavior matches intent — read the function, confirm it only touches `classifier`-prefixed state_dict keys and leaves everything else byte-identical to the honest update, since this is the entire premise of the experiment.
4. Decide whether this experiment reuses `USE_ADAPTIVE_KRUM` (matching Experiment 1's actual method) or `USE_KRUM` fixed-m (matching the original spec) — recommend **adaptive**, for direct comparability with Experiment 1's already-completed results, but state this decision explicitly in the write-up exactly as Experiment 1's deviation was documented.

---

## ⭐ EXPERIMENT 3 — PRIVACY CONFIGURATION CHECKPOINT MANIFEST — NEXT STEP (v5)

### Why This Matters

Every experiment so far (locked baselines, Condition 5, all six Experiment 1 runs) has produced a CSV of round-by-round metrics and, since the `save_best_checkpoint()` fix, a best-round `.npz`. But there is currently **no single, structured, cross-run index** tying a given checkpoint file to its exact configuration and final metrics — reconstructing "which checkpoint corresponds to which condition" currently requires reading filenames and cross-referencing `experiment_config_*.json` by hand, exactly the kind of manual bookkeeping that has already caused real problems in this project (the original round-20/round-22 unrecoverable-checkpoint incident, the `PROX_MU` mismatch confound in Condition 5). Experiment 3 turns this into a systematic, reloadable system before more conditions (Experiment 2's hybrid runs, any future epsilon/k/m sweeps) make the bookkeeping problem worse.

### What to Build

A `models/` (or `checkpoints/`) directory, one subfolder per completed run, plus a single top-level `manifest.json` (or `.csv`) indexing all of them:

```
models/
├── manifest.json                          ← index of every run below
├── network_baseline/
│   ├── state_dict.npz                     ← best-round weights (reuse save_best_checkpoint output)
│   ├── config.json                        ← full experiment_config_*.json contents
│   └── metrics.json                       ← final metrics summary (best round, best F1-Macro,
│                                              per-class F1, accuracy, round achieved)
├── network_krum/
├── network_dp_krum_eps3/
├── network_dp_krum_eps9/
├── network_dp_krum_eps15/
├── network_he/
├── network_he_krum_hybrid/                ← from Experiment 2, once it exists
├── application_baseline/
├── application_krum/
├── application_dp_krum_eps3/
├── application_dp_krum_eps09/
├── application_dp_krum_eps15/
├── application_he/
└── application_he_krum_hybrid/
```

**The six combinations explicitly named in the task** (baseline, +Krum, +DP, +DP+Krum, +HE, +HE+Krum) — mapped onto what already exists vs. what's still needed:

| Combination | Network model | Application model | Status |
|---|---|---|---|
| Baseline (no defence, no attack) | Locked round-22 baseline | Locked round-20 baseline | **Exists** — checkpoint recoverability was the original problem; needs backfilling into the manifest format |
| +Krum (fixed-m, no attack/DP) | Not yet run standalone | Not yet run standalone | **Missing** — Condition 3 as originally scoped (clean, no DP) was folded into Experiment 1's ε=∞ anchor concept but never actually executed as a standalone row |
| +DP (no Krum, no attack) | Not yet run | Not yet run | **Missing** — a DP-only ablation, isolating DP's utility cost without any Byzantine attack or defence in the picture, doesn't exist yet |
| +DP+Krum (Experiment 1) | 3 runs exist (ε=3/9/15) | 3 runs exist (ε=3/9/15) | **Exists**, but only under Byzantine attack + adaptive Krum — the "clean" (no-attack) +DP+Krum combination implied by this table doesn't exist separately |
| +HE (no Krum, no attack) | Historical ablation timing/RAM numbers only, F1 invalid pre-label-fix | Historical ablation timing/RAM numbers only, F1 invalid pre-label-fix | **Missing valid F1 numbers** — needs a fresh run under the corrected labels |
| +HE+Krum (Experiment 2) | Does not exist | Does not exist | **Blocked on Experiment 2's hybrid implementation** |

**This table itself is a useful output of Experiment 3** — it makes explicit that "baseline/+Krum/+DP/+DP+Krum/+HE/+HE+Krum" as a clean 6-row comparison table doesn't fully exist yet even after Experiment 1, because Experiment 1's DP+Krum rows are specifically *under Byzantine attack*, not a clean ablation of DP+Krum's cost with no attack present. Closing this gap (running the missing "no attack" ablation rows) is itself a concrete, scoped task this experiment surfaces.

### Manifest Schema (`manifest.json`)

```json
{
  "runs": [
    {
      "run_id": "network_dp_krum_eps9",
      "model_type": "network",
      "defence": "adaptive_krum",
      "privacy": {"use_dp": true, "epsilon": 9.0, "delta": 1e-5, "max_grad_norm": 1.5},
      "attack": {"active": true, "clients": [0, 1], "scale": 5.0, "type": "sign_flip"},
      "krum_config": {"method": "mad", "k": 2.5, "min_keep_fraction": 0.5},
      "training": {"prox_mu": 0.02, "rounds": 25, "local_epochs": 5, "lr_decay": false, "ema": false},
      "checkpoint_path": "models/network_dp_krum_eps9/state_dict.npz",
      "config_path": "models/network_dp_krum_eps9/config.json",
      "best_round": 23,
      "best_f1_macro": 0.7778,
      "best_accuracy": 0.9345,
      "krum_detection_rate": 1.0,
      "source_csv": "results_network_dp9.csv",
      "device": "cuda",
      "git_commit": null
    }
  ]
}
```
(`git_commit` deliberately included as a field even though not yet populated anywhere — flagged in the code manifest's Open Items as worth adding for full reproducibility provenance; Experiment 3 is the natural place to actually add it, e.g. via `git rev-parse HEAD` captured at run time.)

### Implementation Notes

- **Reuse, don't reinvent:** `save_best_checkpoint()` and `experiment_config_*.json` already produce most of the raw material — Experiment 3 is primarily an **organizing/indexing** layer on top of existing outputs, not a new training mechanism. A script that walks existing `checkpoint_*_best.npz` / `checkpoint_*_best.json` / `experiment_config_*.json` triples, copies/renames them into the `models/` structure, and appends an entry to `manifest.json` covers most of the backfill work for already-completed runs (locked baselines, Experiment 1's six runs).
- **`build_manifest.py`** (mentioned in the repository structure, `scripts/build_manifest.py`, described as "auto-generates manifests from CSV+config pairs") may already be intended for exactly this — confirm whether it already does some or all of this before writing a second, redundant tool.
- Loading a manifested checkpoint for reuse (e.g. for `confusion_matrix.py`, still an open item) should become a one-line lookup (`manifest["runs"][i]["checkpoint_path"]`) rather than manual filename reconstruction.

### Why This Is a Strong Central Chapter for the Write-Up

As stated in the task: most FL-security projects test privacy (DP/HE) or robustness (Byzantine defence) in isolation. This project's actual contribution — demonstrated concretely by Experiment 1's negative result and, pending Experiment 2, a possible concrete blind spot — is showing *where these interact and where they leave gaps*, evaluated together against the same defence, with a reloadable, reproducible record backing every number. Experiment 3's manifest is what makes that claim auditable rather than asserted.

---

## Open Items — Current State (v5)

### Still Unresolved From Earlier Revisions

- **Client 6 (application model) Password/XSS anomaly** — `per_client_audit.py` still not run.
- **Client 4 (network model) Condition-5-only exclusion anomaly** — still not run against network partitions; note it did *not* recur in Experiment 1's runs, which is itself worth investigating (what differs between the Condition 5 recipe and Experiment 1's recipe that would explain this?).
- **Condition 5's recipe-drift confound** (`PROX_MU=0.1`, LR decay vs. locked baseline's `PROX_MU`/no-decay) — the recommended recipe-matched Condition 1 rerun has not been done. Do not cite Condition 5 deltas over the locked baseline without it.
- **`confusion_matrix.py`** — proposed multiple revisions ago, still never run on any checkpoint, including Experiment 1's best-F1 checkpoints, which now exist and are recoverable (`save_best_checkpoint()` confirmed working) — this is now low-effort to actually do.

### From v4

- **`plot_epsilon_sweep.py`** — still not written. All six Experiment 1 CSVs exist and are fully analyzed by hand; a plotting script would let this be regenerated/extended (e.g. if more epsilon points are added) without repeating manual analysis.
- **Application ε=9 run tagged `dp09` vs. network's `dp9`** — cosmetic filename inconsistency, standardize before running further conditions.
- **Application model's ε=9 > ε=15 non-monotonicity** — needs a repeat run (different seed) to confirm it's real rather than one volatile draw, before leaning on it hard in the paper.
- **A true fixed-m-Krum epsilon sweep does not exist** — if the paper needs a clean apples-to-apples comparison against the standalone Condition 3 fixed-m result, this is a new set of runs, not something already done (see Deviation note in Experiment 1 section).
- **Experiment 1b (head-only DP)** — proposed in v3 as a follow-on only if Krum showed degradation under DP noise. **Since it did not (Headline Result 1), this follow-on is no longer motivated by the original rationale** — if pursued at all, it would need a different justification (e.g. compute/communication savings, formal per-layer guarantees) rather than "preserve Krum's robustness."

### New in v5 — Active Next Steps

- **Experiment 2 (HE vs. Krum) is blocked on new aggregation code**, not just a config flip — `main.py`'s `assert sum([USE_KRUM, USE_ADAPTIVE_KRUM, USE_HE]) <= 1` currently forbids running HE and Krum simultaneously. The hybrid plaintext-slice-Krum + encrypted-slice-HE design is specified in detail above but not yet implemented.
- **`he_aggregate()`'s two known bugs** (no decrypt-before-return, unweighted averaging) block Experiment 2's F1-Macro numbers from being trustworthy even once the hybrid branch exists — fix before, not after, running Experiment 2.
- **Experiment 3 (checkpoint manifest)** — check whether `scripts/build_manifest.py` (referenced in the repository structure but not yet reviewed) already does some or all of this before writing a redundant tool.
- **The "6 combinations" comparison table (baseline/+Krum/+DP/+DP+Krum/+HE/+HE+Krum) has real gaps even after Experiment 1** — standalone +Krum (no attack, no DP), standalone +DP (no Krum, no attack), and a valid post-label-fix +HE (no Krum) ablation all still need to be run; Experiment 1's existing +DP+Krum rows are specifically under Byzantine attack, not the clean ablation this table implies. See the gap table in the Experiment 3 section above.
- **`classifier_head_flip_attack()`'s exact behavior needs confirming** before Experiment 2 is trusted — read the function itself to confirm it only touches `classifier`-prefixed keys, since the entire experiment's premise depends on this being true.

---

## Constraints — Never Violate (Updated v5)

- **Network model: 39 features. Application model: 90 features.** Directly measured, supersedes all earlier estimates (38/40/52/~80-91).
- **LabelEncoder is removed.** Never re-add it. Use `_class_to_idx` manual mapping.
- **DDoS_TCP is class index 7 in the corrected mapping.** y==7 now correctly means DDoS_TCP.
- **The 18% cap targets Normal, not DDoS_TCP** — confirmed via Experiment 1 run logs. Do not revert to the pre-fix "cap DDoS_TCP" framing.
- **`multi_krum()` must return (aggregated_params, selected_indices).** Confirmed fixed, `m` propagates explicitly.
- **`accepted_client_indices` must be tracked in parallel.** ZKP compaction makes raw positions wrong.
- **NaN guard must remain in both `multi_krum()` and `adaptive_multi_krum()`.** Confirmed present in both; zero NaN rounds across all six Experiment 1 runs, but do not remove the guard.
- **No LR decay, no EMA** in the baseline pipeline or Experiment 1. Confirmed still true.
- **No Flower.** Framework is custom FedProx (confirmed, direct Python, sequential-in-process on GPU / `ProcessPoolExecutor` on CPU).
- **Save-best-checkpoint is implemented and confirmed working** — no longer a "required before running" item, it's done. Keep it working; do not regress to round-only checkpointing.
- **`dp_safe=True` (GroupNorm + DPLSTM) must be used for every client, honest and Byzantine alike, in every DP-active run** — confirmed code-reviewed as correct in the actual `main.py`.
- **DP epsilon is per-round, not cumulative** — `dp_epsilon_target` (config value) vs. `dp_epsilon_spent` (Opacus's actual achieved value) are logged as separate CSV columns; use `dp_epsilon_target` as any sweep's x-axis.
- **Any DP-active run with `PROX_MU != 0` must use the DP-safe decoupled proximal step (`_apply_dp_safe_prox_step`), never a loss-based proximal term.** The latter is silently discarded by Opacus's `.grad_sample`-based update — confirmed via the fix documented above. This applies to any future DP+FedProx experiment, not just Experiment 1.
- **~~Do not substitute `adaptive_multi_krum()` into Experiment 1~~ — RETRACTED in v4.** Experiment 1 was actually run with adaptive Krum, by deliberate decision, and those results are the canonical Experiment 1 results going forward. A separate fixed-m sweep remains a distinct, not-yet-done experiment if needed later.
- **`multi_krum()` (fixed-m) must never be deleted or replaced by `adaptive_multi_krum()` in the codebase** — both are needed: fixed-m for Condition 3 and any future fixed-m sweep; adaptive/MAD for Condition 5 and the actual completed Experiment 1.
- **On GPU (CUDA available): never create a `ProcessPoolExecutor` for client training.** Confirmed this causes a fork+CUDA hang. Sequential in-process training is the correct and only currently-implemented GPU path.
- **`FocalLoss.weight` must be a registered buffer, never a plain attribute** — confirmed fixed; do not reintroduce a plain `self.weight = weight` assignment, it silently breaks `.to(device)`.
- **Every `train()`/`test()` call site in `main.py` must pass `device=` explicitly** — both default to `'cpu'`, so a forgotten kwarg at any call site fails silently (wrong device, not an error) rather than crashing. Confirmed all current call sites correct; re-verify after any future edit to the round loop.
- **Training recipe (`PROX_MU`, LR-decay schedule, EMA) must be held identical to whatever produced the number being compared against, every time a delta is reported.** Confirmed `PROX_MU=0.02` (not 0.01) for all Experiment 1 runs — consistent across all six, so internal Experiment 1 comparisons are valid; but this differs from the locked-baseline recipe's undocumented mu, so Experiment 1 numbers still should not be directly diffed against the 0.7293/0.8289 locked baselines without the same recipe-matching caveat as Condition 5.
- **CKKS poly_modulus_degree=4096, 64-bit security** — state honestly in paper.
- **Per-round DP guarantee only** — state composition caveat honestly.
- **Repository working directory is `experiments/Current model/`** — not "Current tests/" as earlier revisions of this document assumed.
- **`USE_HE` must remain mutually exclusive with `USE_KRUM`/`USE_ADAPTIVE_KRUM` until the Experiment 2 hybrid aggregation branch is actually implemented.** Do not simply delete the `assert` to "make it run" — Krum would silently compute distances over an incomplete/mismatched parameter set (some clients' vectors including plaintext-only layers, encrypted layers absent) without the hybrid design's explicit plaintext/encrypted split, producing meaningless results that would look superficially valid.
- **`he_aggregate()` must be fixed (decrypt-before-return, weighted averaging) before Experiment 2's results are trusted** — these are pre-existing, previously-flagged bugs, not new ones, but they were never blocking until HE's F1 numbers were about to be reported directly rather than just its timing/RAM ablation numbers.
- **Any new checkpoint-producing run should be added to Experiment 3's manifest at creation time, not backfilled later "eventually"** — the whole point of Experiment 3 is to prevent a repeat of the original round-20/round-22 unrecoverable-checkpoint problem; treat manifest entry as part of "a run is complete," not an optional afterward step.
- **Supervisor framing: "make FL more secure and more private"** — research contribution orientation.



Week 9 summary:


-Built the HE+Krum hybrid aggregation branch, unblocking Experiment 2. main.py previously asserted USE_HE and USE_KRUM/USE_ADAPTIVE_KRUM were mutually exclusive; added USE_HE_KRUM_HYBRID, which splits each client's parameters into "sensitive" (classifier.*, CKKS-encrypted via he_local.py) and "bulk" (everything else, plaintext) using split_sensitive_bulk(). Adaptive Krum scores only the plaintext slice; the encrypted slice is aggregated afterward, restricted to whichever clients Krum selected from plaintext evidence alone. This also replaced the old full-model USE_HE path's two known bugs (no decrypt-before-return, unweighted averaging) with a correct implementation, without touching the old path.
- Found and fixed a critical bug in the stealthy attack itself before trusting any results from it. The first version of classifier_head_flip_attack's call site skipped local training entirely for Byzantine clients, returning last round's unmodified global model for the backbone. This made Krum's detection trivial for the wrong reason — it was catching "this client never trained," not "this client's encrypted head is poisoned." Fixed: Byzantine clients now train normally on the full model first, and only the trained classifier head gets flipped/scaled before encryption.

- Headline result, replicated on both models: 0% Byzantine detection, every round, with the corrected attack. krum_score_ratio stayed flat at ≈0.38 (application) / ≈0.237 (network) across all 25 rounds — the attackers didn't just blend in, they scored as more trustworthy than the average honest client. Best F1-Macro collapsed from a 0.73/0.83 clean baseline to 0.13 (application) and 0.72 (network); Normal-class F1 (application) and Vulnerability_scanner F1 (network) both sat at 0.0000 every single round while aggregate accuracy still climbed past 90% on the network model — a clean demonstration that aggregate accuracy can hide a fully destroyed class.
- Surfaced a persistent honest-client exclusion anomaly on the network model (clients 4/5/10 dropped every round regardless of attack), connecting to a previously-documented, unresolved Condition-5 anomaly that had grown from 1 excluded client to 3. Round-25 instability also recurred (5th documented occurrence in this project) — recommended round 24 as the headline number instead of the final round.

- Designed and built the Layer 2 mitigation: extended defences/zkp.py with a ciphertext-bound head-norm guard. Each client computes the L2 norm of its classifier-head delta (trained head minus the round's starting global head) before encrypting, signs it bound to a hash of the actual ciphertext bytes being submitted (so a client can't swap in a different ciphertext after the fact), and the server runs a MAD-threshold outlier check over all verified clients' committed norms — a magnitude-only analogue of Krum, applied to the one number the encrypted slice reveals, run as a pre-filter before Krum. Explicitly documented limitation: catches magnitude attacks, not a bounded-magnitude directional attack under threshold — same split the project already draws between ZKP and Krum for the full-model case. Reviewed and rejected a weaker alternative implementation with no ciphertext binding, confirmed via a standalone proof-of-concept that it would have been trivially bypassable.

- Confirmed the mitigation works — five runs, 100% detection in every one, across both models and three different attacked-client configurations (default clients 1,2; extreme-data clients 4,10; ordinary-data clients 2,7). Vulnerability_scanner F1 recovered from 0.0000 to 0.7864 (matching/exceeding baseline); application per-class F1 recovered broadly across previously-dead classes. Best F1-Macro landed within ~0.001 of the clean baseline for the default-client runs on both models.
- Cross-attack-configuration finding: Krum's extra (non-attacker) exclusions are consistently the same clients (network: 4, 10; application: 6, 7) across every attack configuration tested, including ones that don't target them at all — strong evidence the exclusion is driven by partition size/composition, not the attack. Confirmed directly via a new print_data_split() diagnostic added to main.py: clients 4/10 (network) hold 3–6x the fleet-median sample count and ~73% of the entire Vulnerability_scanner class between them.

- Added --byzantine <clients> and --krum-k <float> CLI args (change which/how many clients attack and the MAD sensitivity multiplier without editing the file), and raised ADAPTIVE_KRUM_K's default 2.5→3.5 plus added a new ADAPTIVE_KRUM_HYBRID_ASSUMED_F knob (default lowered from NUM_BYZANTINE to 1) — both justified by the data-split evidence and the norm guard's now-proven track record.

- Assembled manifests, configs, metrics, and run notes for all five mitigated runs plus a cross-configuration comparison table.