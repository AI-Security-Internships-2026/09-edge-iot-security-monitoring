# Split Protocol (DAT1)

This document describes the TRAIN / VALIDATION / TEST split discipline
enforced by `data_loader.py` and referenced throughout
`experiments/configs/hyperparams.json`'s per-parameter provenance
fields. It supersedes any earlier copy of this file that may exist
under `experiments/Current model/` -- this is the single authoritative
location (`docs/SPLIT_PROTOCOL.md`).

## Pipeline order

```
DATA -> separate untouched TEST holdout -> fit preprocessing on TRAIN only -> transform all splits -> FL partitioning/training
```

Concretely, per model (Network and Application):

1. The global raw dataset is loaded once.
2. A stratified 80/10/10 TRAIN/VALIDATION/TEST split is performed once,
   seeded from the global experiment seed, and written to
   `experiments/Current model/splits/TVT_global_<model>_<seed>.npz` so
   every later run against the same seed reuses the exact same rows.
3. `VarianceThreshold` is fit on TRAIN rows only (Network model path;
   skipped for the Application model per existing documented rule).
4. `StandardScaler` is fit on the TRAIN post-VT matrix (Network) or the
   raw TRAIN feature matrix (Application). Both fitted objects are
   pickled to `scalers_<model>_<seed>.pkl` and never refit between
   invocations.
5. All three splits are transformed with the fitted scaler(s).
6. Only after step 5, the preprocessed TRAIN matrix is partitioned
   across 10 clients via `Dirichlet(alpha)`.
7. Each client further splits its own TRAIN shard 90/10 into
   client-local training vs. client-local validation, used only for
   local-progress logging / early stopping -- never as a paper metric.
8. The global TEST holdout is never loaded by any client-training
   dataloader. It is loaded exactly once, via `get_global_test_holdout()`,
   after the round loop completes, for the single paper-citable
   evaluation.

## What each split is for

| Split | Used for | Never used for |
|---|---|---|
| TRAIN | Client-local training; fitting `VarianceThreshold`/`StandardScaler`; client-local validation (early-stopping signal only) | Any reported paper metric |
| VALIDATION | Selecting/justifying any tunable hyperparameter (FedProx mu, DP clipping norm C, MAD-k, class-weight multipliers, etc.) | Client training; final paper metrics |
| TEST | Exactly one final evaluation per experiment, after Round 25 | Any tuning decision, of any kind, ever |

## Per-parameter provenance

This table must stay in sync with `experiments/configs/hyperparams.json`'s
own `validated_on_split` fields -- treat the JSON as the source of
truth and this table as its human-readable mirror; if they ever
disagree, the JSON wins and this table needs updating.

| Parameter | Validated against | Status |
|---|---|---|
| `fedprox_mu` | VALIDATION-split mu-sweep (mu in {0, 0.005, 0.02, 0.05, 0.1}, 5 seeds x 2 modalities) | **Resolved.** network=0.005 (val macro-F1 0.868), application=0.0 (val macro-F1 0.806). No single mu is optimal for both modalities, so this is per-model_type, not a flat scalar. |
| `dp_max_grad_norm` (C) | -- | **Open.** No VALIDATION-split clipping-norm sweep has been run yet. Currently an inherited pre-DAT1 literal (1.5). |
| `adaptive_krum_k` | -- | **Open.** No VALIDATION-split MAD-k sweep result has been adopted as the operating default yet (a sweep exists for E5b sensitivity analysis purposes, but the *default* value has not been re-derived from it). Currently an inherited pre-DAT1 literal (2.5). |
| `adaptive_krum_hybrid_assumed_f` | N/A by design | Not a value fit to any split -- an operator-chosen assumed attacker-count cap. Experiment 2's assumed-f-mismatch ablation is the closest thing to sensitivity evidence for it. |
| `class_weight_multipliers_application` (Uploading/XSS/Fingerprinting) | -- | **Open.** Originally chosen pre-DAT1 against each client's own local held-out split (test-adjacent). Must be re-derived from a per-class VALIDATION-F1 sweep before being cited in the paper as deliberately tuned. See `task.py:build_criterion_application()`'s docstring for the full history. |
| `trimmed_mean_beta` | -- | **Open.** New BAS1 (Issue 3) baseline-comparison knob, not yet swept against any split. |

## Determinism and integrity guards

- Two pipeline invocations with the same global seed produce
  byte-identical `TVT_global_<seed>.npz` artifacts (not merely
  identical decoded index arrays) and identical scaler
  `.mean_`/`.var_`/`.support_` arrays.
- At pipeline startup, a SHA-256 hash of the TEST-holdout row index
  list is computed and logged; a mismatch against a prior run with the
  same seed raises loudly rather than silently proceeding.
- The global TEST holdout CSV/split path is never opened during client
  local-train steps -- enforced both by a static check on `main.py`'s
  source (the `get_global_test_holdout(` call site occurs exactly once,
  textually after the round loop) and by runtime file-access
  instrumentation during an actual 1-round/10-client run.
