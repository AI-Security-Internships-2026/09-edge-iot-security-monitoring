# Split Protocol

One page describing how TRAIN / VALIDATION / TEST are used in this
pipeline, per Issue DAT1. This is the canonical reference — if any
code or paper text disagrees with this document, the code/text is
wrong, not this document.

## The three splits

Every experiment uses a **stratified 80 / 10 / 10** split, computed
once per `(model_type, seed)` pair and reused identically thereafter
(`data_loader.py: _get_or_build_tvt_indices`, artifacts under
`splits/TVT_global_<model_type>_<seed>.npz`, integrity-checked via a
SHA-256 hash sidecar on every load).

| Split | Size | Used for | Never used for |
|---|---|---|---|
| **TRAIN** | 80% | Fitting `VarianceThreshold`/`StandardScaler` (network/application respectively); the Dirichlet client partition; each client's own local training; computing class counts for loss-weighting (`get_class_counts_network/application`) | Any tuning decision that gets reported as a "held-out" result |
| **VALIDATION** | 10% | Multiplier/hyperparameter tuning decisions (FedProx μ, MAD-k, class-weight multipliers, DP clipping norm `C`); early-stopping decisions | Fitting scalers; client training; the final reported paper metric |
| **TEST** | 10% | Exactly **one** final evaluation per experiment, against the final global model, after the last FL round | Anything else. Never opened by any client during training. Never used to pick a hyperparameter, ever. |

## Client-local train/val is a different thing

Within a client's own TRAIN shard (after Dirichlet partitioning), each
client further splits 90/10 into **client-local-train** vs
**client-local-val** (`_dirichlet_partition`'s returned 4-tuple). This
is used only for that client's own local progress logging / potential
early-stopping — it is **not** the paper's VALIDATION split above, and
it is absolutely not the paper's TEST metric. Do not report numbers
from this split as a paper result.

## Per-parameter provenance table

| Parameter | Split used to pick it | Procedure |
|---|---|---|
| `VarianceThreshold` mask (network model) | TRAIN | `.fit()` once on TRAIN post-Normal-cap rows; pickled, never refit |
| `StandardScaler` mean/variance | TRAIN | `.fit()` once on TRAIN (post-VT for network, raw for application); pickled, never refit |
| Class-weight multipliers (Uploading/XSS/Fingerprinting) | ⚠️ **Currently unvalidated** — see flag in `task.py:build_criterion_application()`. Originally chosen against what was, pre-DAT1, a test-adjacent metric. Must be re-derived against VALIDATION before being cited as deliberately tuned. | Pending (DAT1 Task 2) |
| FedProx μ | TBD — not yet swept per-VALIDATION | Pending (Issue 4/5 μ-sweep work) |
| DP clipping norm `C` | TBD | Pending |
| MAD-k threshold | TBD | Pending |
| Final reported Macro-F1 / per-class F1 / accuracy | TEST | Computed exactly once, after Round 25, via `get_global_test_holdout(model_type, seed)` against the final global model |

## Reproducibility guarantee

Because the split artifacts and fitted scalers are cached to disk and
never regenerated once they exist for a given `(model_type, seed)`,
re-running the same seed always trains/evaluates against the exact
same rows and the exact same fitted preprocessing — this is what makes
multi-seed mean±std numbers meaningful (see
`tests/test_data_pipeline_dat1.py`'s determinism tests) and what the
TEST-holdout SHA-256 hash check guards against silently drifting.
