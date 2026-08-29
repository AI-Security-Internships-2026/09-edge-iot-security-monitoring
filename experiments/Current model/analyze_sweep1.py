#!/usr/bin/env python3
"""
analyze_sweep1.py — Full analysis of Experiment 1 (DP vs. Krum epsilon sweep)
for the FL-IDS edge IoT security project.

WHAT THIS DOES
--------------
Loads every per-round results CSV produced by `main.py` for Experiment 1
(one CSV per {model, epsilon} condition — e.g. results_network_dp3.csv,
results_application_dp09.csv, ...) and produces:

  1. A per-condition summary (best round, best F1-Macro/accuracy, last-5-round
     stability, Krum detection rate, krum_score_ratio drift, DP calibration
     error).
  2. The two headline findings from the project's own analysis template:
       - Headline 1: does DP epsilon measurably erode Krum's Byzantine-
         detection separation (krum_score_ratio) or detection rate?
       - Headline 2: does DP epsilon delay/suppress rare-class learning
         (e.g. Fingerprinting), independent of Krum?
  3. Monotonicity check of F1-Macro / accuracy vs. epsilon per model, flagging
     any non-monotonic point (e.g. the documented application ε=9 > ε=15
     anomaly) instead of assuming it.
  4. Round-25 (or configurable last-round) instability check.
  5. Krum selection audit: per-round detection, any honest-client collateral
     exclusion pattern across conditions.
  6. A markdown report (headline-style, like the project's own analysis docs)
     and a tidy CSV summary table, plus optional PNG plots.

USAGE
-----
    python3 analyze_sweep1.py --results-dir /path/to/results --output-dir ./sweep1_analysis

    # Only network model, skip plots:
    python3 analyze_sweep1.py --results-dir . --model network --no-plots

    # Custom file pattern (default matches results_<model>_<tag>.csv):
    python3 analyze_sweep1.py --results-dir . --pattern "results_*.csv"

The script is schema-tolerant: it doesn't assume every column exists. If a
column (e.g. krum_score_ratio, dp_epsilon_spent) is missing from a given CSV,
that specific analysis section is skipped for that file with a note in the
report rather than the script crashing.

No third-party deps beyond pandas / numpy / (optional) matplotlib.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Known class name sets (used to auto-detect per-class F1 columns in a CSV
# that otherwise mixes metric columns and class columns together). Extend
# these lists if your class taxonomy changes.
# ----------------------------------------------------------------------------
NETWORK_CLASSES = [
    "Normal", "DDoS_UDP", "DDoS_ICMP", "Ransomware", "DDoS_HTTP",
    "DDoS_TCP", "Vulnerability_scanner", "MITM",
]
APPLICATION_CLASSES = [
    "Normal", "SQL_injection", "Uploading", "Backdoor", "Port_Scanning",
    "XSS", "Password", "Fingerprinting",
]
ALL_KNOWN_CLASSES = sorted(set(NETWORK_CLASSES) | set(APPLICATION_CLASSES))

# Rare / historically-interesting classes to always highlight if present.
RARE_CLASS_WATCHLIST = ["Fingerprinting", "SQL_injection", "Uploading", "MITM", "XSS"]

# Candidate column names for each logical field, in priority order. CSV
# schemas drift over the life of a project (this one's has); we try each
# candidate and use the first that exists.
COLUMN_ALIASES = {
    "round": ["round", "round_num", "Round"],
    "accuracy": ["accuracy", "acc"],
    "f1_macro": ["f1_macro", "f1-macro", "F1_Macro", "f1_macro_avg"],
    "krum_detected": ["krum_detected_byzantine", "krum_detected"],
    "krum_score_ratio": ["krum_score_ratio"],
    "dp_epsilon_target": ["dp_epsilon_target", "epsilon_target", "epsilon"],
    "dp_epsilon_spent": ["dp_epsilon_spent", "epsilon_spent", "achieved_epsilon"],
    "is_mean": ["is_mean"],
    "num_dropped": ["num_dropped", "krum_num_dropped"],
    "selected_indices": ["selected_indices", "accepted_client_indices"],
}


def resolve_col(df: pd.DataFrame, field_name: str) -> Optional[str]:
    """Return the actual column name in df for a logical field, or None."""
    for cand in COLUMN_ALIASES.get(field_name, []):
        if cand in df.columns:
            return cand
    return None


def detect_class_columns(df: pd.DataFrame) -> list[str]:
    """Return columns in df that look like per-class F1 columns."""
    cols = [c for c in df.columns if c in ALL_KNOWN_CLASSES]
    if cols:
        return cols
    # Fallback: numeric columns not already claimed by known metric fields,
    # excluding obvious non-class columns.
    claimed = set()
    for aliases in COLUMN_ALIASES.values():
        claimed.update(aliases)
    exclude_patterns = re.compile(
        r"(round|epoch|epsilon|delta|krum|dp_|client|loss|lr|seed|tag|model|"
        r"num_|threshold|std|mean_|is_|accuracy|acc$|f1_macro|ratio)",
        re.IGNORECASE,
    )
    fallback = []
    for c in df.columns:
        if c in claimed:
            continue
        if exclude_patterns.search(c):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            fallback.append(c)
    return fallback


# ----------------------------------------------------------------------------
# Filename parsing: results_<model>_<tag>.csv  where tag often encodes
# epsilon, e.g. "dp3", "dp09", "dp15". We extract a numeric epsilon from the
# tag when possible, else fall back to the dp_epsilon_target column.
# ----------------------------------------------------------------------------
FNAME_RE = re.compile(
    r"results_(?P<model>[A-Za-z0-9]+)_(?P<tag>[A-Za-z0-9]+)\.csv$", re.IGNORECASE
)
TAG_PREFIX_RE = re.compile(r"^dp", re.IGNORECASE)


def parse_filename(path: str) -> tuple[Optional[str], Optional[str]]:
    m = FNAME_RE.search(os.path.basename(path))
    if not m:
        return None, None
    return m.group("model"), m.group("tag")


def tag_to_epsilon(tag: str) -> Optional[float]:
    """Parse an epsilon value out of a tag like 'dp3', 'dp09' (=9), or
    'dp0p5' (=0.5, 'p' standing in for a decimal point — a common
    filename-safe convention since '.' is awkward in filenames)."""
    remainder = TAG_PREFIX_RE.sub("", tag)
    if not remainder:
        return None
    # 'p' -> '.' handles 'dp0p5' -> '0.5', 'dp2p25' -> '2.25', etc.
    remainder_dot = remainder.replace("p", ".").replace("P", ".")
    try:
        if "." in remainder_dot:
            return float(remainder_dot)
        # No decimal point: treat as a (possibly zero-padded) integer,
        # e.g. 'dp09' -> 9, 'dp15' -> 15, 'dp3' -> 3.
        return float(int(remainder_dot))
    except ValueError:
        return None


# ----------------------------------------------------------------------------
# Data model for a single condition (one CSV = one {model, epsilon} run)
# ----------------------------------------------------------------------------
@dataclass
class ConditionResult:
    path: str
    model: str
    tag: str
    epsilon: Optional[float]
    df: pd.DataFrame
    class_cols: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # populated by analyze()
    summary: dict = field(default_factory=dict)


def load_condition(path: str) -> ConditionResult:
    df = pd.read_csv(path)
    model, tag = parse_filename(path)
    if model is None:
        # best-effort fallback: use the 'model' column if present
        model_col = resolve_col(df, "round")  # noop, just to keep pattern
        model = str(df["model"].iloc[0]) if "model" in df.columns else "unknown"
        tag = os.path.splitext(os.path.basename(path))[0]

    is_mean_col = resolve_col(df, "is_mean")
    if is_mean_col:
        df = df[df[is_mean_col].fillna(0) == 0].copy()

    granularity_warning = None
    round_col_probe = resolve_col(df, "round")
    if round_col_probe:
        dup_counts = df[round_col_probe].value_counts()
        if len(dup_counts) > 0 and dup_counts.max() > 1:
            # Multiple rows share the same round number. This is expected if
            # the CSV is per-client (one row per client per round) rather
            # than one row per round. Try to auto-detect a client/row-type
            # column so we can collapse to one row per round; if we can't,
            # warn loudly rather than silently averaging across duplicated
            # rows (which corrupts krum_detection_rate, rare-class
            # wake-round counts, and score-ratio trend calculations).
            client_col = None
            for cand in ["client_id", "client_idx", "client_index", "client", "client_num"]:
                if cand in df.columns:
                    client_col = cand
                    break
            rows_per_round = int(dup_counts.max())
            if client_col:
                # Prefer an explicit mean/summary sentinel if the client
                # column has one (e.g. 'MEAN', 'mean', 'server', 'global').
                sentinel_mask = df[client_col].astype(str).str.lower().isin(
                    ["mean", "avg", "average", "server", "global", "summary"]
                )
                if sentinel_mask.any():
                    df = df[sentinel_mask].copy()
                    granularity_warning = (
                        f"ℹ {rows_per_round} rows/round detected via '{client_col}' — "
                        f"filtered down to the explicit summary/mean row per round "
                        f"(a standard per-client-plus-summary logging convention). "
                        f"High confidence this is correct if your pipeline always writes "
                        f"one such sentinel row per round."
                    )
                else:
                    # No sentinel row: average numeric columns across clients
                    # per round as a best-effort round-level summary. This is
                    # a reasonable default for continuous metrics (accuracy,
                    # f1, score ratios) but may NOT be correct for
                    # detection-flag columns, which often need a different
                    # aggregation (e.g. "was ANY Byzantine client caught").
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    df = df.groupby(round_col_probe, as_index=False)[numeric_cols].mean()
                    granularity_warning = (
                        f"⚠ UNVERIFIED AGGREGATION: {rows_per_round} rows/round detected via "
                        f"'{client_col}' with no mean/summary sentinel row — averaged numeric "
                        f"columns across clients per round as a best-effort fallback. "
                        f"detection-flag columns (e.g. krum_detected_byzantine) likely need "
                        f"a different aggregation than a plain mean (e.g. 'any client "
                        f"flagged' rather than 'fraction of client-rows flagged'). Re-check "
                        f"against your CSV's real schema before trusting this run's numbers."
                    )
            else:
                granularity_warning = (
                    f"⚠ {rows_per_round} rows share the same round number and no "
                    f"is_mean/client-id column was found to disambiguate them. "
                    f"Per-round metrics below (krum_detection_rate, rare-class wake "
                    f"rounds, score-ratio trend) are almost certainly WRONG — they are "
                    f"being computed over all {rows_per_round} rows per round (e.g. "
                    f"per-client rows) rather than one row per round. Add the real "
                    f"is_mean / client-id column name to COLUMN_ALIASES in this script, "
                    f"or pre-aggregate your CSV to one row per round, before trusting "
                    f"these numbers."
                )

    eps_col = resolve_col(df, "dp_epsilon_target")
    epsilon = tag_to_epsilon(tag) if tag else None
    if epsilon is None and eps_col:
        try:
            epsilon = float(df[eps_col].dropna().iloc[0])
        except (IndexError, ValueError):
            epsilon = None

    class_cols = detect_class_columns(df)

    # ------------------------------------------------------------------
    # Derive f1_macro when it isn't its own column (confirmed from the
    # project's main.py: round_f1_macro = mean_f1.mean() is computed
    # in-memory and used for checkpointing, but never written to the CSV
    # as its own column — only the per-class F1 columns are persisted).
    # This must run AFTER row-granularity resolution (so it's computed on
    # one row per round, e.g. the MEAN row) to match main.py's semantics
    # exactly: F1-Macro = mean of that round's per-class F1 values.
    # ------------------------------------------------------------------
    if resolve_col(df, "f1_macro") is None and class_cols:
        df["f1_macro"] = df[class_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        f1_derived_note = (
            f"f1_macro column not present — derived as the row-wise mean of the "
            f"{len(class_cols)} detected per-class F1 columns ({', '.join(class_cols)}), "
            f"matching main.py's own `round_f1_macro = mean_f1.mean()` computation."
        )
    else:
        f1_derived_note = None

    # Defensive numeric coercion: several columns are written as the
    # literal string "N/A" on rows/files where the field doesn't apply
    # (e.g. krum_detected_byzantine on files where Krum isn't active,
    # dp_epsilon_spent on non-mean client rows before aggregation). Left
    # as-is, pandas reads these as object dtype and a later .mean()
    # raises instead of just ignoring the N/A as missing data.
    for field_name in ["krum_detected", "krum_score_ratio", "dp_epsilon_target",
                        "dp_epsilon_spent", "accuracy", "f1_macro"]:
        col = resolve_col(df, field_name)
        if col:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    cr = ConditionResult(
        path=path, model=model.lower(), tag=tag, epsilon=epsilon,
        df=df, class_cols=class_cols,
    )
    if granularity_warning:
        cr.warnings.insert(0, granularity_warning)
    if f1_derived_note:
        cr.warnings.append(f1_derived_note)
    return cr


# ----------------------------------------------------------------------------
# Per-condition analysis
# ----------------------------------------------------------------------------
def analyze_condition(cr: ConditionResult, last_n_stability: int = 5,
                       instability_round: Optional[int] = None,
                       instability_drop_thresh: float = 0.03) -> dict:
    df = cr.df
    round_col = resolve_col(df, "round")
    acc_col = resolve_col(df, "accuracy")
    f1_col = resolve_col(df, "f1_macro")

    out: dict = {"path": cr.path, "model": cr.model, "tag": cr.tag, "epsilon": cr.epsilon}

    if round_col is None:
        cr.warnings.append("No round column found — cannot order rounds; skipping most analysis.")
        out["error"] = "no_round_column"
        return out
    df = df.sort_values(round_col)
    n_rounds = df[round_col].max()

    # --- best round by F1-Macro ---
    if f1_col:
        best_idx = df[f1_col].idxmax()
        out["best_round"] = int(df.loc[best_idx, round_col])
        out["best_f1_macro"] = float(df.loc[best_idx, f1_col])
        out["final_f1_macro"] = float(df[f1_col].iloc[-1])
        last_n = df[f1_col].tail(last_n_stability)
        out["last_n_f1_std"] = float(last_n.std()) if len(last_n) > 1 else None
    else:
        cr.warnings.append("No f1_macro column found.")

    if acc_col:
        out["best_accuracy"] = float(df.loc[df[acc_col].idxmax(), acc_col]) if f1_col is None else float(df.loc[best_idx, acc_col])
        out["final_accuracy"] = float(df[acc_col].iloc[-1])
        last_n_acc = df[acc_col].tail(last_n_stability)
        out["last_n_acc_std"] = float(last_n_acc.std()) if len(last_n_acc) > 1 else None
    else:
        cr.warnings.append("No accuracy column found.")

    # --- Krum detection ---
    krum_col = resolve_col(df, "krum_detected")
    if krum_col:
        out["krum_detection_rate"] = float(df[krum_col].mean())
        out["krum_rounds_full_detection"] = bool((df[krum_col] >= 0.999).all())
    ksr_col = resolve_col(df, "krum_score_ratio")
    if ksr_col:
        out["krum_score_ratio_mean"] = float(df[ksr_col].mean())
        out["krum_score_ratio_std"] = float(df[ksr_col].std())
        # trend: slope of a simple linear fit vs round, as a normalized % change
        try:
            slope = np.polyfit(df[round_col], df[ksr_col], 1)[0]
            start_val = df[ksr_col].iloc[0]
            pct_change_per_round = (slope / start_val * 100) if start_val else None
            out["krum_score_ratio_trend_slope"] = float(slope)
            out["krum_score_ratio_pct_change_total"] = (
                float((df[ksr_col].iloc[-1] - df[ksr_col].iloc[0]) / df[ksr_col].iloc[0] * 100)
                if df[ksr_col].iloc[0] else None
            )
        except Exception:
            pass

    # --- DP calibration ---
    tgt_col = resolve_col(df, "dp_epsilon_target")
    spent_col = resolve_col(df, "dp_epsilon_spent")
    if tgt_col and spent_col:
        tgt = df[tgt_col].astype(float)
        spent = df[spent_col].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_err = np.abs((spent - tgt) / tgt.replace(0, np.nan)) * 100
        out["dp_calibration_mean_pct_error"] = float(pct_err.mean())
        out["dp_calibration_max_pct_error"] = float(pct_err.max())
    else:
        if not tgt_col:
            cr.warnings.append("No dp_epsilon_target column — calibration check skipped.")
        if not spent_col:
            cr.warnings.append("No dp_epsilon_spent column — calibration check skipped.")

    # --- Round-N instability check (defaults to last round) ---
    # Prefer f1_macro; fall back to accuracy if f1_macro isn't in this CSV's
    # schema, so a column-name mismatch doesn't silently blank this check.
    instability_metric_col = f1_col or acc_col
    instability_metric_name = "f1_macro" if f1_col else ("accuracy" if acc_col else None)
    target_round = instability_round or int(n_rounds)
    if instability_metric_col and target_round in set(df[round_col]):
        row_now = df[df[round_col] == target_round][instability_metric_col].iloc[0]
        prior_rounds = df[df[round_col] < target_round]
        if not prior_rounds.empty:
            row_prior = prior_rounds[instability_metric_col].iloc[-1]
            drop = row_prior - row_now
            out["last_round_drop_metric"] = instability_metric_name
            out["last_round_f1_drop_from_prior"] = float(drop)
            out["last_round_instability_flag"] = bool(drop >= instability_drop_thresh)
    else:
        out["last_round_instability_flag"] = None  # not evaluated, not "no instability found"

    # --- rare-class trajectories ---
    rare_present = [c for c in RARE_CLASS_WATCHLIST if c in cr.class_cols]
    rare_traj = {}
    for c in rare_present:
        series = df[[round_col, c]].dropna()
        if series.empty:
            continue
        wake_round = None
        nonzero = series[series[c] > 0.01]
        if not nonzero.empty:
            wake_round = int(nonzero[round_col].iloc[0])
        rare_traj[c] = {
            "final_value": float(series[c].iloc[-1]),
            "max_value": float(series[c].max()),
            "wake_round": wake_round,
            "rounds_at_zero": int((series[c] <= 0.01).sum()),
        }
    if rare_traj:
        out["rare_class_trajectories"] = rare_traj

    # --- per-client accuracy spread (heterogeneity), if per-client cols exist ---
    per_client_cols = [c for c in df.columns if re.match(r"(client|per_client)_?\d+_?(acc|accuracy)?", c, re.I)
                        and pd.api.types.is_numeric_dtype(df[c])]
    if per_client_cols:
        spread = df[per_client_cols].std(axis=1).mean()
        out["per_client_spread_mean_std"] = float(spread)

    return out


# ----------------------------------------------------------------------------
# Cross-condition (sweep-level) analysis
# ----------------------------------------------------------------------------
def cross_condition_analysis(conditions: list[ConditionResult]) -> dict:
    rows = []
    for cr in conditions:
        s = dict(cr.summary)
        rows.append(s)
    summary_df = pd.DataFrame(rows)

    result: dict = {"summary_table": summary_df}

    if summary_df.empty:
        return result

    monotonicity = {}
    # Prefer F1-Macro when present; fall back to accuracy so a missing
    # f1_macro column (schema mismatch) doesn't silently blank this section.
    metric_col = "best_f1_macro" if "best_f1_macro" in summary_df.columns and \
        summary_df["best_f1_macro"].notna().any() else \
        ("best_accuracy" if "best_accuracy" in summary_df.columns else None)
    metric_label = {"best_f1_macro": "F1-Macro", "best_accuracy": "Accuracy"}.get(metric_col)

    if metric_col:
        for model, grp in summary_df.groupby("model"):
            grp = grp.dropna(subset=["epsilon", metric_col]).sort_values("epsilon")
            if len(grp) < 2:
                continue
            vals = grp[metric_col].values
            eps_vals = grp["epsilon"].values
            is_monotonic = np.all(np.diff(vals) >= -1e-9)
            violations = []
            for i in range(1, len(vals)):
                if vals[i] < vals[i - 1] - 1e-9:
                    violations.append((float(eps_vals[i - 1]), float(vals[i - 1]),
                                        float(eps_vals[i]), float(vals[i])))
            monotonicity[model] = {
                "metric_used": metric_label,
                "is_monotonic_increasing_with_epsilon": bool(is_monotonic),
                "violations": violations,
                "epsilon_order": eps_vals.tolist(),
                "metric_order": vals.tolist(),
            }
    result["monotonicity"] = monotonicity

    # Krum robustness to epsilon: does detection rate or score ratio degrade
    # as epsilon decreases (more noise)?
    krum_robustness = {}
    if "krum_score_ratio_mean" in summary_df.columns:
        for model, grp in summary_df.groupby("model"):
            grp = grp.dropna(subset=["epsilon", "krum_score_ratio_mean"]).sort_values("epsilon")
            if len(grp) < 2:
                continue
            lo = grp.iloc[0]  # lowest epsilon = most noise
            hi = grp.iloc[-1]  # highest epsilon = least noise
            pct_change = (
                (hi["krum_score_ratio_mean"] - lo["krum_score_ratio_mean"])
                / lo["krum_score_ratio_mean"] * 100
                if lo["krum_score_ratio_mean"] else None
            )
            krum_robustness[model] = {
                "eps_low": float(lo["epsilon"]), "eps_high": float(hi["epsilon"]),
                "score_ratio_low_eps": float(lo["krum_score_ratio_mean"]),
                "score_ratio_high_eps": float(hi["krum_score_ratio_mean"]),
                "pct_change_low_to_high_eps": float(pct_change) if pct_change is not None else None,
                "detection_rate_all_conditions": grp["krum_detection_rate"].tolist()
                if "krum_detection_rate" in grp else None,
            }
    result["krum_robustness_vs_epsilon"] = krum_robustness

    return result


# ----------------------------------------------------------------------------
# Report generation
# ----------------------------------------------------------------------------
def fmt(x, digits=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def df_to_markdown_table(df: pd.DataFrame, float_digits: int = 4) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table without
    depending on the optional `tabulate` package (which pandas' own
    `.to_markdown()` requires but doesn't ship by default)."""
    if df.empty:
        return "_(empty)_"

    def cell(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        if isinstance(v, (bool, np.bool_)):
            return str(bool(v))
        if isinstance(v, (float, np.floating)):
            return f"{v:.{float_digits}f}"
        return str(v)

    headers = list(df.columns)
    rows = [[cell(v) for v in row] for row in df.itertuples(index=False, name=None)]

    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def build_markdown_report(conditions: list[ConditionResult], cross: dict,
                           instability_round_label: str) -> str:
    lines = []
    lines.append("# Experiment 1 — DP vs. Krum Epsilon Sweep: Analysis Report")
    lines.append("")
    lines.append(f"Generated from {len(conditions)} condition file(s).")
    lines.append("")

    granularity_flags = [cr for cr in conditions if any(w.startswith("⚠") for w in cr.warnings)]
    confident_flags = [cr for cr in conditions if any(w.startswith("ℹ") for w in cr.warnings)]
    if granularity_flags:
        lines.append("> ## ⚠ DATA-QUALITY WARNING — READ BEFORE TRUSTING ANY NUMBER BELOW")
        lines.append(">")
        lines.append(f"> {len(granularity_flags)}/{len(conditions)} condition file(s) have more than "
                      f"one row per round with no confidently-identified summary row, so a best-effort "
                      f"or no aggregation was applied. Per-round metrics (Krum detection rate, "
                      f"rare-class wake rounds, score-ratio trend) computed from these files may be "
                      f"**incorrect** until this is resolved. See each file's entry under Data-Quality "
                      f"Notes at the bottom of this report for the exact row count detected and what "
                      f"was auto-applied.")
        lines.append("")
    elif confident_flags:
        lines.append(f"> ℹ {len(confident_flags)}/{len(conditions)} condition file(s) had multiple rows "
                      f"per round and were auto-collapsed to an explicit per-round summary row "
                      f"(see Data-Quality Notes at the bottom for detail).")
        lines.append("")

    # --- Per-condition table ---
    summary_df = cross.get("summary_table", pd.DataFrame())
    lines.append("## Per-Condition Summary")
    lines.append("")
    if not summary_df.empty:
        cols_order = [
            "model", "tag", "epsilon", "best_round", "best_f1_macro", "final_f1_macro",
            "last_n_f1_std", "best_accuracy", "final_accuracy", "last_n_acc_std",
            "krum_detection_rate", "krum_score_ratio_mean", "krum_score_ratio_pct_change_total",
            "dp_calibration_mean_pct_error", "dp_calibration_max_pct_error",
            "last_round_f1_drop_from_prior", "last_round_instability_flag",
        ]
        present_cols = [c for c in cols_order if c in summary_df.columns]
        disp = summary_df[present_cols].sort_values(["model", "epsilon"])
        lines.append(df_to_markdown_table(disp, float_digits=4))
    else:
        lines.append("_No conditions loaded._")
    lines.append("")

    # --- Headline 1: Krum robustness to epsilon ---
    lines.append("## Headline Finding 1 — Does DP Noise Erode Krum's Detection?")
    lines.append("")
    kr = cross.get("krum_robustness_vs_epsilon", {})
    has_krum_cols = any(
        "krum_score_ratio_mean" in cr.summary or "krum_detection_rate" in cr.summary
        for cr in conditions
    )
    n_with_epsilon = len({cr.model for cr in conditions if cr.epsilon is not None})
    if not kr:
        if not has_krum_cols:
            lines.append("_krum_score_ratio / krum_detected_byzantine columns not found in the data — "
                          "cannot evaluate this finding. Check your CSV schema against COLUMN_ALIASES "
                          "in the script._")
        else:
            lines.append(f"_krum_score_ratio and krum_detected_byzantine were found and are populated "
                          f"per condition (see the summary table above), but this section compares the "
                          f"trend **across** epsilon values, which needs at least 2 conditions with a "
                          f"known epsilon for the same model. Only {len(conditions)} condition file(s) "
                          f"were loaded this run — pass the whole sweep directory to see this section._")
    for model, info in kr.items():
        lines.append(f"**{model.title()} model:**")
        lines.append(
            f"- krum_score_ratio at ε={fmt(info['eps_low'],1)} (most noise): "
            f"{fmt(info['score_ratio_low_eps'],1)}"
        )
        lines.append(
            f"- krum_score_ratio at ε={fmt(info['eps_high'],1)} (least noise): "
            f"{fmt(info['score_ratio_high_eps'],1)}"
        )
        pct = info.get("pct_change_low_to_high_eps")
        lines.append(f"- Change, lowest→highest ε: {fmt(pct,2)}%")
        det = info.get("detection_rate_all_conditions")
        if det:
            all_perfect = all(d is not None and d >= 0.999 for d in det if d is not None)
            lines.append(
                f"- Detection rate across all ε conditions: {[fmt(d,3) for d in det]} "
                f"{'(100% in every condition — Krum unaffected by DP noise in this range)' if all_perfect else '(NOT uniformly 100% — investigate which condition dropped)'}"
            )
        verdict = (
            "Consistent with the *negative* result documented in the project write-up: "
            "DP noise does not measurably erode Krum's separation in this ε range."
            if pct is not None and abs(pct) < 5
            else "Score ratio moved by more than a marginal amount across the ε range — "
                 "worth checking whether this is a real DP↔Krum interaction or run-to-run noise."
        )
        lines.append(f"- **Verdict:** {verdict}")
        lines.append("")

    # --- Headline 2: rare-class suppression ---
    lines.append("## Headline Finding 2 — Does DP Noise Delay/Suppress Rare-Class Learning?")
    lines.append("")
    any_rare = False
    for cr in conditions:
        rare = cr.summary.get("rare_class_trajectories")
        if not rare:
            continue
        any_rare = True
        lines.append(f"**{cr.model} / ε={fmt(cr.epsilon,1)} ({cr.tag}):**")
        for cname, info in rare.items():
            lines.append(
                f"- {cname}: wakes at round {info['wake_round'] if info['wake_round'] else 'never'}, "
                f"final={fmt(info['final_value'],3)}, max={fmt(info['max_value'],3)}, "
                f"rounds at ~zero={info['rounds_at_zero']}"
            )
        lines.append("")
    if not any_rare:
        lines.append("_No watch-listed rare classes (Fingerprinting, SQL_injection, Uploading, "
                      "MITM, XSS) found as columns in the data — cannot evaluate this finding._")
        lines.append("")
    else:
        lines.append(
            "Compare `wake_round` and `rounds at ~zero` across ε for the same model/class: if "
            "lower ε consistently wakes later / stays at zero longer, that reproduces the "
            "documented dose-dependent rare-class suppression effect."
        )
        lines.append("")

    # --- Monotonicity check ---
    lines.append("## F1-Macro / Accuracy vs. Epsilon — Monotonicity Check")
    lines.append("")
    mono = cross.get("monotonicity", {})
    if not mono:
        lines.append("_Not enough conditions per model (need ≥2 with a known epsilon and either "
                      "f1_macro or accuracy) to check._")
    for model, info in mono.items():
        metric = info.get("metric_used", "Metric")
        lines.append(f"**{model.title()} model** (using {metric} — f1_macro unavailable, "
                      f"falling back to accuracy — if this says Accuracy, fix the f1_macro "
                      f"column alias first):" if metric == "Accuracy" else f"**{model.title()} model:**")
        lines.append(f"- Epsilons (ascending): {info['epsilon_order']}")
        lines.append(f"- Best {metric} per epsilon: {[fmt(v,4) for v in info['metric_order']]}")
        if info["is_monotonic_increasing_with_epsilon"]:
            lines.append(f"- **Monotonic**: more privacy budget (higher ε) never hurts best {metric}.")
        else:
            lines.append("- **NOT monotonic** — flagged violation(s):")
            for lo_e, lo_v, hi_e, hi_v in info["violations"]:
                lines.append(
                    f"  - ε={fmt(lo_e,1)} ({metric}={fmt(lo_v,4)}) → ε={fmt(hi_e,1)} "
                    f"({metric}={fmt(hi_v,4)}): {metric} dropped despite higher ε. Treat as "
                    f"needing repeat-seed confirmation before calling it a genuine sweet spot."
                )
        lines.append("")

    # --- Round instability ---
    lines.append(f"## {instability_round_label} Instability Check")
    lines.append("")
    evaluated = [cr for cr in conditions if cr.summary.get("last_round_instability_flag") is not None]
    not_evaluated = [cr for cr in conditions if cr.summary.get("last_round_instability_flag") is None]
    flagged = [cr for cr in evaluated if cr.summary.get("last_round_instability_flag")]
    if not evaluated:
        lines.append("_Not evaluated for any condition — no f1_macro or accuracy column, or the "
                      "target round wasn't present in the data. This is NOT the same as "
                      "'no instability found.'_")
    else:
        metric = evaluated[0].summary.get("last_round_drop_metric", "metric")
        if flagged:
            for cr in flagged:
                lines.append(
                    f"- **{cr.model} / ε={fmt(cr.epsilon,1)} ({cr.tag})**: {metric} dropped by "
                    f"{fmt(cr.summary.get('last_round_f1_drop_from_prior'),4)} into the checked round "
                    f"— consider reporting best-round rather than final-round numbers for this run."
                )
        else:
            lines.append(f"_No condition showed a drop in {metric} above the configured threshold "
                          f"(evaluated {len(evaluated)}/{len(conditions)} conditions)._")
        if not_evaluated:
            lines.append(f"_Note: {len(not_evaluated)} condition(s) could not be evaluated "
                          f"(missing column or round) — see Data-Quality Notes._")
    lines.append("")

    # --- DP calibration ---
    lines.append("## DP Calibration Accuracy")
    lines.append("")
    calib_rows = [cr for cr in conditions if "dp_calibration_mean_pct_error" in cr.summary]
    if calib_rows:
        for cr in calib_rows:
            lines.append(
                f"- {cr.model} / ε={fmt(cr.epsilon,1)}: mean error "
                f"{fmt(cr.summary['dp_calibration_mean_pct_error'],3)}%, max error "
                f"{fmt(cr.summary['dp_calibration_max_pct_error'],3)}%"
            )
    else:
        lines.append("_dp_epsilon_target / dp_epsilon_spent columns not found — skipped._")
    lines.append("")

    # --- Warnings / data-quality notes ---
    lines.append("## Data-Quality Notes")
    lines.append("")
    any_warn = False
    for cr in conditions:
        if cr.warnings:
            any_warn = True
            lines.append(f"**{os.path.basename(cr.path)}:**")
            for w in cr.warnings:
                lines.append(f"- {w}")
    if not any_warn:
        lines.append("_No missing-column warnings — all expected fields were found in every file._")
    lines.append("")

    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Plotting (optional, matplotlib)
# ----------------------------------------------------------------------------
def make_plots(conditions: list[ConditionResult], summary_df: pd.DataFrame, output_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots (pip install matplotlib).")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. F1-Macro vs epsilon, per model
    if {"model", "epsilon", "best_f1_macro"}.issubset(summary_df.columns):
        fig, ax = plt.subplots(figsize=(7, 5))
        for model, grp in summary_df.dropna(subset=["epsilon"]).groupby("model"):
            grp = grp.sort_values("epsilon")
            ax.plot(grp["epsilon"], grp["best_f1_macro"], marker="o", label=model)
        ax.set_xlabel("DP epsilon (target)")
        ax.set_ylabel("Best F1-Macro")
        ax.set_title("Best F1-Macro vs. Epsilon")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "f1_macro_vs_epsilon.png"), dpi=150)
        plt.close(fig)

    # 2. krum_score_ratio vs epsilon
    if {"model", "epsilon", "krum_score_ratio_mean"}.issubset(summary_df.columns):
        fig, ax = plt.subplots(figsize=(7, 5))
        for model, grp in summary_df.dropna(subset=["epsilon", "krum_score_ratio_mean"]).groupby("model"):
            grp = grp.sort_values("epsilon")
            ax.plot(grp["epsilon"], grp["krum_score_ratio_mean"], marker="o", label=model)
        ax.set_xlabel("DP epsilon (target)")
        ax.set_ylabel("Mean krum_score_ratio")
        ax.set_title("Krum Score Ratio vs. Epsilon")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "krum_score_ratio_vs_epsilon.png"), dpi=150)
        plt.close(fig)

    # 2b. krum detection rate vs epsilon
    if {"model", "epsilon", "krum_detection_rate"}.issubset(summary_df.columns):
        fig, ax = plt.subplots(figsize=(7, 5))
        any_data = False
        for model, grp in summary_df.dropna(subset=["epsilon", "krum_detection_rate"]).groupby("model"):
            grp = grp.sort_values("epsilon")
            if grp.empty:
                continue
            any_data = True
            ax.plot(grp["epsilon"], grp["krum_detection_rate"] * 100, marker="o", label=model)
        if any_data:
            ax.set_xlabel("DP epsilon (target)")
            ax.set_ylabel("Krum detection rate (%)")
            ax.set_title("Krum Byzantine Detection Rate vs. Epsilon")
            # detection rate is a proportion in [0,1]; fix the y-axis so a flat
            # 100% line reads as "no degradation" rather than looking noisy
            ymax = max(100.0, float((summary_df["krum_detection_rate"].max() or 1.0) * 100) + 5)
            ymin = min(0.0, float((summary_df["krum_detection_rate"].min() or 0.0) * 100) - 5)
            ax.set_ylim(ymin, ymax)
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "krum_detection_rate_vs_epsilon.png"), dpi=150)
        plt.close(fig)

    # 3. Rare-class trajectory per model (one figure per model, one line per epsilon)
    by_model: dict[str, list[ConditionResult]] = {}
    for cr in conditions:
        by_model.setdefault(cr.model, []).append(cr)

    round_col_cache = {}
    for model, crs in by_model.items():
        rare_classes_present = set()
        for cr in crs:
            rare_classes_present.update(c for c in RARE_CLASS_WATCHLIST if c in cr.class_cols)
        for cname in rare_classes_present:
            fig, ax = plt.subplots(figsize=(9, 6))
            ordered_crs = sorted(crs, key=lambda c: (c.epsilon is None, c.epsilon))
            # Use a continuous colormap sampled across all series instead of
            # matplotlib's default 10-color cycle, which repeats (and
            # therefore visually conflates) two different epsilon series
            # whenever there are more than 10 conditions for one model.
            n_series = max(len(ordered_crs), 1)
            cmap = plt.get_cmap("viridis")
            colors = [cmap(i / max(n_series - 1, 1)) for i in range(n_series)]
            for color, cr in zip(colors, ordered_crs):
                round_col = resolve_col(cr.df, "round")
                if round_col is None or cname not in cr.df.columns:
                    continue
                d = cr.df[[round_col, cname]].dropna().sort_values(round_col)
                ax.plot(d[round_col], d[cname], marker=".", label=f"ε={fmt(cr.epsilon,1)}", color=color)
            ax.set_xlabel("Round")
            ax.set_ylabel(f"{cname} F1")
            ax.set_title(f"{model.title()} model — {cname} F1 vs. Round, by Epsilon")
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, ncol=1 if n_series <= 12 else 2)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fname = f"rare_class_{model}_{cname}.png".replace(" ", "_")
            fig.savefig(os.path.join(output_dir, fname), dpi=150)
            plt.close(fig)

    print(f"Plots written to {output_dir}/")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True, help="Directory containing the result CSVs.")
    ap.add_argument("--pattern", default="results_*_*.csv",
                     help="Glob pattern (relative to --results-dir) to find result CSVs. "
                          "Default matches results_<model>_<tag>.csv.")
    ap.add_argument("--output-dir", default="./sweep1_analysis",
                     help="Where to write the report, summary CSV, and plots.")
    ap.add_argument("--model", choices=["network", "application"], default=None,
                     help="Restrict analysis to one model (default: both).")
    ap.add_argument("--last-n-stability", type=int, default=5,
                     help="Window size (rounds) for the last-N-round stability std-dev.")
    ap.add_argument("--instability-round", type=int, default=None,
                     help="Round number to check for a late-round instability drop "
                          "(default: each run's own final round).")
    ap.add_argument("--instability-drop-thresh", type=float, default=0.03,
                     help="F1-Macro drop (absolute) into the checked round that triggers "
                          "an instability flag.")
    ap.add_argument("--no-plots", action="store_true", help="Skip generating PNG plots.")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.results_dir, args.pattern)))
    if not paths:
        print(f"No files matched '{args.pattern}' in {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    conditions: list[ConditionResult] = []
    for p in paths:
        try:
            cr = load_condition(p)
        except Exception as e:
            print(f"WARNING: failed to load {p}: {e}", file=sys.stderr)
            continue
        if args.model and cr.model != args.model:
            continue
        conditions.append(cr)

    if not conditions:
        print("No conditions loaded after filtering — nothing to analyze.", file=sys.stderr)
        sys.exit(1)

    instability_label = f"Round {args.instability_round}" if args.instability_round else "Final-Round"

    for cr in conditions:
        cr.summary = analyze_condition(
            cr,
            last_n_stability=args.last_n_stability,
            instability_round=args.instability_round,
            instability_drop_thresh=args.instability_drop_thresh,
        )

    cross = cross_condition_analysis(conditions)

    os.makedirs(args.output_dir, exist_ok=True)

    # Write summary CSV
    summary_df = cross["summary_table"]
    summary_csv_path = os.path.join(args.output_dir, "sweep1_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8")
    print(f"Summary table written to {summary_csv_path}")

    # Write JSON (full machine-readable dump)
    json_path = os.path.join(args.output_dir, "sweep1_summary.json")
    json_payload = {
        "conditions": [
            {**cr.summary, "path": cr.path, "warnings": cr.warnings} for cr in conditions
        ],
        "monotonicity": cross.get("monotonicity", {}),
        "krum_robustness_vs_epsilon": cross.get("krum_robustness_vs_epsilon", {}),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, default=str)
    print(f"Machine-readable summary written to {json_path}")

    # Write markdown report
    report = build_markdown_report(conditions, cross, instability_label)
    report_path = os.path.join(args.output_dir, "sweep1_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report written to {report_path}")

    if not args.no_plots:
        make_plots(conditions, summary_df, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
