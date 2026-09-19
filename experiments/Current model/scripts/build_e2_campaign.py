#!/usr/bin/env python3
"""
build_e2_campaign.py

Generates the E2 section of experiments/configs/EXP1_campaign.json
(Issue 5 Task 1: "create experiments/configs/EXP1_campaign.json ... every
required experiment cell must be marked REUSE | NEW_RUN | RERUN").

All 90 E2 cells are marked NEW_RUN per the reuse audit: no valid
Issue #22-23 results exist yet to reuse for this comparison.

Grid: 6 aggregators x 5 attacks x 3 seeds = 90 cells.
  - Krum and Multi-Krum are the same code path in this codebase
    (USE_KRUM = AGGREGATOR in ("krum","multi_krum")) -- counted as ONE
    aggregator condition ("multi_krum"), per the issue's own instruction.
  - Seeds {42, 123, 456} -- a subset of the 5-seed set {42,123,456,789,
    2024} used elsewhere in the campaign, so E2 stays directly
    paired-seed-comparable with E3/E5/E6 if those later need the same
    seeds, while meeting E2's stated minimum of 3.

Base condition (fixed across all 90 cells): Edge-IIoTset, network IDS,
alpha=0.7, no DP, f=2 Byzantine (clients 1,2 -- main.py's own default).
"""

import json
import os

OUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "experiments", "configs", "EXP1_campaign_E2.json"
)

SEEDS = [42, 123, 456]

# (canonical_aggregator_slug, --aggregator CLI value)
AGGREGATORS = [
    ("fedavg", "fedavg"),
    ("multi_krum", "multi_krum"),   # aliases "krum" in this codebase
    ("median", "median"),
    ("trimmed_mean", "trimmed_mean"),
    ("adaptive_krum", "adaptive_krum"),
    ("calibrated_krum", "calibrated_krum"),
]

ATTACKS = ["sign_flip", "gaussian", "minmax", "minsum", "bounded_directional"]

HETERO_FIT_PATH = "experiments/configs/hetero_fit_coeffs_a0.7.json"


def attack_flags(attack, model_type="network"):
    """Attack-specific CLI flags, matching main.py's own documented
    defaults for anything not being deliberately overridden."""
    if attack == "gaussian":
        # main.py's own default (50.0 for network, 30.0 for application)
        # -- explicit here so the manifest is self-describing without
        # having to cross-reference main.py's _GAUSSIAN_STD_DEFAULT.
        return {"gaussian_std": 50.0 if model_type == "network" else 30.0}
    if attack in ("minmax", "minsum"):
        return {
            # None = auto (5x coalition spread) UNLESS Task 2's
            # check_attack_difficulty.py Check 1/3 sweep found this
            # needs an explicit override to land in the required
            # krum_score_ratio band -- fill in the frozen value here
            # once that check passes, and do not change it afterward
            # (Task 2: "freeze all attack parameters before final TEST
            # evaluation").
            "minmax_dev_type": "std",
            "minmax_search_iters": 15,
            "minmax_gamma_init": None,
        }
    if attack == "bounded_directional":
        return {
            # No live norm guard exists under any E2 aggregator (E2
            # never uses pure_norm_guard/USE_NORM_GUARD) -- tau is left
            # None so main.py estimates it ONLINE each round from this
            # run's own honest classifier-head norms
            # (estimate_norm_guard_tau, same MAD-k rule the real guard
            # uses). This is condition-appropriate self-calibration,
            # not an unfrozen parameter: the ESTIMATION RULE (k=3.5
            # default, margin, direction) is what's frozen here, not a
            # single numeric tau carried over from a different
            # alpha/DP condition (the E6 pilot's frozen tau was fit
            # under alpha=0.3 + epsilon=5 + DP-on and is NOT valid
            # under E2's alpha=0.7/no-DP condition -- same class of
            # mismatch as the hetero-fit alpha issue).
            "bounded_tau": None,
            "bounded_margin": 0.05,
            "bounded_direction": "negate",
        }
    return {}


def build():
    cells = []
    cell_id = 0
    for agg_slug, agg_cli in AGGREGATORS:
        for attack in ATTACKS:
            for seed in SEEDS:
                cell_id += 1
                # main.py's own _TAG construction is
                # f"{MODEL_TYPE}_{tag}_seed{seed}" -- it appends
                # _seed{seed} automatically, so `tag` here must NOT
                # include the seed itself (else filenames end up
                # "..._seed42_seed42").
                tag = f"e2_{agg_slug}_{attack}"
                cli = {
                    "positional_model_type": "network",
                    "ablation_mode": "krum_baseline",
                    "aggregator": agg_cli,
                    "attack_type": attack,
                    "alpha": 0.7,
                    "seed": seed,
                    "tag": tag,
                    "dataset": "edge_iiotset",
                    # no --epsilon: krum_baseline hardcodes USE_DP=False
                    "byzantine": "1,2",  # f=2, main.py's own default
                }
                cli.update(attack_flags(attack))
                if agg_slug == "calibrated_krum":
                    cli["hetero_fit_coeffs_json"] = HETERO_FIT_PATH
                    # DP calibration is a documented no-op here (USE_DP
                    # is always False under krum_baseline, so every
                    # client's noise_multiplier is None) -- leaving
                    # --no-dp-calibration UNSET anyway, since main.py
                    # already handles that correctly (contributes 0);
                    # explicitly disabling it would just be redundant,
                    # not wrong.

                cells.append({
                    "cell_id": f"E2-{cell_id:03d}",
                    "experiment": "E2",
                    "table": "Table 2",
                    "figure": "Figure 5",
                    "aggregator": agg_slug,
                    "attack_type": attack,
                    "seed": seed,
                    "alpha_dirichlet": 0.7,
                    "epsilon": None,
                    "dataset": "edge_iiotset",
                    "model_type": "network",
                    "num_byzantine": 2,
                    "byzantine_clients_1indexed": "1,2",
                    "cli_args": cli,
                    "expected_manifest": f"experiment_config_network_{tag}_seed{seed}.json",
                    "expected_log_csv": f"results_network_{tag}_seed{seed}.csv",
                    "expected_krum_scores_csv": (
                        f"per_client_krum_scores_network_{tag}_seed{seed}.csv"
                        if agg_slug in ("multi_krum", "adaptive_krum", "calibrated_krum")
                        else None
                    ),
                    "reuse_audit": {
                        "existing_result": None,
                        "valid": None,
                        "action": "NEW_RUN",
                        "reason": (
                            "No valid Issue #22-23 result exists for this "
                            "exact (aggregator, attack, seed, alpha=0.7, "
                            "no-DP) combination -- confirmed via "
                            "user-supplied reuse audit at campaign-build "
                            "time."
                        ),
                    },
                    "prerequisites": (
                        ["hetero_fit_coeffs_a0.7.json (scripts/refit_hetero_alpha0.7.py)"]
                        if agg_slug == "calibrated_krum" else []
                    ) + [
                        "check_attack_difficulty.py Check 1/3 PASS (minmax/minsum only)"
                        if attack in ("minmax", "minsum") else None,
                    ],
                })
                cells[-1]["prerequisites"] = [p for p in cells[-1]["prerequisites"] if p]

    manifest = {
        "experiment": "E2",
        "title": "Aggregator x Byzantine Attack Comparison",
        "issue": 24,
        "base_condition": {
            "dataset": "edge_iiotset",
            "task": "Network IDS",
            "alpha_dirichlet": 0.7,
            "dp_epsilon": None,
            "num_byzantine": 2,
            "byzantine_clients_1indexed": "1,2",
            "min_paired_seeds": 3,
        },
        "aggregators": [slug for slug, _ in AGGREGATORS],
        "attacks": ATTACKS,
        "seeds": SEEDS,
        "n_cells": len(cells),
        "metrics_reported": [
            "byzantine_tpr", "honest_fpr", "macro_f1",
            "attack_success_rate", "rare_class_recall", "rare_class_aucpr",
        ],
        "metric_interpretation_note": (
            "byzantine_tpr / honest_fpr are only non-trivially meaningful "
            "for the Krum-family aggregators (multi_krum, adaptive_krum, "
            "calibrated_krum), which make an explicit per-client accept/"
            "reject decision (krum_selected / krum_detected_byzantine "
            "columns in results_*.csv). fedavg, median, and trimmed_mean "
            "never reject a client -- they aggregate everyone (median/"
            "trimmed_mean do so robustly, fedavg does not) -- so their "
            "TPR/FPR will read exactly 0%/0% BY CONSTRUCTION, not because "
            "the attack went undetected. This is a correct result, not a "
            "bug, and must be stated explicitly wherever Table 2 is "
            "written up: for those three aggregators, robustness shows "
            "up in Macro-F1 and Attack Success Rate degradation, not in "
            "TPR/FPR."
        ),
        "cells": cells,
    }
    return manifest


if __name__ == "__main__":
    manifest = build()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(manifest['cells'])} E2 cells to {OUT_PATH}")
