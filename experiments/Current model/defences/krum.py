import math

import numpy as np


def adaptive_multi_krum(all_params, weights, num_byzantine=2, k=2.5, method="mad",
                         min_keep_fraction=0.5, return_diagnostics=False):
    """
    Adaptive Multi-Krum via Dynamic Thresholding (MAD / Z-Score).

    Companion to multi_krum() in this file — same NaN guard, same flatten
    convention, same (aggregated_params, selected_indices) contract. Do
    NOT replace multi_krum() with this; run them as separate conditions
    and compare (fixed-m Krum vs adaptive-threshold Krum).

    The difference from multi_krum(): instead of always keeping a fixed
    count `m` of the lowest-scoring clients, this computes each client's
    standard Krum distance score, then classifies a client as an outlier
    based on how far its score sits from the round's own score
    distribution (median + k * MAD, or mean + k * std). Consequences:

        - All-honest round, even with high non-IID variance -> the score
          distribution is tight, threshold stays close to the pack,
          ~0 clients dropped.
        - A cluster of extreme Byzantine clients -> their scores sit far
          out in the tail, threshold sits between the honest cluster and
          the attackers, all of them get dropped regardless of how many
          there are (not capped at a fixed count the way m is).

    Parameters
    ----------
    all_params        : list of parameter lists, one per client
    weights            : list of sample counts per client
    num_byzantine      : assumed number of Byzantine clients (f). Used
                          ONLY to size the neighbour count for the
                          underlying Krum score — exactly like
                          multi_krum(), this is fixed at n - f - 2 and is
                          NOT the thing that decides how many clients get
                          dropped. Passing a wrong f still gives usable
                          scores; it does not cap or floor the outlier
                          count the way it effectively does in multi_krum().
    k                   : sensitivity multiplier on the dispersion term.
                          threshold = center(scores) + k * spread(scores).
                          This is the actual tuning knob for this
                          function — the equivalent of m in multi_krum().
                          Larger k → more permissive (fewer clients
                          dropped). Start around 2.5–3.0 and sweep, the
                          same way you swept m=6 vs m=7.
    method              : "mad" (default) or "zscore".
                          "mad"    → threshold = median(S) + k * 1.4826 * MAD(S)
                          "zscore" → threshold = mean(S) + k * std(S)
                          MAD is robust to several simultaneous outliers
                          dragging the center/spread estimate itself;
                          z-score is not (a handful of extreme Byzantine
                          scores inflate mean and std, potentially hiding
                          themselves). Default to "mad"; keep "zscore"
                          available for an ablation comparing the two.
    min_keep_fraction   : safety floor on how many FINITE clients can be
                          dropped by thresholding, expressed as a
                          fraction of the finite client count. Prevents a
                          single wild non-IID round from thresholding out
                          more than a small minority and starving
                          FedProx. Set to 0 to disable (not recommended
                          for n<=10).
    return_diagnostics  : if True, returns a third element — a dict with
                          the raw per-position Krum scores plus the
                          threshold/center/spread used to compute the
                          cutoff this round. Needed for the epsilon-sweep
                          diagnostic (krum_score_ratio = mean Byzantine
                          score / mean honest score) — the caller (which
                          knows original client IDs, this function does
                          not) is responsible for mapping diagnostics
                          "scores" positions back to client IDs via
                          accepted_client_indices and splitting into
                          Byzantine vs honest, exactly as krum_selected_ids
                          / krum_discarded_ids are already computed at the
                          main.py call site. Default False preserves the
                          original 2-tuple return for existing callers.

    Algorithm
    ---------
    1. Flatten each client's full parameter set into a single vector
    2. Quarantine any client whose update contains NaN or Inf values —
       identical NaN guard to multi_krum(): score=inf, always dropped.
    3. If too few finite clients remain to compute meaningful scores,
       fall back to averaging all finite clients (same fallback as
       multi_krum()).
    4. Compute pairwise squared Euclidean distances between finite clients
    5. Score each client by summing distances to its (n - f - 2) nearest
       neighbours — same fixed theoretical neighbour count as multi_krum().
    6. Compute threshold = center(scores) + k * spread(scores) over the
       finite scores (method = "mad" or "zscore").
    7. Any finite client with score > threshold is dropped. NaN/Inf
       clients are already excluded via step 2 (score=inf always exceeds
       the threshold). Enforce min_keep_fraction floor if triggered.
    8. Return the weighted average of the kept clients.

    Returns
    -------
    (aggregated_params, selected_indices) : tuple, or
    (aggregated_params, selected_indices, diagnostics) if return_diagnostics=True

        aggregated_params  : list[np.ndarray] — weighted average of kept
                             (trusted) clients' parameters
        selected_indices   : list[int] — 0-indexed client indices kept
                             this round. Same semantics as multi_krum()'s
                             selected_indices — check membership against
                             BYZANTINE_CLIENTS for detection rate.
        diagnostics        : dict with keys:
                               "scores"       : list[float], length n,
                                                per-position Krum score
                                                (inf for NaN/Inf-quarantined
                                                or, in the too-few-finite
                                                fallback path, uncomputed
                                                positions).
                               "threshold"    : float or None (None in
                                                the too-few-finite fallback
                                                path, where no threshold
                                                was computed).
                               "center"       : float or None (same caveat).
                               "spread"       : float or None (same caveat).
                               "num_dropped"  : int.
                               "num_nan"      : int — count of NaN/Inf-
                                                quarantined clients this
                                                round (feeds nan_this_round
                                                at the main.py call site).
                               "fallback_triggered" : bool — True if either
                                                the too-few-finite-clients
                                                path or the min_keep_fraction
                                                safety floor path fired.
    """
    n = len(all_params)
    f = num_byzantine
    theoretical_neighbours = n - f - 2

    # HARD requirement: need a positive neighbour count to compute Krum
    # scores at all — below this the distance-sum computation is
    # meaningless (sums zero or negative-count neighbours).
    if theoretical_neighbours <= 0:
        raise ValueError(
            f"Cannot compute Krum scores: n={n}, f={f} → theoretical "
            f"neighbour count={theoretical_neighbours} (n - f - 2). "
            f"Need f <= n - 3."
        )

    # SOFT requirement: Blanchard et al.'s formal Byzantine-robustness
    # guarantee only holds for n >= 2f + 3, i.e. f < n/2 - 1. The function
    # still RUNS below this bound (the hard check above is the only actual
    # blocker) — but detection is no longer theoretically guaranteed at
    # this ratio. Warn loudly instead of raising, since exploring past the
    # guarantee is exactly the point of a Byzantine-fraction sweep — but
    # every result obtained this way needs to be visibly flagged as
    # "beyond the guarantee," not indistinguishable from a validated
    # in-bound result.
    if f >= n / 2 - 1:
        print(f"  ⚠️  WARNING: f={f} Byzantine clients out of n={n} exceeds "
              f"the theoretical Krum safety bound (f < n/2 - 1). "
              f"Byzantine-robustness is NOT guaranteed at this ratio — "
              f"treat this round's detection result as exploratory, not "
              f"a validated robust-detection outcome.")
    if method not in ("mad", "zscore"):
        raise ValueError(f"method={method!r} must be 'mad' or 'zscore'.")

    # ── Step 1: flatten each client's parameters into one long vector ──
    flat = []
    for params in all_params:
        flat.append(np.concatenate([p.flatten() for p in params]))
    flat = np.array(flat)   # shape: (n, total_params)

    # ── Step 2: NaN/Inf guard (identical to multi_krum) ────────────────
    scores       = np.full(n, np.inf)
    nan_clients  = set()
    finite_clients = []

    for i in range(n):
        if not np.all(np.isfinite(flat[i])):
            nan_clients.add(i)
            print(f"  ⚠  Client {i+1} update contains NaN/Inf "
                  f"— quarantined (score=inf, will be discarded)")
        else:
            finite_clients.append(i)

    # ── Step 3: fallback if too few finite clients to score meaningfully ──
    # Mirrors multi_krum()'s fallback trigger condition (needs at least
    # theoretical_neighbours + 1 finite clients to compute a neighbour sum).
    min_required = theoretical_neighbours + 1
    if len(finite_clients) < min_required:
        print(f"  ⚠  Only {len(finite_clients)} finite clients available, "
              f"need at least {min_required} to score. Averaging all finite clients.")
        finite_params  = [all_params[i] for i in finite_clients]
        finite_weights = [weights[i]    for i in finite_clients]
        total = sum(finite_weights)
        result = []
        for layer_idx in range(len(finite_params[0])):
            layer_avg = sum(
                p[layer_idx] * (w / total)
                for p, w in zip(finite_params, finite_weights)
            )
            result.append(layer_avg)

        if return_diagnostics:
            diagnostics = {
                "scores": scores.tolist(),   # all inf — no scores computed
                "threshold": None,
                "center": None,
                "spread": None,
                "num_dropped": n - len(finite_clients),
                "num_nan": len(nan_clients),
                "fallback_triggered": True,
            }
            return result, finite_clients, diagnostics
        return result, finite_clients

    # ── Step 4: pairwise squared distances (finite clients only) ───────
    distances = np.zeros((n, n))
    for i in finite_clients:
        for j in finite_clients:
            if j <= i:
                continue
            d = float(np.sum((flat[i] - flat[j]) ** 2))
            distances[i][j] = d
            distances[j][i] = d

    # ── Step 5: score each finite client (fixed theoretical neighbour count) ──
    for i in finite_clients:
        finite_distances = sorted(
            distances[i][j] for j in finite_clients if j != i
        )
        kk = min(theoretical_neighbours, len(finite_distances))
        scores[i] = sum(finite_distances[:kk])

    finite_scores = np.array([scores[i] for i in finite_clients])

    # ── Step 6: dynamic threshold over finite scores ────────────────────
    if method == "mad":
        center = float(np.median(finite_scores))
        mad = float(np.median(np.abs(finite_scores - center)))
        spread = 1.4826 * mad
    else:  # zscore
        center = float(np.mean(finite_scores))
        spread = float(np.std(finite_scores))

    threshold = center + k * spread if spread > 0 else center + 1e-9

    # ── Step 7: threshold + safety floor ────────────────────────────────
    kept    = [i for i in finite_clients if scores[i] <= threshold]
    dropped = [i for i in finite_clients if scores[i] > threshold] + sorted(nan_clients)

    fallback_triggered = False
    min_keep = int(np.ceil(min_keep_fraction * len(finite_clients))) if min_keep_fraction > 0 else 0
    if len(kept) < min_keep:
        print(f"  ⚠  Thresholding kept only {len(kept)}/{len(finite_clients)} finite "
              f"clients, below floor of {min_keep}. Falling back to lowest-{min_keep}-score clients.")
        fallback_triggered = True
        ranked_finite = sorted(finite_clients, key=lambda i: scores[i])
        kept = ranked_finite[:min_keep]
        dropped = [i for i in range(n) if i not in kept]

    ranked = np.argsort(scores)  # ascending, for display only

    print(f"\n  Adaptive Multi-Krum ({method}): n={n}, f={f} assumed, "
          f"neighbours={theoretical_neighbours}, k={k}")
    if nan_clients:
        print(f"  Quarantined (NaN/Inf): clients {sorted([c+1 for c in nan_clients])}")
    print(f"  center={center:.4e}  spread={spread:.4e}  threshold={threshold:.4e}")
    print(f"  Scores (lower = more trusted):")
    for i in ranked:
        tag = "✓ SELECTED " if i in kept else "✗ DISCARDED"
        score_str = f"{scores[i]:.4e}" if np.isfinite(scores[i]) else "NaN/Inf"
        print(f"    Client {i+1:>2}  score={score_str:<14}  {tag}")
    print(f"  Kept {len(kept)}/{n} clients (dropped {len(dropped)})")

    # ── Step 8: weighted average of kept clients ─────────────────────────
    kept_params  = [all_params[i] for i in kept]
    kept_weights = [weights[i]    for i in kept]
    total = sum(kept_weights)

    result = []
    for layer_idx in range(len(kept_params[0])):
        layer_avg = sum(
            p[layer_idx] * (w / total)
            for p, w in zip(kept_params, kept_weights)
        )
        result.append(layer_avg)

    if return_diagnostics:
        diagnostics = {
            "scores": scores.tolist(),
            "threshold": threshold,
            "center": center,
            "spread": spread,
            "num_dropped": len(dropped),
            "num_nan": len(nan_clients),
            "fallback_triggered": fallback_triggered,
        }
        return result, kept, diagnostics
    return result, kept


# ---------------------------------------------------------------------------
# BAS1 (Issue 3) -- restored/added robust-aggregation baselines.
# adaptive_multi_krum() above is UNCHANGED (byte-for-byte, per the issue's
# explicit constraint) -- everything below is new or restored.
# ---------------------------------------------------------------------------

def fedavg(all_params, weights=None):
    """
    Explicit, no-defence FedAvg baseline. Functionally identical to
    main.py's pre-existing fedprox_aggregate() -- same formula, made a
    first-class, importable, dispatch-table aggregator instead of
    living only as a main.py-local helper.

    weights=None -> unweighted mean (every client weight=1). This is
    the one behavioral difference from fedprox_aggregate(), which
    always requires an explicit weights list.
    """
    n = len(all_params)
    if weights is None:
        weights = [1.0] * n
    total = sum(weights)
    result = []
    for layer_idx in range(len(all_params[0])):
        layer_avg = sum(
            p[layer_idx] * (w / total)
            for p, w in zip(all_params, weights)
        )
        result.append(layer_avg)
    return result


def coordinate_median(all_params, weights=None):
    """
    Per-coordinate median across clients (Yin et al. 2018 baseline).

    JUDGMENT CALL: `weights` is accepted only for dispatch-table
    signature symmetry -- NOT used. Plain coordinate-wise median is
    the standard literature definition of this baseline; a weighted
    median is a different, not-specified-here construct.

    No NaN guard: median's whole point is tolerating up to
    floor(n/2)-1 arbitrarily-valued clients without needing to detect
    them first, unlike Krum. A literal NaN will still sort
    inconsistently -- if that becomes a live problem, add an explicit
    isfinite pre-filter; not reproduced here to keep this matching the
    literature baseline's own contract.
    """
    n_layers = len(all_params[0])
    result = []
    for layer_idx in range(n_layers):
        stacked = np.stack([p[layer_idx] for p in all_params], axis=0)
        result.append(
            np.median(stacked, axis=0).astype(all_params[0][layer_idx].dtype)
        )
    return result


def trimmed_mean(all_params, weights=None, beta=0.1):
    """
    Per-coordinate trimmed mean: sort the n client values at each
    coordinate, drop floor(beta * n) from EACH tail, average the rest.

    JUDGMENT CALLS:
      1. Trim count = floor(beta*n) per tail (conservative; never
         trims more than beta specifies). With n=5, beta=0.1 (the
         issue's stated default) floors to 0 -- i.e. beta=0.1 on 5
         clients is a no-op, not a bug. The test fixture uses
         beta=0.2 so the trim is non-trivial and hand-verifiable; the
         module default stays 0.1 per the issue text.
      2. If 2*trim_count >= n, falls back to plain fedavg() rather
         than raising or returning garbage.
      3. `weights` applies AFTER trimming, to survivors only (weighted
         mean of the untrimmed remainder) -- trimming decides "which
         values are outliers" by rank alone; weighting then decides
         how much each survivor counts.
    """
    n = len(all_params)
    trim_count = int(np.floor(beta * n))
    if weights is None:
        weights = [1.0] * n

    if 2 * trim_count >= n:
        return fedavg(all_params, weights)

    n_layers = len(all_params[0])
    result = []
    for layer_idx in range(n_layers):
        stacked = np.stack([p[layer_idx] for p in all_params], axis=0)
        w = np.array(weights, dtype=np.float64)

        order = np.argsort(stacked, axis=0)
        keep_idx = order[trim_count:n - trim_count]

        kept_vals = np.take_along_axis(stacked, keep_idx, axis=0)
        kept_w = w[keep_idx]
        layer_result = np.sum(kept_vals * kept_w, axis=0) / np.sum(kept_w, axis=0)
        result.append(layer_result.astype(all_params[0][layer_idx].dtype))
    return result


def multi_krum(all_params, weights, num_byzantine=2, m=None,
               return_diagnostics=False):
    """
    Fixed-M Multi-Krum (Blanchard, Guerraoui, Stainer 2017 canonical
    form). RESTORED per Issue 3 (BAS1) Task 1.

    JUDGMENT CALL (documented, not silent): the issue text proposes
    signature multi_krum(updates_list, n_clients, f_byzantine,
    m_select=None), but main.py's real, live call site
    (`global_params, selected_positions = multi_krum(accepted_params,
    accepted_weights, num_byzantine=NUM_BYZANTINE, m=effective_m)`)
    already uses (all_params, weights, num_byzantine=, m=). Restoring
    the issue's proposed signature would silently break that existing
    call site. This restores multi_krum to match the REAL call site's
    convention, mirroring adaptive_multi_krum's own style since it's
    the sibling function in this same file.

    Same score (sum of distances to the n-f-2 theoretically nearest
    other clients) and the same NaN/Inf quarantine + too-few-finite
    fallback as adaptive_multi_krum() in this file, reused for
    consistency. UNLIKE adaptive_multi_krum, which keeps however many
    clients clear a dynamic per-round threshold, this always keeps
    exactly `m` clients -- the m lowest-scoring ones.

    Parameters
    ----------
    all_params : list of parameter lists, one per client.
    weights : list of sample counts per client -- REQUIRED, since the
        aggregation step is a weighted average of the m kept clients.
    num_byzantine : assumed f, sizes the neighbour count (n - f - 2).
    m : how many lowest-scoring clients to keep. If None, defaults to
        max(1, n - f - 1) (Blanchard's suggested m = n - f). main.py's
        real call site never relies on this default -- it always
        passes m=effective_m explicitly -- this default just makes the
        function independently testable.
    return_diagnostics : default False preserves the exact 2-tuple
        return main.py's real call site expects.

    Returns
    -------
    (aggregated_params, selected_indices), or a 3-tuple with
    diagnostics -- same shape as adaptive_multi_krum().
    """
    n = len(all_params)
    f = num_byzantine
    theoretical_neighbours = n - f - 2
    if theoretical_neighbours <= 0:
        raise ValueError(
            f"Cannot compute Krum scores: n={n}, f={f} -> theoretical "
            f"neighbour count={theoretical_neighbours} (n - f - 2). "
            f"Need f <= n - 3."
        )
    if f >= n / 2 - 1:
        print(f"  WARNING: f={f} Byzantine clients out of n={n} exceeds "
              f"the theoretical Krum safety bound (f < n/2 - 1). "
              f"Byzantine-robustness is NOT guaranteed at this ratio.")

    if m is None:
        m = max(1, n - f - 1)
    if m > n:
        raise ValueError(f"m={m} cannot exceed n={n}.")

    flat = np.array([np.concatenate([p.flatten() for p in params])
                      for params in all_params])

    scores = np.full(n, np.inf)
    nan_clients = set()
    finite_clients = []
    for i in range(n):
        if not np.all(np.isfinite(flat[i])):
            nan_clients.add(i)
            print(f"  Client {i+1} update contains NaN/Inf -- quarantined "
                  f"(score=inf, will be discarded)")
        else:
            finite_clients.append(i)

    min_required = theoretical_neighbours + 1
    if len(finite_clients) < min_required:
        print(f"  Only {len(finite_clients)} finite clients available, "
              f"need at least {min_required} to score. Averaging all finite clients.")
        finite_params = [all_params[i] for i in finite_clients]
        finite_weights = [weights[i] for i in finite_clients]
        result = fedavg(finite_params, finite_weights)
        if return_diagnostics:
            diag = {"scores": scores.tolist(), "threshold": None, "m": m,
                    "num_dropped": n - len(finite_clients),
                    "num_nan": len(nan_clients), "fallback_triggered": True}
            return result, finite_clients, diag
        return result, finite_clients

    distances = np.zeros((n, n))
    for i in finite_clients:
        for j in finite_clients:
            if j <= i:
                continue
            d = float(np.sum((flat[i] - flat[j]) ** 2))
            distances[i][j] = d
            distances[j][i] = d

    for i in finite_clients:
        finite_distances = sorted(distances[i][j] for j in finite_clients if j != i)
        kk = min(theoretical_neighbours, len(finite_distances))
        scores[i] = sum(finite_distances[:kk])

    m_eff = min(m, len(finite_clients))
    # Stable sort by (score, index) -> deterministic tie-breaking:
    # lowest original index wins any exact-score tie.
    ranked_finite = sorted(finite_clients, key=lambda i: (scores[i], i))
    kept = sorted(ranked_finite[:m_eff])
    dropped = [i for i in range(n) if i not in kept]

    print(f"\n  Multi-Krum (fixed-m): n={n}, f={f} assumed, "
          f"neighbours={theoretical_neighbours}, m={m_eff}")
    if nan_clients:
        print(f"  Quarantined (NaN/Inf): clients {sorted(c+1 for c in nan_clients)}")
    for i in sorted(finite_clients, key=lambda i: scores[i]):
        tag = "SELECTED " if i in kept else "DISCARDED"
        print(f"    Client {i+1:>2}  score={scores[i]:.4e}  {tag}")
    print(f"  Kept {len(kept)}/{n} clients (dropped {len(dropped)})")

    kept_params = [all_params[i] for i in kept]
    kept_weights = [weights[i] for i in kept]
    result = fedavg(kept_params, kept_weights)

    if return_diagnostics:
        diag = {"scores": scores.tolist(), "m": m_eff,
                "num_dropped": len(dropped), "num_nan": len(nan_clients),
                "fallback_triggered": False}
        return result, kept, diag
    return result, kept


# BAS1 (Issue 3) Tests Required section imports a bare name `krum`
# directly, separate from `multi_krum`:
#   from ... import fedavg, krum, multi_krum, coordinate_median, trimmed_mean, adaptive_multi_krum
# This is a plain alias to the SAME function object -- not a second,
# potentially-divergent implementation. "krum" and "multi_krum" must
# never drift apart; if you need to change fixed-M Multi-Krum's
# behavior, change multi_krum() above and this alias follows for free.
krum = multi_krum


# ===========================================================================
# Issue 4 -- DP- and Heterogeneity-Calibrated Adaptive Krum
# ===========================================================================
#
# JUDGMENT CALL #1 (signature): the ticket's proposed signature is
# calibrated_adaptive_multi_krum(updates_list, n_clients, f_byzantine,
# per_client_metadata, expected_dp_noise_std, expected_hetero_std,
# baseline_honest_std_from_prior_round, mad_multiplier_k, use_dp_
# calibration, use_hetero_calibration). This codebase's REAL, live
# convention for every other aggregator in this file (adaptive_multi_krum,
# multi_krum) is (all_params, weights, num_byzantine=, k=, method=,
# min_keep_fraction=, return_diagnostics=) -- n_clients is redundant with
# len(all_params), and f_byzantine == num_byzantine. This function follows
# THIS file's real, established convention instead of the ticket's
# proposed one, exactly as multi_krum()'s own docstring already did for
# the same reason (its own "JUDGMENT CALL" note). expected_dp_noise_std /
# expected_hetero_std are not separately-injected callables here -- they
# are implemented as the two module-level functions dp_variance() and
# hetero_variance() below, called internally; this keeps the calibration
# math auditable in one place instead of behind an opaque callback the
# caller could silently swap per-run without it showing up in any log.
#
# JUDGMENT CALL #2 (per_client_metadata schema, a REAL deviation from the
# ticket's literal text): the ticket's schema is
#   {client_id: {"n_samples": int, "class_entropy": float,
#                "epsilon": float or None}}
# "epsilon" alone is NOT sufficient to compute a noise-variance
# contribution -- Opacus's actual per-step noise std is sigma * C, where
# sigma (the noise MULTIPLIER) is what the RDP accountant solves for
# against (target_epsilon, target_delta, sample_rate, steps); recovering
# sigma FROM epsilon inside this function would mean re-invoking Opacus's
# own accountant a second time, redundantly, and risking a different
# answer from whatever sigma the real client actually trained under.
# This codebase ALREADY computes and logs the real per-client sigma
# directly (main.py: dp_noise_multiplier_by_client[i], sourced from
# dp_persistent_client_state.py's state["sigma"] = dp_optimizer.
# noise_multiplier) -- so per_client_metadata here is REQUIRED to carry
# "noise_multiplier" (the real sigma), with "epsilon" kept as an OPTIONAL
# informational field only (not used in any calculation). Flagged here,
# explicitly, rather than silently redefining the schema without saying
# so:
#   per_client_metadata = {
#       client_id: {
#           "n_samples":       int,
#           "class_entropy":   float,
#           "noise_multiplier": float or None,   # REAL sigma; None/absent
#                                                  # -> treated as a non-DP
#                                                  # client (var_dp=0 for
#                                                  # any pair involving it)
#           "epsilon":         float or None,     # informational only
#       },
#       ...
#   }


def dp_variance(sigma_i, sigma_j, dp_max_grad_norm, dp_batch_size_i,
                 dp_batch_size_j, total_params):
    """
    Defensible closed-form starting point (per the handoff doc), derived
    against THIS project's actual DP-SGD mechanics, not copied from an
    unrelated paper:

    Opacus's DP-SGD adds i.i.d. Gaussian noise with std (sigma * C) to
    the SUMMED per-sample clipped gradients for a batch (C =
    dp_max_grad_norm, confirmed 1.0 in this codebase's config). After
    averaging over a batch of size B, the noise contribution to a single
    client's REPORTED per-parameter update has variance
    approximately (sigma * C)^2 / B (standard DP-SGD noise-averaging
    result). Krum's raw_dist_ij is a SUM of squared per-parameter
    differences over the full flattened parameter vector (total_params
    dimensions, not a normalized/per-parameter distance) -- so the
    noise's EXPECTED contribution to that summed squared distance scales
    linearly with total_params (D independent per-parameter noise terms,
    each contributing its own variance to the corresponding squared-
    difference coordinate).

    For a PAIR of clients (i, j) being compared, their noise draws are
    independent, so the variance of (noise_i - noise_j) at each
    parameter is var(noise_i) + var(noise_j) -- summed over all
    total_params dimensions:

        var_dp_ij = total_params * [ (sigma_i*C)^2 / B_i
                                    + (sigma_j*C)^2 / B_j ]

    A client with sigma=None (no real DP accounting -- e.g. a Byzantine
    client in this codebase's build_dp_client_states(), which excludes
    Byzantine clients entirely) contributes 0 to this term, not a
    fabricated default -- absence of DP noise is a real, valid state,
    not missing data.

    CROSS-CHECK REQUIRED (per the handoff doc, NOT performed here -- this
    function is pure and has no access to Task 1's logged data): before
    trusting this formula, verify against Task 1's real per_client_krum_
    scores.csv that higher noise_multiplier sigma (i.e. tighter epsilon)
    correlates with higher measured raw Krum score dispersion among
    honest clients, in the direction this formula predicts. If it does
    not, this formula is wrong or incomplete for this model's actual
    per-parameter gradient scale -- do not simply proceed as if this
    derivation is validated.
    """
    var_i = 0.0 if sigma_i is None else (sigma_i * dp_max_grad_norm) ** 2 / dp_batch_size_i
    var_j = 0.0 if sigma_j is None else (sigma_j * dp_max_grad_norm) ** 2 / dp_batch_size_j
    return total_params * (var_i + var_j)


def hetero_variance(meta_i, meta_j, alpha_dirichlet, fit_coeffs=None):
    """
    REVISED (post-hoc fix, see fit_hetero_variance_regression() below for
    the full rationale): the ORIGINAL version of this function applied
    coefficients fit against each client's own ABSOLUTE raw_krum_score
    (a summed-over-neighbours quantity, whose neighbour count depends on
    f_byzantine) to a PAIRWISE DIFFERENCE input at inference time -- two
    stacked mismatches (absolute-vs-diff, and aggregate-sum-scale-vs-
    single-pair-scale) that made this term behave inconsistently across
    different f_byzantine settings (confirmed: clean at f=2, the setting
    the original fit happened to be built from, broken at f=1 and f=3).

    THIS version's fit_coeffs are produced by a genuinely pairwise,
    per-neighbour-normalized, LOG-SCALE regression (see
    fit_hetero_variance_regression()) -- i.e. `fit_coeffs` now means:

        log1p(|per_neighbour_score_i - per_neighbour_score_j|) ~=
            intercept + coef_n_samples_diff * |n_samples_i - n_samples_j|
                      + coef_entropy_diff   * |entropy_i - entropy_j|

    which is back-transformed here to a variance-scale prediction (see
    below). A log-scale fit was necessary, not just a style choice: the
    same real data fit on the RAW (non-log) pairwise scale produces a
    LARGE, WRONG-SIGNED entropy coefficient with an implausibly high R^2
    (an outlier-dominated OLS artifact -- raw Krum-distance pairs are
    extremely heavy-tailed: mean pairwise diff ~10-15x the median in the
    real data this was checked against), while the log-scale fit on the
    identical data gives small, correctly-signed (positive) coefficients
    for BOTH terms and a moderate, non-suspicious R^2. Do not refit this
    on the raw (non-log) scale without re-confirming that heavy-tail
    problem is gone.

    `fit_coeffs` schema (produced by fit_hetero_variance_regression()):
        {"intercept": b0, "coef_n_samples_diff": b1,
         "coef_entropy_diff": b2, "r_squared": r2, "n_pairs_fit": n,
         "fit_target": "log1p(abs(pairwise per-neighbour raw_krum_score diff))",
         "schema_version": 2}
    `schema_version` is checked explicitly below -- an old (schema_version
    1 / absent) coeffs file must not be silently applied here, since it
    was fit for a different quantity entirely (see above) and would
    silently reproduce the exact bug this rewrite fixes.

    fit_coeffs=None (the honest default until a real sanity run has
    actually produced data to fit against): returns 0.0 and prints a
    ONE-TIME warning (module-level flag, not per-call-spam) rather than
    fabricating a plausible-looking constant. This means
    use_hetero_calibration=True with fit_coeffs=None is silently a
    no-op for the hetero term specifically (dp term, if enabled, is
    unaffected) -- callers should treat "fit_coeffs is None" as
    equivalent to hetero calibration not actually being available yet,
    not as hetero calibration having been validated to contribute
    nothing.

    alpha_dirichlet is accepted for signature completeness /
    forward-compatibility (a future refit could stratify by alpha) but
    is NOT currently part of the fitted formula -- the regression is fit
    on n_samples/class_entropy DIFFERENCES directly, which already
    reflect whatever alpha produced them; using alpha as a second
    explicit regressor as well as the primary heterogeneity variable
    would double-count the same underlying effect. Not a bug, but do
    not extend this to use alpha_dirichlet without first checking
    whether it adds real explanatory power beyond n_samples/entropy
    differences.
    """
    if fit_coeffs is None:
        if not hetero_variance._warned:
            print("  ⚠️  hetero_variance(): fit_coeffs is None -- no "
                  "empirical regression has been fit against real "
                  "logged data yet. Returning 0.0 for every pair "
                  "(hetero calibration term is currently a no-op, not "
                  "validated-to-be-zero). Run "
                  "fit_hetero_variance_regression() against a real "
                  "per_client_krum_scores.csv before trusting "
                  "use_hetero_calibration=True's results.")
            hetero_variance._warned = True
        return 0.0

    schema_version = fit_coeffs.get("schema_version", 1)
    assert schema_version == 2, (
        f"hetero_variance() received fit_coeffs with schema_version="
        f"{schema_version!r} (expected 2). schema_version 1 (or absent, "
        f"the original format) was fit against each client's ABSOLUTE "
        f"raw_krum_score on the raw (non-log) scale -- applying those "
        f"coefficients to this function's pairwise-diff/log-scale "
        f"formula would silently reproduce the exact f-dependent bug "
        f"this rewrite fixes. Re-fit with the current "
        f"fit_hetero_variance_regression() to get a schema_version=2 "
        f"coeffs file."
    )

    n_diff = abs(meta_i["n_samples"] - meta_j["n_samples"])
    e_diff = abs(meta_i["class_entropy"] - meta_j["class_entropy"])
    predicted_log = (fit_coeffs["intercept"]
                      + fit_coeffs["coef_n_samples_diff"] * n_diff
                      + fit_coeffs["coef_entropy_diff"] * e_diff)
    # Back-transform: the fit predicts log1p(expected |pairwise diff|),
    # i.e. an expected-STD-scale quantity (mean absolute pairwise
    # difference is a standard robust proxy for spread). Squaring it
    # converts that std-scale quantity into the VARIANCE this function's
    # contract requires (it feeds into sqrt(var_dp + var_hetero + ...)
    # downstream, which expects a variance, not a std). Clip at 0 before
    # squaring (expm1 of a very negative prediction could in principle
    # dip fractionally below 0 from floating-point error at the low end;
    # squaring first would hide that instead of clamping it).
    predicted_absdiff = max(0.0, math.expm1(predicted_log))
    return predicted_absdiff ** 2


hetero_variance._warned = False


def fit_hetero_variance_regression(per_client_krum_scores_csv):
    """
    Fits hetero_variance()'s regression against a REAL Task 1
    per_client_krum_scores.csv (produced by main.py's Issue 4 Task 1
    logger). Uses ONLY rows where ground_truth_client_label == "honest"
    (per the ticket: this is meant to model legitimate heterogeneity-
    induced dispersion, not attacker behaviour), and only rows from
    runs where use_dp_calibration was OFF / DP was inactive for that
    row's client, so the fitted "hetero" term is not itself
    contaminated by an unremoved DP-noise contribution -- caller is
    responsible for filtering the input CSV to DP-inactive honest rows
    before calling this (kept as an explicit precondition rather than
    silently guessed here, since "was DP active for this row" is not a
    column in Task 1's schema as specified).

    REWRITTEN (schema_version 2) to fix two stacked problems found in
    the original (schema_version 1, absolute-value, raw-scale) version:

    Problem 1 -- wrong quantity fit vs. wrong quantity applied. The
    original version regressed each client's own ABSOLUTE raw_krum_score
    (a value SUMMED over that client's n-f-2 nearest neighbours, so its
    raw magnitude depends on f_byzantine) against that client's own
    ABSOLUTE n_samples/class_entropy -- then hetero_variance() applied
    those coefficients to a PAIRWISE DIFFERENCE of two different
    clients' n_samples/entropy. Confirmed against real 5-seed f-sweep
    data: this made the hetero term behave correctly at f=2 (the
    setting the original fit happened to come from) but break at f=1
    and f=3 -- an f-dependent scale mismatch, not a random bug.

    Fix: this version (a) groups honest rows by (round_id,
    alpha_dirichlet_config, active_epsilon_config) -- i.e. by "the same
    real cohort of honest clients that round", (b) derives each round's
    neighbour count directly from that round's own honest headcount
    (honest_count - 2 -- exactly Krum's n-f-2, since every OTHER client
    that round is by construction either honest or the f byzantine
    clients being assumed-excluded; no external f_byzantine metadata
    needed), (c) divides each client's raw_krum_score by that count to
    approximate a single-pair distance rather than an f-dependent sum,
    and (d) regresses the PAIRWISE DIFFERENCE of that per-neighbour
    score between every pair of honest clients in the same cohort
    against |n_samples_i - n_samples_j| and |entropy_i - entropy_j| --
    i.e. fits and is applied on the exact same quantity.

    Problem 2 -- raw-scale OLS is outlier-dominated on this data. Fit on
    the raw (non-log) pairwise-diff scale, the same real data produces a
    large, WRONG-SIGNED coef_entropy_diff with an implausibly high R^2 --
    checked directly: mean pairwise diff was ~15x the median in the real
    data this was fit against, i.e. a strongly right-skewed target, which
    plain least-squares fits by chasing the few extreme pairs at the
    expense of getting the sign wrong for the typical case. Fix: fit
    log1p(pairwise diff) instead. On the identical real data this flips
    both coefficients to the correct (positive) sign and gives a modest,
    non-suspicious R^2 instead of an inflated one. hetero_variance()
    back-transforms (expm1, then squares to convert the std-scale
    prediction into the variance this function's contract requires) --
    see that function's docstring.

    Returns {"intercept", "coef_n_samples_diff", "coef_entropy_diff",
    "r_squared", "n_pairs_fit", "fit_target", "schema_version": 2} --
    reports fit quality HONESTLY; a poor R^2 must be reported as such by
    the caller, not hidden. `schema_version` lets hetero_variance() refuse
    to silently apply an old-format coeffs file (see its docstring).
    """
    import csv as _csv
    from collections import defaultdict

    rounds = defaultdict(list)
    with open(per_client_krum_scores_csv, newline="") as f:
        for row in _csv.DictReader(f):
            if row["ground_truth_client_label"] != "honest":
                continue
            try:
                score = float(row["raw_krum_score"])
            except (ValueError, TypeError):
                continue
            if not np.isfinite(score) or score <= 0:
                continue
            key = (row["round_id"], row["alpha_dirichlet_config"],
                   row["active_epsilon_config"])
            rounds[key].append({
                "score": score,
                "n_samples": float(row["client_n_samples"]),
                "entropy": float(row["client_class_entropy"]),
            })

    diff_targets, ns_diffs, ent_diffs = [], [], []
    for key, clients in rounds.items():
        neighbours = len(clients) - 2   # Krum's n-f-2, derived directly
                                          # from this round's real honest
                                          # headcount -- no external
                                          # f_byzantine metadata needed.
        if neighbours < 1:
            continue   # too few honest clients logged this round to
                        # form a meaningful neighbour-normalized score
        for c in clients:
            c["score_per_neighbour"] = c["score"] / neighbours
        for i in range(len(clients)):
            for j in range(i + 1, len(clients)):
                a, b = clients[i], clients[j]
                diff_targets.append(
                    abs(a["score_per_neighbour"] - b["score_per_neighbour"])
                )
                ns_diffs.append(abs(a["n_samples"] - b["n_samples"]))
                ent_diffs.append(abs(a["entropy"] - b["entropy"]))

    if len(diff_targets) < 10:
        raise ValueError(
            f"Only {len(diff_targets)} usable honest-honest pairs found "
            f"in {per_client_krum_scores_csv!r} -- too few to fit a "
            f"defensible regression. Re-run the sanity sweep (more "
            f"rounds/seeds) before fitting."
        )

    y_log = np.log1p(np.array(diff_targets))
    X = np.column_stack([
        np.ones(len(y_log)),
        np.array(ns_diffs),
        np.array(ent_diffs),
    ])
    coeffs, residuals, rank, sv = np.linalg.lstsq(X, y_log, rcond=None)
    y_pred = X @ coeffs
    ss_res = float(np.sum((y_log - y_pred) ** 2))
    ss_tot = float(np.sum((y_log - y_log.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    if r_squared > 0.95:
        print(f"  ⚠️  fit_hetero_variance_regression(): r_squared="
              f"{r_squared:.4f} is suspiciously high for this kind of "
              f"noisy, real-world data -- worth double-checking this "
              f"isn't the same outlier-domination artifact this rewrite "
              f"was meant to fix (e.g. too few distinct rounds/cohorts "
              f"pooled) before trusting it.")

    return {
        "intercept": float(coeffs[0]),
        "coef_n_samples_diff": float(coeffs[1]),
        "coef_entropy_diff": float(coeffs[2]),
        "r_squared": r_squared,
        "n_pairs_fit": len(diff_targets),
        "fit_target": "log1p(abs(pairwise per-neighbour raw_krum_score diff))",
        "schema_version": 2,
    }


def calibrated_adaptive_multi_krum(all_params, weights, per_client_metadata,
                                    num_byzantine=2, k=2.5, method="mad",
                                    min_keep_fraction=0.5,
                                    baseline_honest_std_from_prior_round=None,
                                    dp_max_grad_norm=1.0,
                                    alpha_dirichlet=0.7,
                                    hetero_fit_coeffs=None,
                                    use_dp_calibration=True,
                                    use_hetero_calibration=True,
                                    return_diagnostics=False):
    """
    DP- and Heterogeneity-Calibrated Adaptive Multi-Krum (Issue 4).

    Same NaN/Inf quarantine, too-few-finite fallback, and finite-client
    bookkeeping as adaptive_multi_krum() above (deliberately duplicated,
    not refactored to share code with it in this revision -- adaptive_
    multi_krum() is explicitly required to stay UNCHANGED per Issue 3's
    constraint, and inserting a shared helper touches its file region).

    Algorithm -- ONE DELIBERATE, FLAGGED CORRECTION to the ticket's
    literal pseudocode, everything else matching it exactly (using
    this file's established (all_params, weights, num_byzantine=)
    parameter convention instead of the ticket's proposed one -- see
    the module-level JUDGMENT CALL notes above):

        for each client pair i, j (finite clients only):
            raw_dist_ij = L2(flat_i, flat_j)   [SQUARED, matches multi_krum's
                                                 and adaptive_multi_krum's own
                                                 convention in this same file]
            var_dp     = dp_variance(...)      if use_dp_calibration else 0
            var_hetero = hetero_variance(...)  if use_hetero_calibration else 0
            expected_var_ij = var_dp + var_hetero
                              + baseline_honest_std_from_prior_round**2
            calibrated_dist_ij = raw_dist_ij / (expected_var_ij + EPS)

    CORRECTION FLAGGED: the ticket's literal pseudocode divides
    raw_dist_ij by expected_STD_ij (i.e. sqrt(var_dp + var_hetero +
    baseline_std**2)), not expected_VARIANCE_ij. That is a units
    mismatch given THIS codebase's established convention (confirmed
    in multi_krum()/adaptive_multi_krum() above: `d = np.sum((flat[i]
    - flat[j]) ** 2)` -- raw_dist is already a SQUARED distance, i.e.
    variance-like units). Dividing a squared (variance-scale) quantity
    by a std-scale quantity does not produce a dimensionless,
    comparable-across-pairs ratio -- verified empirically against
    tests/test_calibrated_krum.py's fixture: dividing by std left the
    honest-heterogeneous clients' calibrated scores several times
    larger than the honest cluster's own calibrated scores even with
    the injected perturbation's variance EXACTLY matched by
    dp_variance()'s prediction, because raw_dist and the divisor
    weren't on comparable scales. Dividing by the VARIANCE instead
    (as implemented below) is dimensionally consistent -- raw_dist
    (variance-scale) / expected_var_ij (variance-scale) = a proper
    dimensionless ratio -- and empirically collapses the honest-
    heterogeneous clients' calibrated distances to the same order of
    magnitude as the honest cluster's own, while leaving uncalibrated
    (no injected noise) malicious distances clearly elevated, exactly
    as the ticket's own stated intent describes. If a future
    empirical validation against Task 1's real logged data finds THIS
    correction wrong instead, revisit -- but leaving the literal std-
    division in was verified to not work at all on a controlled
    synthetic fixture where the ground truth is known exactly, so it
    was not left in silently.

        per-client calibrated_scores = sum of each client's
            (n - f - 2) smallest CALIBRATED distances (identical
            neighbour-counting rule to plain Krum/Adaptive Krum, applied
            to the calibrated matrix instead of the raw one).
        MAD threshold applied to calibrated_scores (same mechanism as
        adaptive_multi_krum's own MAD/z-score thresholding, reused
        conceptually, not by direct call, since the input here is
        calibrated_scores, not raw scores).

    Parameters
    ----------
    all_params, weights : same as adaptive_multi_krum().
    per_client_metadata : dict, keyed by the SAME 0-indexed client
        position used in all_params/weights (i.e. per_client_metadata[i]
        describes all_params[i]) -- see module-level docstring above for
        the required schema (n_samples, class_entropy, noise_multiplier,
        epsilon).
    num_byzantine, k, method, min_keep_fraction : same meaning as
        adaptive_multi_krum() (k is this function's mad_multiplier_k --
        named `k` for consistency with its sibling function in this
        file, not renamed to match the ticket's `mad_multiplier_k`
        purely for naming symmetry with the rest of this module).
    baseline_honest_std_from_prior_round : float or None. None is
        ONLY valid for round 1 (bootstrap case) -- see the round-1
        bootstrap note below. Any later round must pass a real float
        (the caller's responsibility, per the ticket's explicit
        round-to-round state management requirement).
    dp_max_grad_norm : this run's real DP clipping norm C (confirmed
        1.0 in this codebase's config -- passed explicitly rather than
        hardcoded so a future config change can't silently desync this
        function from the real value).
    alpha_dirichlet : this run's real Dirichlet alpha (passed through
        for hetero_variance's signature; see that function's docstring
        for why it is not currently an active regressor).
    hetero_fit_coeffs : see hetero_variance() -- None means the hetero
        term is currently a no-op (not validated-zero).
    use_dp_calibration, use_hetero_calibration : independent ablation
        toggles, exactly as required by the ticket -- verified
        independently toggleable by tests/test_calibrated_krum.py.
    return_diagnostics : if True, returns a 3rd element containing this
        round's honest-cluster raw-score std, for the CALLER to thread
        into next round's baseline_honest_std_from_prior_round (this
        function does not maintain any state itself between calls --
        matches this file's existing style of pure, stateless
        aggregator functions; state persistence is main.py's job, the
        same way it already owns dp_states/DP-engine persistence).

    Round-1 bootstrap (explicit, per the ticket): pass
    baseline_honest_std_from_prior_round=None for round 1 ONLY. This
    function then computes an initial estimate by running the SAME
    neighbour-sum scoring on the RAW (uncalibrated) distance matrix
    first -- i.e. exactly what plain adaptive_multi_krum() would have
    produced for this round's data -- and uses the std of the RAW
    scores among clients that a plain-adaptive-Krum MAD threshold (same
    k) would keep, as round 1's baseline. This is the "bootstrap from
    round 1's own plain Adaptive Krum honest-cluster score estimate"
    the ticket asks for, computed inline rather than requiring the
    caller to run adaptive_multi_krum() separately and pass its result
    in.

    JUDGMENT CALL (off-by-one, stated explicitly per the ticket's
    warning about silently biasing every downstream Honest-FPR number):
    "prior round" state is ALWAYS the value computed at the END of the
    PREVIOUS round's call to this function (returned in diagnostics as
    "new_baseline_honest_std"), fed into THIS round's
    baseline_honest_std_from_prior_round argument by the caller. Round
    T's calibration therefore reflects round T-1's honest-cluster
    dispersion, never round T's own (which would be circular -- you'd
    need the calibrated result to compute the calibration). This means
    round 1 is the ONLY round that bootstraps from an uncalibrated
    estimate; every subsequent round uses a genuinely-calibrated prior
    estimate.

    Returns
    -------
    (aggregated_params, selected_indices) or
    (aggregated_params, selected_indices, diagnostics) -- same shape
    convention as adaptive_multi_krum(). diagnostics additionally
    contains "new_baseline_honest_std" (float) -- the value the CALLER
    must carry into next round's baseline_honest_std_from_prior_round.
    """
    EPS = 1e-9
    n = len(all_params)
    f = num_byzantine
    theoretical_neighbours = n - f - 2
    if theoretical_neighbours <= 0:
        raise ValueError(
            f"Cannot compute Krum scores: n={n}, f={f} -> theoretical "
            f"neighbour count={theoretical_neighbours} (n - f - 2). "
            f"Need f <= n - 3."
        )
    if method not in ("mad", "zscore"):
        raise ValueError(f"method={method!r} must be 'mad' or 'zscore'.")

    flat = np.array([np.concatenate([p.flatten() for p in params])
                      for params in all_params])
    total_params = flat.shape[1]

    scores = np.full(n, np.inf)
    nan_clients = set()
    finite_clients = []
    for i in range(n):
        if not np.all(np.isfinite(flat[i])):
            nan_clients.add(i)
            print(f"  ⚠  Client {i+1} update contains NaN/Inf -- "
                  f"quarantined (score=inf, will be discarded)")
        else:
            finite_clients.append(i)

    min_required = theoretical_neighbours + 1
    if len(finite_clients) < min_required:
        print(f"  ⚠  Only {len(finite_clients)} finite clients available, "
              f"need at least {min_required} to score. Averaging all "
              f"finite clients.")
        finite_params = [all_params[i] for i in finite_clients]
        finite_weights = [weights[i] for i in finite_clients]
        result = fedavg(finite_params, finite_weights)
        if return_diagnostics:
            diag = {"scores": scores.tolist(), "threshold": None,
                    "center": None, "spread": None,
                    "num_dropped": n - len(finite_clients),
                    "num_nan": len(nan_clients), "fallback_triggered": True,
                    "new_baseline_honest_std": baseline_honest_std_from_prior_round}
            return result, finite_clients, diag
        return result, finite_clients

    # ── Raw pairwise squared distances (finite clients only) ───────────
    raw_dist = np.zeros((n, n))
    for i in finite_clients:
        for j in finite_clients:
            if j <= i:
                continue
            d = float(np.sum((flat[i] - flat[j]) ** 2))
            raw_dist[i][j] = d
            raw_dist[j][i] = d

    # ── Round-1 bootstrap: plain (uncalibrated) neighbour-sum scores ───
    def _neighbour_sum_scores(dist_matrix):
        s = np.full(n, np.inf)
        for i in finite_clients:
            dlist = sorted(dist_matrix[i][j] for j in finite_clients if j != i)
            kk = min(theoretical_neighbours, len(dlist))
            s[i] = sum(dlist[:kk])
        return s

    def _pairwise_std_within(cluster, dist_matrix):
        """
        std of PAIRWISE raw distances among `cluster`'s members --
        NOT the std of neighbour-sum SCORES. This distinction matters:
        expected_std_ij calibrates a SINGLE pairwise distance
        (raw_dist_ij), and var_dp/var_hetero are likewise derived at
        the per-pair, full-vector level (see dp_variance()'s
        docstring) -- so baseline_honest_std_from_prior_round must be
        on that SAME pairwise-distance scale, not the much larger
        scale of an aggregated (n-f-2)-neighbour SUM. Using the score
        std here was tried first and found to swamp the DP/hetero
        terms in tests/test_calibrated_krum.py (the aggregate-sum
        scale is roughly theoretical_neighbours times larger),
        silently making calibration a near no-op -- fixed here rather
        than left in.
        """
        pair_vals = [dist_matrix[a][b] for idx_a, a in enumerate(cluster)
                     for b in cluster[idx_a + 1:]]
        if len(pair_vals) < 2:
            return None
        return float(np.std(pair_vals))

    if baseline_honest_std_from_prior_round is None:
        raw_scores_bootstrap = _neighbour_sum_scores(raw_dist)
        finite_raw_scores = np.array([raw_scores_bootstrap[i] for i in finite_clients])
        if method == "mad":
            b_center = float(np.median(finite_raw_scores))
            b_mad = float(np.median(np.abs(finite_raw_scores - b_center)))
            b_spread = 1.4826 * b_mad
        else:
            b_center = float(np.mean(finite_raw_scores))
            b_spread = float(np.std(finite_raw_scores))
        b_threshold = b_center + k * b_spread if b_spread > 0 else b_center + EPS
        bootstrap_kept = [i for i in finite_clients if raw_scores_bootstrap[i] <= b_threshold]
        pairwise_std = _pairwise_std_within(bootstrap_kept, raw_dist)
        baseline_honest_std_from_prior_round = (
            pairwise_std if pairwise_std is not None
            # Fallback if bootstrap_kept has <2 members (degenerate):
            # spread/sqrt(theoretical_neighbours) converts the
            # neighbour-SUM spread back to an approximate single-pair
            # scale (sum of `theoretical_neighbours` roughly-iid terms
            # has std ~ sqrt(theoretical_neighbours) times a single
            # term's std, for the variances actually in play here).
            else float(b_spread) / np.sqrt(max(theoretical_neighbours, 1))
        )
        print(f"  [Calibrated Krum] Round-1 bootstrap: baseline_honest_std "
              f"(pairwise-distance scale) = "
              f"{baseline_honest_std_from_prior_round:.4e} "
              f"(from {len(bootstrap_kept)} plain-Adaptive-Krum-kept clients)")

    # ── Calibrate every pairwise distance ───────────────────────────────
    calibrated_dist = np.zeros((n, n))
    for i in finite_clients:
        for j in finite_clients:
            if j <= i:
                continue
            meta_i = per_client_metadata[i]
            meta_j = per_client_metadata[j]

            var_dp = 0.0
            if use_dp_calibration:
                var_dp = dp_variance(
                    meta_i.get("noise_multiplier"), meta_j.get("noise_multiplier"),
                    dp_max_grad_norm,
                    meta_i["n_samples"], meta_j["n_samples"],
                    total_params,
                )
            var_hetero = 0.0
            if use_hetero_calibration:
                var_hetero = hetero_variance(
                    meta_i, meta_j, alpha_dirichlet, fit_coeffs=hetero_fit_coeffs
                )

            expected_var_ij = (
                var_dp + var_hetero + baseline_honest_std_from_prior_round ** 2
            )
            calibrated_dist[i][j] = raw_dist[i][j] / (expected_var_ij + EPS)
            calibrated_dist[j][i] = calibrated_dist[i][j]

    calibrated_scores = _neighbour_sum_scores(calibrated_dist)
    finite_calibrated_scores = np.array([calibrated_scores[i] for i in finite_clients])

    # ── MAD/z-score threshold over CALIBRATED scores ────────────────────
    if method == "mad":
        center = float(np.median(finite_calibrated_scores))
        mad = float(np.median(np.abs(finite_calibrated_scores - center)))
        spread = 1.4826 * mad
    else:
        center = float(np.mean(finite_calibrated_scores))
        spread = float(np.std(finite_calibrated_scores))
    threshold = center + k * spread if spread > 0 else center + EPS

    kept = [i for i in finite_clients if calibrated_scores[i] <= threshold]
    dropped = [i for i in finite_clients if calibrated_scores[i] > threshold] + sorted(nan_clients)

    fallback_triggered = False
    min_keep = int(np.ceil(min_keep_fraction * len(finite_clients))) if min_keep_fraction > 0 else 0
    if len(kept) < min_keep:
        print(f"  ⚠  Calibrated thresholding kept only {len(kept)}/"
              f"{len(finite_clients)} finite clients, below floor of "
              f"{min_keep}. Falling back to lowest-{min_keep}-calibrated-"
              f"score clients.")
        fallback_triggered = True
        ranked_finite = sorted(finite_clients, key=lambda i: calibrated_scores[i])
        kept = ranked_finite[:min_keep]
        dropped = [i for i in range(n) if i not in kept]

    print(f"\n  Calibrated Adaptive Multi-Krum ({method}): n={n}, f={f} "
          f"assumed, neighbours={theoretical_neighbours}, k={k}, "
          f"use_dp_calibration={use_dp_calibration}, "
          f"use_hetero_calibration={use_hetero_calibration}")
    if nan_clients:
        print(f"  Quarantined (NaN/Inf): clients {sorted(c+1 for c in nan_clients)}")
    print(f"  calibrated center={center:.4e}  spread={spread:.4e}  "
          f"threshold={threshold:.4e}")
    print(f"  Kept {len(kept)}/{n} clients (dropped {len(dropped)})")

    kept_params = [all_params[i] for i in kept]
    kept_weights = [weights[i] for i in kept]
    result = fedavg(kept_params, kept_weights)

    # Raw (uncalibrated) PAIRWISE-distance std among THIS round's kept/
    # honest cluster -- same pairwise-distance scale fix as the round-1
    # bootstrap above (see _pairwise_std_within()'s docstring); what the
    # caller carries forward as next round's baseline_honest_std_from_
    # prior_round (see the off-by-one judgment call in the docstring
    # above).
    new_baseline_honest_std = _pairwise_std_within(kept, raw_dist)
    if new_baseline_honest_std is None:
        new_baseline_honest_std = baseline_honest_std_from_prior_round
    raw_scores_this_round = _neighbour_sum_scores(raw_dist)

    if return_diagnostics:
        diag = {
            "scores": calibrated_scores.tolist(),
            "raw_scores": raw_scores_this_round.tolist(),
            "threshold": threshold, "center": center, "spread": spread,
            "num_dropped": len(dropped), "num_nan": len(nan_clients),
            "fallback_triggered": fallback_triggered,
            "new_baseline_honest_std": new_baseline_honest_std,
        }
        return result, kept, diag
    return result, kept


def public_noise_multiplier_map(client_ids, sigma_by_client):
    """
    Server-visible DP noise multiplier for every client in `client_ids`,
    for use as per_client_metadata[...]["noise_multiplier"] in
    calibrated_adaptive_multi_krum().

    WHY THIS EXISTS (E6 / Issue 5 leak fix): main.py builds
    `sigma_by_client` from the persistent DP client states, and
    Byzantine clients are deliberately EXCLUDED from those states (they
    run no Opacus engine). Reading `sigma_by_client.get(client_id)`
    directly therefore returned None for exactly the attackers and a real
    sigma for exactly the honest clients -- i.e. the aggregator's
    "metadata" encoded the ground-truth Byzantine labels, shrinking every
    honest<->Byzantine pair's variance divisor relative to honest<->honest
    pairs and handing Calibrated Krum an oracle advantage.

    A real server only knows the publicly-declared DP configuration
    (target epsilon, delta, clip norm), which is identical for every
    client regardless of whether it is honest. So every client that has
    no entry in `sigma_by_client` is assigned the MEDIAN of the sigmas
    that do exist -- the attacker is assumed to DECLARE the standard DP
    configuration. Whether it actually adds that noise is precisely the
    signal calibration should be able to exploit, not something the
    metadata may reveal.

    Returns {client_id: sigma_or_None}. If `sigma_by_client` is empty
    (no DP active anywhere) every client maps to None, unchanged from
    the previous behaviour.
    """
    known = [float(s) for s in sigma_by_client.values() if s is not None]
    fallback = float(np.median(known)) if known else None
    out = {}
    for cid in client_ids:
        s = sigma_by_client.get(cid)
        out[cid] = float(s) if s is not None else fallback
    return out
