import numpy as np


def multi_krum(all_params, weights, num_byzantine=2, m=None):
    """
    Multi-Krum Byzantine-robust aggregation (Blanchard et al., NeurIPS 2017).

    Parameters
    ----------
    all_params     : list of parameter lists, one per client
    weights        : list of sample counts per client
    num_byzantine  : assumed number of Byzantine clients (f)
    m              : number of clients to SELECT (keep) after scoring.
                     Defaults to the theoretical Blanchard et al. guarantee
                     m = n - f - 2 if not given.

                     IMPORTANT — what m does and does NOT change:
                     The NEIGHBOUR-SCORING step (how many nearest neighbours
                     each client's distance score is summed over) ALWAYS
                     uses the theoretical n - f - 2 value, regardless of
                     what m is passed here. That neighbour count is what
                     gives Krum its formal robustness guarantee — it's not
                     an arbitrary knob. Passing a LARGER m than the
                     theoretical value (e.g. m = n - f - 1, keeping one
                     more client than the strict guarantee) does NOT change
                     which clients Krum considers most suspicious, or how
                     it ranks them — it only changes how far down that same
                     honest ranking you keep. This is a deliberate,
                     documented trade for extra legitimate-client data at
                     the cost of a thinner safety margin, not a change to
                     Krum's actual detection logic.

                     Hard constraint: m must satisfy 1 <= m <= n - f. Above
                     n - f, you are guaranteed to keep at least one
                     Byzantine client regardless of how well Krum ranks
                     them, since there are only n - f legitimate clients
                     to fill m slots from — this defeats the point of
                     Byzantine-robust selection entirely, so it raises
                     rather than silently proceeding.

    Algorithm
    ---------
    1. Flatten each client's full parameter set into a single vector
    2. Quarantine any client whose update contains NaN or Inf values
       (sign-flip attacks at high scale cause numerical overflow)
    3. Compute pairwise squared Euclidean distances between finite clients
    4. Score each client by summing distances to its (n - f - 2) nearest
       neighbours — legitimate clients cluster together (low score);
       Byzantine outliers score high. This neighbour count is FIXED at the
       theoretical value regardless of m (see m's docstring above).
    5. Select the m lowest-scoring clients (m defaults to n - f - 2, but
       can be widened up to n - f via the m parameter)
    6. Return their weighted average as the global update

    Guarantee: tolerates up to f Byzantine clients when f < n/2 - 1, when
    m is left at its default n - f - 2. Widening m trades some of that
    margin for more legitimate-client data — see m's docstring.

    NaN guard: Byzantine updates with extreme scale values overflow to NaN.
    These clients are assigned score=inf before distance computation,
    ensuring they are always ranked last and discarded.

    Returns
    -------
    (aggregated_params, selected_indices) : tuple
        aggregated_params  : list[np.ndarray] — weighted average of
                             selected (trusted) clients' parameters
        selected_indices   : list[int] — 0-indexed client indices that
                             were selected/trusted this round. Callers
                             comparing against a 0-indexed Byzantine
                             client list (e.g. BYZANTINE_CLIENTS=[0, 1])
                             can check membership directly against this
                             list to compute a per-round detection rate:
                             any BYZANTINE_CLIENTS index NOT present in
                             selected_indices was correctly discarded.
    """
    n = len(all_params)
    f = num_byzantine
    theoretical_m = n - f - 2      # the value Krum's formal guarantee is built on

    if theoretical_m <= 0:
        raise ValueError(
            f"Too many Byzantine clients: n={n}, f={f} → theoretical m={theoretical_m}. "
            f"Require f < n/2 - 1, i.e. f ≤ {n // 2 - 2}."
        )

    if m is None:
        m = theoretical_m
    if m < 1:
        raise ValueError(f"m={m} must be at least 1.")
    if m > n - f:
        raise ValueError(
            f"m={m} exceeds n-f={n - f}. Selecting more clients than there "
            f"are legitimate clients guarantees at least one Byzantine "
            f"client is kept, regardless of Krum's ranking — this defeats "
            f"the purpose of Byzantine-robust selection. Reduce m or "
            f"increase n (fewer assumed Byzantine clients f)."
        )

    # ── Step 1: flatten each client's parameters into one long vector ──
    flat = []
    for params in all_params:
        flat.append(np.concatenate([p.flatten() for p in params]))
    flat = np.array(flat)   # shape: (n, total_params)

    # ── Step 2: NaN/Inf guard ─────────────────────────────────────────
    # Sign-flip attacks at high scale (e.g. 5x) cause gradient overflow.
    # NaN values in a client's update corrupt pairwise distance scores
    # for ALL clients, making Krum selection unpredictable.
    # Fix: detect NaN/Inf clients and assign score=inf BEFORE distance
    # computation so they are always ranked last and discarded.
    scores       = np.full(n, np.inf)   # default all to inf
    nan_clients  = set()
    finite_clients = []

    for i in range(n):
        if not np.all(np.isfinite(flat[i])):
            nan_clients.add(i)
            print(f"  ⚠  Client {i+1} update contains NaN/Inf "
                  f"— quarantined (score=inf, will be discarded)")
        else:
            finite_clients.append(i)

    if len(finite_clients) < m:
        # Not enough finite clients to select m — fall back to
        # averaging all finite clients to avoid total failure.
        # selected_indices here is every finite client, since all of
        # them were used in the fallback average (no discarding based
        # on distance score — the NaN guard already excluded the
        # non-finite ones from finite_clients before we got here).
        print(f"  ⚠  Only {len(finite_clients)} finite clients available, "
              f"need {m}. Averaging all finite clients.")
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
        return result, finite_clients

    # ── Step 3: pairwise squared distances (finite clients only) ──────
    distances = np.zeros((n, n))
    for i in finite_clients:
        for j in finite_clients:
            if j <= i:
                continue
            d = float(np.sum((flat[i] - flat[j]) ** 2))
            distances[i][j] = d
            distances[j][i] = d

    # ── Step 4: score each finite client ──────────────────────────────
    # Sum of squared distances to (n - f - 2) nearest finite neighbours.
    # This ALWAYS uses theoretical_m, not the caller-supplied m — see
    # the m parameter's docstring for why this must stay fixed.
    neighbours = theoretical_m
    for i in finite_clients:
        # Distances from client i to all OTHER finite clients
        finite_distances = sorted(
            distances[i][j] for j in finite_clients if j != i
        )
        # Take the nearest `neighbours` distances
        k = min(neighbours, len(finite_distances))
        scores[i] = sum(finite_distances[:k])

    # ── Step 5: select m clients with lowest scores ───────────────────
    ranked    = np.argsort(scores)       # ascending — NaN clients at end
    selected  = ranked[:m].tolist()
    discarded = ranked[m:].tolist()

    print(f"\n  Multi-Krum: n={n}, f={f} assumed, "
          f"theoretical_m={theoretical_m}, using m={m} selected")
    if nan_clients:
        print(f"  Quarantined (NaN/Inf): "
              f"clients {sorted([c+1 for c in nan_clients])}")
    print(f"  Scores (lower = more trusted):")
    for i in ranked:
        tag = "✓ SELECTED " if i in selected else "✗ DISCARDED"
        score_str = f"{scores[i]:.4e}" if np.isfinite(scores[i]) else "NaN/Inf"
        print(f"    Client {i+1:>2}  score={score_str:<14}  {tag}")

    # ── Step 6: weighted average of selected clients ──────────────────
    selected_params  = [all_params[i] for i in selected]
    selected_weights = [weights[i]    for i in selected]
    total = sum(selected_weights)

    result = []
    for layer_idx in range(len(selected_params[0])):
        layer_avg = sum(
            p[layer_idx] * (w / total)
            for p, w in zip(selected_params, selected_weights)
        )
        result.append(layer_avg)

    return result, selected


def adaptive_multi_krum(all_params, weights, num_byzantine=2, k=2.5, method="mad",
                         min_keep_fraction=0.5):
    """
    Adaptive Multi-Krum via Dynamic Thresholding (MAD / Z-Score).

    Companion to multi_krum() above — same NaN guard, same flatten
    convention, same (aggregated_params, selected_indices) contract. Do
    NOT use this in place of multi_krum(); run them as separate
    conditions and compare (fixed-m Krum vs adaptive-threshold Krum).

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
    (aggregated_params, selected_indices) : tuple
        aggregated_params  : list[np.ndarray] — weighted average of kept
                             (trusted) clients' parameters
        selected_indices   : list[int] — 0-indexed client indices kept
                             this round. Same semantics as multi_krum()'s
                             selected_indices — check membership against
                             BYZANTINE_CLIENTS for detection rate.
    """
    n = len(all_params)
    f = num_byzantine
    theoretical_neighbours = n - f - 2

    if theoretical_neighbours <= 0:
        raise ValueError(
            f"Too many Byzantine clients: n={n}, f={f} → theoretical neighbour "
            f"count={theoretical_neighbours}. Require f < n/2 - 1, i.e. f ≤ {n // 2 - 2}."
        )
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

    min_keep = int(np.ceil(min_keep_fraction * len(finite_clients))) if min_keep_fraction > 0 else 1
    if len(kept) < min_keep:
        print(f"  ⚠  Thresholding kept only {len(kept)}/{len(finite_clients)} finite "
              f"clients, below floor of {min_keep}. Falling back to lowest-{min_keep}-score clients.")
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

    return result, kept
