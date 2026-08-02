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