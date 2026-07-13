import numpy as np


def multi_krum(all_params, weights, num_byzantine=2):
    """
    Multi-Krum Byzantine-robust aggregation (Blanchard et al., NeurIPS 2017).

    Parameters
    ----------
    all_params     : list of parameter lists, one per client
    weights        : list of sample counts per client
    num_byzantine  : assumed number of Byzantine clients (f)

    Algorithm
    ---------
    1. Flatten each client's full parameter set into a single vector
    2. Quarantine any client whose update contains NaN or Inf values
       (sign-flip attacks at high scale cause numerical overflow)
    3. Compute pairwise squared Euclidean distances between finite clients
    4. Score each client by summing distances to its (n - f - 2) nearest
       neighbours — legitimate clients cluster together (low score);
       Byzantine outliers score high
    5. Select m = n - f - 2 lowest-scoring clients
    6. Return their weighted average as the global update

    Guarantee: tolerates up to f Byzantine clients when f < n/2 - 1.
    With n=10, f=2: m = 6 clients selected each round.

    NaN guard: Byzantine updates with extreme scale values overflow to NaN.
    These clients are assigned score=inf before distance computation,
    ensuring they are always ranked last and discarded.
    """
    n = len(all_params)
    f = num_byzantine
    m = n - f - 2      # number of clients to keep

    if m <= 0:
        raise ValueError(
            f"Too many Byzantine clients: n={n}, f={f} → m={m}. "
            f"Require f < n/2 - 1, i.e. f ≤ {n // 2 - 2}."
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
        # averaging all finite clients to avoid total failure
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
        return result

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
    # Only finite clients are considered as neighbours.
    neighbours = n - f - 2
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

    print(f"\n  Multi-Krum: n={n}, f={f} assumed, m={m} selected")
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

    return result