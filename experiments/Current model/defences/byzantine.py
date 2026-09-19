import numpy as np


# ===========================================================================
# Issue 5 Task 1 -- STEALTHY BYZANTINE ATTACK FAMILIES
# ===========================================================================
# Added: Min-Max / Min-Sum optimized distance-aware poisoner (Fang et al.,
# USENIX Security 2020, "Local Model Poisoning Attacks to Byzantine-Robust
# Federated Learning") and a bounded-magnitude directional poisoner.
#
# Both follow the SAME "train first, then craft" convention already
# established by *_trained() above: the attacker(s) train normally on
# real local data first; these functions only ever transform an already
# HONESTLY-TRAINED parameter list (or list of such lists, for the
# coalition-aware Min-Max/Min-Sum pair). Nothing here trains a model or
# touches an untouched global_params snapshot directly -- see
# sign_flip_attack()'s docstring above for why that pattern was wrong
# for the easy-control attacks and is avoided here from the start.
#
# THREAT MODEL NOTE (applies to minmax_attack_trained / minsum_attack_trained):
# this is the AGR-agnostic / partial-knowledge attacker from Fang et al.
# Section IV -- the colluding Byzantine coalition sees only its OWN
# members' honestly-trained updates, not the honest clients' updates
# (which would require server-side collusion, not modeled here). This is
# a strictly weaker attacker than the "AGR-tailored" full-knowledge
# variant; label results as evaluated against a partial-knowledge
# attacker, not implied to generalize to a stronger, full-knowledge one.
# ===========================================================================


def _flatten(params):
    """Flatten a list[np.ndarray] into one 1-D float64 vector + the
    shape/size metadata needed to invert the operation exactly."""
    shapes = [p.shape for p in params]
    sizes = [p.size for p in params]
    flat = np.concatenate([p.ravel() for p in params]).astype(np.float64)
    return flat, shapes, sizes


def _unflatten(flat, shapes, sizes):
    """Inverse of _flatten() -- rebuild list[np.ndarray] (float32, to
    match every other attack function's output dtype in this file)."""
    out = []
    idx = 0
    for shape, size in zip(shapes, sizes):
        out.append(flat[idx:idx + size].reshape(shape).astype(np.float32))
        idx += size
    return out


def _direction_and_coalition_matrix(byzantine_trained_params_list, dev_type):
    """Shared setup for minmax_attack_trained / minsum_attack_trained:
    flattens the coalition's honest updates into a matrix, computes the
    coalition mean, and derives the perturbation direction per dev_type.
    Returns (W, shapes, sizes, w_avg, direction, dev_type_used)."""
    flats = []
    shapes = sizes = None
    for p in byzantine_trained_params_list:
        f, shapes, sizes = _flatten(p)
        flats.append(f)
    W = np.stack(flats, axis=0)  # (n_coalition, d)
    w_avg = W.mean(axis=0)

    if dev_type == "std":
        if W.shape[0] < 2:
            direction = -w_avg / (np.linalg.norm(w_avg) + 1e-12)
            dev_type_used = "unit_vec (auto-fallback: coalition size 1, " \
                             "cannot estimate a coordinate-wise std from " \
                             "a single sample)"
        else:
            direction = W.std(axis=0)
            dev_type_used = "std"
    elif dev_type == "sign":
        direction = -np.sign(w_avg)
        dev_type_used = "sign"
    elif dev_type == "unit_vec":
        direction = -w_avg / (np.linalg.norm(w_avg) + 1e-12)
        dev_type_used = "unit_vec"
    else:
        raise ValueError(f"Unknown dev_type: {dev_type!r} "
                          f"(expected 'std', 'sign', or 'unit_vec')")

    return W, shapes, sizes, w_avg, direction, dev_type_used


def minmax_attack_trained(byzantine_trained_params_list, dev_type="std",
                           gamma_init=None, search_iters=15,
                           return_diagnostics=False, global_params=None):
    """
    Min-Max optimized distance-aware poisoner (Fang et al., USENIX
    Security 2020). See this module's header comment for the full
    AGR-agnostic / partial-knowledge threat-model discussion.

    Crafts ONE poisoned update, to be broadcast identically by every
    colluding Byzantine client this round (coalition-optimal per Fang
    et al. -- sending distinct vectors only fragments the effect on a
    pairwise-distance-based defense like Krum). The update is
    w_avg + gamma * direction, where gamma is the LARGEST scale found
    (via binary search) such that the crafted update's worst-case (max)
    L2 distance to any coalition member does not exceed the coalition's
    own max pairwise distance -- i.e. it stays "no farther from the pack
    than the pack already is from itself", the Min-Max evasion criterion.

    Parameters
    ----------
    byzantine_trained_params_list : list[list[np.ndarray]]
        One entry per colluding Byzantine client: that client's own
        HONESTLY-trained local parameters (train() called normally
        first). >= 2 entries needed for dev_type='std' to mean anything;
        with exactly 1 entry it auto-falls-back to 'unit_vec' (logged in
        diagnostics, not silent).
    dev_type : {'std', 'sign', 'unit_vec'}
        Perturbation-direction estimator:
          'std'      -- per-coordinate std across the coalition (Fang et
                        al.'s reported strongest variant; default here).
          'sign'     -- -sign(coalition mean).
          'unit_vec' -- -mean / ||mean||_2.
    gamma_init : float or None
        Upper bound for the gamma binary search. Defaults to 5x the
        coalition's own max pairwise L2 distance if not given. Tune UP
        (via --minmax-gamma-init) if Task 2's difficulty check shows
        plain Adaptive Krum TPR stuck at 100% (attack not evasive
        enough); tune DOWN if TPR collapses toward 0% (too aggressive).
    search_iters : int
        Binary-search iterations for gamma (default 15; ~2^-15 relative
        precision against the search range -- plenty for this purpose).
    return_diagnostics : bool
        If True, also return a dict with the resolved gamma and the
        bound it was searched against.
    global_params : list[np.ndarray] or None
        Current global model, for diagnostics ONLY (Issue 5 Task 2:
        "log update norm and distance statistics"). Does NOT affect the
        crafted vector itself -- the craft is still w_avg + gamma *
        direction, anchored to the coalition's own honestly-trained
        params exactly as before. If provided, the returned diagnostics
        include crafted_update_l2_norm = ||crafted - global_params||,
        i.e. how far the poisoned broadcast update is from what the
        server currently holds, for logging/plausibility-checking
        across rounds and conditions. If None, that key is omitted
        (not zero -- omitted, so callers can't mistake "not measured"
        for "measured as zero").

    Returns
    -------
    crafted_params : list[np.ndarray]
    diagnostics : dict   (only if return_diagnostics=True)
    """
    assert len(byzantine_trained_params_list) >= 1, \
        "minmax_attack_trained needs at least one honestly-trained " \
        "Byzantine-coalition update to craft against."

    W, shapes, sizes, w_avg, direction, dev_type_used = \
        _direction_and_coalition_matrix(byzantine_trained_params_list, dev_type)

    if W.shape[0] >= 2:
        max_pair_dist = 0.0
        for i in range(W.shape[0]):
            for j in range(i + 1, W.shape[0]):
                dist = np.linalg.norm(W[i] - W[j])
                if dist > max_pair_dist:
                    max_pair_dist = dist
    else:
        # No internal pairwise distance to bound against with a single
        # attacker -- use ||w_avg|| as a conservative search ceiling.
        max_pair_dist = float(np.linalg.norm(w_avg)) + 1e-12

    gamma_hi = gamma_init if gamma_init is not None else 5.0 * max_pair_dist
    gamma_hi = max(gamma_hi, 1e-12)

    def _feasible(gamma):
        candidate = w_avg + gamma * direction
        worst = 0.0
        for i in range(W.shape[0]):
            dist = np.linalg.norm(candidate - W[i])
            if dist > worst:
                worst = dist
        return worst <= max_pair_dist

    if _feasible(gamma_hi):
        gamma_star = gamma_hi
    else:
        lo, hi = 0.0, gamma_hi
        for _ in range(search_iters):
            mid = (lo + hi) / 2.0
            if _feasible(mid):
                lo = mid
            else:
                hi = mid
        gamma_star = lo

    crafted_flat = w_avg + gamma_star * direction
    crafted_params = _unflatten(crafted_flat, shapes, sizes)

    if return_diagnostics:
        # Issue 5 Task 2: "verify Min-Max ... generate valid stealthy
        # updates" -- re-derive the ACHIEVED worst-case distance at
        # gamma_star (not just the pass/fail bool _feasible() used
        # during the search) so the feasibility margin is inspectable,
        # not just asserted.
        worst_at_star = 0.0
        for i in range(W.shape[0]):
            dist = np.linalg.norm(crafted_flat - W[i])
            if dist > worst_at_star:
                worst_at_star = dist
        constraint_ratio = float(worst_at_star / max_pair_dist) if max_pair_dist > 0 else float("nan")

        diag = {
            "attack": "minmax",
            "dev_type_used": dev_type_used,
            "gamma": float(gamma_star),
            "gamma_search_upper_bound": float(gamma_hi),
            "coalition_size": int(W.shape[0]),
            "coalition_max_pairwise_distance": float(max_pair_dist),
            "constraint_worst_distance": float(worst_at_star),
            "constraint_ratio": constraint_ratio,
            "constraint_satisfied": bool(constraint_ratio <= 1.0 + 1e-9),
        }
        if global_params is not None:
            global_flat, _, _ = _flatten(global_params)
            diag["crafted_update_l2_norm"] = float(
                np.linalg.norm(crafted_flat - global_flat))
        return crafted_params, diag
    return crafted_params


def minsum_attack_trained(byzantine_trained_params_list, dev_type="std",
                           gamma_init=None, search_iters=15,
                           return_diagnostics=False, global_params=None):
    """
    Min-Sum optimized distance-aware poisoner (Fang et al., USENIX
    Security 2020). Same coalition-broadcast convention and
    AGR-agnostic threat model as minmax_attack_trained() above -- see
    that function's docstring and this module's header comment.

    Differs from Min-Max only in the feasibility bound used during the
    gamma binary search: instead of bounding the crafted update's WORST
    (max) distance to any coalition member, Min-Sum bounds the crafted
    update's SUM of squared distances to ALL coalition members by the
    smallest such sum achieved by any actual coalition member. This
    mimics Krum's own scoring function directly (Krum selects whichever
    client has smallest sum-of-squared-distances-to-neighbours), so a
    Min-Sum-crafted update targets looking "Krum-central" rather than
    merely "not the single farthest outlier" (Min-Max's target).

    Parameters and return value: identical shape to
    minmax_attack_trained() -- see that docstring for the full
    parameter reference, including global_params (diagnostics-only,
    does not affect the crafted vector).
    """
    assert len(byzantine_trained_params_list) >= 1, \
        "minsum_attack_trained needs at least one honestly-trained " \
        "Byzantine-coalition update to craft against."

    W, shapes, sizes, w_avg, direction, dev_type_used = \
        _direction_and_coalition_matrix(byzantine_trained_params_list, dev_type)

    if W.shape[0] >= 2:
        sums = []
        for i in range(W.shape[0]):
            s = sum(np.linalg.norm(W[i] - W[j]) ** 2
                     for j in range(W.shape[0]) if j != i)
            sums.append(s)
        min_sum = min(sums)
    else:
        min_sum = float(np.linalg.norm(w_avg)) ** 2 + 1e-12

    gamma_hi = gamma_init if gamma_init is not None else \
        5.0 * float(np.sqrt(min_sum / max(W.shape[0], 1)))
    gamma_hi = max(gamma_hi, 1e-12)

    def _feasible(gamma):
        candidate = w_avg + gamma * direction
        total = sum(np.linalg.norm(candidate - W[i]) ** 2
                     for i in range(W.shape[0]))
        return total <= min_sum

    if _feasible(gamma_hi):
        gamma_star = gamma_hi
    else:
        lo, hi = 0.0, gamma_hi
        for _ in range(search_iters):
            mid = (lo + hi) / 2.0
            if _feasible(mid):
                lo = mid
            else:
                hi = mid
        gamma_star = lo

    crafted_flat = w_avg + gamma_star * direction
    crafted_params = _unflatten(crafted_flat, shapes, sizes)

    if return_diagnostics:
        total_at_star = sum(np.linalg.norm(crafted_flat - W[i]) ** 2
                             for i in range(W.shape[0]))
        constraint_ratio = float(total_at_star / min_sum) if min_sum > 0 else float("nan")

        diag = {
            "attack": "minsum",
            "dev_type_used": dev_type_used,
            "gamma": float(gamma_star),
            "gamma_search_upper_bound": float(gamma_hi),
            "coalition_size": int(W.shape[0]),
            "coalition_min_sum_sq_distance": float(min_sum),
            "constraint_worst_distance": float(total_at_star),
            "constraint_ratio": constraint_ratio,
            "constraint_satisfied": bool(constraint_ratio <= 1.0 + 1e-9),
        }
        if global_params is not None:
            global_flat, _, _ = _flatten(global_params)
            diag["crafted_update_l2_norm"] = float(
                np.linalg.norm(crafted_flat - global_flat))
        return crafted_params, diag
    return crafted_params


def bounded_directional_attack_trained(trained_params, global_params, tau,
                                        margin=0.05, direction="negate",
                                        model_state_keys=None):
    """
    Bounded-magnitude directional Byzantine attack (Issue 5 Task 1).

    Crafts a poisoned update pointed in an adversarial DIRECTION (negate
    the client's own trained DELTA entirely -- broadly adversarial --
    or negate only the classifier-head slice of the delta -- targeted,
    e.g. flipping a rare-class logit like MITM toward Normal), then
    rescales so the quantity the guard actually checks sits strictly
    under the HMAC norm guard's rejection threshold.

    FIX (this revision, v2): the previous version of this function
    rescaled the ABSOLUTE crafted parameter vector's norm to
    target_norm, using only `trained_params` -- it never received
    `global_params` at all. The real HMAC norm guard (see
    defences/hmac_norm_guard.py and he_local.py's
    generate_head_norm_proof() call site) verifies the DELTA norm
    ||trained_head - global_head||, not the absolute parameter
    magnitude, so that version was fixed to rescale the DELTA instead.
    That fix is preserved here.

    FIX (this revision, v3 -- NEW): for direction='classifier_head_negate',
    the v2 code still rescaled the WHOLE delta vector (honest backbone +
    flipped head) to a single shared target_norm calibrated from
    HEAD-ONLY honest norms (tau comes from estimate_norm_guard_tau() /
    --bounded-tau, both computed over classifier-head-only delta norms
    -- see hmac_norm_guard.py's generate_head_norm_proof(), which only
    ever measures the classifier-head slice, never the full model).
    Rescaling a vector multiplies every coordinate by the same scalar,
    which preserves each slice's SHARE of the total norm exactly. Since
    the backbone is ~94% of parameters and carries real magnitude, the
    head slice's own norm after a whole-vector rescale to tau-margin
    ends up well UNDER tau-margin, not sitting at the guard's actual
    boundary -- the attack still passed the guard, but was materially
    weaker than "hug the threshold" implies, because it was spending
    most of its rescaled norm budget on the (already-honest,
    guard-irrelevant) backbone rather than on the head slice the guard
    actually inspects.

    This version rescales ONLY the classifier-head sub-vector to
    target_norm for direction='classifier_head_negate' -- the backbone
    keeps its full, un-rescaled honest delta untouched. This makes the
    head-slice norm land exactly at tau-margin (the real quantity the
    guard checks), which is a materially stronger and more honest test
    of the guard's documented magnitude-only blind spot. direction=
    'negate' (the broadly-adversarial, whole-model variant) is
    UNCHANGED by this fix -- it is not the variant Experiment 2's
    Phase 1.3 pilot uses, and its own guard-interaction question (the
    same dilution logic applies to it too, since the guard is
    head-only in every mode of this codebase) is left as a separate,
    not-yet-addressed item rather than silently changed alongside this
    fix.

    PURPOSE (per issue spec): this attack is DESIGNED to defeat the
    magnitude-only HMAC norm guard by construction -- the attacker
    truthfully reports a delta-norm below tau, so the guard has no
    basis to reject it. Task 2's difficulty check expects norm-guard
    TPR = 0% against this attack; that is the documented, EXPECTED
    result (an honestly-reported known limitation), not a bug to fix.
    What Calibrated Krum is expected to still catch is the DIRECTION
    (content) of the update, which a magnitude-only check cannot see.

    Parameters
    ----------
    trained_params : list[np.ndarray]
        The attacking client's own honestly-trained local parameters.
    global_params : list[np.ndarray]
        The global parameters this client received at the START of the
        round, before local training -- same object main.py already
        threads through for FedProx's proximal term and
        classifier_head_flip_attack's key lookup. Required so this
        function can compute and rescale the DELTA, the quantity the
        guard actually verifies, rather than an absolute magnitude the
        guard never looks at.
    tau : float
        Assumed/estimated norm-guard threshold for this round, over
        the classifier-head-only delta norm (matching what
        estimate_norm_guard_tau() / the guard's own
        mad_threshold_head_norms() actually compute against). Caller
        supplies this -- see estimate_norm_guard_tau() below for one
        way to derive it online from the prior round's verified honest
        head-norms.
    margin : float
        Safety margin subtracted from tau (default 0.05) so the crafted
        delta-norm stays strictly under threshold even given estimation
        error in tau. Increase this for more headroom when tau is only
        a rough estimate.
    direction : {'negate', 'classifier_head_negate'}
        'negate'                 -- flip the entire trained DELTA, then
                                     rescale the whole delta vector to
                                     tau-margin. UNCHANGED by the v3 fix
                                     above -- this variant's own
                                     guard-dilution question is a
                                     separate, not-yet-addressed item.
        'classifier_head_negate' -- flip only the classifier-head
                                     slice of the delta (requires
                                     model_state_keys); the backbone
                                     keeps its full, UN-rescaled honest
                                     delta. ONLY the classifier-head
                                     sub-vector is rescaled, to
                                     tau-margin -- matching what the
                                     HMAC norm guard actually measures
                                     (the classifier-head-only delta
                                     norm, never the full model's).
    model_state_keys : list[str] or None
        Required when direction='classifier_head_negate'; same
        convention as classifier_head_flip_attack()'s model_state_keys.

    Returns
    -------
    crafted_params : list[np.ndarray]
        global_params + a delta redirected adversarially. For
        direction='negate': the WHOLE delta is rescaled so
        ||crafted_params - global_params||_2 = max(tau - margin, 0).
        For direction='classifier_head_negate': only the
        classifier-head SLICE of the delta is rescaled so
        ||crafted_head - global_head||_2 = max(tau - margin, 0); the
        backbone slice is returned at its full, honestly-trained delta.
    """
    if tau is None:
        raise ValueError(
            "bounded_directional_attack_trained requires a numeric tau "
            "(assumed norm-guard threshold) -- pass --bounded-tau "
            "explicitly, or ensure estimate_norm_guard_tau() has a "
            "prior round's honest norms to estimate from."
        )
    target_norm = max(tau - margin, 0.0)

    honest_delta = [t - g for t, g in zip(trained_params, global_params)]

    if direction == "negate":
        # UNCHANGED by the v3 fix -- see docstring's `direction` entry
        # and the module-level note above for why this variant's own
        # guard-dilution behavior is left alone here.
        raw_delta = [-d for d in honest_delta]

        flat, shapes, sizes = _flatten(raw_delta)
        current_norm = np.linalg.norm(flat)
        if current_norm < 1e-12:
            # Degenerate all-zero direction -- shouldn't happen for a
            # real trained update, but fall back to returning
            # global_params unmodified (zero delta) rather than divide
            # by zero.
            return [g.copy() for g in global_params]
        scaled_delta = flat * (target_norm / current_norm)
        delta_parts = _unflatten(scaled_delta, shapes, sizes)
        return [g + d for g, d in zip(global_params, delta_parts)]

    elif direction == "classifier_head_negate":
        if model_state_keys is None:
            raise ValueError(
                "direction='classifier_head_negate' requires "
                "model_state_keys (same convention as "
                "classifier_head_flip_attack)."
            )

        # v3 fix: rescale ONLY the classifier-head sub-vector to
        # target_norm -- this is the quantity the HMAC norm guard
        # actually measures (see generate_head_norm_proof(), which
        # only ever hashes/norms the classifier-head slice). The
        # backbone keeps its full, un-rescaled honest delta -- Krum's
        # bulk-slice scoring (in the HE+Krum hybrid pipeline) or no
        # check at all (pure_norm_guard mode) is what sees the
        # backbone, not this guard.
        sensitive_idx = [i for i, k in enumerate(model_state_keys)
                          if 'classifier' in k]
        if not sensitive_idx:
            raise ValueError(
                "direction='classifier_head_negate': no key in "
                "model_state_keys matched the 'classifier' prefix -- "
                "cannot locate the classifier-head slice to attack."
            )

        # Flip only the head slice's sign; backbone entries pass
        # through unmodified (still the honest delta).
        raw_delta = []
        for i, d in enumerate(honest_delta):
            raw_delta.append(-d if i in sensitive_idx else d)

        head_only = [raw_delta[i] for i in sensitive_idx]
        flat_head, shapes_head, sizes_head = _flatten(head_only)
        current_head_norm = np.linalg.norm(flat_head)
        if current_head_norm < 1e-12:
            # Degenerate all-zero head delta -- shouldn't happen for a
            # real trained update; fall back to returning
            # global_params' head slice unmodified (zero delta) rather
            # than divide by zero, backbone still honest.
            scaled_head = [np.zeros_like(h) for h in head_only]
        else:
            scaled_head_flat = flat_head * (target_norm / current_head_norm)
            scaled_head = _unflatten(scaled_head_flat, shapes_head, sizes_head)

        final_delta = list(raw_delta)
        for pos, i in enumerate(sensitive_idx):
            final_delta[i] = scaled_head[pos]

        return [g + d for g, d in zip(global_params, final_delta)]

    else:
        raise ValueError(f"Unknown direction: {direction!r} "
                          f"(expected 'negate' or 'classifier_head_negate')")

def estimate_norm_guard_tau(prior_round_honest_head_norms, k):
    """
    Best-effort ONLINE estimate of the HMAC norm guard's MAD-k
    rejection threshold, for bounded_directional_attack_trained() when
    no explicit --bounded-tau override is supplied.

    Mirrors the median + k*MAD form used by
    defences/hmac_norm_guard.py:mad_threshold_head_norms() -- if the two
    formulas ever drift apart, THIS is what the ATTACKER assumes, not
    necessarily what the guard actually enforces; a real attacker
    estimating this online is in exactly that position, so a mismatch
    here is realistic, not a bug. For the Task 2 calibration run
    specifically, prefer passing --bounded-tau explicitly with the
    guard's own computed threshold so the "norm-guard TPR = 0% by
    design" check is exact rather than approximate.

    Parameters
    ----------
    prior_round_honest_head_norms : array-like of float
        Verified classifier-head L2 norms from HONEST clients only, in
        the PREVIOUS round (the attacker cannot see this round's norms
        before deciding its own update -- this is a causal estimate).
    k : float
        Same MAD multiplier as HEAD_NORM_GUARD_K.

    Returns
    -------
    tau_estimate : float
    """
    norms = np.asarray(prior_round_honest_head_norms, dtype=np.float64)
    if norms.size == 0:
        raise ValueError(
            "estimate_norm_guard_tau needs at least one prior-round "
            "honest norm to estimate against -- on round 1 (no prior "
            "round exists yet), pass --bounded-tau explicitly instead."
        )
    median = np.median(norms)
    mad = np.median(np.abs(norms - median))
    return float(median + k * mad)


def sign_flip_attack(global_params, scale=5.0):
    """
    NAIVE sign-flip -- KEPT FOR REPRODUCIBILITY OF PRIOR RESULTS ONLY.
    DO NOT USE THIS FOR NEW EXPERIMENTS. See sign_flip_attack_trained()
    below for the literature-standard version.

    ------------------------------------------------------------------
    WHY THIS IS NON-STANDARD (confirmed via literature search):
    ------------------------------------------------------------------
    This function operates on `global_params` -- last round's
    UNTOUCHED global model -- never on anything the client actually
    trained. The canonical sign-flip attack, per multiple independent
    papers, computes the attacker's OWN honest update first and THEN
    negates/scales it:

      - RSA (Byzantine-Robust Stochastic Aggregation Methods):
        "a Byzantine worker i first calculates the true value, and
        then sends sigma times that value to the master, where sigma
        is a negative constant." sigma=-4 is their tested value --
        this codebase's scale=5.0 (network) / scale=2.0 (application)
        sit in the same range, so the MAGNITUDE was never the problem,
        only the missing "calculates the true value" step.
      - SpectralKrum: "for benign update b, submit d=-b" -- b is the
        benign (i.e. actually locally-computed) update.
      - FedSV: "switching the sign of the weights" -- the client's own
        computed weights, not the untouched global model.

    Every one of these defines the attack as: TRAIN NORMALLY, THEN
    NEGATE. This function skips the training step entirely, which
    creates two problems for any Krum/Byzantine-detection experiment
    that uses it:
      1. It's a strictly easier target -- the "attack" carries zero
         client-specific training variance, so Byzantine clients are
         bitwise-identical to each other every single round, on top of
         being an extreme outlier. A distance-based defense catching
         this is catching "this client never trained," not "this
         client's trained update is corrupted" -- the same failure
         mode already found and fixed in classifier_head_flip_attack's
         pre-fix version (see that function's docstring).
      2. Any "detection rate is ε-invariant" finding produced using
         this function is a real result, but about a narrower,
         easier-than-canonical attacker -- it should be labeled as
         such (e.g. "against a replay-and-negate attacker") rather
         than implied to generalize to the standard literature attack.

    Scale guidance (still valid, unaffected by the above):
      network model     -> scale=5.0  (large gradients, won't NaN)
      application model -> scale=2.0  (smaller gradients, prevents overflow)
      RSA's own tested value is sigma=-4 -- confirms this codebase's
      2.0-5.0 range is not unusually aggressive or unusually weak.
    """
    return [-scale * p for p in global_params]


def sign_flip_attack_trained(trained_params, scale=5.0):
    """
    Literature-standard sign-flip attack (Blanchard et al. 2017 family;
    confirmed against RSA, SpectralKrum, FedSV formulations -- see
    sign_flip_attack()'s docstring above for the full citation trail).

    Takes the client's OWN locally-trained parameters (computed via a
    normal train() call on that client's real local data, exactly like
    an honest client would) and negates + scales them. This is what
    "sign-flip" means in every independent source checked: compute the
    true/honest update first, THEN flip it -- not replay-and-negate the
    untouched global model.

    Call this AFTER training the attacking client normally, e.g.:

        train(model, X_tr, y_tr, criterion, epochs=..., lr=...,
              global_params=global_params, mu=prox_mu, device=device)
        trained_params = get_model_parameters(model)
        params = sign_flip_attack_trained(trained_params, scale=ATTACK_SCALE)

    This mirrors exactly how classifier_head_flip_attack() is already
    invoked (train first, corrupt the result) -- same fix pattern,
    applied to the plain full-model sign-flip path that was missed the
    first time.

    Parameters
    ----------
    trained_params : list[np.ndarray]
        The attacking client's own locally-trained parameters -- NOT
        global_params. Caller is responsible for actually training
        first; this function does not train anything itself, to keep
        it a pure, easily-testable transform (matching
        classifier_head_flip_attack's structure).
    scale : float
        Same convention as sign_flip_attack() -- 5.0 (network) / 2.0
        (application) by default in this codebase, within RSA's tested
        sigma=-4 range.

    Returns
    -------
    poisoned : list[np.ndarray]
        Negated, scaled version of trained_params.
    """
    return [-scale * p for p in trained_params]


def gaussian_attack(global_params, std=10.0):
    """
    Gaussian noise attack — adds large random noise to parameters.

    Less targeted than sign-flip but harder for the server to predict.
    std=10.0 produces noise roughly 10x larger than typical gradient
    magnitudes in the CNN-LSTM.

    NOTE: this ALSO operates on global_params, not a trained update --
    same non-standard pattern flagged in sign_flip_attack()'s docstring
    above. KEPT FOR REPRODUCIBILITY / AS A REFERENCE ONLY, same status
    as sign_flip_attack(). See gaussian_attack_trained() below for the
    literature-standard, train-first version -- use that one for any
    new experiment.
    """
    return [p + np.random.normal(0, std, p.shape).astype(np.float32)
            for p in global_params]


def gaussian_attack_trained(trained_params, std=10.0):
    """
    Gaussian noise attack applied to a client's OWN locally-trained
    parameters -- same "train first, then corrupt" fix pattern as
    sign_flip_attack_trained() and classifier_head_flip_attack() above.
    Adds i.i.d. Gaussian noise on top of an otherwise-honest trained
    update, rather than perturbing the untouched global model directly
    (see gaussian_attack()'s docstring for why the untouched-global-
    model version carries the same non-standard-attacker problem
    sign_flip_attack() did).

    A genuinely different attack signature from sign-flip, worth having
    alongside it: no consistent direction (each attacking client's
    noise draw is independent), so two Byzantine clients under this
    attack are NOT bitwise-identical to each other the way the naive
    (untrained) versions of either attack were -- a meaningfully
    different geometric shape for Krum's distance-based scoring to
    contend with than a coordinated negation.

    Call this AFTER training the attacking client normally, exactly
    like sign_flip_attack_trained():

        train(model, X_tr, y_tr, criterion, epochs=..., lr=...,
              global_params=global_params, mu=prox_mu, device=device)
        trained_params = get_model_parameters(model)
        params = gaussian_attack_trained(trained_params, std=GAUSSIAN_STD)

    Parameters
    ----------
    trained_params : list[np.ndarray]
        The attacking client's own locally-trained parameters -- NOT
        global_params.
    std : float
        Standard deviation of the added Gaussian noise. Default 10.0
        matches gaussian_attack()'s existing default, so any earlier
        informal experimentation with that magnitude stays a valid
        reference point for this version too.

    Returns
    -------
    poisoned : list[np.ndarray]
        trained_params with i.i.d. Gaussian noise (mean 0, std=std)
        added elementwise to every layer.
    """
    return [p + np.random.normal(0, std, p.shape).astype(np.float32)
            for p in trained_params]


def zero_gradient_attack(global_params):
    """
    Zero gradient attack — Byzantine client sends all zeros.

    Represents a lazy/inactive Byzantine client. Less aggressive than
    sign-flip but still distorts the aggregate by contributing no
    useful gradient signal. Used to test sensitivity to passive attacks.

    NOTE: this one is arguably fine as-is -- "send all zeros" doesn't
    have a meaningful "trained-then-negated" version; a lazy client
    sending zeros IS the attack, independent of what it would have
    computed. No fix needed here. Callers may pass either global_params
    or a trained client's params -- output is identical either way
    since only the input's SHAPE is used, all values become zero.
    """
    return [np.zeros_like(p) for p in global_params]


def classifier_head_flip_attack(global_params, model_state_keys, scale=5.0):
    """
    Targeted Byzantine attack: flips only classifier-head parameters,
    leaving the backbone (CNN + LSTM layers) clean.

    Designed to test whether partial HE creates a Krum blind spot: the
    classifier head is the part of the model that gets CKKS-encrypted
    before reaching the server, while the backbone (~94% of params)
    arrives in plaintext. Krum computes pairwise distances on the full
    flattened parameter vector it receives — if the classifier-head
    slice arrives as ciphertext, Krum's distance computation only ever
    sees the backbone, and a Byzantine client that keeps its backbone
    clean while poisoning only the classifier head can evade detection
    entirely. This function generates exactly that attack.

    FIX (this revision): signature corrected back to match main.py's
    actual call site and the original documented experiment spec.
    A previous version of this function took a full `model` (nn.Module)
    as its second argument and returned a 3-tuple
    (poisoned, poisoned_keys, clean_keys) — but main.py has always
    called it as:
        model_state_keys = list(model.state_dict().keys())
        params = classifier_head_flip_attack(
            global_params, model_state_keys, scale=ATTACK_SCALE
        )
    i.e. passing an already-extracted list of key STRINGS (not a model
    object) and expecting a single list back, not a tuple. The old
    signature would crash with AttributeError the first time
    BYZANTINE_HEAD_ONLY=True was actually exercised (list has no
    .state_dict() method), and even patched around that, unpacking a
    3-tuple into `params` would have silently broken every downstream
    consumer (HE encryption, weighted aggregation) that expects params
    to be a flat list of arrays. This version matches the call site
    exactly — no changes needed in main.py.

    NOTE: despite the parameter being named `global_params` here, the
    ACTUAL call site in main.py passes this function TRAINED params
    (trained_params = get_model_parameters(model), after a real
    train() call) -- this function itself is agnostic to that
    distinction, it just flips whatever's in the classifier-head slice
    of whatever list it's given. The parameter name is a holdover and
    slightly misleading; the call site is what makes this the
    literature-correct "train first, then poison" pattern. Contrast
    with sign_flip_attack() above, which genuinely does receive
    untouched global_params -- that's the actual bug, not this
    function's naming.

    Parameters
    ----------
    global_params : list[np.ndarray]
        In practice, the caller's locally-trained parameters (see NOTE
        above) — despite the parameter name, this is not necessarily
        the untouched global model.
    model_state_keys : list[str]
        Ordered state_dict() key names, in the same order as
        global_params. Caller extracts this once via
        list(model.state_dict().keys()) — this function does not need
        the model object itself, only the key names, so the
        classifier-vs-backbone split is driven by real key names
        rather than a hardcoded index range that could silently drift
        if the architecture changes.
    scale : float
        Sign-flip magnitude applied to classifier-head params only.
        Matches sign_flip_attack's scale convention for comparability.

    Returns
    -------
    poisoned : list[np.ndarray]
        Full parameter list — classifier-head entries flipped and
        scaled, everything else returned as a clean copy of the
        original input (i.e. this client sends back exactly what it
        computed for the backbone, no poisoning there).
    """
    poisoned = []
    for key, param in zip(model_state_keys, global_params):
        if 'classifier' in key:
            poisoned.append(-scale * param)
        else:
            poisoned.append(param.copy())
    return poisoned


# ===========================================================================
# Issue 5 Task 1 -- attack dispatch table
# ===========================================================================
# Acceptance item: "5 attack families selectable via --attack flag; all 5
# selectable via identical --attack CLI flag interface." This codebase's
# existing CLI flag is named --attack-type (see main.py) and already spells
# the easy-control attack "sign_flip" with an underscore; both the issue's
# literal spelling ("signflip") and this codebase's established spelling
# are provided as aliases below so neither convention breaks. Defined here,
# at module end, so every referenced function already exists above.
ATTACK_DISPATCH = {
    "signflip":            sign_flip_attack_trained,
    "sign_flip":           sign_flip_attack_trained,
    "gaussian":            gaussian_attack_trained,
    "minmax":              minmax_attack_trained,
    "minsum":              minsum_attack_trained,
    "bounded_directional": bounded_directional_attack_trained,
}