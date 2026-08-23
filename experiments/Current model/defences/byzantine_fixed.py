import numpy as np


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
    above. Not yet used in any completed experiment per the master doc,
    so no prior results depend on it, but apply the same trains-first
    fix here before using it in any new experiment that needs a
    literature-comparable attack.
    """
    return [p + np.random.normal(0, std, p.shape).astype(np.float32)
            for p in global_params]


def zero_gradient_attack(global_params):
    """
    Zero gradient attack — Byzantine client sends all zeros.

    Represents a lazy/inactive Byzantine client. Less aggressive than
    sign-flip but still distorts the aggregate by contributing no
    useful gradient signal. Used to test sensitivity to passive attacks.

    NOTE: this one is arguably fine as-is -- "send all zeros" doesn't
    have a meaningful "trained-then-negated" version; a lazy client
    sending zeros IS the attack, independent of what it would have
    computed. No fix needed here.
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
