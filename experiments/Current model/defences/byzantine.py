import numpy as np


def sign_flip_attack(global_params, scale=5.0):
    """
    Sign-flip Byzantine attack (Blanchard et al. 2017, standard benchmark).

    Flips the sign of every parameter value and scales up by `scale`.
    This pushes the global model in exactly the wrong direction.

    Scale guidance:
      network model     → scale=5.0  (large gradients, won't NaN)
      application model → scale=2.0  (smaller gradients, prevents overflow)

    scale=5.0 on the network model causes complete collapse (NaN) in
    FedAvg but is detectable by Multi-Krum due to extreme outlier distance.

    scale=2.0 on the application model causes measurable F1 degradation
    without NaN overflow, producing a cleaner Krum evaluation.
    """
    return [-scale * p for p in global_params]


def gaussian_attack(global_params, std=10.0):
    """
    Gaussian noise attack — adds large random noise to parameters.

    Less targeted than sign-flip but harder for the server to predict.
    std=10.0 produces noise roughly 10x larger than typical gradient
    magnitudes in the CNN-LSTM.
    """
    return [p + np.random.normal(0, std, p.shape).astype(np.float32)
            for p in global_params]


def zero_gradient_attack(global_params):
    """
    Zero gradient attack — Byzantine client sends all zeros.

    Represents a lazy/inactive Byzantine client. Less aggressive than
    sign-flip but still distorts the aggregate by contributing no
    useful gradient signal. Used to test sensitivity to passive attacks.
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

    Parameters
    ----------
    global_params : list[np.ndarray]
        Current global model parameters, in state_dict order.
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
        original global params (i.e. this client sends back exactly
        what it received for the backbone, no poisoning there).
    """
    poisoned = []
    for key, param in zip(model_state_keys, global_params):
        if 'classifier' in key:
            poisoned.append(-scale * param)
        else:
            poisoned.append(param.copy())
    return poisoned
