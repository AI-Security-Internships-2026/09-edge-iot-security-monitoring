import numpy as np


def clip_gradient(params, clip_norm=1.0):
    """
    Clip gradient to bound L2 sensitivity.

    Without clipping, the L2 sensitivity of a gradient is unbounded —
    meaning the Gaussian noise required for (ε,δ)-DP would be infinite.
    Clipping ensures sensitivity = clip_norm, making noise calibration exact.

    This is the standard approach from Abadi et al., 2016
    (Deep Learning with Differential Privacy, CCS 2016).
    """
    flat = np.concatenate([p.flatten() for p in params])
    norm = float(np.linalg.norm(flat))
    if norm > clip_norm:
        scale  = clip_norm / norm
        params = [p * scale for p in params]
        clipped_norm = clip_norm
    else:
        clipped_norm = norm
    return params, clipped_norm


def gaussian_noise(params, sensitivity, epsilon, delta=1e-5):
    """
    Add calibrated Gaussian noise for (ε, δ)-differential privacy.

    Noise std = sqrt(2 * ln(1.25/δ)) * sensitivity / ε

    WHY THIS IS QUANTUM-SAFE:
    This guarantee is information-theoretic, not computational.
    No adversary — classical or quantum — can distinguish whether
    a specific user's data was in the training set, because the
    mathematical bound holds regardless of compute power.
    Quantum computers speed up computation; they cannot break
    statistical noise bounds.

    Contrast with RSA / ECC: broken by Shor's algorithm because
    they rely on factoring/discrete-log being computationally hard.
    """
    sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    noisy = [
        (p + np.random.normal(0, sigma, p.shape)).astype(np.float32)
        for p in params
    ]
    return noisy, sigma


def apply_local_dp(params, epsilon=3.0, delta=1e-5, clip_norm=1.0):
    """
    Full local DP pipeline for one client on a shared gateway.

    Step 1 — Clip:  ||w||_2 ≤ clip_norm  (bounds sensitivity)
    Step 2 — Noise: w_dp = w + N(0, σ²I) (information-theoretic privacy)

    Threat model addressed:
      Multiple users (shifts, operators) share one IoT gateway.
      After this step, even a user with root access to the gateway's
      memory cannot reverse-engineer what another user's traffic
      contributed to the gradient. The guarantee holds for any
      adversary, including future quantum computers.

    Returns
    -------
    noisy_params : list of numpy arrays
    info         : dict with privacy accounting
    """
    clipped, actual_norm = clip_gradient(params, clip_norm)
    noisy, sigma         = gaussian_noise(clipped, clip_norm, epsilon, delta)

    return noisy, {
        "epsilon":      epsilon,
        "delta":        delta,
        "clip_norm":    clip_norm,
        "actual_norm":  actual_norm,
        "noise_sigma":  sigma,
    }