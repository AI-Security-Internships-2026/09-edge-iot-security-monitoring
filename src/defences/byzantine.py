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

