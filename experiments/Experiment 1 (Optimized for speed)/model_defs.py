"""
Model definitions — kept dependency-free on purpose (beyond torch, and
opacus ONLY when dp_safe=True is actually requested — see below).

This module is imported by train_worker.py, which runs inside a
256MB-constrained subprocess alongside torch. It must NEVER import
pandas, scikit-learn, or data_loader — those add 80-150MB of RSS for
functionality train_worker.py never uses (evaluation metrics, class
counts, dataset loading), and were previously being loaded anyway
because train_worker.py did `from task import get_model, ...`, which
executes task.py's entire module body, including its sklearn/pandas
imports, regardless of which names were actually requested.

task.py (server-side / evaluation-side, no RAM constraint) imports
its model class and parameter helpers FROM this module instead of
defining them itself, so there's exactly one definition of the
architecture and no duplication.

DP_SAFE ARCHITECTURE (NEW)
---------------------------
Opacus's per-sample gradient computation is incompatible with two
layers the original architecture used unconditionally:

  1. nn.BatchNorm1d — its running statistics are computed ACROSS the
     batch, which directly conflicts with the per-sample independence
     Opacus's clipping mechanism requires. Fixed by swapping to
     nn.GroupNorm (num_groups=8) when dp_safe=True — GroupNorm
     normalizes within each sample independently, so it has no
     cross-sample dependency and is fully compatible.

  2. nn.LSTM — Opacus does not support per-sample gradients for the
     standard cuDNN-backed nn.LSTM implementation. Fixed by swapping
     to opacus.layers.DPLSTM (a drop-in, same-interface replacement)
     when dp_safe=True. opacus is imported lazily, only inside this
     branch, so the default (non-DP) path has zero opacus dependency,
     consistent with this module's "dependency-free beyond torch"
     design intent for train_worker.py's constrained subprocess.

CRITICAL: dp_safe changes state_dict key names AND parameter shapes
(GroupNorm/DPLSTM have different internal parameter structure than
BatchNorm1d/LSTM). A model built with dp_safe=True cannot load a
checkpoint saved with dp_safe=False, and vice versa — set_model_parameters()
will fail on mismatched state_dict keys/shapes if this isn't kept
consistent across checkpoint init, training, and eval within one run.
Callers MUST use the same dp_safe value everywhere for a given
experiment condition (this is exactly why main.py derives
DP_SAFE = USE_DP once, at the top of the file, rather than passing a
possibly-inconsistent value at each call site).
"""

import torch
import torch.nn as nn

# Default — overridden dynamically via get_model(num_features=)
NUM_FEATURES = 52


# ── Model architecture ──────────────────────────────────────────────

class CNN_LSTM(nn.Module):
    """
    1D CNN + LSTM for network traffic classification.
    num_features is passed in dynamically so both models
    (network=40 features, application=52 features) use the
    same architecture class without hardcoding.

    dp_safe=False (default): nn.BatchNorm1d + nn.LSTM — original
        architecture, used for all non-DP-SGD runs (baselines,
        Krum-only, HE-only, etc.) so those results stay comparable
        to prior runs that used this exact architecture.
    dp_safe=True: nn.GroupNorm + opacus.layers.DPLSTM — required for
        any run using Opacus DP-SGD (USE_DP=True). See module
        docstring for why each swap is necessary.
    """
    def __init__(self, num_features=NUM_FEATURES, num_classes=8,
                 dp_safe=False):
        super().__init__()
        self.dp_safe = dp_safe

        if dp_safe:
            norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            norm2 = nn.GroupNorm(num_groups=8, num_channels=128)
        else:
            norm1 = nn.BatchNorm1d(64)
            norm2 = nn.BatchNorm1d(128)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            norm1,
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            norm2,
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        with torch.no_grad():
            dummy   = torch.zeros(1, 1, num_features)
            cnn_out = self.cnn(dummy)
            lstm_in = cnn_out.shape[1]

        if dp_safe:
            # Lazy import — keeps opacus out of the dependency chain
            # entirely for non-DP runs (e.g. train_worker.py's default
            # constrained subprocess path, or any USE_DP=False run).
            from opacus.layers import DPLSTM
            self.lstm = DPLSTM(
                input_size  = lstm_in,
                hidden_size = 64,
                num_layers  = 1,
                batch_first = True,
            )
        else:
            self.lstm = nn.LSTM(
                input_size  = lstm_in,
                hidden_size = 64,
                num_layers  = 1,
                batch_first = True,
            )

        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        x = h_n.squeeze(0)
        return self.classifier(x)


def get_model(num_features=NUM_FEATURES, num_classes=8, dp_safe=False):
    return CNN_LSTM(num_features=num_features, num_classes=num_classes,
                     dp_safe=dp_safe)


# ── Parameter helpers ────────────────────────────────────────────────
# Architecture-agnostic — work identically regardless of dp_safe, since
# they operate on state_dict() rather than assuming specific layer types.
# NOTE: the state_dict keys/shapes themselves DO differ between
# dp_safe=True and dp_safe=False models (see module docstring) — these
# helpers don't hide that; they just don't need to know about it.

def get_model_parameters(model):
    return [v.cpu().numpy() for v in model.state_dict().values()]


def get_model_parameter_keys(model):
    """Ordered state_dict key names — used by the client/server to know
    which parameter indices belong to which named layer without either
    side needing to hold a live torch model."""
    return list(model.state_dict().keys())


def set_model_parameters(model, parameters):
    keys       = list(model.state_dict().keys())
    state_dict = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    model.load_state_dict(state_dict, strict=True)
    return model