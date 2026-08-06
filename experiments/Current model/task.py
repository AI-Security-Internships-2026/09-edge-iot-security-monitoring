import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score

from data_loader import (
    NETWORK_NAMES, NUM_NETWORK_CLASSES,
    APP_NAMES,     NUM_APP_CLASSES,
    get_class_counts_network, get_class_counts_application
)

# Model architecture + parameter helpers now live in model_defs.py
# (kept dependency-free so train_worker.py's subprocess doesn't have
# to import pandas/sklearn along with this file — see model_defs.py's
# docstring for why that mattered).
from model_defs import (
    CNN_LSTM, NUM_FEATURES, get_model,
    get_model_parameters, get_model_parameter_keys, set_model_parameters
)


# ── Loss ─────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017).
    Concentrates gradient on hard misclassified examples.
    gamma=2.0 for network model (easier problem).
    gamma=2.0 for application model too, as of this revision — see
    build_criterion_application() docstring for why 3.0 was dropped.
    """
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce  = nn.functional.cross_entropy(
            inputs, targets, weight=self.weight, reduction='none'
        )
        pt  = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def _inverse_sqrt_weights(counts):
    """
    Inverse square root class weights.
    Softer than raw inverse — prevents extreme ratios between
    very rare and moderate classes from destabilising training.
    Normalised so mean weight = 1.0.
    """
    w = torch.tensor([1.0 / (c ** 0.5) for c in counts])
    w = w / w.sum() * len(counts)
    return w


def build_criterion_network():
    """
    Loss for Model 1 (network-layer attacks).
    Inverse sqrt weights handle class imbalance.
    gamma=2.0 standard for this problem difficulty.

    NOTE: counts are now computed LIVE from the corrected data cache
    via get_class_counts_network(), not a hardcoded NETWORK_COUNTS
    table. The old table was derived under the LabelEncoder bug (see
    data_loader.py header comment / verify_label_bug.py) and no longer
    exists — computing live means this can never drift out of sync
    with whatever data was actually used to build the cache again.
    """
    counts = get_class_counts_network()
    return FocalLoss(
        gamma  = 2.0,
        weight = _inverse_sqrt_weights(counts)
    )


def build_criterion_application():
    """
    Loss for Model 2 (application-layer attacks).

    HISTORY (read before changing weights again):
      1. Original manual overrides (w[3]*=5.0 etc.) were tuned while
         the LabelEncoder bug was still active — the per-class F1
         numbers being watched at the time were attached to the WRONG
         class names, so those specific multipliers were meaningless
         once labels were fixed.
      2. After the label fix, those same stale multipliers stacked
         with the real (much more severe, ~132:1 Normal:Fingerprinting)
         class imbalance and caused Normal to collapse to F1=0.0 —
         the model had almost no gradient incentive to get the
         majority class right.
      3. Overrides were removed entirely and MAX_WEIGHT_RATIO clamping
         added instead — this produced a sane, non-collapsed baseline
         (round 25: F1-Macro=0.5565, Normal=0.73, but Uploading=0.36,
         XSS=0.17, Fingerprinting=0.27 still weak).
      4. THIS REVISION: small, targeted boosts (1.3-1.5x) reintroduced
         for the three classes that are ACTUALLY weak on that real
         round-25 baseline — applied AFTER the ratio clamp, so they're
         bounded on both ends and can't reproduce failure mode #2.
         check_feature_signal.py already confirmed the underlying
         features carry real signal for these classes (tcp_payload_*
         especially) — this is a loss-weighting change, not a feature
         fix, because the features were never the problem.

    IMPORTANT: change PROX_MU (in main.py) and these weights in
    SEPARATE experiments, not the same run — otherwise you can't tell
    which change caused any observed improvement.

    gamma dropped 3.0 -> 2.0 in the previous revision (matching the
    network model) — 3.0 was tuned under the old, much milder apparent
    imbalance and was likely too aggressive against the real ratio.
    Left at 2.0 here; revisit only after PROX_MU and these weights have
    each been tested in isolation.
    """
    counts = get_class_counts_application()
    w = _inverse_sqrt_weights(counts)

    MAX_WEIGHT_RATIO = 5.0
    w = torch.clamp(w, min=w.max() / MAX_WEIGHT_RATIO)

    name_to_idx = {name: i for i, name in enumerate(APP_NAMES)}

    # Targeted boosts for classes still underperforming on the REAL
    # corrected-data round-25 baseline. Small, bounded multipliers —
    # re-tune these specific values against your NEXT run's actual
    # numbers rather than trusting them as final; this is a starting
    # point, not a converged answer.
    w[name_to_idx['Uploading']]      *= 1.3
    w[name_to_idx['XSS']]            *= 1.5
    w[name_to_idx['Fingerprinting']] *= 1.3

    w = w / w.mean()
    return FocalLoss(gamma=2.0, weight=w)


# ── FedProx proximal term ───────────────────────────────────────────

def _proximal_term(model, global_params):
    """
    FedProx: (mu/2) * ||w - w_global||^2
    Matched by parameter name to avoid BatchNorm buffer misalignment.
    Returns None when global_params is None (plain FedAvg mode).
    """
    if global_params is None:
        return None

    state_keys  = list(model.state_dict().keys())
    global_dict = dict(zip(state_keys, global_params))

    total = None
    for name, param in model.named_parameters():
        g    = torch.tensor(global_dict[name], dtype=torch.float32)
        term = torch.sum((param - g) ** 2)
        total = term if total is None else total + term
    return total


# ── Training ─────────────────────────────────────────────────────────

def train(model, X_train, y_train, criterion,
          epochs=5, lr=0.001, global_params=None, mu=0.01):
    """
    Local training step.

    global_params=None  → plain FedAvg
    global_params=list  → FedProx (proximal term added to loss)

    StepLR(step_size=3, gamma=0.95) — gentle decay, less aggressive
    than previous step_size=2 gamma=0.9 which caused collapse rounds.

    Gradient clipping (max_norm=1.0) prevents explosion.

    DEFENCE HOOK — DP-SGD:
        wrap optimizer with Opacus PrivacyEngine before epoch loop
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.95
    )

    X = torch.FloatTensor(X_train)
    y = torch.LongTensor(y_train)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y),
        batch_size=256, shuffle=True
    )

    for _ in range(epochs):
        for X_b, y_b in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            prox = _proximal_term(model, global_params)
            if prox is not None:
                loss = loss + (mu / 2) * prox
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()
        scheduler.step()

    return model


# ── Evaluation ───────────────────────────────────────────────────────

def test(model, X_test, y_test, num_classes):
    """
    Returns (loss, accuracy, per_class_f1_array).
    Uses standard CrossEntropyLoss for evaluation so loss values
    are comparable across rounds and experiments.
    """
    model.eval()
    X = torch.FloatTensor(X_test)
    y = torch.LongTensor(y_test)

    with torch.no_grad():
        out   = model(X)
        loss  = nn.CrossEntropyLoss()(out, y).item()
        preds = torch.argmax(out, dim=1).numpy()

    accuracy     = float((preds == y_test).mean())
    per_class_f1 = f1_score(
        y_test, preds,
        average       = None,
        labels        = np.arange(num_classes),
        zero_division = 0
    )
    return loss, accuracy, per_class_f1