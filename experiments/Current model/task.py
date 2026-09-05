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

# DAT1 Task 2 -- the application-model class-weight multipliers below
# (Uploading/XSS/Fingerprinting) are now config-driven rather than
# inline numeric literals. See config_loader.py and
# experiments/configs/hyperparams.json.
from config_loader import load_hyperparams_config

_hp_cfg = load_hyperparams_config()


# ── Loss ─────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017).
    Concentrates gradient on hard misclassified examples.
    gamma=2.0 for network model (easier problem).
    gamma=2.0 for application model too, as of this revision — see
    build_criterion_application() docstring for why 3.0 was dropped.

    GPU FIX: weight is now registered via register_buffer() instead of
    a plain attribute assignment. A plain `self.weight = weight`
    attribute is invisible to nn.Module's .to(device)/.cuda() machinery
    — only registered parameters and buffers get moved. Previously,
    calling criterion.to(device) in main.py silently did nothing to
    this tensor, which would have caused a CPU-vs-GPU device-mismatch
    RuntimeError the moment cross_entropy tried to use it against
    GPU-resident logits (or, if weight is None, no error but silently
    wrong — no, N/A here since weight is always a real tensor in this
    codebase's call sites). register_buffer(..., persistent=False)
    keeps it out of state_dict() (this is a fixed loss-weighting
    tensor, not a trainable/checkpointed parameter) while still making
    it participate correctly in .to()/.cuda()/.cpu() calls.
    """
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('weight', weight, persistent=False)

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


def build_criterion_network(seed=42):
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

    DAT1: `seed` must be the run's actual --seed, not left at the
    default — counts now come from that seed's TRAIN split ONLY
    (never VAL or TEST, per DAT1's no-tuning-on-test requirement), and
    different seeds have different TRAIN splits, so this must match
    whatever seed load_partition_network() is using for this same run.

    No device kwarg here deliberately — main.py already calls
    .to(_DEVICE) on the returned criterion. With the register_buffer
    fix above, that .to() call now actually works.
    """
    counts = get_class_counts_network(seed=seed)
    return FocalLoss(
        gamma  = 2.0,
        weight = _inverse_sqrt_weights(counts)
    )


def build_criterion_application(seed=42):
    """
    Loss for Model 2 (application-layer attacks).

    ⚠ DAT1 FLAG — READ BEFORE TRUSTING THESE MULTIPLIERS:
    The 1.3–1.5x boosts below (history item 4) were originally chosen
    by observing "round-25" per-class performance under the OLD
    pipeline's _dirichlet_partition() X_test/y_test — which was, at
    the time, each client's own local held-out split, NOT a proper
    global TRAIN/VAL/TEST holdout (that didn't exist yet). Functionally,
    this means the metric used to pick these multipliers was the
    closest thing to a test-like signal available at the time, which
    is exactly the tuning-on-test risk DAT1 flags. Now that a genuine
    global TEST holdout exists (get_global_test_holdout()) that is
    NEVER touched until final evaluation, these specific multiplier
    values are UNVALIDATED against that holdout and should be treated
    as provisional, carried over from before this pipeline fix — they
    need to be re-derived by tuning against VALIDATION-split metrics
    only (per-round local-val performance, or a proper held-out VAL
    pass) before being cited in the paper as deliberately chosen, not
    just inherited. This is DAT1 Task 2 (move to config + validation-
    only provenance requirement) — not yet done; flagged here in the
    interim so this isn't silently carried forward as if resolved.

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
    counts = get_class_counts_application(seed=seed)
    w = _inverse_sqrt_weights(counts)

    MAX_WEIGHT_RATIO = 5.0
    w = torch.clamp(w, min=w.max() / MAX_WEIGHT_RATIO)

    name_to_idx = {name: i for i, name in enumerate(APP_NAMES)}

    # Targeted boosts for classes still underperforming on the REAL
    # corrected-data round-25 baseline. Small, bounded multipliers —
    # DAT1 Task 2: now read from config rather than hardcoded, and
    # explicitly flagged UNVALIDATED there (see hyperparams.json) until
    # re-derived against VALIDATION-split per-class F1, per the
    # docstring history above -- re-tune the config values against
    # your NEXT run's actual numbers rather than trusting them as
    # final; this is a starting point, not a converged answer.
    multipliers = _hp_cfg["class_weight_multipliers_application"]
    w[name_to_idx['Uploading']]      *= multipliers['Uploading']
    w[name_to_idx['XSS']]            *= multipliers['XSS']
    w[name_to_idx['Fingerprinting']] *= multipliers['Fingerprinting']

    w = w / w.mean()
    return FocalLoss(gamma=2.0, weight=w)


# ── FedProx proximal term ───────────────────────────────────────────

def _proximal_term(model, global_params):
    """
    FedProx: (mu/2) * ||w - w_global||^2
    Matched by parameter name to avoid BatchNorm buffer misalignment.
    Returns None when global_params is None (plain FedAvg mode).

    GPU FIX: g used to be created via torch.tensor(..., dtype=float32)
    with no device argument, which always lands on CPU regardless of
    where `param` (and the rest of the model) actually live. Comparing
    a CPU tensor against a CUDA tensor in (param - g) would raise a
    device-mismatch RuntimeError the first time this ran on GPU. Now
    built directly on param.device.
    """
    if global_params is None:
        return None

    state_keys  = list(model.state_dict().keys())
    global_dict = dict(zip(state_keys, global_params))

    total = None
    for name, param in model.named_parameters():
        g    = torch.tensor(global_dict[name], dtype=torch.float32,
                            device=param.device)
        term = torch.sum((param - g) ** 2)
        total = term if total is None else total + term
    return total


# ── Training ─────────────────────────────────────────────────────────

def train(model, X_train, y_train, criterion,
          epochs=5, lr=0.001, global_params=None, mu=0.01, device='cpu'):
    """
    Local training step.

    global_params=None  → plain FedAvg
    global_params=list  → FedProx (proximal term added to loss)

    StepLR(step_size=3, gamma=0.95) — gentle decay, less aggressive
    than previous step_size=2 gamma=0.9 which caused collapse rounds.

    Gradient clipping (max_norm=1.0) prevents explosion.

    GPU FIX (device kwarg, new): caller (main.py's _train_one_client)
    already moves `model` to `device` before calling this. X_train/
    y_train arrive as numpy arrays and are converted to CPU tensors by
    torch.FloatTensor/LongTensor regardless — those must be moved to
    `device` too, per-batch, exactly like the DP-SGD path in main.py
    already does (DataLoader always yields CPU tensors regardless of
    the source TensorDataset's device). Defaults to 'cpu' so any
    existing non-device-aware call site keeps working unmodified.
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
            X_b = X_b.to(device)
            y_b = y_b.to(device)
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

def test(model, X_test, y_test, num_classes, device='cpu'):
    """
    Returns (loss, accuracy, per_class_f1_array).
    Uses standard CrossEntropyLoss for evaluation so loss values
    are comparable across rounds and experiments.

    GPU FIX (device kwarg, new): X/y moved to `device` before the
    forward pass. Predictions are explicitly brought back to CPU via
    .cpu() before .numpy() — calling .numpy() directly on a CUDA
    tensor raises immediately ("can't convert cuda:0 device type
    tensor to numpy"), and y_test (still a plain numpy array from the
    caller) needs preds in numpy form on the CPU side to compare
    against. Defaults to 'cpu' so any existing non-device-aware call
    site keeps working unmodified.
    """
    model.eval()
    X = torch.FloatTensor(X_test).to(device)
    y = torch.LongTensor(y_test).to(device)

    with torch.no_grad():
        out   = model(X)
        loss  = nn.CrossEntropyLoss()(out, y).item()
        preds = torch.argmax(out, dim=1).cpu().numpy()

    accuracy     = float((preds == y_test).mean())
    per_class_f1 = f1_score(
        y_test, preds,
        average       = None,
        labels        = np.arange(num_classes),
        zero_division = 0
    )
    return loss, accuracy, per_class_f1
