import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, recall_score, average_precision_score
from sklearn.preprocessing import label_binarize

from data_loader import (
    NETWORK_NAMES, NUM_NETWORK_CLASSES,
    APP_NAMES,     NUM_APP_CLASSES,
    get_class_counts_network, get_class_counts_application
)

from model_defs import (
    CNN_LSTM, NUM_FEATURES, get_model,
    get_model_parameters, get_model_parameter_keys, set_model_parameters
)

from config_loader import load_hyperparams_config

_hp_cfg = load_hyperparams_config()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('weight', weight, persistent=False)

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(
            inputs, targets, weight=self.weight, reduction='none'
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def _inverse_sqrt_weights(counts):
    w = torch.tensor([1.0 / (c ** 0.5) for c in counts])
    w = w / w.sum() * len(counts)
    return w


def build_criterion_network(seed=42):
    counts = get_class_counts_network(seed=seed)
    return FocalLoss(gamma=2.0, weight=_inverse_sqrt_weights(counts))


def build_criterion_application(seed=42):
    counts = get_class_counts_application(seed=seed)
    w = _inverse_sqrt_weights(counts)
    MAX_WEIGHT_RATIO = 5.0
    w = torch.clamp(w, min=w.max() / MAX_WEIGHT_RATIO)
    name_to_idx = {name: i for i, name in enumerate(APP_NAMES)}
    multipliers = _hp_cfg["class_weight_multipliers_application"]
    w[name_to_idx['Uploading']] *= multipliers['Uploading']
    w[name_to_idx['XSS']] *= multipliers['XSS']
    w[name_to_idx['Fingerprinting']] *= multipliers['Fingerprinting']
    w = w / w.mean()
    return FocalLoss(gamma=2.0, weight=w)


def _proximal_term(model, global_params):
    if global_params is None:
        return None
    state_keys = list(model.state_dict().keys())
    global_dict = dict(zip(state_keys, global_params))
    total = None
    for name, param in model.named_parameters():
        g = torch.tensor(global_dict[name], dtype=torch.float32, device=param.device)
        term = torch.sum((param - g) ** 2)
        total = term if total is None else total + term
    return total


def train(model, X_train, y_train, criterion,
          epochs=5, lr=0.001, global_params=None, mu=0.01, device='cpu'):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.95)

    X = torch.FloatTensor(X_train)
    y = torch.LongTensor(y_train)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y), batch_size=256, shuffle=True
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
    return model


def test(model, X_test, y_test, num_classes, device='cpu', return_extended=False):
    model.eval()
    X = torch.FloatTensor(X_test).to(device)
    y = torch.LongTensor(y_test).to(device)

    with torch.no_grad():
        out = model(X)
        loss = nn.CrossEntropyLoss()(out, y).item()
        preds = torch.argmax(out, dim=1).cpu().numpy()
        if return_extended:
            probs = torch.softmax(out, dim=1).cpu().numpy()

    accuracy = float((preds == y_test).mean())
    per_class_f1 = f1_score(y_test, preds, average=None,
                             labels=np.arange(num_classes), zero_division=0)

    if not return_extended:
        return loss, accuracy, per_class_f1

    per_class_recall = recall_score(y_test, preds, average=None,
                                     labels=np.arange(num_classes), zero_division=0)

    y_true_bin = label_binarize(y_test, classes=np.arange(num_classes))
    if num_classes == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    per_class_aucpr = np.full(num_classes, np.nan)

    # A model that has diverged during training (e.g. an unmitigated
    # Byzantine attack under an aggregator with no robustness -- FedAvg
    # under sign-flip is the known case) can produce non-finite (NaN/inf)
    # softmax probabilities. average_precision_score's own internal
    # assert_all_finite() raises a hard ValueError on that input, which
    # previously killed the entire run/process rather than recording
    # what is itself a legitimate, reportable result: this
    # (aggregator, attack) combination causes catastrophic divergence.
    # Guard per-class so a run isn't lost to this -- report NaN for any
    # class whose probs aren't finite, same convention already used
    # above for a class absent from the test set, and print a clear,
    # one-time diagnostic distinguishing this cause from that one so
    # it's traceable in the run's stdout/log rather than silently
    # blending into ordinary "no support" NaNs.
    _diverged_logged = False
    for c in range(num_classes):
        if y_true_bin[:, c].sum() == 0:
            continue
        class_probs = probs[:, c]
        if not np.all(np.isfinite(class_probs)):
            if not _diverged_logged:
                n_nonfinite_rows = int((~np.isfinite(probs).all(axis=1)).sum())
                print(f"  [WARNING] test(): model output contains non-finite "
                      f"probabilities ({n_nonfinite_rows}/{len(probs)} rows "
                      f"affected) -- likely training divergence (e.g. an "
                      f"unmitigated Byzantine attack). Affected per-class "
                      f"AUC-PR values are reported as NaN rather than "
                      f"crashing the run; per-class F1/Recall above still "
                      f"reflect the (likely very poor) argmax predictions.")
                _diverged_logged = True
            continue  # leave per_class_aucpr[c] as NaN
        per_class_aucpr[c] = average_precision_score(y_true_bin[:, c], class_probs)

    return loss, accuracy, per_class_f1, per_class_recall, per_class_aucpr
