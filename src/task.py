import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score

from data_loader import (
    NETWORK_NAMES, NUM_NETWORK_CLASSES, NETWORK_COUNTS,
    APP_NAMES,     NUM_APP_CLASSES,     APP_COUNTS
)

# Default — overridden dynamically in main.py via get_model(num_features=)
NUM_FEATURES = 52


# ── Model architecture ────────────────────────────────────────────────

class CNN_LSTM(nn.Module):
    """
    1D CNN + LSTM for network traffic classification.
    num_features is passed in dynamically so both models
    (network=40 features, application=52 features) use the
    same architecture class without hardcoding.
    """
    def __init__(self, num_features=NUM_FEATURES, num_classes=8):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        with torch.no_grad():
            dummy   = torch.zeros(1, 1, num_features)
            cnn_out = self.cnn(dummy)
            lstm_in = cnn_out.shape[1]

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


def get_model(num_features=NUM_FEATURES, num_classes=8):
    return CNN_LSTM(num_features=num_features, num_classes=num_classes)


# ── Parameter helpers ─────────────────────────────────────────────────

def get_model_parameters(model):
    return [v.cpu().numpy() for v in model.state_dict().values()]


def set_model_parameters(model, parameters):
    keys       = list(model.state_dict().keys())
    state_dict = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    model.load_state_dict(state_dict, strict=True)
    return model


# ── Loss ──────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017).
    Concentrates gradient on hard misclassified examples.
    gamma=2.0 for network model (easier problem).
    gamma=3.0 for application model (harder — HTTP attacks look alike).
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
    very rare (SQL_injection 1k) and moderate (Backdoor 50k) classes
    from destabilising training.
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
    """
    return FocalLoss(
        gamma  = 2.0,
        weight = _inverse_sqrt_weights(NETWORK_COUNTS)
    )


def build_criterion_application():
    """
    Loss for Model 2 (application-layer attacks).

    Backdoor (50k samples) and XSS (51k samples) together make up
    52% of the application subset but are the hardest to separate
    because both are HTTP-based DVWA attacks. Inverse sqrt gives them
    LOW weights (they look like majority classes) but they're failing.
    Manual overrides fix this by boosting their weights explicitly.

    gamma=3.0 (vs 2.0 for network) — harder problem needs more focus
    on misclassified examples.
    """
    w = _inverse_sqrt_weights(APP_COUNTS)

    # Manual overrides for feature-confused classes
    w[3] = w[3] * 5.0   # Backdoor      — 50k samples but was F1=0.28
    w[5] = w[5] * 4.0   # XSS           — confused with Backdoor
    w[6] = w[6] * 3.0   # Password      — HTTP-based, looks like XSS
    w[7] = w[7] * 3.0   # Fingerprinting — underperforming
    w[4] = w[4] * 1.5   # Port_Scanning  — moderate boost

    w = w / w.mean()
    return FocalLoss(gamma=3.0, weight=w)


# ── FedProx proximal term ─────────────────────────────────────────────

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


# ── Training ──────────────────────────────────────────────────────────

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


# ── Evaluation ────────────────────────────────────────────────────────

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