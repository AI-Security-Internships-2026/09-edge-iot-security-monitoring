"""
Model architecture + parameter helpers — deliberately dependency-free
beyond torch. Imported by BOTH:
  - train_worker.py (the RAM-constrained subprocess), which needs
    ONLY the model class and parameter helpers, never the loss
    functions, evaluation code, or data_loader/sklearn/pandas stack.
  - task.py (server side), which imports from here instead of
    redefining the model, so there's a single source of truth.

Keeping this file free of sklearn/pandas/data_loader imports is what
lets train_worker.py fit inside the 256MB cgroup alongside torch.
"""

import torch
import torch.nn as nn

NUM_FEATURES = 52  # default — overridden dynamically via get_model(num_features=)


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


def get_model_parameters(model):
    return [v.cpu().numpy() for v in model.state_dict().values()]


def get_model_parameter_keys(model):
    """Ordered state_dict key names — used by client/server to know
    which parameter indices belong to which named layer without
    either side needing to hold a live torch model."""
    return list(model.state_dict().keys())


def set_model_parameters(model, parameters):
    keys       = list(model.state_dict().keys())
    state_dict = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    model.load_state_dict(state_dict, strict=True)
    return model