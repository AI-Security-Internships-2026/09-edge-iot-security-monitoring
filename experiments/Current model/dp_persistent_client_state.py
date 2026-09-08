"""
dp_persistent_client_state.py

PRV1 Task 1 -- single-process, in-memory version.

This replaces the earlier multiprocessing design (dp_persistent_worker.py,
one dedicated OS process per client, communicating over a Pipe). That
design satisfied Task 1's literal requirement ("the SAME engine instance
... for all NUM_ROUNDS rounds") but did so by forking a new OS process
per client -- unsafe on this codebase's CUDA-only execution target,
since CUDA is already touched in the main process (torch.cuda.is_available()
at module level in main.py, model/criterion construction before the
round loop) before any fork would happen. This codebase already hit and
fixed exactly this hazard once before, for the training/eval pool
("fork+CUDA hang" -- GPU client training/eval now runs sequentially,
in-process, with no ProcessPoolExecutor at all). Reintroducing a fork
for the DP path would very likely reproduce that same hang.

Task 1's actual requirement -- "one PrivacyEngine per client, created
exactly once before Round 1, never recreated/detached/reassigned, reused
for that client's entire lifespan" -- does not require a separate OS
process. It requires that the SAME Python object survive, unmodified,
across the round loop. A plain dict living in the one process that's
already running everything sequentially (GPU) satisfies this trivially,
with no IPC, no fork-safety concerns, and no Pipe serialization limits.

Byzantine clients are excluded entirely from this module -- they have no
real DP accounting to preserve (see main.py's client_cfg["byzantine_clients"]
handling); their training still goes through _train_one_client(), unchanged.

If USE_DP is False, main.py never imports this module at all (regression
guard, PRV1 acceptance item 4).
"""

import torch
import torch.utils.data as tud

from task import get_model, get_model_parameters, set_model_parameters


def build_dp_client_states(clients_data, byzantine_clients, use_byzantine_attack,
                            sample_features, num_classes, dp_safe, learning_rate,
                            dp_batch_size, dp_target_epsilon, dp_target_delta,
                            dp_max_grad_norm, total_epochs_per_client, device):
    """
    Called EXACTLY ONCE, before Round 1. For every DP-active (i.e.
    non-Byzantine, when USE_DP=True) client: builds a model + Adam
    optimizer, moves the model to `device` BEFORE wrapping it (so
    Opacus's GradSampleModule hooks bind to the tensors that will
    actually be trained on, not a set that gets replaced by a later
    .to(device) call), then calls PrivacyEngine.make_private_with_epsilon()
    ONCE, calibrated against the client's REAL Dirichlet-partition size
    (via the DataLoader) and the FULL total_epochs_per_client horizon
    (= NUM_ROUNDS * LOCAL_EPOCHS for this run -- never hardcoded).

    Returns {client_idx: state_dict}, where state_dict holds the model,
    optimizer, loader, and the (never-to-be-recreated) engine, plus
    diagnostic fields ("sigma", "sample_rate") main.py logs at startup
    and writes into the per-client final-epsilon manifest.
    """
    from opacus import PrivacyEngine

    states = {}
    for i, (X_tr, y_tr, X_te, y_te) in enumerate(clients_data):
        if use_byzantine_attack and i in byzantine_clients:
            continue  # no real DP accounting to build for Byzantine clients

        base_model = get_model(num_features=sample_features,
                                num_classes=num_classes,
                                dp_safe=dp_safe).to(device)
        base_optimizer = torch.optim.Adam(base_model.parameters(), lr=learning_rate)

        X_t = torch.FloatTensor(X_tr)
        y_t = torch.LongTensor(y_tr)
        loader = tud.DataLoader(tud.TensorDataset(X_t, y_t),
                                 batch_size=dp_batch_size, shuffle=True)

        # PRV1 Task 1, steps 2-3, literally: engine created EXACTLY
        # ONCE, here, calibrated against THIS client's real dataset
        # size (via `loader`) and the FULL total_epochs_per_client
        # horizon. Never recreated, detached, or reassigned anywhere
        # below (steps 4-5) -- run_dp_client_round() only ever reads
        # from this SAME object.
        engine = PrivacyEngine(accountant="rdp")
        dp_model, dp_optimizer, dp_loader = engine.make_private_with_epsilon(
            module=base_model, optimizer=base_optimizer, data_loader=loader,
            target_epsilon=dp_target_epsilon, target_delta=dp_target_delta,
            epochs=total_epochs_per_client, max_grad_norm=dp_max_grad_norm,
        )

        states[i] = {
            "model": dp_model,
            "optimizer": dp_optimizer,
            "loader": dp_loader,
            "engine": engine,
            "device": device,
            "sigma": dp_optimizer.noise_multiplier,
            "sample_rate": dp_batch_size / len(X_tr),
        }
    return states


def run_dp_client_round(client_idx, dp_states, global_params, criterion,
                         local_epochs, learning_rate, prox_mu, dp_delta):
    """
    Called once per round, per DP-active client. Loads this round's
    global_params IN PLACE into the SAME wrapped model built in
    build_dp_client_states() (load_state_dict copies data into existing
    parameter tensors rather than replacing them, which is what keeps
    Opacus's per-parameter hooks valid across rounds), trains
    local_epochs worth of passes through the SAME optimizer/engine, and
    reads that SAME engine's cumulative-so-far epsilon -- never a fresh
    accountant, never re-derived.

    Returns (params, cumulative_epsilon).
    """
    state = dp_states[client_idx]
    dp_model = state["model"]
    dp_optimizer = state["optimizer"]
    dp_loader = state["loader"]
    engine = state["engine"]
    device = state["device"]

    real_model = dp_model._module if hasattr(dp_model, "_module") else dp_model
    set_model_parameters(real_model, global_params)

    global_dict = (dict(zip(list(real_model.state_dict().keys()), global_params))
                   if prox_mu else None)

    dp_model.train()
    for _ in range(local_epochs):
        for X_b, y_b in dp_loader:
            X_b = X_b.to(device)
            y_b = y_b.to(device)
            dp_optimizer.zero_grad()
            loss_val = criterion(dp_model(X_b), y_b)
            loss_val.backward()
            dp_optimizer.step()
            if global_dict is not None:
                with torch.no_grad():
                    for name, param in real_model.named_parameters():
                        if name not in global_dict:
                            continue
                        g = torch.as_tensor(global_dict[name], dtype=param.dtype,
                                             device=param.device)
                        param -= learning_rate * prox_mu * (param - g)

    params = get_model_parameters(real_model)
    # PRV1's real, composed, cumulative-so-far epsilon: read every
    # round directly off the SAME never-reset engine -- monotonically
    # non-decreasing round over round, never a per-round-reset value.
    cumulative_epsilon = engine.get_epsilon(delta=dp_delta)
    return params, cumulative_epsilon


def get_final_epsilons(dp_states, dp_delta):
    """
    Called EXACTLY ONCE, after the last round. Reads final_total_epsilon
    directly off each DP-active client's own never-reset engine -- the
    same object that has been alive (and the only thing ever queried)
    since build_dp_client_states(). No separate/central accountant
    exists to re-derive this value from.

    Returns {client_idx: final_total_epsilon}.
    """
    return {i: state["engine"].get_epsilon(delta=dp_delta)
            for i, state in dp_states.items()}
