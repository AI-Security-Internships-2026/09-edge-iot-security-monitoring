"""
tests/test_dp_accounting_composition.py

PRV1 Task 3, implemented per the issue's literal spec, against the
single-process persistent design (src/dp_persistent_client_state.py) --
NOT the earlier multiprocessing/dp_persistent_worker.py design, which
this revision removes (fork after CUDA has been touched in the parent
process is unsafe on this codebase's CUDA-only execution target; see
main.py's module docstring).

  Path A (BUGGY -- kept ONLY here, in the test harness, exactly as the
  issue instructs; never used in production code): a fresh
  PrivacyEngine recreated every round, its own get_epsilon() read
  immediately -- reproduces the OLD bug's "epsilon_reported_per_call".

  Path B (CORRECT -- the real production code path, via
  src/dp_persistent_client_state.py): one PrivacyEngine per client,
  created once, reused unmodified across both rounds --
  "epsilon_after_round_2".

Assert epsilon_after_round_2 > epsilon_reported_per_call. If equal, RDP
composition is not accumulating and the fix is wrong.

Run with: pytest tests/test_dp_accounting_composition.py -v -s
(run from the repo root, with `src/` on PYTHONPATH -- see the sys.path
insert below, or `cd src && pytest ../tests/...`)
"""
import ast
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = _REPO_ROOT
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np
import torch
import torch.utils.data as tud
from opacus import PrivacyEngine

from task import get_model, FocalLoss
from dp_persistent_client_state import (
    build_dp_client_states, run_dp_client_round, get_final_epsilons,
)

NUM_FEATURES  = 6
NUM_CLASSES   = 3
DATASET_LEN   = 10
BATCH_SIZE    = 5
LOCAL_EPOCHS  = 2
NUM_ROUNDS    = 2
TARGET_EPS    = 5.0
TARGET_DELTA  = 1e-5
MAX_GRAD_NORM = 1.0


def _synthetic_client_data(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(DATASET_LEN, NUM_FEATURES)).astype(np.float32)
    y = rng.integers(0, NUM_CLASSES, size=DATASET_LEN).astype(np.int64)
    return X, y


def _path_a_buggy_per_round_reset_epsilon():
    """
    Reproduces the OLD, BUGGY behavior EXACTLY: a fresh PrivacyEngine
    every round, queried immediately after that round's local epochs.
    This function must NEVER be called from production code -- it
    exists solely so this test can demonstrate the difference against
    Path B.
    """
    X, y = _synthetic_client_data(seed=1)
    X_t, y_t = torch.FloatTensor(X), torch.LongTensor(y)

    last_round_epsilon = None
    for _round_num in range(NUM_ROUNDS):
        model = get_model(num_features=NUM_FEATURES, num_classes=NUM_CLASSES,
                           dp_safe=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loader = tud.DataLoader(tud.TensorDataset(X_t, y_t),
                                 batch_size=BATCH_SIZE, shuffle=True)

        # THE BUG: brand-new engine every round, calibrated as if THIS
        # round were the entire experiment (epochs=LOCAL_EPOCHS, not
        # the full NUM_ROUNDS*LOCAL_EPOCHS horizon).
        engine = PrivacyEngine(accountant="rdp")
        model, optimizer, loader = engine.make_private_with_epsilon(
            module=model, optimizer=optimizer, data_loader=loader,
            target_epsilon=TARGET_EPS, target_delta=TARGET_DELTA,
            epochs=LOCAL_EPOCHS, max_grad_norm=MAX_GRAD_NORM,
        )
        criterion = FocalLoss(gamma=2.0)
        model.train()
        for _ in range(LOCAL_EPOCHS):
            for X_b, y_b in loader:
                optimizer.zero_grad()
                loss = criterion(model(X_b), y_b)
                loss.backward()
                optimizer.step()

        last_round_epsilon = engine.get_epsilon(delta=TARGET_DELTA)
        # BUG: this is reported/logged as "the epsilon" every round,
        # with no memory of any prior round's composition.

    return last_round_epsilon


def _path_b_correct_persistent_engine_epsilon():
    """
    The real production path: build_dp_client_states() once, then
    run_dp_client_round() each round, reusing the SAME engine. Returns
    (epsilons_by_round, final_epsilon) -- final_epsilon is Path B's
    "epsilon_after_round_2", read via get_final_epsilons() after the
    round loop (same object, so it must match round 2's own value
    exactly).
    """
    X, y = _synthetic_client_data(seed=1)   # SAME seed/data as Path A,
                                             # apples-to-apples comparison
                                             # of the accounting logic only.
    clients_data = [(X, y, X, y)]   # single client; X_te/y_te unused here

    dp_states = build_dp_client_states(
        clients_data, byzantine_clients=[], use_byzantine_attack=False,
        sample_features=NUM_FEATURES, num_classes=NUM_CLASSES, dp_safe=True,
        learning_rate=0.001, dp_batch_size=BATCH_SIZE,
        dp_target_epsilon=TARGET_EPS, dp_target_delta=TARGET_DELTA,
        dp_max_grad_norm=MAX_GRAD_NORM,
        total_epochs_per_client=NUM_ROUNDS * LOCAL_EPOCHS,
        device=torch.device("cpu"),
    )

    from task import get_model as _gm, get_model_parameters as _gmp
    global_params = _gmp(_gm(num_features=NUM_FEATURES,
                              num_classes=NUM_CLASSES, dp_safe=True))

    criterion = FocalLoss(gamma=2.0)
    epsilons_by_round = []
    for _round_num in range(NUM_ROUNDS):
        params, cumulative_epsilon = run_dp_client_round(
            0, dp_states, global_params, criterion,
            local_epochs=LOCAL_EPOCHS, learning_rate=0.001,
            prox_mu=0.0, dp_delta=TARGET_DELTA,
        )
        global_params = params
        epsilons_by_round.append(cumulative_epsilon)

    final_epsilon = get_final_epsilons(dp_states, TARGET_DELTA)
    return epsilons_by_round, final_epsilon[0]


def test_composition_accumulates_correctly_vs_buggy_per_round_reset():
    """
    PRV1 Task 3's required assertion: epsilon_after_round_2 (correct,
    persistent-engine composition) must be STRICTLY GREATER than
    epsilon_reported_per_call (buggy, per-round-reset). If equal (or
    less), RDP composition is not accumulating and the fix is wrong.
    """
    epsilon_reported_per_call = _path_a_buggy_per_round_reset_epsilon()

    epsilons_by_round, final_epsilon = _path_b_correct_persistent_engine_epsilon()
    epsilon_after_round_2 = epsilons_by_round[-1]

    print(f"\n  Path A (BUGGY, per-round-reset)  epsilon_reported_per_call = {epsilon_reported_per_call:.4f}")
    print(f"  Path B (CORRECT, persistent)      epsilon_after_round_1     = {epsilons_by_round[0]:.4f}")
    print(f"  Path B (CORRECT, persistent)      epsilon_after_round_2     = {epsilon_after_round_2:.4f}")
    print(f"  Path B final_total_epsilon (post-round-loop)                = {final_epsilon:.4f}")

    assert epsilon_after_round_2 > epsilon_reported_per_call, (
        f"epsilon_after_round_2 ({epsilon_after_round_2:.4f}) must be "
        f"strictly greater than epsilon_reported_per_call "
        f"({epsilon_reported_per_call:.4f}) -- if not, RDP composition "
        f"is not accumulating across rounds and the PRV1 fix is wrong."
    )

    # Sanity: within Path B itself, epsilon must be monotonically
    # non-decreasing round over round (composition never decreases).
    assert epsilons_by_round[0] <= epsilons_by_round[1], (
        "Path B's own per-round epsilon trace is not monotonically "
        "non-decreasing -- composition is broken."
    )

    # The final value (read via get_final_epsilons(), after the round
    # loop) must exactly match what round 2 itself already reported --
    # same engine, never reset.
    assert abs(final_epsilon - epsilon_after_round_2) < 1e-9, (
        "final_total_epsilon does not match round 2's own reported "
        "cumulative epsilon -- something reset or recreated the engine "
        "between the last round and the final read."
    )


def test_ast_audit_no_privacy_engine_in_train_one_client():
    """
    PRV1 acceptance item ("Tests Required" -> "Grep audit"): the
    PrivacyEngine(...) constructor must never appear inside
    _train_one_client() or any per-round function in main.py when
    USE_DP=True. Implemented via AST (not a naive substring search) so
    a docstring merely mentioning "PrivacyEngine(" in prose can never
    produce a false positive -- only an actual `PrivacyEngine(...)`
    call expression in the function's real code counts.
    """
    main_py_path = os.path.join(_SRC_DIR, "main.py")
    with open(main_py_path) as f:
        source = f.read()

    tree = ast.parse(source, filename=main_py_path)

    target_functions = {"_train_one_client", "_eval_one_client",
                        "_run_training_wave", "_run_eval_wave"}
    offending = []

    class _CallVisitor(ast.NodeVisitor):
        def __init__(self, func_name):
            self.func_name = func_name

        def visit_Call(self, node):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name == "PrivacyEngine":
                offending.append(self.func_name)
            self.generic_visit(node)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in target_functions:
            _CallVisitor(node.name).visit(node)

    assert not offending, (
        f"PrivacyEngine(...) constructor call found (via AST) inside: "
        f"{offending} -- PRV1's Task 1 lifecycle change must be the "
        f"ONLY code path used when USE_DP=True."
    )
