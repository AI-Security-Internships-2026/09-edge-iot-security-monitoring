"""
config_loader.py

DAT1 Task 2 -- loads experiments/configs/hyperparams.json and provides
validated access to the tunables that were previously hardcoded numeric
literals in main.py/task.py (FedProx mu, DP clipping norm C, the
adaptive-Krum MAD-k multiplier, the HE+Krum-hybrid assumed-f cap, and
the application-model class-weight multipliers).

Deliberately strict rather than permissive: a config file missing a
required key, or missing a 'validated_on_split' provenance field on a
scalar tunable, must fail loudly at load time -- not silently ship an
undocumented/unprovenanced hyperparameter into a run whose numbers will
later be cited in the paper.
"""
import json
import os


DEFAULT_HYPERPARAMS_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "configs", "hyperparams.json"
))

# Every key DAT1 Task 2 requires the config file to expose.
REQUIRED_KEYS = (
    "fedprox_mu",
    "dp_max_grad_norm",
    "adaptive_krum_k",
    "adaptive_krum_hybrid_assumed_f",
    "class_weight_multipliers_application",
    "fedprox_mu_sweep_default",
    "mad_k_sweep_default",
    "byzantine_f_sweep_default",
)

# The scalar tunables that must additionally carry a 'validated_on_split'
# provenance field -- i.e. a statement of which split (if any) actually
# justifies the value, per DAT1's no-tuning-on-test requirement.
PROVENANCE_REQUIRED_KEYS = (
    "fedprox_mu",
    "dp_max_grad_norm",
    "adaptive_krum_k",
    "adaptive_krum_hybrid_assumed_f",
)


def load_hyperparams_config(path=None):
    """
    Loads and validates the hyperparameters config file.

    Raises AssertionError if:
      - the file is missing any key in REQUIRED_KEYS, or
      - one of PROVENANCE_REQUIRED_KEYS is present but missing its
        'validated_on_split' field.

    Returns the parsed config dict on success.
    """
    if path is None:
        path = DEFAULT_HYPERPARAMS_PATH

    with open(path) as f:
        cfg = json.load(f)

    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    assert not missing, (
        f"hyperparams config at {path!r} is missing required key(s): "
        f"{missing}. DAT1 Task 2 requires all of {REQUIRED_KEYS} to be "
        f"present -- a config file cannot silently drop a required "
        f"tunable or sweep array."
    )

    for key in PROVENANCE_REQUIRED_KEYS:
        assert "validated_on_split" in cfg[key], (
            f"{key!r} in {path!r} is missing a 'validated_on_split' "
            f"provenance field -- every tunable value must state which "
            f"split (if any) justifies it (DAT1's no-tuning-on-test "
            f"requirement)."
        )

    return cfg


def get_value(cfg, key):
    """
    Extracts the scalar 'value' field of a provenance-wrapped tunable,
    e.g. cfg['fedprox_mu'] == {"value": 0.02, "validated_on_split": "..."}
    -> get_value(cfg, 'fedprox_mu') == 0.02.

    Only meaningful for keys in PROVENANCE_REQUIRED_KEYS; sweep-array
    and class-weight-multiplier keys are read directly from cfg[key]
    by their own call sites instead, since they aren't single scalars.
    """
    return cfg[key]["value"]
