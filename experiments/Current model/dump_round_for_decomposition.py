"""Closes the BatchNorm-counter todo. Paste decompose() into main.py just before the
aggregation branch (where accepted_params exists) for ONE non-DP E3 run, call it once at round 5,
then paste the printed shares into the paper. Needs model_state_keys = list(model.state_dict().keys())."""
import itertools, numpy as np

def decompose(accepted_params, model_state_keys, byz_ids=(), client_ids=None):
    def part(k):
        return "counter" if k.endswith("num_batches_tracked") else \
               "bn_stats" if k.endswith(("running_mean", "running_var")) else "trainable"
    groups = {}
    for idx, k in enumerate(model_state_keys):
        groups.setdefault(part(k), []).append(idx)
    ids = client_ids or list(range(len(accepted_params)))
    tot = {g: [] for g in groups}
    for a, b in itertools.combinations(range(len(accepted_params)), 2):
        d = {g: float(sum(np.sum((np.asarray(accepted_params[a][i], float) -
                                  np.asarray(accepted_params[b][i], float)) ** 2) for i in idx))
             for g, idx in groups.items()}
        s = sum(d.values()) or 1.0
        for g in d: tot[g].append(d[g] / s)
    print("median share of d_ij by component:", {g: round(float(np.median(v)), 4) for g, v in tot.items()})
