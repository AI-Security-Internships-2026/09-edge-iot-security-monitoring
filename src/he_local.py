"""
Local (single-process) HE adapter for main.py — PARTIAL encryption.

main.py was written against a `defences.homomorphic` module with this
exact function signature set (create_ckks_context, encrypt_params,
aggregate_encrypted, decrypt_params, benchmark_he). That module was
superseded during the Docker RAM work by he_aggregation.py, which has
a lower-level API tuned for partial (classifier-head-only) encryption.

This version matches the Docker client's design exactly: only the
"sensitive" classifier-head layers (~6% of the model's parameters) are
CKKS-encrypted; everything else ("bulk" — the CNN+LSTM feature
extractor, ~94% of params) is sent DP-noised but in plaintext and
weighted-averaged normally. This keeps local results directly
comparable to the Docker numbers, rather than measuring a different
(full-model) encryption design.

Because this is a single process (no client/server network boundary),
the one CKKS context created here holds the secret key throughout —
there's no public/private context split like in the Docker version,
since there's no untrusted party to hide the secret key from.
"""

import time
import numpy as np

import he_aggregation as he

# Standard TenSEAL parameters for n=8192, 128-bit security (RLWE).
# [60, 40, 40, 60] = 200 total coefficient-modulus bits, under the
# ~218-bit ceiling TenSEAL enforces for 128-bit security at this
# degree. Two 40-bit middle primes give headroom (this pipeline only
# ever consumes one level for ct+ct / ct*scalar), and global_scale is
# set to match the middle primes' bit-length, which is standard
# practice for this parameter set.
HE_COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]
HE_GLOBAL_SCALE = 2 ** 40

# Same convention as Docker's client.py: only classifier-head layers
# are treated as "sensitive" and go through CKKS. Everything else is
# "bulk" — plaintext, DP-noised, plain weighted average.
SENSITIVE_PREFIX = "classifier"


def create_ckks_context(poly_degree):
    """Create the single local CKKS context (holds the secret key)."""
    return he.create_server_context(
        poly_modulus_degree=poly_degree,
        coeff_mod_bit_sizes=HE_COEFF_MOD_BIT_SIZES,
        global_scale=HE_GLOBAL_SCALE,
    )


def split_sensitive_bulk(keys, params):
    """Same split logic as Docker's client.py: classifier-head layers
    are 'sensitive', everything else is 'bulk'."""
    sensitive_idx = [i for i, k in enumerate(keys) if k.startswith(SENSITIVE_PREFIX)]
    bulk_idx      = [i for i, k in enumerate(keys) if not k.startswith(SENSITIVE_PREFIX)]
    sensitive = [params[i] for i in sensitive_idx]
    bulk      = [params[i] for i in bulk_idx]
    return sensitive, bulk, sensitive_idx, bulk_idx


def encrypt_params(raw_params, keys, he_context, poly_degree):
    """
    Encrypt ONLY the classifier-head ("sensitive") layers of one
    client's update. The bulk layers are kept as plain numpy arrays.

    Returns a dict with both halves plus the index mapping needed to
    reassemble the full parameter list later, in decrypt_params().
    """
    sensitive, bulk, sensitive_idx, bulk_idx = split_sensitive_bulk(keys, raw_params)

    sensitive_enc = he.encrypt_param_list(
        sensitive, he_context, chunk_size=poly_degree // 2
    )

    n_sensitive = sum(p.size for p in sensitive)
    n_bulk      = sum(p.size for p in bulk)

    return {
        "mode":          "partial_he",
        "n_chunks":      sensitive_enc["n_chunks"],
        "sensitive_enc": sensitive_enc,
        "bulk":          bulk,
        "sensitive_idx": sensitive_idx,
        "bulk_idx":      bulk_idx,
        "n_layers":      len(keys),
        "pct_encrypted": 100 * n_sensitive / (n_sensitive + n_bulk),
    }


def aggregate_encrypted(accepted_params, accepted_weights, he_context):
    """
    Aggregate a list of encrypt_params() outputs (one per accepted
    client). Sensitive layers are homomorphically weighted-summed
    (still encrypted on return); bulk layers are weighted-averaged in
    plaintext immediately, since they were never encrypted.
    """
    sensitive_lists = [c["sensitive_enc"] for c in accepted_params]
    agg_sensitive = he.aggregate_encrypted_param_lists(
        sensitive_lists, accepted_weights, he_context
    )

    bulk_lists = [c["bulk"] for c in accepted_params]
    agg_bulk = he.plaintext_weighted_sum(bulk_lists, accepted_weights)

    ref = accepted_params[0]
    return {
        "agg_sensitive": agg_sensitive,
        "agg_bulk":      agg_bulk,
        "sensitive_idx": ref["sensitive_idx"],
        "bulk_idx":      ref["bulk_idx"],
        "n_layers":      ref["n_layers"],
    }


def decrypt_params(enc_aggregate):
    """
    Decrypt the sensitive half and merge it with the already-plaintext
    bulk half, back into a single ordered parameter list matching the
    model's original state_dict layer order.
    """
    sensitive_params = he.decrypt_param_list(enc_aggregate["agg_sensitive"])

    merged = [None] * enc_aggregate["n_layers"]
    for pos, layer_i in enumerate(enc_aggregate["sensitive_idx"]):
        merged[layer_i] = sensitive_params[pos]
    for pos, layer_i in enumerate(enc_aggregate["bulk_idx"]):
        merged[layer_i] = enc_aggregate["agg_bulk"][pos]

    return merged


def benchmark_he(he_context, dummy_params, dummy_keys, poly_degree, num_clients):
    """
    Quick timing benchmark on a dummy model, printed once at startup so
    you know roughly what per-round HE overhead to expect before the
    real run starts.
    """
    n_total = sum(p.size for p in dummy_params)
    print(f"  Benchmarking HE on dummy model ({n_total:,} params total, "
          f"poly_degree={poly_degree})...")

    t0 = time.time()
    enc = encrypt_params(dummy_params, dummy_keys, he_context, poly_degree)
    t_enc = time.time() - t0
    print(f"    Encrypt (1 client, sensitive only): {t_enc:.2f}s  "
          f"chunks={enc['n_chunks']}  "
          f"({enc['pct_encrypted']:.1f}% of params encrypted, rest plaintext)")

    t0 = time.time()
    agg = aggregate_encrypted(
        [enc] * num_clients, [1] * num_clients, he_context
    )
    t_agg = time.time() - t0
    print(f"    Aggregate ({num_clients} clients): {t_agg:.2f}s")

    t0 = time.time()
    _ = decrypt_params(agg)
    t_dec = time.time() - t0
    print(f"    Decrypt: {t_dec:.2f}s")

    est_round = t_enc * num_clients + t_agg + t_dec
    print(f"    Estimated per-round HE overhead (~{num_clients} clients, "
          f"serial): {est_round:.1f}s")
