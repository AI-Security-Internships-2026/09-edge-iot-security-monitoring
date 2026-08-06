"""
Homomorphic Encryption module using TenSEAL (CKKS scheme).

CRYPTOGRAPHIC BUG FIXED IN THIS VERSION
----------------------------------------
The original code had each client call `ts.context(...)` independently
inside client.py. That generates a NEW random keypair per client. CKKS
ciphertexts encrypted under different keys cannot be validly summed —
`enc_A(x) + enc_B(y)` does not decrypt to `x + y` under either key. The
old server-side homomorphic aggregation would have run without raising
an exception but produced silently meaningless numbers.

Fix: there is now exactly ONE context, generated once by the SERVER (or,
for a local single-process run, by main.py — see he_local.py). The
server/owner keeps the secret key locally and only ever gives clients a
PUBLIC context (`serialize(save_secret_key=False)`), which can encrypt
but not decrypt. Every client's ciphertexts are therefore under the
same public key, so homomorphic addition/scaling is valid, and only the
holder of the secret key can decrypt the aggregate.

RAM / SIZE FIXES APPLIED (Docker constrained-client defaults)
----------------------------------------------------------------
1. No `generate_galois_keys()` — only needed for ciphertext rotation,
   which this pipeline never does. Galois keys are typically the
   single largest allocation in a CKKS context.
2. No `generate_relin_keys()` — only needed after ciphertext x
   ciphertext multiplication (produces a degree-2 ciphertext). This
   pipeline only ever does ciphertext + ciphertext and ciphertext x
   scalar, neither of which needs it.
3. Coefficient modulus chain shortened from a depth-2
   [60, 40, 40, 60] chain to a depth-1 [37, 28, 37] chain (first/last
   primes are the encrypt/decrypt boundary primes, middle prime is the
   one working level this pipeline actually uses).
4. poly_modulus_degree dropped 8192 -> 4096 to match the shorter chain
   (102 total coefficient-modulus bits stays under the ~109-bit ceiling
   TenSEAL enforces for 128-bit security at n=4096 — chains close to
   that ceiling were tried and rejected by TenSEAL itself, so this
   value was verified empirically, not just estimated).
   global_scale dropped 2**40 -> 2**28 to match. This was measured
   directly against realistic post-clip gradient magnitudes (values
   around 0.01, consistent with a 79k-parameter vector clipped to
   L2 norm <= 1.0): max absolute error ~1.5e-5, ~0.15% relative —
   far below the DP noise floor already added in Layer 1, so it adds
   no meaningful degradation. (An earlier, even smaller scale of 2**20
   was tested and rejected — it produced ~1e-2 absolute error, which
   would have swamped real gradient values at this magnitude.)
5. Ciphertexts are base64-encoded instead of hex-encoded (hex is 2
   bytes of text per 1 byte of binary; base64 is ~1.33 bytes per byte
   — a 33% cut in payload size and JSON string memory for free).

These four values (POLY_MODULUS_DEGREE, COEFF_MOD_BIT_SIZES,
GLOBAL_SCALE) remain the MODULE-LEVEL DEFAULTS below, used automatically
by create_server_context() when called with no arguments — this is
what Docker's server.py/client.py do, so their behavior is completely
unchanged by the additions in this file. create_server_context() now
also accepts explicit overrides for callers (e.g. a local, non-RAM-
constrained research run encrypting the FULL model rather than just a
classifier head) that want a different, more standard parameter set —
see he_local.py.
"""

import base64
import numpy as np

try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False
    print("TenSEAL not installed — HE layer disabled")
    print("Install with: pip install tenseal")


# Depth-1 chain: this pipeline only ever does ct+ct and ct*scalar.
# Verified empirically (see module docstring) against realistic
# post-clip gradient magnitudes and against TenSEAL's own security
# parameter validation for n=4096. These are DEFAULTS used when
# create_server_context() is called with no arguments (Docker path).
POLY_MODULUS_DEGREE = 4096
COEFF_MOD_BIT_SIZES = [37, 28, 37]
GLOBAL_SCALE = 2 ** 28


# ── Server-side: the ONE context with the secret key ──────────────────

def create_server_context(poly_modulus_degree=None,
                           coeff_mod_bit_sizes=None,
                           global_scale=None):
    """
    Create the single CKKS context for the whole FL run. Holds the
    secret key — never send this object or its secret serialization
    to a client.

    With no arguments, uses the Docker-tuned defaults above (depth-1,
    n=4096) — this is exactly what server.py's _get_he_context() and
    the previous version of this function did, so existing Docker
    behavior is unchanged.

    Pass explicit overrides for a different security/RAM tradeoff,
    e.g. the standard n=8192, [60,40,40,60], scale=2**40 configuration
    used by a local, non-RAM-constrained full-model-encryption run
    (see he_local.py).
    """
    if not TENSEAL_AVAILABLE:
        return None

    poly_modulus_degree = poly_modulus_degree or POLY_MODULUS_DEGREE
    coeff_mod_bit_sizes = coeff_mod_bit_sizes or COEFF_MOD_BIT_SIZES
    global_scale        = global_scale or GLOBAL_SCALE

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )
    context.global_scale = global_scale
    # Intentionally NOT calling generate_galois_keys() / generate_relin_keys().
    return context


def serialize_public_context(context):
    """Serialize WITHOUT the secret key. Safe to send to clients."""
    public_bytes = context.serialize(save_secret_key=False)
    return base64.b64encode(public_bytes).decode("ascii")


# ── Client-side: public-only context, can encrypt, cannot decrypt ─────

def load_client_context(public_context_b64):
    """Client-side: load the public context distributed by the server."""
    if not TENSEAL_AVAILABLE:
        raise RuntimeError("TenSEAL not available")
    raw = base64.b64decode(public_context_b64)
    return ts.context_from(raw)


def encrypt_flat_array(flat_array, context, chunk_size=None):
    """
    Encrypt a flat float array in chunks under the shared public context.
    Returns list of base64-encoded ciphertext strings.
    """
    if not TENSEAL_AVAILABLE:
        raise RuntimeError("TenSEAL not available")
    if chunk_size is None:
        chunk_size = POLY_MODULUS_DEGREE // 2

    chunks_b64 = []
    for i in range(0, len(flat_array), chunk_size):
        chunk = flat_array[i:i + chunk_size].tolist()
        enc = ts.ckks_vector(context, chunk)
        chunks_b64.append(base64.b64encode(enc.serialize()).decode("ascii"))
        del enc, chunk
    return chunks_b64


# ── Generic full-parameter-list helpers ────────────────────────────────
# (used by he_local.py for local, full-model encryption; the Docker
# client.py instead does the flatten/chunk/pack steps inline for just
# the "sensitive" classifier-head slice — these are equivalent logic,
# generalized to any parameter list, added here so it's not duplicated
# a third time.)

def encrypt_param_list(params, context, chunk_size=None):
    """
    Encrypt an entire list of numpy parameter arrays: flatten -> chunk
    -> encrypt. Returns a wire-style dict describing the ciphertext.
    """
    if not TENSEAL_AVAILABLE:
        raise RuntimeError("TenSEAL not available")
    if chunk_size is None:
        chunk_size = POLY_MODULUS_DEGREE // 2

    flat = np.concatenate([p.flatten() for p in params]).astype(np.float64)
    chunks_b64 = encrypt_flat_array(flat, context, chunk_size)
    return {
        "mode":       "ckks",
        "chunks":     chunks_b64,
        "shapes":     [list(p.shape) for p in params],
        "sizes":      [int(p.size) for p in params],
        "total":      int(flat.size),
        "chunk_size": chunk_size,
        "n_chunks":   len(chunks_b64),
    }


def aggregate_encrypted_param_lists(encrypted_clients, weights, context):
    """
    Homomorphically weighted-sum a list of encrypt_param_list() outputs
    (one per client). Returns the STILL-ENCRYPTED aggregate as a dict
    (chunks + the shape/size metadata needed to decrypt and reshape
    later) — decryption is a separate step, see decrypt_param_list().

    `context` must hold the secret key.
    """
    if not TENSEAL_AVAILABLE:
        raise RuntimeError("TenSEAL not available")

    all_chunks = [c["chunks"] for c in encrypted_clients]
    aggregated_chunks = he_weighted_sum(all_chunks, weights, context)

    ref = encrypted_clients[0]
    return {
        "chunks": aggregated_chunks,
        "shapes": ref["shapes"],
        "sizes":  ref["sizes"],
        "total":  ref["total"],
    }


def decrypt_param_list(enc_aggregate):
    """
    Decrypt the output of aggregate_encrypted_param_lists() back into a
    plain list of numpy arrays matching the original per-layer shapes.
    """
    flat = decrypt_aggregate(enc_aggregate["chunks"], enc_aggregate["total"])
    result = []
    offset = 0
    for shape, size in zip(enc_aggregate["shapes"], enc_aggregate["sizes"]):
        result.append(
            flat[offset:offset + size].reshape(shape).astype(np.float32)
        )
        offset += size
    return result


# ── Server-side: aggregate + decrypt (secret context required) ────────

def deserialize_chunk(chunk_b64, context):
    return ts.ckks_vector_from(context, base64.b64decode(chunk_b64))


def he_weighted_sum(all_client_chunks, weights, context):
    """
    Homomorphic weighted sum across clients' chunk lists.
    all_client_chunks: list (per client) of list (per chunk) of base64 str
    Returns list of aggregated (still-encrypted) CKKSVector, one per chunk.
    """
    if not TENSEAL_AVAILABLE:
        raise RuntimeError("TenSEAL not available")

    total = sum(weights)
    n_chunks = len(all_client_chunks[0])

    aggregated = []
    for chunk_idx in range(n_chunks):
        agg = None
        for client_chunks, w in zip(all_client_chunks, weights):
            vec = deserialize_chunk(client_chunks[chunk_idx], context)
            scaled = vec * (w / total)
            agg = scaled if agg is None else agg + scaled
            del vec, scaled
        aggregated.append(agg)
    return aggregated


def decrypt_aggregate(aggregated_chunks, total_len):
    """Decrypt the final aggregated ciphertext. Requires the secret context."""
    flat = []
    for chunk in aggregated_chunks:
        flat.extend(chunk.decrypt())
    return np.array(flat[:total_len], dtype=np.float32)


# ── Plaintext fallback / bulk-layer averaging ──────────────────────────

def plaintext_weighted_sum(all_params, weights):
    """Plain weighted average — used for the non-HE 'bulk' layers, and
    as a fallback if HE is unavailable/fails."""
    total = sum(weights)
    n_layers = len(all_params[0])
    result = []
    for layer_idx in range(n_layers):
        layer_avg = sum(p[layer_idx] * (w / total)
                         for p, w in zip(all_params, weights))
        result.append(layer_avg.astype(np.float32))
    return result
