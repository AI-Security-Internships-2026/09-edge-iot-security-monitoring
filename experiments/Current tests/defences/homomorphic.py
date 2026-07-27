"""
CKKS Homomorphic Encryption module.

What this implements
--------------------
Server-side aggregation where the server NEVER sees plaintext gradients.
All arithmetic (weighted addition) happens directly on ciphertext.

Why CKKS is quantum-safe
------------------------
CKKS is built on the Ring Learning With Errors (RLWE) problem.
Breaking RLWE requires solving high-dimensional lattice problems.
Shor's algorithm (which breaks RSA / ECC) works only against
factoring and discrete logarithm. No polynomial-time quantum
algorithm is known for lattice problems.

NIST's 2024 post-quantum standards (CRYSTALS-Kyber, CRYSTALS-Dilithium,
FALCON) are all lattice-based — same mathematical foundation as CKKS.

Why NOT Trimmed Mean under HE
------------------------------
Trimmed Mean requires sorting values across clients per coordinate.
Sorting requires comparisons. Comparisons in CKKS require 15-30 levels
of multiplicative depth per comparison via polynomial approximation.
For 50,000 parameters × 10 clients this would take days on CPU.
This is an open research problem (2024-2025), not an implementation issue.

Solution: ZKP rejects outlier gradients BEFORE encryption.
If every accepted client has ||w||_2 ≤ threshold (proven by ZKP),
simple homomorphic FedAvg is sufficient — no sorting needed.
"""

import numpy as np
import time

try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False
    print("[HE WARNING] TenSEAL not installed. "
          "pip install tenseal  to enable CKKS encryption.\n"
          "Falling back to plaintext simulation for research demonstration.")


# ── Context ───────────────────────────────────────────────────────────

def create_ckks_context(poly_modulus_degree=8192):
    """
    Create CKKS encryption context with post-quantum security.

    poly_modulus_degree=8192  → 128-bit post-quantum security
    poly_modulus_degree=16384 → 256-bit post-quantum security (~4x slower)

    coeff_mod_bit_sizes=[60,40,40,60] gives 3 multiplication levels
    which is sufficient for weighted summation (only addition needed).

    DEPLOYMENT NOTE:
    The context created here contains BOTH public and secret key.
    In production:
      - Gateway keeps: full context (secret key for decryption)
      - Server receives: public_context only (for homomorphic ops)
      - Server CANNOT decrypt — only gateway can

    In this simulation, we use the full context everywhere since
    everything runs on one machine. This is standard in FL simulation.
    """
    if not TENSEAL_AVAILABLE:
        print("  [HE] Using plaintext simulation (TenSEAL unavailable)")
        return None

    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree  = poly_modulus_degree,
        coeff_mod_bit_sizes  = [60, 40, 40, 60]
    )
    ctx.global_scale = 2 ** 40
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()

    cap = poly_modulus_degree // 2
    print(f"  CKKS context ready: "
          f"poly_modulus_degree={poly_modulus_degree}, "
          f"capacity={cap} floats/chunk, "
          f"128-bit post-quantum security (RLWE)")
    return ctx


# ── Encryption ────────────────────────────────────────────────────────

def encrypt_params(params, context, poly_modulus_degree=8192):
    """
    Encrypt model parameters under CKKS.

    Parameters are flattened into one long vector and split into
    chunks of size poly_modulus_degree // 2 (CKKS vector capacity).

    For a CNN-LSTM with ~50,000 parameters and degree=8192 (cap=4096):
      ceil(50000 / 4096) = 13 chunks per client

    Returns a dict that can be passed directly to aggregate_encrypted().
    """
    # Flatten all layers into one array
    flat   = np.concatenate([p.flatten() for p in params]).astype(np.float64)
    shapes = [p.shape for p in params]
    sizes  = [p.size for p in params]
    total  = len(flat)

    if context is None:
        # Plaintext simulation
        return {
            "mode":   "plaintext",
            "flat":   flat.astype(np.float32),
            "shapes": shapes,
            "sizes":  sizes,
            "total":  total,
        }

    chunk_size = poly_modulus_degree // 2
    chunks     = []
    for i in range(0, total, chunk_size):
        chunk = flat[i:i + chunk_size].tolist()
        chunks.append(ts.ckks_vector(context, chunk))

    return {
        "mode":       "ckks",
        "chunks":     chunks,
        "shapes":     shapes,
        "sizes":      sizes,
        "total":      total,
        "chunk_size": chunk_size,
        "n_chunks":   len(chunks),
    }


# ── Homomorphic Aggregation ───────────────────────────────────────────

def aggregate_encrypted(encrypted_updates, weights):
    """
    Homomorphic FedAvg: weighted sum on ciphertext.

    global_enc = Σ_i  enc(w_i) * (n_i / N)

    CKKS supports:
      enc(a) + enc(b) = enc(a + b)   ← homomorphic addition
      enc(a) * scalar = enc(a*scalar) ← plaintext scalar multiply

    The server performs this computation without ever seeing w_i.
    The weights n_i / N are public information (sample counts).

    Byzantine safety:
      ZKP verification happens BEFORE this function is called.
      Only clients that passed ZKP verification contribute here.
      If ZKP rejected a client, their update is dropped before encryption.
    """
    total = sum(weights)
    first = encrypted_updates[0]

    if first["mode"] == "plaintext":
        # Simulation: plaintext weighted average
        result = np.zeros(first["total"], dtype=np.float32)
        for enc, w in zip(encrypted_updates, weights):
            result += enc["flat"] * float(w / total)
        return {
            "mode":   "plaintext",
            "flat":   result,
            "shapes": first["shapes"],
            "sizes":  first["sizes"],
            "total":  first["total"],
        }

    # CKKS: chunk-by-chunk homomorphic weighted sum
    result_chunks = None
    for enc, w in zip(encrypted_updates, weights):
        scale  = float(w / total)
        scaled = [chunk * scale for chunk in enc["chunks"]]
        if result_chunks is None:
            result_chunks = scaled
        else:
            result_chunks = [r + s for r, s in zip(result_chunks, scaled)]

    return {
        "mode":       "ckks",
        "chunks":     result_chunks,
        "shapes":     first["shapes"],
        "sizes":      first["sizes"],
        "total":      first["total"],
        "chunk_size": first["chunk_size"],
        "n_chunks":   first["n_chunks"],
    }


# ── Decryption ────────────────────────────────────────────────────────

def decrypt_params(encrypted_result):
    """
    Decrypt the aggregated result back to a list of numpy arrays.

    DEPLOYMENT NOTE:
    In production, this runs on the GATEWAY (client side), not the server.
    The server broadcasts enc(global_model) to all gateways.
    Each gateway decrypts using its local secret key.
    The server never holds the secret key and never calls this function.

    In simulation, this runs on the same machine — acceptable for research.
    """
    if encrypted_result["mode"] == "plaintext":
        flat   = encrypted_result["flat"]
        shapes = encrypted_result["shapes"]
        sizes  = encrypted_result["sizes"]
    else:
        flat_list = []
        for chunk in encrypted_result["chunks"]:
            flat_list.extend(chunk.decrypt())
        flat   = np.array(flat_list[:encrypted_result["total"]],
                          dtype=np.float32)
        shapes = encrypted_result["shapes"]
        sizes  = encrypted_result["sizes"]

    # Reconstruct original parameter structure
    params = []
    offset = 0
    for shape, size in zip(shapes, sizes):
        params.append(flat[offset:offset + size].reshape(shape))
        offset += size
    return params


# ── Benchmarking ──────────────────────────────────────────────────────

def benchmark_he(context, sample_params, poly_modulus_degree=8192,
                 n_clients=10):
    """
    Measure per-round HE overhead. Call once before training loop.
    Reports realistic time estimates for experiment documentation.
    """
    print("\n  ── HE Benchmark ──────────────────────────────")

    t0  = time.time()
    enc = encrypt_params(sample_params, context, poly_modulus_degree)
    t_enc = time.time() - t0

    t0  = time.time()
    agg = aggregate_encrypted([enc, enc], [1, 1])
    t_agg = time.time() - t0

    t0  = time.time()
    decrypt_params(agg)
    t_dec = time.time() - t0

    mode = "CKKS" if context is not None else "Plaintext simulation"
    print(f"  Mode:       {mode}")
    print(f"  Parameters: {enc['total']:,}")

    if context is not None:
        print(f"  Chunks:     {enc['n_chunks']} "
              f"({poly_modulus_degree // 2} elements each)")

    print(f"  Encrypt (1 client):   {t_enc:.2f}s")
    print(f"  HE-Add (2 clients):   {t_agg:.2f}s")
    print(f"  Decrypt (aggregate):  {t_dec:.2f}s")
    est = t_enc * n_clients + t_agg * (n_clients - 1) + t_dec
    print(f"  Estimated per-round:  ~{est:.0f}s ({est/60:.1f} min)")
    print(f"  Estimated 25 rounds:  ~{est*25/3600:.1f} hours")
    print("  ─────────────────────────────────────────────")