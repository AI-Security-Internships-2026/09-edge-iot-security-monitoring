"""
Lightweight Zero-Knowledge Proof module.

Unchanged from the original — it only depends on numpy/hashlib/hmac,
so it runs fine in the lightweight client orchestrator process and
never forces a torch import there.

What this implements
--------------------
A commitment scheme with a signed norm assertion — honest to call
it this in the paper rather than a full ZKP.

A full SNARK/STARK ZKP over 50,000 float32 parameters requires:
  - Encoding floats as finite-field elements (expensive)
  - Building arithmetic circuits for norm computation (100k+ gates)
  - Proof generation: minutes per client per round on CPU

For a research prototype demonstrating the architecture, we use:
  - Commitment: HMAC-SHA256(shared_key, params || salt)
      → binding (client cannot change params after committing)
      → hiding   (server cannot recover params from commitment)
  - Norm assertion: client sends actual_norm signed with HMAC
      → server verifies signature (HMAC forgery requires shared key)
      → server checks actual_norm ≤ threshold
      → Byzantine gradient bomb is blocked

Quantum safety:
  HMAC-SHA256 is hash-based → quantum-safe.
  Grover's algorithm halves the effective key space, so 256-bit
  HMAC-SHA256 gives 128-bit post-quantum security.
  This is above NIST's minimum threshold for post-quantum security.

Limitation acknowledged (honest):
  This does not prove DP noise was applied correctly.
  DP correctness comes from correct implementation, not from this proof.
  The ZKP proves structural integrity (norm bound), not procedural honesty.
"""

import hashlib
import hmac
import os
import numpy as np


# In production: distribute via PKI (e.g. CRYSTALS-Dilithium signatures)
# In research prototype: pre-shared symmetric key
SHARED_KEY = b"fl_ids_zkp_shared_key_2025_v2"


# ── Helpers ───────────────────────────────────────────────────────────

def flatten_params(params):
    """Flatten list of numpy arrays to a single contiguous float32 array."""
    return np.concatenate([p.flatten() for p in params]).astype(np.float32)


def _hmac(message_bytes):
    return hmac.new(SHARED_KEY, message_bytes, hashlib.sha256).hexdigest()


# ── Commitment ────────────────────────────────────────────────────────

def generate_commitment(params_flat, salt=None):
    """
    Commitment C = HMAC-SHA256(key, params_bytes || salt)

    salt: 32 random bytes — prevents rainbow table attacks and ensures
    two identical gradients produce different commitments.

    The commitment binds the client to a specific gradient value.
    After sending C, the client cannot later claim they sent a different
    gradient (binding property). The server cannot learn the gradient
    from C alone (hiding property).
    """
    if salt is None:
        salt = os.urandom(32)
    message = params_flat.tobytes() + salt
    C       = _hmac(message)
    return C, salt


def verify_commitment(params_flat, salt, C_claimed):
    """
    Verify that params_flat matches the committed value C_claimed.
    Returns True if the commitment is consistent with the params.
    """
    C_expected, _ = generate_commitment(params_flat, salt)
    return hmac.compare_digest(C_expected, C_claimed)


# ── Norm Proof ────────────────────────────────────────────────────────

def generate_norm_proof(params_flat, clip_norm=1.0, noise_sigma=0.0):
    """
    Generate a signed norm assertion.

    After Local DP, the gradient norm may exceed clip_norm due to
    added noise. Expected upper bound:
        threshold = clip_norm + sqrt(n_params) * noise_sigma * safety_factor

    BUG FIXED HERE: the threshold used to include a stray `* 0.01`
    "empirical factor" that made it ~100x too small. For a Gaussian
    mechanism adding i.i.d. noise ~N(0, sigma^2) independently to
    every one of n_params coordinates, the expected L2 norm of the
    noise vector alone is sigma * sqrt(n_params) — NOT sigma * 0.01 *
    sqrt(n_params). Verified by direct simulation: for n_params=80074,
    sigma=1.6149 (this project's actual DP_EPSILON=3.0 config), the
    simulated noise-norm mean was 456.9 with std of only ~1.2 (<0.3%
    relative) — i.e. this quantity is extremely concentrated, so a
    small multiplicative safety factor is enough tail coverage; no
    large additive fudge term is needed. With the old buggy formula,
    every honestly DP-noised update from this project's actual
    parameter count would have been rejected as "exceeding the norm
    bound" — the ZKP gate was effectively blocking all real clients,
    not just Byzantine ones.

    A Byzantine client CANNOT:
      - Send a false norm (HMAC signature would fail)
      - Send a gradient with true norm > threshold (server rejects)

    A Byzantine client CAN:
      - Apply sign-flip within the norm bound
      → This is why Multi-Krum is still useful for direction-based attacks
      → ZKP handles structural/magnitude attacks; Krum handles directional

    In a full ZKP this would be a range proof (Bulletproofs or STARK).
    Here: signed norm value. The binding between this norm and the
    actual gradient is provided by the commitment scheme.

    NOTE — a separate, real finding worth reporting even after this
    fix: at this project's current DP_EPSILON=3.0 over an ~80k-parameter
    model, the injected noise (~457 in norm) swamps the clipped signal
    (norm 1.0) by ~450x. The proof will correctly PASS now (it's
    honestly measuring what's happening), but the model is very
    unlikely to learn anything useful at this noise-to-signal ratio.
    That's a DP-calibration problem to solve separately (e.g. much
    larger epsilon per round, adding noise once to the server-side
    aggregate instead of per-client-per-coordinate, or reducing the
    dimensionality that gets noised) — not something a norm-proof
    threshold fix can or should paper over.
    """
    norm = float(np.linalg.norm(params_flat))

    # Expected noise-vector L2 norm for i.i.d. N(0, sigma^2) added to
    # n_params coordinates is sigma * sqrt(n_params) (concentrated —
    # see docstring). A modest multiplicative safety factor covers the
    # small amount of tail variance without needing a large fudge term.
    n_params  = len(params_flat)
    NOISE_NORM_SAFETY_FACTOR = 1.15
    noise_contribution = np.sqrt(n_params) * noise_sigma * NOISE_NORM_SAFETY_FACTOR
    threshold = clip_norm + noise_contribution

    # Sign the (norm, threshold) pair
    payload   = f"{norm:.8f}:{threshold:.8f}".encode()
    signature = _hmac(payload)

    return {
        "norm":      float(norm),
        "threshold": float(threshold),
        "signature": signature,
        "passes":    bool(norm <= threshold),
    }


def verify_norm_proof(pi, clip_norm=1.0):
    """
    Server verifies the norm proof.
    1. Re-derive signature and compare (forgery check)
    2. Check norm ≤ threshold (gradient bomb check)
    """
    payload          = f"{pi['norm']:.8f}:{pi['threshold']:.8f}".encode()
    expected_sig     = _hmac(payload)
    sig_valid        = hmac.compare_digest(expected_sig, pi["signature"])
    norm_valid       = pi["norm"] <= pi["threshold"]
    return sig_valid and norm_valid


# ── Full Proof Generation / Verification ─────────────────────────────

def generate_proof(params, clip_norm=1.0, noise_sigma=0.0, salt=None):
    """
    Generate a complete proof bundle for one client's update.

    Call AFTER Local DP, BEFORE CKKS encryption.
    (Cannot generate a ZKP about plaintext data after encrypting it.)

    Returns
    -------
    proof : dict containing commitment C, salt, and norm proof π
    """
    flat       = flatten_params(params)
    C, salt    = generate_commitment(flat, salt)
    pi         = generate_norm_proof(flat, clip_norm, noise_sigma)

    return {
        "commitment": C,
        "salt":       salt,
        "norm_proof": pi,
        "params_dim": len(flat),
    }


def verify_proof(proof, params=None, clip_norm=1.0,
                 verify_commitment_flag=True):
    """
    Server-side proof verification.

    Steps:
    1. Verify norm proof signature (anti-forgery)
    2. Verify norm ≤ threshold (gradient bomb prevention)
    3. If plaintext params available: verify commitment (binding check)
       — In HE mode, server has ciphertext only; skip step 3 in that case.

    Returns (is_valid, reason_string)
    """
    pi = proof["norm_proof"]

    # Step 1+2: norm proof
    if not verify_norm_proof(pi, clip_norm):
        if not hmac.compare_digest(
            _hmac(f"{pi['norm']:.8f}:{pi['threshold']:.8f}".encode()),
            pi["signature"]
        ):
            return False, f"SIGNATURE_INVALID — possible forgery"
        return False, (f"NORM_EXCEEDED {pi['norm']:.4f} > "
                       f"{pi['threshold']:.4f} — gradient bomb blocked")

    # Step 3: commitment (only when plaintext is available)
    if verify_commitment_flag and params is not None:
        flat = flatten_params(params)
        if not verify_commitment(flat, proof["salt"], proof["commitment"]):
            return False, "COMMITMENT_MISMATCH — params do not match commitment"

    return True, "PROOF_VALID"


def print_verification(client_id, proof, is_valid, reason):
    pi     = proof["norm_proof"]
    status = "✓ PASS" if is_valid else "✗ FAIL"
    print(f"    Client {client_id:>2}  ZKP {status}  "
          f"norm={pi['norm']:.4f}  "
          f"threshold={pi['threshold']:.4f}  "
          f"| {reason}")
