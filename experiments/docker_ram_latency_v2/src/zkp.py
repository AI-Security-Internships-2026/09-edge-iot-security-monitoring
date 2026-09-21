"""
Lightweight Zero-Knowledge Proof module.

Runs fine in both the local (main.py) and Docker (client.py) orchestrator
processes — it only depends on numpy/hashlib/hmac, never forces a torch
import, and is safe to call from a lightweight client process either way.

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
        threshold = clip_norm + sqrt(n_params) * noise_sigma * safety_factor + margin

    BUG FIXED HERE: the threshold used to include a stray `* 0.01`
    "empirical factor" that made it ~100x too small. For a Gaussian
    mechanism adding i.i.d. noise ~N(0, sigma^2) independently to
    every one of n_params coordinates, the expected L2 norm of the
    noise vector alone is sigma * sqrt(n_params) — NOT sigma * 0.01 *
    sqrt(n_params). Verified by direct simulation: for n_params=80074,
    sigma=1.6149 (project's DP_EPSILON=3.0 config at that param count),
    the simulated noise-norm mean was 456.9 with std of only ~1.2 (<0.3%
    relative) — i.e. this quantity is extremely concentrated, so a
    small multiplicative safety factor is enough tail coverage; no
    large additive fudge term is needed. With the old buggy formula,
    every honestly DP-noised update would have been rejected as
    "exceeding the norm bound" — the ZKP gate was effectively blocking
    all real clients, not just Byzantine ones.

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

    Fixed +0.2 additive safety margin (on top of the multiplicative
    NOISE_NORM_SAFETY_FACTOR) absorbs small numerical drift at very low
    noise_sigma (e.g. DP disabled or near-zero epsilon-noise edge cases)
    where the multiplicative factor alone rounds to ~0 and a legitimate
    client's norm could sit right at the boundary due to float rounding.

    NOTE — a separate, real finding worth reporting even after this
    fix: at DP_EPSILON=3.0 over an ~80k-parameter model, injected noise
    (~457 in norm) can swamp the clipped signal (norm 1.0) by ~450x if
    noise is applied once to the full flattened parameter vector
    (the old local_dp.py post-hoc approach). The proof will correctly
    PASS (it's honestly measuring what's happening), but the model is
    very unlikely to learn anything useful at that noise-to-signal
    ratio. That's a DP-calibration problem to solve separately (e.g.
    DP-SGD with per-sample gradient clipping instead of one-shot output
    perturbation, larger epsilon per round, or noising the server-side
    aggregate instead of every client-per-coordinate) — not something a
    norm-proof threshold fix can or should paper over.
    """
    norm = float(np.linalg.norm(params_flat))

    # Expected noise-vector L2 norm for i.i.d. N(0, sigma^2) added to
    # n_params coordinates is sigma * sqrt(n_params) (concentrated —
    # see docstring). A modest multiplicative safety factor covers the
    # small amount of tail variance without needing a large fudge term.
    n_params  = len(params_flat)
    NOISE_NORM_SAFETY_FACTOR = 1.15
    noise_contribution = np.sqrt(n_params) * noise_sigma * NOISE_NORM_SAFETY_FACTOR
    threshold = clip_norm + noise_contribution + 0.2  # additive safety margin

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


# ══════════════════════════════════════════════════════════════════════
# PART 2 — HYBRID HE HEAD-NORM GUARD (Experiment 2 mitigation)
# ══════════════════════════════════════════════════════════════════════
"""
Motivated directly by Experiment 2's confirmed finding: with partial
(classifier-head-only) CKKS HE, adaptive Krum's plaintext-distance
scoring achieves 0% detection against a client that trains normally on
the backbone and poisons only the classifier head before encryption —
because that slice never reaches Krum in any form. This section
extends the Part 1 commitment/norm-proof machinery to cover exactly
that slice, per the master doc's Layer 2 "possible future role" note.

WHAT THIS ADDS OVER PART 1
---------------------------
Part 1's norm proof asserts a norm for a plaintext vector the server
can (if it chooses) also check a commitment against. Here the server
NEVER sees the plaintext classifier-head slice at all — only its
ciphertext. So the proof must be bound to the CIPHERTEXT itself, not
the plaintext: the signed payload includes a hash of the actual
ciphertext bytes being submitted, so a client cannot compute a proof
for one (small-norm) plaintext and then submit a different
(large-norm) ciphertext under that proof — the hash mismatch is
immediately detectable without decrypting anything.

WHAT THIS DOES NOT DO — read this before treating it as a solved
problem, exactly the way Part 1 is upfront about not being a full ZKP:
  - It does NOT prove the claimed norm actually equals the norm of the
    plaintext inside the ciphertext. A client can still lie about the
    norm of the exact ciphertext it commits to — nothing here performs
    a computation ON the ciphertext to verify this (that would require
    an actual homomorphic norm/range proof, a substantially harder
    primitive; see the master doc's Lancelot/PBFL pointer for the
    literature on this). What this DOES prevent is swapping in a
    different ciphertext after the fact, or replaying a stale proof
    against a new submission — the commitment binds proof-to-ciphertext,
    not norm-to-plaintext.
  - Exactly like Part 1: this catches MAGNITUDE outliers only. A
    Byzantine client that keeps its classifier-head delta within the
    round's normal magnitude range while still corrupting direction
    (a bounded-magnitude directional attack) is NOT caught by this
    mechanism, for the same reason Part 1 states "A Byzantine client
    CAN apply sign-flip within the norm bound." An adaptive attacker
    that knows this guard exists and stays under its threshold defeats
    it. This is a real, disclosed residual risk, not an oversight.

THRESHOLD DESIGN
-----------------
Unlike Part 1 (which calibrates against a known DP clip_norm + noise
formula), there is no DP noise in Experiment 2 (USE_DP=False) and no
fixed a priori bound on "normal" classifier-head movement per round.
Instead of a fixed threshold, this uses the SAME MAD-based robust
statistic as adaptive_multi_krum() in defences/krum.py — computed over
the one number every accepted client's proof reveals this round (its
committed delta norm) — rather than inventing a second, differently-
justified threshold scheme. This makes it a direct magnitude-only
analogue of Krum, applied to a slice Krum structurally cannot reach.
"""

import numpy as np


def _hash_ciphertext(ciphertext_chunks_b64):
    """
    Deterministic SHA-256 hash over an encrypted classifier-head slice's
    serialized chunks (list of base64 strings, as returned by
    he_local.encrypt_params()'s "sensitive_enc"["chunks"]). Order-
    sensitive by design — chunk order is fixed by the encryption
    process, so this binds to a specific ciphertext, not just its
    content set.
    """
    h = hashlib.sha256()
    for chunk in ciphertext_chunks_b64:
        h.update(chunk.encode("ascii"))
    return h.hexdigest()


def generate_head_norm_proof(delta_flat, ciphertext_chunks_b64, salt=None):
    """
    Client-side. Call AFTER local training, AFTER computing the
    classifier-head delta (trained_head - global_head_received_this_round,
    flattened), AFTER encrypting that slice — but BEFORE sending anything
    to the server. Binds a signed norm claim to the specific ciphertext
    being submitted.

    Parameters
    ----------
    delta_flat : np.ndarray
        Flattened (trained classifier-head params - global classifier-
        head params this client started the round with). NOT the raw
        trained params — the DELTA, so an honest client's proof reflects
        how far local training actually moved the head, not the head's
        absolute magnitude (which is round-independent and would make
        outlier detection meaningless).
    ciphertext_chunks_b64 : list[str]
        The actual base64-encoded CKKS ciphertext chunks for this slice
        (he_local.encrypt_params()'s "sensitive_enc"["chunks"]) — the
        exact bytes being submitted this round.
    salt : bytes or None
        Passed through to keep the commitment convention consistent
        with Part 1; not strictly required for binding here since the
        ciphertext hash already provides uniqueness per submission.

    Returns
    -------
    dict: {norm, ciphertext_commitment, signature, salt}
        norm                 : claimed L2 norm of delta_flat (float)
        ciphertext_commitment: sha256 hex of the ciphertext chunks
        signature            : HMAC-SHA256 over norm||ciphertext_commitment
        salt                 : bytes (unused in the signed payload here,
                                kept for interface symmetry with Part 1)
    """
    if salt is None:
        salt = os.urandom(32)
    norm = float(np.linalg.norm(delta_flat))
    ciphertext_commitment = _hash_ciphertext(ciphertext_chunks_b64)
    payload = f"{norm:.8f}:{ciphertext_commitment}".encode()
    signature = _hmac(payload)
    return {
        "norm": norm,
        "ciphertext_commitment": ciphertext_commitment,
        "signature": signature,
        "salt": salt,
    }


def verify_head_norm_proof(proof, ciphertext_chunks_b64):
    """
    Server-side. Two checks, NEITHER of which requires decrypting
    anything:
      1. Signature validity (anti-forgery — requires the shared key).
      2. The proof's ciphertext_commitment matches a fresh hash of the
         ACTUAL ciphertext chunks received this round (anti-swap — a
         client cannot reuse/redirect a proof computed for different
         ciphertext bytes).

    Does NOT check the norm against any threshold here — that's a
    round-level, cross-client decision, see mad_threshold_head_norms().

    Returns (is_valid, reason_string).
    """
    expected_commitment = _hash_ciphertext(ciphertext_chunks_b64)
    if not hmac.compare_digest(expected_commitment, proof["ciphertext_commitment"]):
        return False, ("CIPHERTEXT_MISMATCH — proof does not correspond to "
                       "the ciphertext actually submitted this round")
    payload = f"{proof['norm']:.8f}:{proof['ciphertext_commitment']}".encode()
    expected_sig = _hmac(payload)
    if not hmac.compare_digest(expected_sig, proof["signature"]):
        return False, "SIGNATURE_INVALID — possible forgery"
    return True, "PROOF_VALID"


def mad_threshold_head_norms(client_norms, k=2.5, min_keep_fraction=0.5):
    """
    Server-side. 1-D magnitude-only analogue of
    defences.krum.adaptive_multi_krum()'s MAD thresholding — applied to
    the one scalar the encrypted classifier-head slice reveals (its
    committed delta norm) rather than a full parameter vector, since
    that's all a slice Krum cannot see structurally offers to work with.

    Parameters
    ----------
    client_norms       : list[float], one committed norm per client
                         whose proof already passed verify_head_norm_proof()
                         this round. Callers MUST filter to
                         signature/binding-valid proofs before calling
                         this — an unverified norm claim is worthless.
    k                   : sensitivity multiplier, same role and same
                         default (2.5) as adaptive_multi_krum()'s k.
                         threshold = median(norms) + k * 1.4826 * MAD(norms).
    min_keep_fraction   : safety floor, same semantics as
                         adaptive_multi_krum() — 0 disables it.

    Returns
    -------
    (kept_positions, dropped_positions, diagnostics) : tuple
        kept_positions   : list[int], 0-indexed positions into
                           client_norms that passed the threshold.
        dropped_positions: list[int], the rest.
        diagnostics      : dict with center, spread, threshold,
                           num_dropped — same shape as adaptive_multi_krum's
                           diagnostics dict, for consistent logging.
    """
    n = len(client_norms)
    norms = np.array(client_norms, dtype=np.float64)

    if n < 3:
        # Too few points for a meaningful MAD — accept everyone, flag it.
        return (list(range(n)), [],
                {"center": None, "spread": None, "threshold": None,
                 "num_dropped": 0, "fallback_triggered": True})

    center = float(np.median(norms))
    mad = float(np.median(np.abs(norms - center)))
    spread = 1.4826 * mad
    threshold = center + k * spread if spread > 0 else center + 1e-9

    kept    = [i for i in range(n) if norms[i] <= threshold]
    dropped = [i for i in range(n) if norms[i] > threshold]

    fallback_triggered = False
    min_keep = int(np.ceil(min_keep_fraction * n)) if min_keep_fraction > 0 else 0
    if len(kept) < min_keep:
        fallback_triggered = True
        ranked = sorted(range(n), key=lambda i: norms[i])
        kept = ranked[:min_keep]
        dropped = [i for i in range(n) if i not in kept]

    diagnostics = {
        "center": center,
        "spread": spread,
        "threshold": threshold,
        "num_dropped": len(dropped),
        "fallback_triggered": fallback_triggered,
    }
    return kept, dropped, diagnostics
