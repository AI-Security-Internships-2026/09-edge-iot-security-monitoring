"""
Binary wire-format helpers.

WHY THIS FILE EXISTS (extra RAM finding on top of the ones already fixed):

The original code sent every numpy array over HTTP as a nested Python
list via `p.tolist()` inside a plain `json=` payload:

    "params": [p.tolist() for p in raw_params]

That is a much bigger memory cost than it looks. A float32 numpy array
is 4 bytes/value in one contiguous C buffer. `.tolist()` converts every
single value into a full Python float object (~24 bytes each, plus list
overhead) — roughly 6-8x the memory of the original array, and `json`
then re-serializes each of those Python floats back into ASCII digits
(even more bytes on the wire), and the receiving end reverses the whole
thing. For a 79k-parameter model this is wasted allocation happening on
every round, on both ends, stacking on top of the HE/DP work.

Fix: keep parameters as raw float32 bytes the whole time. Base64-encode
those bytes (base64 is JSON-safe as plain text) instead of exploding them
into a Python list. This is used for:
  - the global model broadcast (get_global_model / get_round_result)
  - the "bulk" (non-HE) plaintext parameters in submit_update

No torch import here — safe to use from the lightweight client
orchestrator process.
"""

import base64
import numpy as np


def pack_array(arr):
    """numpy array -> base64 string of raw float32 bytes."""
    return base64.b64encode(
        np.ascontiguousarray(arr, dtype=np.float32).tobytes()
    ).decode("ascii")


def unpack_array(b64_str, shape):
    """base64 string + shape -> numpy array (owns its own memory)."""
    raw = base64.b64decode(b64_str)
    return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()


def pack_param_list(params):
    """list[np.ndarray] -> wire dict {data: [...], shapes: [...]}"""
    return {
        "data":   [pack_array(p) for p in params],
        "shapes": [list(p.shape) for p in params],
    }


def unpack_param_list(wire):
    """wire dict -> list[np.ndarray]"""
    return [unpack_array(d, tuple(s))
            for d, s in zip(wire["data"], wire["shapes"])]
