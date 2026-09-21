"""
RAM sampling + stage timing for the Docker RAM/Latency test suite.

Fixes the old mem_limit_mb self-report bug (Contradiction #18 in the
project's master doc): the old client.py hardcoded a "200" label into
every result JSON's config.mem_limit_mb field regardless of the real
declared/enforced limit. This module reads the ACTUAL cgroup-enforced
memory and CPU limits directly from /sys/fs/cgroup at call time, so the
new result JSONs report a real, live-read value instead of a stale
constant -- supports both cgroup v1 and v2 layouts (Docker Desktop/WSL2
and native Linux hosts commonly differ here).
"""

import json
import os
import threading
import time


# ── Real cgroup limit reading (replaces the old hardcoded "200") ──────

def _read_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p) as f:
                    return f.read().strip()
            except OSError:
                continue
    return None


def read_cgroup_memory_limit_mb():
    """Returns the real enforced memory limit in MB, or None if
    unconstrained/unreadable (e.g. 'max' on cgroup v2 means no limit)."""
    raw = _read_first_existing([
        "/sys/fs/cgroup/memory.max",                       # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",      # cgroup v1
    ])
    if raw is None or raw == "max":
        return None
    try:
        val = int(raw)
    except ValueError:
        return None
    # cgroup v1 uses a very large sentinel (e.g. 9223372036854771712) for
    # "unlimited" -- treat anything above 1TB as effectively unlimited.
    if val > (1 << 40):
        return None
    return round(val / (1024 * 1024), 1)


def read_cgroup_cpu_limit():
    """Returns the real enforced CPU limit as a fraction of a core
    (e.g. 0.5), or None if unconstrained/unreadable."""
    # cgroup v2: "quota period" in one file, "max" = unlimited
    raw = _read_first_existing(["/sys/fs/cgroup/cpu.max"])
    if raw is not None:
        parts = raw.split()
        if len(parts) == 2 and parts[0] != "max":
            try:
                return round(int(parts[0]) / int(parts[1]), 3)
            except (ValueError, ZeroDivisionError):
                return None
        return None

    # cgroup v1: quota and period in separate files
    quota = _read_first_existing(["/sys/fs/cgroup/cpu/cpu.cfs_quota_us"])
    period = _read_first_existing(["/sys/fs/cgroup/cpu/cpu.cfs_period_us"])
    if quota is not None and period is not None:
        try:
            q, p = int(quota), int(period)
            if q <= 0:
                return None
            return round(q / p, 3)
        except (ValueError, ZeroDivisionError):
            return None
    return None


def read_current_rss_mb():
    """Current resident set size in MB, read directly from
    /proc/self/status (works identically in-container or bare-metal)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return None


# ── Background RAM sampler ─────────────────────────────────────────────

class RamSampler:
    """Samples RSS on a background thread at a fixed interval for the
    lifetime of a `with RamSampler() as sampler:` block. Call
    sampler.summary() afterward for peak/avg/samples."""

    def __init__(self, interval_s=0.25):
        self.interval_s = interval_s
        self._samples = []  # list of (elapsed_s, rss_mb)
        self._stop_evt = threading.Event()
        self._thread = None
        self._t0 = None

    def _run(self):
        while not self._stop_evt.is_set():
            rss = read_current_rss_mb()
            if rss is not None:
                self._samples.append((time.time() - self._t0, rss))
            self._stop_evt.wait(self.interval_s)

    def __enter__(self):
        self._t0 = time.time()
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self.interval_s)

    def summary(self):
        if not self._samples:
            return {"peak_mb": None, "avg_mb": None, "n_samples": 0}
        vals = [v for _, v in self._samples]
        return {
            "peak_mb": round(max(vals), 1),
            "avg_mb": round(sum(vals) / len(vals), 1),
            "n_samples": len(vals),
        }


# ── Stage timer ──────────────────────────────────────────────────────

class StageTimer:
    """Collects named stage durations across a run:
        with StageTimer() as t:
            with t.stage("train"):
                ...
            with t.stage("encrypt"):
                ...
        t.durations -> {"train": 12.3, "encrypt": 0.4}
    """

    def __init__(self):
        self.durations = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    class _Stage:
        def __init__(self, parent, name):
            self.parent = parent
            self.name = name
            self.t0 = None

        def __enter__(self):
            self.t0 = time.time()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed = time.time() - self.t0
            self.parent.durations[self.name] = self.parent.durations.get(self.name, 0.0) + elapsed
            return False

    def stage(self, name):
        return StageTimer._Stage(self, name)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
