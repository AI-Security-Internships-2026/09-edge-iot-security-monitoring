"""bench_common.py -- shared utilities for the E8 resource-profiling harness.
Pure stdlib + numpy + psutil (NO torch), so the server container stays light and the wire protocol can be tested anywhere.

Project layout assumed (this folder is a SIBLING of defences/, model_defs.py, data_loader.py):
    <PROJECT_ROOT>/main.py, model_defs.py, task.py, data_loader.py, config_loader.py, defences/krum.py, e8_harness/
PROJECT_ROOT defaults to the parent of this folder; override with the PROJECT_ROOT environment variable."""
import io, json, os, socket, sys, threading, time
from contextlib import contextmanager
import numpy as np

try:
    import psutil
except ImportError as e:
    raise SystemExit("psutil is required (pip install psutil)") from e

HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(HARNESS_DIR))
for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "defences")):
    if p not in sys.path:
        sys.path.insert(0, p)


class RamSampler:
    """Samples this process' RSS every `interval_s` seconds (fields ram_peak_mb / ram_avg_mb / ram_n_samples)."""
    def __init__(self, interval_s=0.05):
        self.interval_s, self._proc = interval_s, psutil.Process(os.getpid())
        self._samples, self._stop, self._thread = [], threading.Event(), None
    def _loop(self):
        while not self._stop.is_set():
            try: self._samples.append(self._proc.memory_info().rss / 2**20)
            except Exception: pass
            time.sleep(self.interval_s)
    def start(self):
        self._samples.clear(); self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True); self._thread.start()
    def stop(self):
        self._stop.set()
        if self._thread: self._thread.join(timeout=2 * self.interval_s + 1)
    def summary(self):
        if not self._samples: return {"ram_peak_mb": None, "ram_avg_mb": None, "ram_n_samples": 0}
        return {"ram_peak_mb": round(max(self._samples), 1), "ram_avg_mb": round(sum(self._samples) / len(self._samples), 1),
                "ram_n_samples": len(self._samples)}


@contextmanager
def timer():
    class _T: elapsed = None
    t, start = _T(), time.perf_counter()
    yield t
    t.elapsed = time.perf_counter() - start


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2)
    print(f"[bench_common] wrote {path}")


def load_config(path):
    with open(path) as f: return json.load(f)


def read_cgroup_limits():
    """(mem_limit_mb, cpu_cores) actually enforced on this container (cgroup v2), else (None, None)."""
    mem_mb = cpu = None
    try:
        v = open("/sys/fs/cgroup/memory.max").read().strip()
        if v != "max": mem_mb = int(v) / 2**20
        q, per = open("/sys/fs/cgroup/cpu.max").read().split()
        if q != "max": cpu = int(q) / int(per)
    except (FileNotFoundError, ValueError, OSError):
        pass
    return mem_mb, cpu


def read_cgroup_mem_peak_mb():
    """Container-level peak memory (cgroup v2 memory.peak), a cross-check on the process RSS sampler."""
    for p in ("/sys/fs/cgroup/memory.peak",):
        try: return round(int(open(p).read().strip()) / 2**20, 1)
        except (FileNotFoundError, ValueError, OSError): pass
    return None


# ---------------- parameter (de)serialisation: list of numpy arrays, as get_model_parameters() returns ----------------
def pack_params(params):
    buf = io.BytesIO(); np.savez(buf, **{f"p{i:04d}": np.asarray(a) for i, a in enumerate(params)}); return buf.getvalue()

def unpack_params(payload):
    z = np.load(io.BytesIO(payload)); return [z[k] for k in sorted(z.files)]


# ---------------- wire protocol: [4B header_len][8B payload_len][json header][payload] ; server replies b"ack" ----------------
def recv_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(min(1 << 16, n - len(buf)))
        if not chunk: raise ConnectionError("connection closed mid-message")
        buf += chunk
    return bytes(buf)

def send_msg(host, port, header, payload, connect_timeout_s=120.0):
    """Connect (retrying while the server is still starting; NOT timed), then time send + ack. Returns (seconds, ok)."""
    deadline, sock = time.time() + connect_timeout_s, None
    while sock is None:
        try: sock = socket.create_connection((host, port), timeout=30)
        except OSError:
            if time.time() > deadline: return float("nan"), False
            time.sleep(0.5)
    with sock:
        h = json.dumps(header).encode()
        t0 = time.perf_counter()
        sock.sendall(len(h).to_bytes(4, "big") + len(payload).to_bytes(8, "big") + h + payload)
        ok = sock.recv(16) == b"ack"
        return time.perf_counter() - t0, ok

def recv_msg(conn):
    hl = int.from_bytes(recv_exact(conn, 4), "big"); pl = int.from_bytes(recv_exact(conn, 8), "big")
    header = json.loads(recv_exact(conn, hl)); payload = recv_exact(conn, pl)
    conn.sendall(b"ack")
    return header, payload
