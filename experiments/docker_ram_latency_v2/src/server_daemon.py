"""
Communication-timing daemon. Runs as its own long-lived container
(`server` service) CONCURRENTLY with client0/client1, listening for
their per-round submissions over real HTTP on the Docker compose
network. This is what makes "communication" a genuinely measured
number rather than a shared-volume file write: the client's
send_artifact_over_network() call in client_runner.py times the actual
request/response round trip against this daemon.

This process does NOT do the aggregation math itself (that's
server_aggregate.py, run afterward as a one-shot job reading the
artifacts both this daemon AND the clients already wrote to the shared
results volume) -- it exists purely to give clients something real to
talk to over the network, and to record its own receive-side timing
for comparison against the client's send-side timing.

Exits automatically once it has received NUM_REAL_CLIENTS * ROUNDS
submissions, so `docker compose up --abort-on-container-exit` naturally
tears everything down together once training is done.

ENV VARS
--------
  OUT_DIR              shared results dir, required
  NUM_REAL_CLIENTS      default 2
  ROUNDS                default 3
  PORT                  default 8080
"""

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mem_profiler import write_json  # noqa: E402

OUT_DIR = os.environ["OUT_DIR"]
NUM_REAL_CLIENTS = int(os.environ.get("NUM_REAL_CLIENTS", 2))
ROUNDS = int(os.environ.get("ROUNDS", 3))
PORT = int(os.environ.get("PORT", 8080))

EXPECTED_TOTAL = NUM_REAL_CLIENTS * ROUNDS

_lock = threading.Lock()
_received = []  # list of {client_id, round, recv_time_s, payload_bytes, wall_time}
_start_time = time.time()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # quiet -- we log ourselves below with more useful info

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/submit":
            self.send_response(404)
            self.end_headers()
            return

        qs = parse_qs(parsed.query)
        client_id = qs.get("client_id", ["?"])[0]
        round_idx = qs.get("round", ["?"])[0]

        length = int(self.headers.get("Content-Length", 0))
        t0 = time.time()
        body = self.rfile.read(length)
        try:
            _ = json.loads(body)  # deserialize cost is part of the real receive-side cost
        except json.JSONDecodeError:
            pass
        recv_time_s = time.time() - t0

        with _lock:
            _received.append({
                "client_id": client_id,
                "round": round_idx,
                "recv_time_s": round(recv_time_s, 5),
                "payload_bytes": length,
                "wall_time_s": round(time.time() - _start_time, 3),
            })
            n_done = len(_received)

        print(f"[server_daemon] received client={client_id} round={round_idx} "
              f"bytes={length} recv_time={recv_time_s:.4f}s "
              f"({n_done}/{EXPECTED_TOTAL})")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

        if n_done >= EXPECTED_TOTAL:
            threading.Thread(target=_finish_and_shutdown, args=(self.server,), daemon=True).start()


def _finish_and_shutdown(httpd):
    # Small delay so the HTTP response above actually flushes to the
    # client before we tear the socket down.
    time.sleep(0.5)
    with _lock:
        summary = {
            "expected_total": EXPECTED_TOTAL,
            "received_total": len(_received),
            "submissions": _received,
            "avg_recv_time_s": round(
                sum(r["recv_time_s"] for r in _received) / max(1, len(_received)), 5
            ),
            "total_bytes_received": sum(r["payload_bytes"] for r in _received),
        }
    write_json(os.path.join(OUT_DIR, "server_communication_summary.json"), summary)
    print(f"[server_daemon] all {EXPECTED_TOTAL} submissions received -- shutting down.")
    httpd.shutdown()


def main():
    print(f"[server_daemon] listening on :{PORT}, expecting "
          f"{NUM_REAL_CLIENTS} clients x {ROUNDS} rounds = {EXPECTED_TOTAL} submissions")
    httpd = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
