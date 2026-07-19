"""
Training worker — runs as an ISOLATED SUBPROCESS, one per FL round.

Why a subprocess and not just a function call in client.py
------------------------------------------------------------
Deleting python objects and calling gc.collect() frees memory WITHIN a
process, but CPython usually doesn't hand that memory back to the OS —
it keeps freed blocks in its allocator arenas for reuse. So a PyTorch
training peak and a later TenSEAL encryption peak can effectively stack
inside one long-lived process, even if everything from training is
`del`-ed in between, because the process's RSS high-water mark (which
is what Docker/cgroups report) doesn't come back down on its own.

When a process exits entirely, the OS reclaims 100% of its memory —
guaranteed, no exceptions. So training runs here, in a subprocess that
starts, trains, saves its output to disk, and exits. The parent
(client.py) never imports torch at all, so its own baseline stays
small, and the container's peak becomes
    max(training_subprocess_peak, DP+ZKP+HE_peak)
instead of their sum.

IMPORTANT: this subprocess imports model_defs, NOT task. task.py pulls
in sklearn + pandas + data_loader (for f1_score, class-count constants,
FocalLoss weighting) which this subprocess never uses — importing task
here would add ~80-120MB of sklearn/pandas/scipy on top of torch inside
the same 256MB cgroup the parent client.py process already shares,
which is what was causing the OOM SIGKILL. model_defs.py has zero
dependencies beyond torch.

Usage (called by client.py via subprocess.run):
    python train_worker.py \
        --client-id 0 --model-type network \
        --num-features 40 --num-classes 8 --local-epochs 5 \
        --global-params-path /tmp/global_r3.npz \
        --train-data-path   /tmp/traindata_c0.npz \
        --output-path       /tmp/raw_params_r3_c0.npz \
        --keys-output-path  /tmp/keys_r3_c0.json
"""

import argparse
import json
import os
import sys
import time

# Must be set BEFORE numpy/torch are imported to take effect.
# Each BLAS thread carries its own stack (~1-8MB) — under a tight
# cgroup that's not free, and this subprocess trains on tiny per-round
# batches where multi-threaded BLAS buys nothing anyway.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def log(client_id, msg):
    print(f"[TrainWorker {client_id}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("--num-features", type=int, required=True)
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--global-params-path", type=str, required=True)
    parser.add_argument("--train-data-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--keys-output-path", type=str, required=True)
    parser.add_argument("--app-path", type=str, default="/app")
    args = parser.parse_args()

    t_start = time.time()

    # Import torch and friends ONLY inside this subprocess.
    sys.path.insert(0, args.app_path)
    import numpy as np
    import torch
    import torch.nn as nn
    import psutil
    from model_defs import get_model, set_model_parameters, get_model_parameters

    torch.set_num_threads(1)

    def ram_mb():
        return psutil.Process().memory_info().rss / 1024 / 1024

    log(args.client_id, f"start  RAM={ram_mb():.0f}MB")

    # ── Load inputs ────────────────────────────────────────────────
    global_npz = np.load(args.global_params_path)
    global_params = [global_npz[k] for k in global_npz.files]

    train_npz = np.load(args.train_data_path)
    X_train = train_npz["X_train"]
    y_train = train_npz["y_train"]

    log(args.client_id,
        f"data loaded: X={X_train.shape}  RAM={ram_mb():.0f}MB")

    # ── Build model + load global weights ─────────────────────────
    model = get_model(num_features=args.num_features,
                       num_classes=args.num_classes)
    set_model_parameters(model, global_params)
    del global_params, global_npz

    # ── Train (plain local Adam training loop, matches original) ──
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.FloatTensor(X_train)
    y_t = torch.LongTensor(y_train)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t, y_t),
        batch_size=256, shuffle=True
    )

    for epoch in range(args.local_epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        log(args.client_id,
            f"epoch {epoch + 1}/{args.local_epochs}  "
            f"loss={epoch_loss / len(loader):.4f}  RAM={ram_mb():.0f}MB")

    raw_params = get_model_parameters(model)
    keys = list(model.state_dict().keys())

    # ── Save outputs and exit ──────────────────────────────────────
    np.savez(args.output_path, *raw_params)
    with open(args.keys_output_path, "w") as f:
        json.dump(keys, f)

    elapsed = time.time() - t_start
    log(args.client_id,
        f"done in {elapsed:.1f}s  peakRAM~{ram_mb():.0f}MB  "
        f"(process exiting — memory fully returned to OS)")


if __name__ == "__main__":
    main()