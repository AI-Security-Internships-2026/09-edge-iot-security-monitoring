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
dependencies beyond torch (+ opacus for the DPLSTM layer).

DP-SGD (NEW): differential privacy now happens HERE, inside the actual
training loop, via Opacus — NOT as a one-shot noise-the-final-weights
step after training finishes (that was the old defences/local_dp.py
approach, which noised the entire flattened ~80k-param vector at once
and caused signal-to-noise ratios so bad that loss diverged round over
round instead of converging). Opacus clips and noises PER-SAMPLE
gradients during training, then averages over the batch — this is the
standard DP-SGD approach (Abadi et al. 2016) and gives dramatically
better utility for the same epsilon, since the effective noise on the
aggregated update shrinks with batch size instead of being applied
raw to the full parameter vector.

RAM INSTRUMENTATION: a background-thread RamSampler polls RSS
every RAM_SAMPLE_INTERVAL_S seconds for the duration of the whole
worker body, so the peak/average reported are real measurements
across the run, not a handful of end-of-epoch snapshots. Written to
--memory-output-path as JSON for client.py to read back and fold into
its own per-round results.

Usage (called by client.py via subprocess.run):
    python train_worker.py \
        --client-id 0 --model-type network \
        --num-features 40 --num-classes 8 --local-epochs 5 \
        --global-params-path /tmp/global_r3.npz \
        --train-data-path   /tmp/traindata_c0.npz \
        --output-path       /tmp/raw_params_r3_c0.npz \
        --keys-output-path  /tmp/keys_r3_c0.json \
        --memory-output-path /tmp/mem_r3_c0.json \
        --use-dp --dp-epsilon 5.0 --dp-delta 1e-5 --dp-max-grad-norm 1.0
"""

import argparse
import json
import os
import sys
import threading
import time

# Must be set BEFORE numpy/torch are imported to take effect.
# Each BLAS thread carries its own stack (~1-8MB) — under a tight
# cgroup that's not free, and this subprocess trains on tiny per-round
# batches where multi-threaded BLAS buys nothing anyway.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

RAM_SAMPLE_INTERVAL_S = float(os.environ.get("RAM_SAMPLE_INTERVAL_S", "0.05"))


class RamSampler:
    """Background RAM sampler. Polls RSS every `interval_s` seconds on a
    daemon thread so peak/average reflect what actually happened during
    training, not just a snapshot at epoch boundaries."""
    def __init__(self, interval_s=RAM_SAMPLE_INTERVAL_S):
        self.interval_s = interval_s
        self._samples = []
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        import psutil
        proc = psutil.Process(os.getpid())
        while not self._stop.is_set():
            try:
                self._samples.append(proc.memory_info().rss / 1024 / 1024)
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._stop.clear()
        self._samples = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=1)
        return False

    @property
    def peak(self):
        return max(self._samples) if self._samples else 0.0

    @property
    def avg(self):
        return sum(self._samples) / len(self._samples) if self._samples else 0.0

    @property
    def samples_count(self):
        return len(self._samples)


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
    parser.add_argument("--memory-output-path", type=str, required=True)
    parser.add_argument("--app-path", type=str, default="/app")
    # ── DP-SGD args (NEW) ────────────────────────────────────────────
    parser.add_argument("--use-dp", action="store_true",
                         help="Enable Opacus DP-SGD training (per-sample "
                              "gradient clipping + noise injection).")
    parser.add_argument("--dp-epsilon", type=float, default=5.0,
                         help="Target epsilon for the (epsilon, delta)-DP "
                              "guarantee over this client's local training.")
    parser.add_argument("--dp-delta", type=float, default=1e-5)
    parser.add_argument("--dp-max-grad-norm", type=float, default=1.0,
                         help="Per-sample gradient clipping norm (C).")
    args = parser.parse_args()

    t_start = time.time()

    with RamSampler() as sampler:
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

        # ── Load inputs ────────────────────────────────────────────
        global_npz = np.load(args.global_params_path)
        global_params = [global_npz[k] for k in global_npz.files]

        train_npz = np.load(args.train_data_path)
        X_train = train_npz["X_train"]
        y_train = train_npz["y_train"]

        log(args.client_id,
            f"data loaded: X={X_train.shape}  RAM={ram_mb():.0f}MB")

        # ── Build model + load global weights ───────────────────────
        model = get_model(num_features=args.num_features,
                           num_classes=args.num_classes)
        set_model_parameters(model, global_params)
        del global_params, global_npz

        # ── Train ────────────────────────────────────────────────────
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        X_t = torch.FloatTensor(X_train)
        y_t = torch.LongTensor(y_train)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_t, y_t),
            batch_size=256, shuffle=True
        )

        # ── DP-SGD setup (NEW) ────────────────────────────────────────
        # Wraps model/optimizer/loader in-place. Per-sample gradient
        # clipping + Gaussian noise happens automatically inside
        # loss.backward()/optimizer.step() via Opacus's hooks — the
        # training loop below is UNCHANGED from the non-DP version.
        privacy_engine = None
        achieved_epsilon = None
        noise_multiplier = None

        if args.use_dp:
            from opacus import PrivacyEngine
            privacy_engine = PrivacyEngine()
            model, optimizer, loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=loader,
                epochs=args.local_epochs,
                target_epsilon=args.dp_epsilon,
                target_delta=args.dp_delta,
                max_grad_norm=args.dp_max_grad_norm,
            )
            noise_multiplier = optimizer.noise_multiplier
            log(args.client_id,
                f"DP-SGD engaged: target_eps={args.dp_epsilon} "
                f"delta={args.dp_delta} max_grad_norm={args.dp_max_grad_norm} "
                f"noise_multiplier={noise_multiplier:.4f}")

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

        # ── Unwrap Opacus's GradSampleModule before extracting params ──
        # (Opacus wraps the real nn.Module under ._module; state_dict()
        # keys/shapes on the wrapped model won't match what the server
        # and set_model_parameters() expect, so we must unwrap first.)
        if privacy_engine is not None:
            achieved_epsilon = privacy_engine.get_epsilon(args.dp_delta)
            log(args.client_id, f"DP-SGD complete: achieved_eps={achieved_epsilon:.4f}")
            real_model = model._module
        else:
            real_model = model

        raw_params = get_model_parameters(real_model)
        keys = list(real_model.state_dict().keys())

        # ── Save outputs ─────────────────────────────────────────────
        np.savez(args.output_path, *raw_params)
        with open(args.keys_output_path, "w") as f:
            json.dump(keys, f)

    # sampler.peak / sampler.avg finalized now that the `with` block exited
    with open(args.memory_output_path, "w") as f:
        json.dump({
            "peak_mb": sampler.peak,
            "avg_mb":  sampler.avg,
            "samples": sampler.samples_count,
            "achieved_epsilon": achieved_epsilon,
            "noise_multiplier": noise_multiplier,
        }, f)

    elapsed = time.time() - t_start
    log(args.client_id,
        f"done in {elapsed:.1f}s  peakRAM={sampler.peak:.0f}MB  "
        f"avgRAM={sampler.avg:.0f}MB  (process exiting — memory fully returned to OS)")


if __name__ == "__main__":
    main()