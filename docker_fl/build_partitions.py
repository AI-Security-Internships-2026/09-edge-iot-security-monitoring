"""
Offline partition builder — run ONCE, OUTSIDE the RAM-constrained client
containers, to produce small per-client dataset files.

Why this exists (realism + safety fix)
----------------------------------------
The previous approach had each 256MB client container call
load_partition_*() directly at startup, which pulls the ENTIRE
preprocessed corpus (~568k rows for the network model, several hundred
MB) into the parent client.py process before slicing out just that
client's Dirichlet shard. Two problems with that:

  1. Not realistic: a real IoT gateway never has access to the full
     federation's data or computes a partition at runtime — it only
     ever has its own local traffic logs. Doing the partition inside
     the constrained container misrepresents the deployment.
  2. Not safe: loading the full corpus risks OOM-killing the parent
     client.py process itself (not the training subprocess this time)
     inside the same 256MB cgroup used for the earlier train_worker fix.

Fix: preprocess + partition ONCE, here, outside any RAM limit, using
real Edge-IIoTset data — optionally capped to a manageable PORTION via
--max-rows (stratified by Attack_type, so rare classes like
SQL_injection aren't wiped out by a naive random sample). Save each
client's slice to its own compact .npz. At runtime, client.py just
np.load()s its own file — O(1) relative to corpus size, matching how a
real gateway actually works, and with zero pandas/sklearn import cost
in the constrained container.

Usage
-----
From the docker_fl/ directory, with the raw CSV present at the path
data_loader.py expects (or an existing dnn_preprocessed_cache*.npz):

    python build_partitions.py --model-type network     --num-clients 2 --max-rows 100000
    python build_partitions.py --model-type application --num-clients 2 --max-rows 100000

Or, if pandas/sklearn aren't installed on your host, run it inside a
one-off, UNCONSTRAINED container using the already-built image (no
mem_limit applies to `docker compose run` unless you add one):

    docker compose run --rm --no-deps -v "${PWD}/datasets:/datasets" ^
        fl_server_he python build_partitions.py --model-type network --num-clients 2 --max-rows 100000

Output: /datasets/partitions/client_{id}_{model_type}.npz — mount
./datasets/partitions read-only into the client containers.
"""

import argparse
import os
import sys

sys.path.insert(0, "/app")                                   # in-container
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))  # local host run

from data_loader import save_client_partitions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["network", "application"], required=True)
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=None,
                         help="Cap the corpus to a portion of this many rows "
                              "(stratified by Attack_type) before partitioning. "
                              "Omit to use the full ~568k-row corpus.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="/datasets/partitions")
    args = parser.parse_args()

    print(f"Building {args.num_clients} client partitions for model_type={args.model_type} "
          f"(max_rows={args.max_rows or 'full corpus'})...")

    save_client_partitions(
        model_type=args.model_type,
        num_clients=args.num_clients,
        out_dir=args.out_dir,
        max_rows=args.max_rows,
        test_size=args.test_size,
        alpha=args.alpha,
        seed=args.seed,
    )
    print("Done.")


if __name__ == "__main__":
    main()