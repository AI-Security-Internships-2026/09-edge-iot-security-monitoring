# E8 resource-profiling harness (Table 5)

Measures, for **FedAvg, Adaptive Krum, Calibrated Krum and DP + Calibrated Krum**, at **1.0 and 0.5 vCPU** (`--memory=2048m`):
client train time per round, serialisation and send time, payload size, process RAM, and server-side aggregation latency at n=10 clients.
It uses your real code: `model_defs.get_model`, `data_loader.load_partition_network`, `defences/krum.py`.

## 1. Where to put it
Copy this whole folder next to `main.py`:

```
<project root>/            (the folder containing main.py, model_defs.py, task.py, data_loader.py, defences/)
└── e8_harness/            <- this folder
```
Nothing else needs editing: `PROJECT_ROOT` defaults to the parent of `e8_harness/`, and the code is bind-mounted into the container, so **no pre-existing image is needed**; the Dockerfile builds one (Python 3.11, CPU torch 2.3.1, opacus 1.5.2, numpy 1.26.4 ...). I could not see your `requirements.txt`, so **replace the pins in `Dockerfile` with your own versions** (`pip freeze | grep -i -E "torch|opacus|numpy|pandas|scikit|scipy|psutil"`).

## 2. Three things only you can set
1. **Dataset location.** Open `data_loader.py`, find where it reads `DNN-EdgeIIoT-dataset.csv`. If it is a path inside the project folder, nothing to do. If it is elsewhere, pass `DATA_DIR=<that folder>` (mounted read-only at the same absolute path inside the container).
2. **Rows per client.** Take `train_rows` and `test_rows` from the `config` block of one old `client_0_results.json` and pass them as `TRAIN_ROWS` / `TEST_ROWS`. Train time scales with rows, so rows must match for rows of Table 5 to be comparable. If you skip this and the real data loads, each client uses the whole partition; if real data cannot load, 40,000 synthetic rows are used (the JSON says `"data_source": "synthetic"`).
3. **Run FedAvg here too.** The old FedAvg/DP-SGD/HE rows came from a different image and possibly a different machine, and their payload (171 KB) is not what this harness serialises (about 326 KB: the full state dict incl. BatchNorm buffers, uncompressed `.npz`). Running `fedavg` with this harness gives a baseline measured on the same machine, image and serialisation; use legacy numbers only for the HE rows, and say so in the table note.

## 3. Run
```bash
# 0. optional smoke test, no Docker (tiny data, unthrottled, NOT for Table 5)
python e8_harness/run_e8_local.py calibrated_krum
# 1. real run (Linux/macOS/WSL)
DATA_DIR=/path/to/dataset TRAIN_ROWS=<n> TEST_ROWS=<m> bash e8_harness/run_e8.sh            # all 4 modes x 2 profiles = 8 cells
bash e8_harness/run_e8.sh dp_calibrated_krum                                               # one mode
# Windows PowerShell
.\e8_harness\run_e8.ps1 -DataDir D:\data -TrainRows <n> -TestRows <m>
```
Results go to `./e8_results/<mode>_<profile>vcpu/` (`client_0_results.json`, `client_1_results.json`, `server_<mode>_results.json`, `server_communication_summary.json`, plus `*.log`). Send back the whole `e8_results/` folder.

## 4. What was tested here, and what was not
Tested (numpy only, real `krum.py`): the server in all four modes, the wire protocol, and the JSON output (`python e8_harness/test_harness_offline.py`, with `PROJECT_ROOT` pointing at a folder that has `defences/krum.py`).
**Not tested**: the client (torch, Opacus and your `data_loader` are not available in my environment) and the Docker scripts. Run the smoke test first; if it fails, send me the traceback.

## 5. Choices to be aware of
- Aggregation is timed at n=10 clients (2 real updates + 8 jittered copies), 3 warm-started calls per round (`--repeats`), with the same arguments `main.py` uses (f=2, k=2.5, MAD, keep fraction 0.5, diagnostics on). Calibrated Krum uses `hetero_fit_coeffs.json` (the E5 fit) so the heterogeneity term is active; DP calibration is on only in `dp_calibrated_krum`, with the median honest noise multiplier assigned to synthetic clients, as `public_noise_multiplier_map` does.
- The calibration cost is the difference `calibrated_krum_time_s - adaptive_krum_time_s`.
- `dp_calibrated_krum` uses batch size 512, epsilon 5, clip norm 1.5 (the E5 manifest value; edit `configs/dp_*` if E6 used 1.0), one persistent `PrivacyEngine` over all 6 epochs.
- The client does not receive a global model back and FedProx's proximal step is not applied; this is the same for every mode.
- Sandbox timings of the aggregators (a different CPU) were about 1 ms (FedAvg), 15 ms (Adaptive Krum) and 14-18 ms (Calibrated Krum) per call; do not quote these.
