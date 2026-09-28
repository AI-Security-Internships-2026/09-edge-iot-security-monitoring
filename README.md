# FL-IDS: Privacy-Preserving, Byzantine-Robust Federated Intrusion Detection for Edge IoT

> **CNIT/PNTLab Pisa · TECIP · Scuola Superiore Sant'Anna — AI Security Internship 2026**

A federated-learning intrusion-detection pipeline for IoT/IIoT edge nodes.
Clients train a CNN-LSTM locally on the Edge-IIoTset dataset and share only
model updates. The pipeline lets you study how **privacy** (DP-SGD, partial
CKKS homomorphic encryption) and **robustness** (Byzantine-tolerant
aggregation) interact, rather than evaluating them in isolation.

The active implementation lives in **`experiments/Current model/`**. There is
no `src/main.py` at the repository root — older instructions pointing there
are obsolete.

---

## Repository layout

```
experiments/
  Current model/
    main.py                  <- unified FL training loop (all experiments)
    data_loader.py           <- Edge-IIoTset loading, preprocessing, splits
    model_defs.py            <- CNN-LSTM (BatchNorm/LSTM, or GroupNorm/DPLSTM for DP)
    config_loader.py         <- validated loader for configs/hyperparams.json
    defences/
      krum.py                <- fedavg, coordinate_median, trimmed_mean, multi_krum,
                                adaptive_multi_krum, calibrated_adaptive_multi_krum
      hmac_norm_guard.py     <- HMAC-signed norm assertion (see "What the norm guard is")
      he_aggregation.py, he_local.py   <- CKKS (TenSEAL) partial-HE helpers
    tests/                   <- pytest suite (aggregator hand-calculation fixtures, etc.)
    requirements.txt         <- exact pinned dependencies
  configs/
    hyperparams.json         <- tunables with validated_on_split provenance
    E1_baselines.json        <- baseline campaign definition
    E2_aggregators.json      <- aggregator x attack campaign definition
  docker_ram_latency_v2/     <- separate RAM/latency Docker test suite (own README)
datasets/                    <- NOT committed; you download it (Step 4)
docs/                        <- literature review, proposal, weekly progress
```

---

## Quickstart (5 steps)

Tested target: Python 3.11 on Linux / WSL2. The pinned versions in
`requirements.txt` were frozen from a Python 3.11 virtualenv.

**1. Clone**

```bash
git clone https://github.com/AI-Security-Internships-2026/09-edge-iot-security-monitoring.git
cd 09-edge-iot-security-monitoring
```

**2. Create and activate a virtual environment**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**3. Install pinned dependencies**

```bash
pip install -r "experiments/Current model/requirements.txt"
```

**4. Download the dataset** (about the largest single file in the project;
see [Dataset](#dataset-and-checksum) for the checksum to verify it)

```bash
pip install kaggle        # then place your kaggle.json API token in ~/.kaggle/
mkdir -p "datasets/Edge-IIoTset dataset/Selected dataset for ML and DL"
kaggle datasets download \
  -d mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot \
  -f "Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv" \
  -p "datasets/Edge-IIoTset dataset/Selected dataset for ML and DL"
unzip -o "datasets/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv.zip" \
  -d "datasets/Edge-IIoTset dataset/Selected dataset for ML and DL"
rm "datasets/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv.zip"
```

Dataset page: <https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot>
(also mirrored on IEEE DataPort, DOI 10.21227/mbc1-1h68).

**5. Run the baseline (FedAvg, no attack defence) — a short smoke run first**

```bash
cd "experiments/Current model"
python main.py network --ablation-mode baseline --aggregator fedavg --rounds 2 --tag quickstart
```

`--rounds 2` stops after two rounds so you can confirm the pipeline works end
to end. Drop it to use the default of 25 rounds. Everything routes through CLI
flags; you do not need to edit `ABLATION_MODE` or any constant in the source.
The first run also builds the preprocessed cache under `datasets/`, so it is
slower than later runs.

Use `application` instead of `network` to train the 52-feature application
model.

---

## Dataset and checksum

The pipeline reads exactly one file:

```
datasets/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv
```

Verify your copy before running experiments:

```bash
md5sum "datasets/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"
```

| File | MD5 |
|---|---|
| `DNN-EdgeIIoT-dataset.csv` | `<FILL IN — md5sum of the copy the paper results were produced from>` |

Results are only comparable if this checksum matches the one recorded here.
If the Kaggle copy is ever updated upstream, record the new checksum and note
the change in `docs/weekly-progress.md`.

Dataset licence: free for academic research; cite Ferrag et al., *Edge-IIoTset*,
IEEE Access, 2022, DOI 10.1109/ACCESS.2022.3165809.

---

## Choosing an aggregator

`--aggregator` selects the plaintext aggregation rule. It applies when
`--ablation-mode` is `baseline` or `krum_baseline`; other modes (HE, norm-guard
pipelines) use their own aggregation and ignore it with a warning.

| `--aggregator` | Rule |
|---|---|
| `fedavg` | Weighted average, no defence |
| `median` | Coordinate-wise median |
| `trimmed_mean` | Coordinate-wise trimmed mean (β from `hyperparams.json`) |
| `krum` / `multi_krum` | Fixed-M Multi-Krum (Blanchard et al., 2017) |
| `adaptive_krum` | Multi-Krum with a MAD-based adaptive acceptance threshold |
| `calibrated_krum` | Adaptive Krum with DP-noise-variance calibration (use with `--ablation-mode calibrated_krum_dp_sweep` for DP runs) |

Example — sign-flip attack from clients 1 and 2 against adaptive Krum:

```bash
python main.py network --ablation-mode krum_baseline --aggregator adaptive_krum \
    --attack-type sign_flip --byzantine 1,2 --seed 42 --tag e2_adaptive_krum_sign_flip
```

Attack types are set with `--attack-type`; run `python main.py --help` for the
full list and per-attack parameters.

---

## Reproducing the reported results

- **E1 (baselines)** and **E2 (aggregators × attacks)** are defined in
  `experiments/configs/E1_baselines.json` and `E2_aggregators.json`.
- E2's summary table is produced from the per-cell result CSVs by
  `analyze_e2.py`, which expects them under `e2_results_pull/` and writes
  `table2_full.csv` and `e2_all_cells_raw.csv`.
- Use seeds 42, 123 and 456 for the reported mean ± SD.

---

## Tests

```bash
pip install pytest        # not part of the pinned runtime requirements
cd "experiments/Current model"
python -m pytest tests -v
```

`tests/test_aggregators.py` checks every aggregator against hand-calculated
5-client, 3-dimension fixtures.

---

## What the norm guard is (and is not)

`defences/hmac_norm_guard.py` implements an **HMAC-signed, self-reported
norm assertion**: each client claims the magnitude of its update and signs
that claim with a shared key. It is **not a zero-knowledge proof** and
provides no cryptographic soundness against a malicious client. Limitations:

- The HMAC key is a hardcoded value shared by all parties, so it demonstrates
  the message flow, not real authentication.
- The claimed norm is self-reported; the server cannot verify it against the
  actual update.
- It checks **magnitude only**. It cannot detect attacks that stay within the
  norm bound but point in a harmful direction.

An earlier name for this module implied a zero-knowledge proof. Any reference
to that older name is a naming error, not a change in what the code does.

---

## Docker RAM/latency suite

`experiments/docker_ram_latency_v2/` is a separate harness that measures RAM
and timing only (no accuracy claims). See its own `README.md`. Run
`python verify_results.py results` there before exporting any results.

---

## Workflow and policies

```
Monday     – Review weekly tasks in tasks/week-XX.md
Tue–Thu    – Implementation / experiments
Friday     – Document progress in docs/weekly-progress.md
Friday     – Open weekly Pull Request from your branch → dev
```

| Branch | Purpose |
|---|---|
| `main` | Stable, supervisor-reviewed code only |
| `dev` | Integration branch — merge weekly PRs here |
| `<your-name>-week-XX` | Your working branch for each week |

Never push directly to `main`. One PR per week targeting `dev`, titled
`[Week XX] Brief description`, referencing the weekly task file. A supervisor
or co-student must review before merging.

## Supervisor note

This repository is managed by **CNIT/PNTLab Pisa, TECIP, Scuola Superiore
Sant'Anna**. Contact your supervisor before making architectural changes. All
code must be original or properly attributed. Do **not** commit API keys,
passwords, or large datasets — see `.gitignore`.
