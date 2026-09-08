<!--
BAS1 (Issue 3) Task 3 -- Quickstart section. Path confirmed this session
against the real environment (PowerShell prompt showed
...\09-edge-iot-security-monitoring\experiments\Current model> as the
working directory), matching the master context doc's stated
"Repository working directory is experiments/Current model/". This
resolves the earlier open item about src/ vs experiments/Current model/
layout -- requirements.txt and this Quickstart both belong under
experiments/Current model/.

RESOLVED this session by direct inspection of the real data_loader.py:
  - DATASET_PATH is computed as three dirname() calls up from
    data_loader.py's own location (which the module docstring states is
    experiments/Current model/data_loader.py) + "datasets" /
    "Edge-IIoTset dataset" / "Selected dataset for ML and DL" /
    "DNN-EdgeIIoT-dataset.csv" -- i.e. the RAW dataset file, not the
    preprocessed_DNN.csv the dataset's own published Kaggle recipe
    produces. data_loader.py does its own feature engineering (see
    engineer_text_features(), engineer_method_features(),
    engineer_ua_and_uri_keyword_features()) directly on the raw CSV, and
    that engineering deliberately diverges from the published recipe's
    DROP_COLS list (several columns the published recipe drops outright
    -- http.file_data, http.request.full_uri, http.request.uri.query,
    tcp.payload, http.referer -- are instead engineered into numeric
    features here). Do NOT run the published Kaggle notebook's
    preprocessing steps (dummy encoding, column drops) before placing
    the file -- place the RAW downloaded CSV exactly as Kaggle delivers
    it.
  - Real MD5 confirmed against the actual file at that exact path:
    EE17D434BADE0B980AB0D59764A67DFD (see Step 4 below).
  - First run also builds two local caches (not in this dataset
    placement, safe to .gitignore): a full preprocessed-dataframe cache
    at datasets/dnn_preprocessed_cache.npz, and per-(model_type, seed)
    train/val/test split caches under experiments/Current model/splits/.
    Subsequent runs load from these caches rather than re-processing the
    raw CSV every time.

STILL OPEN:
  - An actual end-to-end run of steps 1-5 from a fresh clone/venv, with
    real terminal output pasted in place of the placeholder at the
    bottom.
-->

## Quickstart

1. **Clone the repository**
   ```bash
   git clone <REPO_URL>
   cd 09-edge-iot-security-monitoring/experiments/Current\ model
   ```
   (Note the literal space in "Current model" -- quote or escape it in
   every shell command below.)

2. **Create and activate a virtual environment (Python 3.11)**

   Windows (PowerShell):
   ```powershell
   python -m venv .venv311
   .\.venv311\Scripts\Activate.ps1
   ```

   macOS/Linux:
   ```bash
   python3.11 -m venv .venv311
   source .venv311/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download and verify the dataset**

   Source: Edge-IIoTset (Ferrag et al., 2022), via Kaggle. Download the
   RAW CSV only -- do NOT run the published Kaggle notebook's own
   preprocessing steps (column drops / dummy encoding); this repo's
   `data_loader.py` does its own feature engineering directly on the
   raw file and expects it unmodified.
   ```bash
   pip install -q kaggle
   kaggle datasets download -d mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot \
     -f "Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"
   unzip DNN-EdgeIIoT-dataset.csv.zip
   rm DNN-EdgeIIoT-dataset.csv.zip
   ```
   Place `DNN-EdgeIIoT-dataset.csv` at (relative to the repo root,
   `09-edge-iot-security-monitoring/`):
   ```
   datasets/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv
   ```
   (This is computed automatically by `data_loader.py`'s `DATASET_PATH`
   as three directories up from `experiments/Current model/` -- get the
   folder names and nesting exactly right, including the literal spaces.)

   Verify the file matches the exact copy used to produce this
   project's reported results:

   Windows (PowerShell):
   ```powershell
   Get-FileHash -Algorithm MD5 "DNN-EdgeIIoT-dataset.csv"
   ```
   macOS/Linux:
   ```bash
   md5sum DNN-EdgeIIoT-dataset.csv
   ```
   Expected MD5: `EE17D434BADE0B980AB0D59764A67DFD`

   **Licence note:** the dataset is not CC BY-NC-SA. Per Ferrag et al.'s
   own distribution terms: free use for academic research is granted in
   perpetuity; commercial use requires asking the lead author (Dr.
   Mohamed Amine Ferrag) first. Cite:
   Ferrag, M.A., Friha, O., Hamouda, D., Maglaras, L., Janicke, H.,
   "Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset
   of IoT and IIoT Applications for Centralized and Federated
   Learning," TechRxiv, 2022. DOI: 10.36227/techrxiv.18857336.v1

5. **Run the baseline FedAvg config**
   ```bash
   python main.py network --ablation-mode baseline
   ```
   You should see Round 1 begin printing within a few seconds (client
   data-split summary, then `[ROUND 1/25]`).

<!-- PASTE REAL TERMINAL OUTPUT HERE from an actual run of steps 1-5,
     once the dataset path/MD5 placeholders above are filled in. -->
