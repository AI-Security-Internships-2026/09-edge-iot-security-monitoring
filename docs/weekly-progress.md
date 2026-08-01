# Weekly Progress Log: Edge IoT Security Monitoring with Federated Learning

**Student:** Muhammad Zarawar Khan
**GitHub username:** Zarawar5555

## Week 1

**Branch:** `zarawar-week-01`
**PR link:**(https://github.com/AI-Security-Internships-2026/09-edge-iot-security-monitoring/pull/1)

### Completed this week
- Read README and proposal
- Set up local environment (Python venv, dependencies)
- created branch with name 'zarawar-week-01'
- Submitted the application for the GitHub Student Developer Pack to integrate Copilot Pro with VS Code
- Ran `src/main.py` successfully
- Wrote personal introduction (below)
- Identified and understood 3 related papers /  articles

### Personal Introduction
Hi, I'm Muhammad Zarawar Khan. I’m an incoming AI Security Intern with a strong interest in IOT and AI, so having been assigned to do research in edge AI is perfect for me. I have a solid foundation in Python and have worked with machine learning libraries like PyTorch, alongside a good understanding of core network security principles. Throughout my time here at CNIT/PNTLab Pisa, I'm looking forward to learn about FL and expand my expertise in this field. I'm very grateful for this opportunity and hope that I can provide value.

### Problems / Blockers
None faced.

### Next week plan
- Read the 5 papers identified this week
- Complete `docs/proposal.md` draft
- Set up dataset download / preprocessing pipeline

## Week 2

**Branch:** `zarawar-week-02` 
**PR link:** (https://github.com/AI-Security-Internships-2026/09-edge-iot-security-monitoring/pull/2)

### Completed this week

- Expanded literature review to 10 papers covering FL baselines, Byzantine robustness, model poisoning, backdoor attacks, and differential privacy (`docs/literature-review.md`)
- Completed full draft of `docs/proposal.md` including problem statement, three research questions, methodology, evaluation metrics, and risk table
- Downloaded Edge-IIoTset dataset and documented source, licence, and preprocessing steps in `datasets/README.md`
- Implemented very simple Flower v1.31 server app using FedAvg strategy (`src/server_app.py`)

### Problems / Blockers

- Unsure whether the three research questions are appropriately scoped for the internship timeline — RQ2 and RQ3 in particular involve a parameter sweep and composed defence evaluation that may be ambitious for the remaining weeks. Would appreciate supervisor feedback on whether to narrow the scope or keep as stretch goals.

### Next week plan

- Implement `task.py` (MLP model, Edge-IIoTset data loading) and `client_app.py` (Flower ClientApp with local training and evaluation)
- Run first end-to-end FL training loop and record accuracy against the 92.49% Rashid et al. benchmark
<<<<<<< HEAD
- Begin implementing Krum aggregation as the first defence baseline


## Week 3

**Branch:** `zarawar-week-03`
**PR link:** (https://github.com/AI-Security-Internships-2026/09-edge-iot-security-monitoring/pull/3)

### Completed this week

- Built Docker containers for the full Flower stack (superlink, supernodes, superexecs) and wrote a `docker-compose.yaml` to run the entire system locally with a single command
- Wrote and deployed the Kubernetes manifests for the same stack (`k8s/superlink.yaml`, `k8s/supernodes.yaml`, `k8s/superexec.yaml`) to a minikube cluster
- Configured the superlink, supernodes, and superexec (serverapp + clientapp) services end-to-end so they connect and communicate correctly across both the Docker Compose and Kubernetes deployments
- Scaled the federated learning setup from 2 clients to 10, with the dataset split into 10 segments so each client mimics a single IoT sensor/device
- Updated all the core code files to support the 10-client setup: `pyproject.toml` (num-partitions), `src/server_app.py` (min_fit/evaluate/available clients), `src/task.py` (model definition and training), and `src/client_app.py` (per-client data loading)
- Implemented a working 10-client FL training loop (`src/main.py`), confirming the global model's loss decreases across rounds with FedAvg aggregation

### Problems / Blockers

- Flower 1.31.0's built-in `flwr run` simulation mode hit a Windows-specific bug that prevents it from creating its local SQLite run-tracking database, and its Ray-based simulation backend doesn't support Python 3.14 (the version installed here). Worked around both by writing a manual FL training loop instead of Flower's native `ServerApp`/`ClientApp` simulation runner — functional for now, but may need to be reconciled with Flower's native orchestration before the final write-up.

### Next week plan

- Plug in the real Edge-IIoTset dataset (replacing the random placeholder data in `client_app.py`) and partition it across the 10 simulated clients
- Re-run the 10-client FL loop with real data and record accuracy against the 92.49% Rashid et al. benchmark
<<<<<<< Updated upstream
=======
>>>>>>> origin/dev
- Begin implementing Krum aggregation as the first defence baseline
=======
- Begin implementing Krum aggregation as the first defence baseline

## Week 4

**Branch:** `zarawar-week-04`
**PR link:** (https://github.com/AI-Security-Internships-2026/09-edge-iot-security-monitoring/pull/4)

### Completed this week

- Integrated the full Edge-IIoTset DNN feature set (1.2GB CSV) into the FL pipeline with `.npz` disk caching so the CSV is only parsed once per machine
- Discovered and fixed a critical silent preprocessing bug: `VarianceThreshold(1e-6)` was dropping all HTTP and UDP features (29 of 52 columns) because they have low variance across the full dataset — destroying the only signal distinguishing Backdoor, XSS, Password, SQL_injection, and Uploading, and causing DDoS_UDP to sit at F1=0.0000 permanently for 20+ rounds
- Applied the fix per-model rather than globally: the network model retains `VarianceThreshold` (reducing to ~40 features) while the application model skips it entirely, preserving all HTTP features (`http.request.uri.query`, `http.file_data`, `http.content_length`, etc.) that are essential for application-layer attack separation
- Applied a VARS-FL-style dataset cap, reducing DDoS_TCP from 72.8% of total rows down to 18% to remove a lab-generation artefact; all other classes left at natural counts
- Updated class weights in both `build_criterion_network()` and `build_criterion_application()` using real per-class sample counts from the preprocessed cache, with manual overrides for feature-confused classes (Backdoor ×5, XSS ×4, Password ×3, Fingerprinting ×3)
- Split the 15-class problem into two specialised 8-class models controlled by a single CLI flag (`python src/main.py network` / `python src/main.py application`): a network-layer model covering volumetric and protocol attacks, and an application-layer model covering stealth and payload attacks, each with its own Focal Loss criterion, feature set, class weights, and output files
- Implemented FedProx (`μ=0.01`) as the local training algorithm
- Added gradient clipping (`max_norm=1.0`) and switched the LR scheduler to a gentler decay (`StepLR step_size=3, gamma=0.95`) to prevent the training collapses observed in earlier runs
- Established pre-defence baselines for both models across 25 rounds — locking in per-class F1 numbers that serve as the comparison point for all subsequent Krum, FLDetector, and DP-SGD experiments
- Expanded literature review from 10 to 12 papers — replaced McMahan FedAvg with FedProx (Li et al., NeurIPS 2020) and added VARS-FL (Lakas & Ferrag, 2026) and Alsaleh et al. (Sensors, 2025), both of which benchmark FL on the same 15-class Edge-IIoTset dataset, I consulted both these papers for guidance and so it was necessary to add them

### Problems / Blockers

- The application-layer model initially produced diverging loss (rounds 18–22) when `PROX_MU=0.1` was used — the proximal term was anchoring clients too strongly to a biased global model. Resolved by reducing to `PROX_MU=0.01`, which stabilised training.

### Next week plan

- Begin implementing Multi-Krum (`m=7`) by writing `src/defences/krum.py` and uncommenting the existing `DEFENCE HOOK` in `main.py`
- Inject simulated Byzantine clients (2 of 10 sending scaled-negative updates) to give Krum something to actually defend against and record detection rate vs. accuracy tradeoff
- Begin FLDetector implementation using per-client update history tracking
>>>>>>> Stashed changes


## Week 5

**Branch:** `zarawar-week-05`
**PR link:** (https://github.com/AI-Security-Internships-2026/09-edge-iot-security-monitoring/pull/5)

### Completed this week

- Implemented a complete Multi-Krum Byzantine-robust aggregation engine (src/defences/krum.py) that evaluates pairwise squared Euclidean distance metrics to safely select m = 6 normal client updates per training round (n = 10, f = 2)
-Built an inline numerical overflow validation guard within the Multi-Krum engine to automatically isolate high-scale sign-flip anomalies by intercepting NaN/Inf values before distance calculation and assigning them an infinite distance score
- Executed comprehensive Round 25 benchmark experiments across all four experimental conditions, proving that Multi-Krum completely prevents catastrophic network model collapse (recovering F1-Macro from 0.012 to 0.857) and partially restores application model performance under attack (recovering F1-Macro from 0.471 to 0.570)
-Deployed a parallel localized penetration testing lab environment using Oracle VirtualBox, bridging a Kali Linux attacker node and a vulnerable Metasploitable 2 target appliance inside a dedicated private Host-Only network configuration (192.168.56.0/24).
-Did a deep dive into all 15 attack types being used in the dataset and demosntrated how a potential attacker might use them to take advantage of our FL system on metasploitable using Kali Linux.

### Problems / Blockers

- none faced.

### Next week plan

- Going to start focsuing more on the privacy part of FL.
- Implement either HE or DP(or anything similar) after choosing what's better suited.


## Week 6

**Branch:** `zarawar-week-6`
**PR link:** (https://github.com/AI-Security-Internships-2026/09-edge-iot-security-monitoring/pull/6)

### Completed this week

**Privacy stack architecture — design and critique**
- Architected a three-layer quantum-safe privacy stack for the multi-user shared IoT gateway scenario, addressing both intra-client privacy (protecting concurrent edge users on the same gateway from each other) and inter-client privacy (server blindness to raw parameters)
- Identified and corrected three flaws in an earlier architecture draft:
  - Trimmed Mean under homomorphic encryption is computationally infeasible (requires 15-30 multiplicative depth levels per comparison across a 50,000-parameter model — days of CPU time). Replaced with homomorphic FedAvg (ciphertext addition + scalar multiplication only), moving outlier protection to the ZKP/commitment layer
  - The ZKP wasn't bound to the ciphertext — a Byzantine client could prove a clean gradient while transmitting a poisoned encrypted payload. Fixed with an HMAC-SHA256 Pedersen-style commitment scheme binding the proof to the actual encrypted data
  - Proving DP noise was correctly sampled is infeasible in a standard ZKP (gigabyte-sized proofs for a 50k-parameter network). Removed from ZKP scope; DP guarantee now rests on correct implementation rather than cryptographic proof

**Layer 1 — Local Differential Privacy (`src/privacy/dp_training.py`)**
- Implemented Opacus-backed DP-SGD wrapper (`PrivacyEngine.make_private_with_epsilon()`), clipping per-sample gradients to `max_grad_norm=1.0` before adding calibrated Gaussian noise
- Replaced `BatchNorm1d` with `GroupNorm` via `ModuleValidator.fix(model)`, since BatchNorm's cross-sample dependencies are incompatible with per-sample DP-SGD tracking

**Layer 2 — Cryptographic commitment (`src/privacy/commitment.py`)**
- Implemented HMAC-SHA256 commitment scheme: `C = Hash(δ || salt)` over the noise-perturbed local update
- Implemented server-side `verify_norm_proof()` checking the update respects a dimensionally-calibrated safety ceiling, and `verify_commitment_opening()` as an audit hook confirming updates match their commitment — mitigates gradient-bomb/parameter-overflow attacks
- Quantum-safe by construction: SHA-256 preimage resistance means Grover's algorithm only halves effective key security

**Layer 3 — Homomorphic Encryption (`src/privacy/he_aggregation.py`, `he_local.py`)**
- Fixed a critical cryptographic bug: clients were generating independent HE keypairs, which breaks the mathematical validity of homomorphic addition across clients. Corrected so the server generates and distributes a single shared public context, keeping the decryption key private
- Implemented partial HE: only the classifier head (~4,680 of 80,074 parameters, 5.8%) is CKKS-encrypted; preceding feature-extraction layers are sent as DP-noised plaintext, cutting server HE workload ~94%

**Resource-constrained optimization / Docker edge emulation**
- Built containerized emulation (`fl_server`, `fl_client_0`, `fl_client_1`) on an isolated Docker bridge network, replicating factory gateway hardware limits
- Diagnosed initial silent memory crashes (`RuntimeError: Training subprocess failed`, kernel SIGKILL) under a 200MB cgroup limit and resolved via five optimization layers:
  - Subprocess training isolation (`train_worker.py`) — PyTorch training runs in a child process that exits after training, forcing full OS memory reclamation; parent client process holds flat at 61-62MB
  - Dependency stripping — moved model definitions to a dependency-free `model_defs.py` after finding `task.py` imports were pulling in pandas/scikit-learn/scipy (80-150MB) inside the training subprocess unnecessarily; also throttled BLAS threading to a single core (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS=1`)
  - CKKS context pruning — removed unused Galois/relinearization keys (pipeline only uses ciphertext-ciphertext addition and ciphertext-scalar multiplication), shortened the coefficient-modulus chain from depth-2/4-level to depth-1/3-level, reduced `poly_modulus_degree` 8192→4096
  - Binary wire format (`wire_format.py`) — replaced JSON float-list serialization (6-8x memory blowup from unpacking numpy arrays into Python objects) with base64-encoded raw `float32` bytes, cutting transmission overhead 33%
  - I/O storage calibration — removed redundant per-round dataset re-serialization; clients now write their training partition to disk once instead of every round
- Established the edge RAM floor: CKKS at n=4096 on this model requires roughly 210-245MB for cryptographic steps alone; combined with the PyTorch runtime, the effective container floor sits around 340-350MB

**Production run validation and debugging**)
- Found and fixed a stray `×0.01` scaling bug in `zkp.py`'s norm-threshold formula that made validation ~100x too strict, causing the server to reject every legitimate update; corrected with a proper `NOISE_NORM_SAFETY_FACTOR = 1.15` applied to the theoretically correct `σ × √n_params` formula
- Achieved a clean end-to-end 3-round production run: 100% update retention (2/2 clients per round, HTTP 200), server-side ZKP re-verification passing independently
- Measured privacy-utility tradeoff directly: moving DP epsilon from 3.0 to 15.0 brought the noise-to-signal ratio down from ~450:1 to ~91:1, but this still manifests as non-monotonic loss between rounds (Client 0: round 1 ends at loss 1.4365, round 2 begins at loss 1.6583 before recovering to 1.4666) — confirms a real limitation of high-dimensional local DP at this model size
- Profiled execution time: local training dominates at 40-60s/round (~60s round-1 CPU warmup, ~40s steady state); combined DP + commitment + partial CKKS encryption overhead is under 0.1s/round; server-side homomorphic decrypt+merge runs at 0.02s/round — cryptographic latency is not a deployment bottleneck
-uploaded all results of tests which were previously done.

### Problems / Blockers

- The privacy-utility tradeoff is not fully resolved: even at ε=15.0 (noise-to-signal ~91:1), local DP noise is large enough to cause visible non-monotonic loss between rounds rather than steady convergence. Would appreciate supervisor input on whether to push epsilon further, move to central DP at the server aggregate, or restrict noise to the classifier head only, since this directly affects what accuracy numbers are achievable for the write-up.
- The model cannot run to it's full potential on the 200mb limit which was given to me, after running tests the ideal ram limit would be 400 mb.

### Next week plans


---

## Week 7

**Branch:** `zarawar-week-7`
**PR link:**

### Completed this week

- **Pure-HE vs. Pure-DP ablation, finalized.** Built `docker-compose.ablation.yml` (independent `he_only_*`/`dp_only_*` groups, separate results volumes, distinct ports, seperate client memory ceiling). Fixed two validity problems: (1) unfair HE comparison — pure HE was only encrypting the 6% classifier head vs. DP's 100% coverage, fixed via a new `HE_FULL_COVERAGE` env var; (2) every prior run had used synthetic placeholder data — fixed via `build_partitions.py` (offline preprocessing into per-client `.npz` partitions), made `load_data()` hard-error instead of silently falling back to synthetic data, and fixed `NUM_FEATURES` being hardcoded to 40 instead of the real post-`VarianceThreshold` count (35)
- Built a `RamSampler` (background-thread RSS polling) for continuous peak/average RAM tracking per stage/round
- **`pure_dp` run completed on real data** (23k/38k-row partitions, 35 features); confirmed DP had been upgraded to real Opacus DP-SGD, a genuine improvement over the old post-hoc noise mechanism. Found the epsilon composition problem — flat ε≈2.99 every round, meaning no cross-round privacy accountant. Attempted composition-tracked DP and hit a utility wall (loss exploded 10x/round at composed ε=3.0); **decision made ("Option A"):** keep per-round DP-SGD, report the caveat honestly, defer a proper ε-sweep to RQ3
- **`pure_he` results reviewed**: confirmed the full-coverage fix worked (CKKS timing scaled ~17x with param count).
- **Final reported RAM/latency numbers:** Pure HE ≈0.2s latency, ≥400MB RAM; Pure DP-SGD (Opacus) far more taxing — ≈600MB RAM, ≈300s/client to train
- Established the HE-vs-DP framing for the write-up: complementary, not competing (HE protects the pipe, DP protects the output); recommended a combined deployment (whole-model DP-SGD + classifier-head-only partial HE) as a future third experiment


- **Defence-folder consolidation completed:** confirmed `docker_fl/` is a full parallel project (own datasets, own results), not a duplicate. Diffed `zkp.py`/`local_dp.py` between `src/` and `docker_fl/` — found real divergences. Merged into one canonical `src/defences/zkp.py`/`local_dp.py`; confirmed `docker-compose.yml` mounts `../src/defences` read-only into all four containers, verified live via `inspect.getsourcefile()`. `krum.py`/`byzantine.py` had no conflict. `defences/homomorphic.py` confirmed dead code — real pure-HE implementation lives in `docker_fl/he_aggregation.py`, porting deferred
- **`main.py` unification completed:** merged the two previously separate `main.py` files (DP/ZKP/HE version and Krum/Byzantine version) into one file with three aggregation branches — HE, Multi-Krum, or plain FedAvg.
- Flagged a terminology note for the write-up: the DP/ZKP/HE `main.py`'s "ZKP" is a plain norm-threshold check, structurally different from the HMAC-commitment `defences/zkp.py`
- **Issue resolution:** deleted the unused scaffold `src/server_app.py` (training has been driven by `app/main.py`/`main.py` on real Edge-IIoTset data since Week 4); closed Issue 7 as a consequence. Clarified metrics separation — `docker_fl/results` (now `RESULTS AND MANIFESTS/Docker test for RAM and Latency/`) intentionally logs hardware metrics on placeholder data (purpose is resource-constraint emulation, not accuracy), while root `results/` holds full real-data benchmarks. Closed Issue 9 (defence-folder duplication); Issue 10 (Krum non-IID exclusion) marked as actively being worked on


- **Repository reorganization completed** (kept in a separate commit from logic changes, `dc7d4af`, for a clean diff): new structure — `experiments/Current tests/` (was `src/`), `experiments/Docker tests for RAM and Latency/` (was `docker_fl/`), consolidated `RESULTS AND MANIFESTS/`. Removed confirmed-safe dead weight: `k8s/` (unused ~4 weeks), `docker-compose.yml`/`superexec.Dockerfile`/`pyproject.toml`, stale root `requirements.txt`/`experiment_config_network.json`/`results_network.csv`, stray accidental terminal-capture junk files, and the outdated `tasks/week-01.md`


- Parameterized `model_defs.py` with an opt-in `dp_safe` flag (BatchNorm1d→GroupNorm, LSTM→DPLSTM) instead of an unconditional swap, so non-DP runs stay byte-identical to existing baselines/checkpoints
- Added a `BYZANTINE_HEAD_ONLY` flag — under `USE_HE=True`, uses `classifier_head_flip_attack()` instead of `sign_flip_attack()` since a full sign-flip at scale=5.0 would trip ZKP's norm gate under HE
- Wrote `scripts/build_manifest.py` (rewritten once after an initial wrong-schema assumption): discovers all runs via `experiment_config_*.json` + `results_*.csv` pairs, computes per-experiment summary stats, sanitizes stray "Flower" references, outputs `manifest.json` + `manifest_summary.csv`
- Confirmed 38 features is the real, reproducible count (not the documented 35) via `check_features.py` and a full column audit — adopted as ground truth for the main experiment


- **Ran and analyzed the first full 25-round ε=15 run** Experiment 1 (DP-SGD ε=15 + Multi-Krum + Byzantine attack, 25 rounds): Krum achieved 100% Byzantine detection every round on both models, confirming DP-SGD and Multi-Krum are compatible at this privacy level. Network held up well (90.8% acc, only 1.5% relative drop from its clean baseline), but application collapsed (47.0% acc, a 38% relative drop). The gap traces almost entirely to XSS, which scored near-zero F1 in 20 of 25 rounds — likely driven by Krum's looser client-exclusion margin (m=6) discarding legitimate non-IID clients that happened to hold XSS signal, on top of a task that's already harder at baseline. Next steps: rerun application under the tightened m=7 Krum config, finish the in-progress oracle-Krum comparison to isolate the exclusion effect, and check per-client XSS sample counts.


- Built a new matched-resource Full-HE vs. Partial-HE ablation (distinct from Week 7's deliberately-varied `pure_he`/`pure_dp` runs); fixed a `docker-compose.yml` header (`version:`/`services:`) lost in the reorg, and — again — `server.py` hardcoding `num_features=40`, fixed to read from environment
- **Received and analyzed the `he_full`/`he_partial` results:** partial HE (classifier head only, 3.6% of params) is ~17–19x faster at client-side encryption and ~14x faster at server-side aggregation than full-model HE, with no measurable difference in peak RAM (within ~1MB — training memory dominates the round's peak regardless of HE scope, at this ~130K-param model size). **Caveat surfaced:** this ablation ran on 35 features / 100k-row-subsampled partitions (from `build_partitions.py --max-rows 100000`), not the main experiment's 38-feature full corpus — internally valid for the full-vs-partial comparison, but not yet directly comparable to the main Krum/DP-SGD results

### Problems / Blockers

-Caught and fixed a critical bug during the 'main.py' merge: ZKP-rejected clients get `continue`d out before aggregation, compacting `accepted_params`, so Krum's `selected_indices` were being compared directly against `BYZANTINE_CLIENTS` (original IDs) — fixed by tracking `accepted_client_indices` in parallel
- Found and fixed three more bugs in the same pass: `get_model()` missing `dp_safe` (fixed via a unified `DP_SAFE = USE_DP` flag across all call sites); the Opacus wrapper never unwrapped before param extraction (fixed via `real_model = model._module if hasattr(model, "_module") else model`); achieved epsilon computed then immediately discarded before logging, leaving `dp_epsilon_spent` always `N/A` (fixed by removing the overwrite). Also fixed an output file-naming collision (results/checkpoints not tagged by DP epsilon condition, risking silent overwrites) via a rename-after-each-run workflow

- Worked through Docker/Windows issues: wrong dataset path, cmd.exe vs. PowerShell mismatches, and a genuine `Errno 12` OOM traced to WSL2's VM having only 3.5GB total memory — fixed via a `.wslconfig` bump
- Epsilon composition gap confirmed structural (flat ε≈2.99/round) — accepted as a documented caveat rather than fixed this week

- `defences/homomorphic.py` still dead code; real HE implementation still needs porting from `docker_fl/he_aggregation.py`

- Also caught and fixed the MEAN-row `krum_detected_byzantine` truthy-collapse bug (`1 if krum_detected else 0` couldn't distinguish 0.5 partial detection from 1.0 full detection) via a new `is_mean` flag in `append_log_row()`; investigated and ruled out a CSV header mismatch (APP_NAMES columns on a network run) as a code bug — traced to a stale/environment file issue, resolved by regenerating headers from the correct directory

- Diagnosed severe network-model slowness (9 hrs / 10 rounds) to network having ~3x application's data volume; applied three fixes on top of an already-revised `main.py`: Parallelization: client training logic moved verbatim into a new top-level function _train_one_client(), The round loop now submits all 10 clients to a 4-worker ProcessPoolExecutor, then processes results in original client order, Thread tuning: main process uses all detected cores (torch.set_num_threads(_CPU_COUNT)); each worker process is capped at cores // 4 to avoid 4 processes each fighting for every core simultaneously. DP batch size: 256 → 512.

- `KRUM_M` still isn't reaching `multi_krum()` — Krum still discards 4 clients instead of the intended 3; blocked on reviewing `krum.py`'s current source to fix the pass-through
- Fix `KRUM_M` propagation into `multi_krum()`

- Added manifests to ensure reproducability.


### Next week plan

- Complete experiment 1 after fixing krum to properly discard 3 clients instead of 4, increase speed of training model.
- Build the combined DP-SGD + partial-HE experiment (Experiment 2)
---