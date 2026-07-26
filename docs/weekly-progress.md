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

**Production run validation and debugging**
- Fixed two JSON serialization crashes in the client→server payload: raw `bytes` salt field, and `numpy.bool_` validation state (neither is JSON-serializable)
- Found and fixed a stray `×0.01` scaling bug in `zkp.py`'s norm-threshold formula that made validation ~100x too strict, causing the server to reject every legitimate update; corrected with a proper `NOISE_NORM_SAFETY_FACTOR = 1.15` applied to the theoretically correct `σ × √n_params` formula
- Achieved a clean end-to-end 3-round production run: 100% update retention (2/2 clients per round, HTTP 200), server-side ZKP re-verification passing independently
- Measured privacy-utility tradeoff directly: moving DP epsilon from 3.0 to 15.0 brought the noise-to-signal ratio down from ~450:1 to ~91:1, but this still manifests as non-monotonic loss between rounds (Client 0: round 1 ends at loss 1.4365, round 2 begins at loss 1.6583 before recovering to 1.4666) — confirms a real limitation of high-dimensional local DP at this model size
- Profiled execution time: local training dominates at 40-60s/round (~60s round-1 CPU warmup, ~40s steady state); combined DP + commitment + partial CKKS encryption overhead is under 0.1s/round; server-side homomorphic decrypt+merge runs at 0.02s/round — cryptographic latency is not a deployment bottleneck
-uploaded all results of tests which were previously done.

### Problems / Blockers

- The privacy-utility tradeoff is not fully resolved: even at ε=15.0 (noise-to-signal ~91:1), local DP noise is large enough to cause visible non-monotonic loss between rounds rather than steady convergence. Would appreciate supervisor input on whether to push epsilon further, move to central DP at the server aggregate, or restrict noise to the classifier head only, since this directly affects what accuracy numbers are achievable for the write-up.
- The model cannot run to it's full potential on the 200mb limit which was given to me, after running tests the ideal ram limit would be 400 mb.

### Next week plans
