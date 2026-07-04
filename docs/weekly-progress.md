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
