**Student:** _Zarawar_
**Supervisor:** _DR. Rana Abubakar_
**Start date:** _8th June_
**Expected end date:** _8th August_

---

## 1. Background

Design a federated-learning framework for IoT edge nodes that collaboratively trains an intrusion detection model without transmitting raw sensor data to a central server.

This project is carried out within the AI Security research agenda of CNIT/PNTLab Pisa (TECIP, Scuola Superiore Sant'Anna).

---

## 2. Problem Statement

IoT deployments are increasingly targeted by sophisticated cyberattacks, yet existing intrusion detection systems require raw traffic data to be sent to a central server, creating unacceptable privacy risks and bandwidth costs for industrial operators. Federated Learning solves the data-locality problem but opens a new attack surface: compromised edge nodes can poison the global model, inject hidden backdoors, or leak training data through gradient inversion. No existing solution combines Byzantine-robust aggregation, backdoor defence, and differential privacy into a single hardened system, and no prior work quantifies how these defences interact when composed together or identifies the minimum viable configuration that satisfies formal privacy, detection, and accuracy constraints simultaneously. This project builds and evaluates exactly that system.

---

## 3. Research Questions

1. _RQ1: Under device-heterogeneous non-IID data distributions on Edge-IIoTset, what is the per-attack-class F1 and recall gap between the proposed FL-IDS and a centralised supervised baseline, and do minority attack categories incur a disproportionately larger federated performance penalty than majority classes?_
2. _RQ2: When Krum, FLDetector, and DP-SGD are composed into a single unified pipeline, does the combined system achieve strictly lower attack success rate and higher poisoning detection rate than any individual defence in isolation, and at what privacy budget ε does DP noise begin to measurably degrade FLDetector's client-level anomaly detection?_
3. _RQ3: What is the minimum viable defence configuration, in terms of privacy budget ε, client participation rate, and aggregation rounds, that simultaneously satisfies a formal DP guarantee, a poisoning detection rate above a defined threshold, and per-class F1 and recall within an acceptable margin of the undefended FedAvg baseline?_

---

## 4. Proposed Methodology

### 4.1 Data Collection / Dataset

The primary dataset is **Edge-IIoTset** (Ferrag et al., 2022), a comprehensive IoT/IIoT cybersecurity dataset generated from a 7-layer real-device testbed covering 14 labelled attack categories including DoS, DDoS, MITM, injection, and scanning attacks.

- **Source:** https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot
- **Licence:** Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
- **Version / date downloaded:** 2026-06-15
- **Size:** 1.6 GB compressed ZIP / 2,219,201 rows with 61 initial features (Targeted DNN Subset)
- **Format:** CSV + PCAP
- **Download command:**
  ```bash
  pip install -q kaggle
  kaggle datasets download -d mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot -f "Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"
  ```
- **Preprocessing steps:**
  1. Ingest the specialised deep learning dataset (DNN-EdgeIIoT-dataset.csv) comprising 1,638 mixed raw attributes
  2. Strip environmental host identifiers and network markers to prevent overfitting: `frame.time`, `ip.src_host`, `ip.dst_host`, `arp.src.proto_ipv4`, `arp.dst.proto_ipv4`, `http.file_data`, `http.request.full_uri`, `icmp.transmit_timestamp`, `http.request.uri.query`, `tcp.options`, `tcp.payload`, `tcp.srcport`, `tcp.dstport`, `udp.port`, and `mqtt.msg`
  3. Execute row sanitisation using `dropna(axis=0, how='any')` and drop duplicated sequences using `drop_duplicates(keep='first')`
  4. Perform text dummy feature mapping on residual string parameters: `http.request.method`, `http.referer`, `http.request.version`, `dns.qry.name.len`, `mqtt.conack.flags`, `mqtt.protoname`, and `mqtt.topic`
  5. Distribute structural matrix rows into non-overlapping client fragments using a Dirichlet distribution (α = 0.5) to simulate device-heterogeneous Non-IID topologies inside the Flower pipeline
- **Train / Val / Test split:** 80% Training / 10% Validation / 10% Testing
- **Mandatory academic citation:** Mohamed Amine Ferrag, Othmane Friha, Djallel Hamouda, Leandros Maglaras, Helge Janicke, "Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning", TechRxiv, 2022, DOI: 10.36227/techrxiv.18857336.v1

The dataset files are stored strictly on local scratch space and are systematically ignored via `.gitignore` to comply with repository space restrictions. Full documentation is in `datasets/README.md`.

### 4.2 Approach

The core contribution of this project is not any single defence technique, since each exists in prior work, but their composition into one unified and configurable system evaluated against the full attack surface simultaneously. Results are analysed at the per-class level to expose minority-category recall degradation that aggregate accuracy metrics would otherwise hide.

Phase 1 trains a supervised PyTorch MLP on the full Edge-IIoTset training split in a centralised setting, recording per-attack-class F1 and recall across all 14 categories to establish the performance ceiling (RQ1). Phase 2 replicates training using Flower with data partitioned across simulated edge clients via Dirichlet Non-IID sharding (α = 0.5) and standard FedAvg aggregation, computing the per-class F1 and recall gap versus the centralised baseline with particular attention to minority attack categories (RQ1). Phase 3 introduces a configurable fraction of malicious clients injecting simultaneous model poisoning and backdoor attacks (Bagdasaryan et al. 2020 constrain-and-scale), then applies each defence in isolation and all three composed, recording attack success rate (ASR), poisoning detection rate (PDR), per-class F1, and per-class recall for each configuration, with ε varied across {3.0, 10.0, ∞} to identify the point at which DP noise degrades FLDetector detection (RQ2). Phase 4 sweeps ε ∈ {1.0, 3.0, 5.0, 10.0, ∞}, client participation rates ∈ {20%, 40%, 60%, 80%, 100%}, and aggregation rounds ∈ {10, 20, 50, 100} to find the minimum viable (ε, participation, rounds) triple satisfying formal DP, PDR threshold, and per-class F1 and recall constraints simultaneously (RQ3).

### 4.3 Evaluation Metrics

| Metric | Description | Relevant RQ |
|---|---|---|
| **Per-class Recall** | Fraction of true attacks in each category correctly detected. The most operationally critical IDS metric, especially for minority attack classes | RQ1, RQ2, RQ3 |
| **Per-class F1-score** | Harmonic mean of precision and recall for each of the 14 attack categories. Exposes class-level degradation hidden by aggregate accuracy | RQ1, RQ2, RQ3 |
| **Federated F1 / Recall Gap** | Per-class delta between centralised baseline and federated model. Quantifies the cost of federation per attack type | RQ1 |
| **Attack Success Rate (ASR)** | Fraction of backdoor-trigger inputs or poisoned samples misclassified as benign | RQ2 |
| **Poisoning Detection Rate (PDR)** | Fraction of malicious clients correctly identified and excluded before aggregation | RQ2 |
| **DP Degradation Threshold** | The ε value at which DP noise begins to measurably reduce FLDetector's PDR | RQ2 |
| **Overall Accuracy** | Aggregate multi-class accuracy, reported for comparison against Rashid et al. 92.49% benchmark | RQ1, RQ3 |
| **Privacy Budget (ε)** | Rényi DP epsilon consumed per training run. Measures formal privacy guarantee strength | RQ2, RQ3 |
| **Minimum Viable Config** | The (ε, participation rate, rounds) triple that jointly satisfies all three constraints in RQ3 | RQ3 |

### 4.4 Tooling

| Tool / Library | Purpose |
|---|---|
| Python 3.10+ | Primary implementation language |
| Flower (`flwr`) | FL client-server orchestration and round management |
| PyTorch (`torch`) | Local model definition, training, and weight update computation |
| TensorFlow Privacy | DP-SGD implementation and Rényi privacy accountant |
| Pandas / NumPy | Dataset loading, preprocessing, and Dirichlet shard partitioning |
| Scikit-learn | Per-class F1, recall, confusion matrix, and classification report |
| Matplotlib / Seaborn | Per-class result visualisation, recall heatmaps, and experiment plots |

All pinned versions are specified in `requirements.txt`.

---

## 5. Expected Outcome

The primary deliverable is a working prototype (`src/`) implementing a Flower-based federated intrusion detection system with three interchangeable aggregation strategies (FedAvg, Krum, FLDetector) and a configurable differential privacy layer, evaluated on Edge-IIoTset. The project will produce a quantitative answer to all three research questions: a per-class F1 and recall breakdown exposing which attack categories suffer most under Non-IID federation (RQ1), a rigorous comparison showing whether defence composition outperforms individual defences and identifying the ε threshold at which DP degrades detection (RQ2), and the minimum viable (ε, participation, rounds) configuration that satisfies formal privacy, detection, and per-class accuracy constraints simultaneously (RQ3). Full experiment results will be documented in `experiments/results/` and written up in a final technical report (`docs/final-report.md`) and presented to the CNIT/PNTLab Pisa research group.

---

## 6. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Scope too broad | High | Implement centralised baseline and FedAvg first (RQ1), then add defences (RQ2) and parameter sweep (RQ3) incrementally. RQ3 is treated as a stretch goal |
| Minority class recall too low to analyse | Medium | Apply class-weighted loss during local training. Oversample using SMOTE locally if insufficient samples exist in some shards |
| Backdoor attacks not fully solvable by composition | Medium | Measure and document the residual ASR as a finding rather than a failure |
| DP noise eliminates FLDetector sensitivity entirely | Medium | Expected at low ε. This is precisely what RQ2 is designed to quantify and a null result is still a valid finding |
| Non-IID partitioning causes non-convergence | Medium | Use Dirichlet (α = 0.5) as standard in literature and tune α as a secondary experiment if time allows |
| Compute resources insufficient for parameter sweep | Low | Flower simulation runs in-process on a single machine. Model kept lightweight (MLP) to fit laptop hardware |

---

_Last updated: 2026-June_