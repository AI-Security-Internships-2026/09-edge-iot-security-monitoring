# Literature Review: Edge IoT Security Monitoring with Federated Learning
Student: Zarawar Khan
Updated: 2026-06-11

## Reference Table (Quick Overview)

| Title (short) | Authors | Year | Method | Dataset | Relevance |

| Edge-IIoTset: FL Mode | Ferrag et al. | 2022 | Centralized & Federated Deep Learning baselines | Edge-IIoTset | Establishes our target dataset parameters & edge evaluation criteria. |
| StatAvg: Flower Baseline | Bouzinis & Jovanović | 2024 | Statistical Averaging for Client Data Normalization | TON_IoT | Direct blueprint for building custom clients and servers using Flower. |
| Secure & Explainable FL | Bilal et al. | 2026 | Systematization of update integrity and poisoning | Edge-IIoTset / ToN_IoT | Informs our security threat models and design constraints. |
| Smart City FL-IDS | ICOSST Cohort | 2025 | Flower Orchestration using FedAvg | IoTID20 | Direct validation of our exact tech stack (Flower + PyTorch). |
| Adaptive FL via HADA | Gutti et al. | 2025 | Hybrid Adaptive-Weight Aggregation & DP | Edge-IIoTset / TabularIoT | Baseline targets for running 100 client node simulations. |
---

## Paper Summary Template

### Paper 1 — Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning
* **Field:** Federated Learning for Industrial IoT Security
* **Full title:** Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning
* **Authors:** Mohamed Amine Ferrag, Othmane Friha, Djallel Hamouda, Leandros Maglaras, Helge Janicke
* **Year:** 2022
* **Venue:** IEEE Access
* **URL / DOI:** https://doi.org/10.1109/ACCESS.2022.3165809
* **Method:** Proposes a multi-layer IoT/IIoT evaluation testbed to generate diverse security telemetry, establishing standard baseline machine learning and deep learning configurations for decentralized setups using collaborative clients.
* **Dataset:** Edge-IIoTset (Contains 14 specialized cyberattack profiles including DoS/DDoS, injection, and malware vectors across 10+ distinct physical IoT devices).
* **Key result:** Validated that deep neural networks deployed in localized federated architectures can achieve comparable detection accuracy to completely centralized models without exporting raw telemetry.
* **Limitation:** The paper focuses on uniform baseline partitions and does not address extreme real-world network edge constraints or severe data distribution skews (non-IID).
* **Relevance to our project:** This provides the core blueprint for selecting relevant network flow attributes (61 optimized correlations) to build a robust local intrusion detection classifier.

### Paper 2 — StatAvg: Mitigating Data Heterogeneity in Federated Learning for Intrusion Detection Systems
* **Field:** Decentralized Optimization & Feature Engineering
* **Full title:** Statistical Averaging (StatAvg) to Alleviate Non-Independently and Identically Distributed Features in FL-based IDS
* **Authors:** Pavlos Bouzinis, Andrej Jovanović, et al.
* **Year:** 2024
* **Venue:** Flower Official Baselines Archive / arXiv Reference
* **URL / DOI:** https://flower.ai/docs/baselines/statavg.html
* **Method:** Introduces an algorithmic extension where localized clients securely communicate general dataset statistical shapes to a centralized server before runtime loops begin, enabling globally uniform normalization transforms.
* **Dataset:** TON_IoT network and OS logs.
* **Key result:** Drastically stabilized global model optimization routines and accelerated neural network loss convergence under conditions with highly unbalanced attack signatures among distinct edge groups.
* **Limitation:** Relies on an added pre-training step that exposes global mean/variance attributes, creating slight informational metadata leaks.
* **Relevance to our project:** Directly demonstrates how to write execution strategies utilizing the `Flower` ecosystem framework (`flwr`) while cleanly targeting cybersecurity-focused datasets.

### Paper 3 — Secure and Explainable Federated Learning for IoT Intrusion Detection: A Comprehensive Survey
* **Field:** Federated Learning Framework Security & Threat Modeling
* **Full title:** Secure and Explainable Federated Learning for IoT Intrusion Detection: A Comprehensive Survey
* **Authors:** Muhammad Ahmad Bilal, Ihtesham Ul Islam, Muhammad Junaid Khan, Shibli Nisar, Maemoona Farooq, Hassan Khan
* **Year:** 2026
* **Venue:** IEEE Open Journal of the Communications Society
* **URL / DOI:** https://doi.org/10.1109/OJCOMS.2026.3681580
* **Method:** Systematizes the coupled design space of decentralized network anomaly detection, evaluating structural vulnerabilities introduced by client federation, including data/update poisoning, backdoor injection, and client participation skews.
* **Dataset:** Cross-evaluates standard IoT telemetry representations (including Edge-IIoTset, ToN_IoT, and CIC-IDS benchmarks).
* **Key result:** Identifies that while vanilla FL prevents raw telemetry exposure, model updates inherently leak statistical signatures. It unifies threat frameworks to map how secure aggregation impacts the reliability of distributed deep learning models.
* **Limitation:** Focuses on a structural taxonomic survey rather than presenting a brand-new proprietary filtering algorithm.
* **Relevance to our project:** Essential for setting up our project's "Risks and Mitigations" section, giving us a blueprint to protect our local PyTorch weights before sending them to the Flower server.

### Paper 4 — Federated Learning for Enhanced Intrusion Detection in Smart City Environments
* **Field:** Practical Flower Framework Orchestration for IoT Nodes
* **Full title:** Federated Learning for Enhanced Intrusion Detection in Smart City Environments
* **Authors:** Anonymous Research Cohort
* **Year:** 2025
* **Venue:** IEEE 18th International Conference on Open Source Systems and Technologies (ICOSST)
* **URL / DOI:** https://doi.org/10.1109/ICOSST.2024.10871154
* **Method:** Implements a completely decentralized, privacy-preserving NIDS using the **Flower framework** coupled with the standard **FedAvg (Federated Averaging)** algorithm to coordinate updates across distributed IoT node simulations.
* **Dataset:** IoTID20 Dataset.
* **Key result:** Achieved an aggregated global model classification accuracy of 98%, demonstrating that distributing the training workload down to edge clients drastically scales down energy consumption compared to centralized cloud processing.
* **Limitation:** The study relies heavily on well-behaved, stratified dataset splits and does not evaluate convergence speeds when network connections drop out.
* **Relevance to our project:** This is a direct practical validation of our target tech stack (Flower + FedAvg) applied to an IoT intrusion detection workflow, proving that our framework plan is highly viable.

### Paper 5 — Federated Learning for Distributed IoT Security: A Privacy-Preserving Approach to Intrusion Detection
* **Field:** Non-IID Optimization and Non-Linear Weight Aggregation
* **Full title:** Federated Learning for Distributed IoT Security: A Privacy-Preserving Approach to Intrusion Detection
* **Authors:** C. Gutti, et al.
* **Year:** 2025
* **Venue:** IEEE Access
* **URL / DOI:** https://doi.org/10.1109/ACCESS.2025.11095679
* **Method:** Introduces a novel weighting algorithm called Hybrid Adaptive-Weight Aggregation (HADA) that integrates SHAP-based feature selection with local Differential Privacy (DP) boundaries to combat data heterogeneity across edge clients.
* **Dataset:** CIC-BCCC-NRC TabularIoTAttack-2024 and **Edge-IIoTset** benchmarks.
* **Key result:** Simulating 100 ARM-class edge devices, the approach sustained a high detection accuracy of 85–89% under normal states, outperforming traditional FedAvg approaches by up to 22 percentage points when subjected to adversarial label-flipping attacks.
* **Limitation:** The added overhead of computing local feature importances increases the processing latency on low-power, constrained hardware components.
* **Relevance to our project:** Directly relates to our goal of evaluating models on **Edge-IIoTset**, giving us a clear benchmark accuracy line to test our prototype implementation against.
---

## Tools and Datasets Identified

| Name | Type | URL | Notes |
|---|---|---|---|
| **Flower (Flwr)** | Framework Library | https://github.com/adap/flower | The core orchestration ecosystem used to establish decentralized client-server loops and handle weight aggregations. |
| **PyTorch** | Deep Learning Framework | https://pytorch.org/ | Used on the local edge clients to construct the neural network architectures (e.g., CNN, Autoencoders) and execute local backpropagation. |
| **Edge-IIoTset** | Benchmark Dataset | https://ieee-dataport.org/documents/edge-iiotset-cyber-security-dataset-iot-and-iiot-applications | A comprehensive multi-class dataset built from actual IoT/IIoT testbeds, tracking 14 cyberattack vectors with distinct device features. Highly suited for simulating decentralized nodes. |
| **ToN_IoT** | Benchmark Dataset | https://research.unsw.edu.au/projects/toniot-datasets | A heterogeneous dataset collecting telemetry from diverse IoT sensors, operating systems logs, and network traffic, perfect for testing model robustness against non-IID data. |
| **IoTID20** | Benchmark Dataset | https://www.kaggle.com/datasets/anandur/iotid20 | A specialized dataset focusing on smart home environments (e.g., SKT NUGU speaker, IoT cameras), isolating typical residential device exploits. |
| **Pandas & Scikit-Learn** | Preprocessing Libraries | https://pandas.pydata.org/ | Essential for client-side data engineering tasks such as normalization (MinMaxScaler), handling missing values, and engineering time-series flow windows. |