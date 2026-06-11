# Literature Review: Edge IoT Security Monitoring with Federated Learning
Student: Zarawar Khan
Updated: 2026-06-12

## Reference Table (Quick Overview)

| Title (short) | Authors / Source | Year | Core Topic / Method | Relevance |
|---|---|---|---|---|
| What is Federated Learning? | Google Cloud Discover | 2024 | Decentralized Data & Privacy Basics | High-level conceptual framework for privacy-preserving AI. |
| FL for Medical & Edge AI | NVIDIA Technical Blog | 2025 | Scaling Distributed Clients at the Edge | Practical insights on edge resource constraints and framework scaling. |
| Edge-IIoTset: FL Baseline | Ferrag et al. | 2022 | Comprehensive Cybersecurity Testbed Dataset | Core dataset blueprint, attack profiles, and baseline criteria. |

---

## Lit Review Summaries (Articles & Papers)

### Item 1 (Article) — What is Federated Learning?
* **Source:** Google Cloud Discover Architecture Guides
* **Year:** 2024
* **URL:** https://cloud.google.com/discover/what-is-federated-learning
* **Core Topic:** Decentralized machine learning architectures and data sovereignty.
* **Key Takeaway:** Breaks down how local models compute updates individually before sending aggregate-only metadata (gradients/weights) back to a central orchestrator, completely eliminating the need to move raw logs off the local machine.
* **Relevance to our project:** Provides the core definitions and operational standards for explaining the absolute data privacy mechanics of our distributed system.

### Item 2 (Article) — What is Federated Learning? A Deep Dive into Distributed AI at the Edge
* **Source:** NVIDIA Technical Developer Blog
* **Year:** 2025
* **URL:** https://developer.nvidia.com/blog/federated-learning-without-the-refactoring-overhead-using-nvidia-flare/
* **Core Topic:** Deploying decentralized machine learning configurations onto resource-constrained embedded and IoT hardware ecosystems.
* **Key Takeaway:** Investigates real-world edge hardware realities, focusing on how communication bottlenecks and network dropouts impact global aggregation models like FedAvg.
* **Relevance to our project:** Directly targets the "Edge IoT" realities of our project, giving us clear implementation guideposts for handling communication loops in Flower.

### Item 3 (Academic Paper) — Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning
* **Field:** Federated Learning for Industrial IoT Security
* **Full Title:** Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning
* **Authors:** Mohamed Amine Ferrag, Othmane Friha, Djallel Hamouda, Leandros Maglaras, Helge Janicke
* **Year:** 2022
* **Venue:** IEEE Access
* **URL / DOI:** https://doi.org/10.1109/ACCESS.2022.3165809
* **Method:** Proposes a multi-layer IoT/IIoT evaluation testbed to generate diverse security telemetry, establishing standard baseline machine learning and deep learning configurations for decentralized setups using collaborative clients.
* **Dataset:** Edge-IIoTset (Contains 14 specialized cyberattack profiles including DoS/DDoS, injection, and malware vectors across 10+ distinct physical IoT devices).
* **Key Result:** Validated that deep neural networks deployed in localized federated architectures can achieve comparable detection accuracy to completely centralized models without exporting raw telemetry.
* **Relevance to our project:** This provides the exact dataset benchmark we are using to extract network flow features and build our local PyTorch classifier scripts.

---

## Tools and Datasets Identified

| Name | Type | URL | Notes |
|---|---|---|---|
| **Flower (Flwr)** | Framework Library | https://github.com/adap/flower | Core ecosystem for client-server orchestration. |
| **PyTorch** | Deep Learning Framework | https://pytorch.org/ | Used to build our local network intrusion classifier models. |
| **Edge-IIoTset** | Benchmark Dataset | https://ieee-dataport.org/ | The cybersecurity dataset we will use to simulate real edge node traffic. |