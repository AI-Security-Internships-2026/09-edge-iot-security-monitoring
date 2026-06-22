# Literature Review: Edge IoT Security Monitoring with Federated Learning

**Student:** Zarawar
**GitHub:** Zarawar5555
**Updated:** June 2026

---

## Reference Table (Quick Overview)

| # | Title (short) | Authors | Year | Method | Dataset | Relevance |
|---|---|---|---|---|---|---|
| 1 | Edge-IIoTset Baseline | Ferrag et al. | 2022 | Centralized & FL Baselines (FedAvg) | Edge-IIoTset | The dataset we will use, this paper introduces and benchmarks it. |
| 2 | Rashid et al. FL-NIDS | Rashid et al. | 2023 | FL with Local CNN + RNN | Edge-IIoTset | Gives us a concrete 92.49% accuracy score to try to beat. |
| 3 | NIDS-FGPA Framework | Wang et al. | 2024 | Gradient Similarity (GSA) + Paillier Encryption | Edge-IIoTset & CIC IoT 2023 | Shows a more advanced aggregation method we can compare against. |
| 4 | How to Backdoor FL | Bagdasaryan et al. | 2020 | Model Replacement + Constrain-and-Scale | CIFAR-10, Reddit (word prediction) | Defines the backdoor attack class our IDS must defend against — the attack-side foundation. |
| 5 | FL Survey Resource-Constrained IoT | Imteaj et al. | 2022 | Survey | N/A | Tells us what hardware and bandwidth limits to design around for edge clients. |
| 6 | FLDetector | Zhang et al. | 2022 | Historical Update Consistency Check | MNIST, CIFAR-10, FEMNIST | Teaches us how to catch a client that suddenly starts sending bad updates. |
| 7 | Krum & Multi-Krum | Blanchard et al. | 2017 | Euclidean Distance Minimization (Krum) | Distributed Baselines | The gold-standard paper for defending FL against attackers — required reading. |
| 8 | Communication-Efficient FL / FedAvg | McMahan et al. | 2017 | FedAvg Algorithm | MNIST, CIFAR-10, Shakespeare | The original FedAvg paper — foundational reference for all FL client-server architecture. |
| 9 | FL Security & Privacy Survey | Mothukuri et al. | 2021 | Survey | N/A | Maps every attack type our IDS needs to defend against. |
| 10 | FedSMOTE-DP Framework | Alsolami | 2026 | DP-SGD + Secure Aggregation + Local SMOTE | WUSTL-EHMS-2020 & CIC-IoMT-2024 | Establishes the exact math and accuracy trade-offs for Differential Privacy in an FL-IDS pipeline. |

---

## Paper 1 — [Edge-IIoTset Baseline]

| Field | Content |
|---|---|
| **Full title** | Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning |
| **Authors** | Mohamed Amine Ferrag, Othmane Friha, Djallel Hamouda, Leandros Maglaras, Helge Janicke |
| **Year** | 2022 |
| **Venue** | IEEE Access, Vol. 10, pp. 40281–40306 |
| **URL / DOI** | https://doi.org/10.1109/ACCESS.2022.3165809 |
| **Method** | Builds a 7-layer IoT/IIoT testbed using 10+ real device types and evaluates both centralized ML and federated FedAvg across 14 labelled attack categories. |
| **Dataset** | Edge-IIoTset — IoT sensor telemetry combined with 14 distinct cyberattack vectors (DoS, DDoS, MITM, Injection, Scanning, etc.) |
| **Key result** | Established standardized baseline performance metrics for decentralized IIoT security monitoring; FL mode achieves comparable accuracy to centralized learning. |
| **Limitation** | Baseline implementation does not address active malicious data or model poisoning attacks from compromised nodes. |
| **Relevance to our project** | This is the dataset our project will use. All simulated edge client nodes will pull data shards from this source. |

### Notes / Quotes:
> Establishes a comprehensive testbed architecture matching real-world industrial protocol configurations (MQTT, Modbus, CoAP).

---

## Paper 2 — [Rashid et al. FL-NIDS]

| Field | Content |
|---|---|
| **Full title** | A Federated Learning-Based Approach for Improving Intrusion Detection in Industrial Internet of Things Networks |
| **Authors** | Md Mamunur Rashid, Shahriar Usman Khan, Fariha Eusufzai, Md. Azharuddin Redwan, Saifur Rahman Sabuj, Mahmoud Elsharief |
| **Year** | 2023 |
| **Venue** | Network (MDPI), Vol. 3, No. 1, pp. 158–179 |
| **URL / DOI** | https://doi.org/10.3390/network3010008 |
| **Method** | Deploys local CNN and RNN architectures on each client within a collaborative FL wrapper; uses parameter update filtering before aggregation. |
| **Dataset** | Edge-IIoTset |
| **Key result** | Achieved 92.49% federated multi-class intrusion detection accuracy across edge client nodes, close to centralized ML (93.92%). |
| **Limitation** | Lacks dynamic client weight adjustments during server aggregation, leading to communication overhead under data drift. |
| **Relevance to our project** | This gives us a specific number to aim for — 92.49% accuracy on the same dataset we are using. |

### Notes / Quotes:
> "This method achieved an accuracy of 92.49% on the Edge-IIoTset dataset, close to the accuracy of traditional centralized machine learning models (93.92%)."

---

## Paper 3 — [NIDS-FGPA Framework]

| Field | Content |
|---|---|
| **Full title** | NIDS-FGPA: A Federated Learning Network Intrusion Detection Algorithm Based on Secure Aggregation of Gradient Similarity Models |
| **Authors** | Jing Wang, Kai Yang, Ming Li |
| **Year** | 2024 |
| **Venue** | PLOS ONE, Vol. 19, No. 10 |
| **URL / DOI** | https://doi.org/10.1371/journal.pone.0308639 |
| **Method** | Converts traffic features to grayscale images, trains a 2DCNN-BiGRU model locally, then uses Gradient Similarity Aggregation (GSA) to filter weak/malicious updates and Paillier Homomorphic Encryption to protect gradients during upload. |
| **Dataset** | Edge-IIoTset & CIC IoT 2023 |
| **Key result** | Achieved 94.5% detection accuracy on Edge-IIoTset and 99.2% on CIC IoT 2023 while filtering out low-quality and deceptive client updates. |
| **Limitation** | Paillier encryption is computationally heavy and adds significant latency on tiny microcontrollers. |
| **Relevance to our project** | Shows us a concrete way to improve on basic FedAvg by filtering bad client updates using cosine similarity — a technique we can try to replicate. |

### Notes / Quotes:
> Rejects divergent client update vectors by tracking the cosine similarity angle between local updates and the global update — if the angle exceeds 90°, the client is dropped.

---

## Paper 4 — [How to Backdoor Federated Learning]

| Field | Content |
|---|---|
| **Full title** | How To Backdoor Federated Learning |
| **Authors** | Eugene Bagdasaryan, Andreas Veit, Yiqing Hua, Deborah Estrin, Vitaly Shmatikov |
| **Year** | 2020 |
| **Venue** | Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics (AISTATS), PMLR Vol. 108, pp. 2938–2948 |
| **URL / DOI** | https://proceedings.mlr.press/v108/bagdasaryan20a.html |
| **Method** | Introduces model replacement as a backdoor attack strategy — a compromised client trains on backdoor-trigger data and scales up its weights to temporarily replace the global model. Also proposes constrain-and-scale, a technique that incorporates evasion of anomaly detection directly into the attacker's loss function, making the backdoored model indistinguishable from benign updates. |
| **Dataset** | CIFAR-10 (image classification) and Reddit corpus (next-word prediction) |
| **Key result** | A single compromised client selected in one round can cause the global model to reach 100% accuracy on the backdoor task while maintaining normal accuracy on the main task — rendering standard anomaly-detection defences ineffective. |
| **Limitation** | The backdoor effect can decay across subsequent rounds as benign clients continue training, unless the attacker is selected repeatedly. Does not test on IoT-specific traffic datasets. |
| **Relevance to our project** | Defines the backdoor attack class as distinct from general model poisoning — the attack our IDS threat model must explicitly account for. Papers 6 and 7 in this review provide the defence side; this paper provides the attack-side foundation that motivates them. |

### Notes / Quotes:
> Introduces model poisoning as a new class of attack, distinct from data poisoning — the attacker directly manipulates model weights rather than training labels, and can incorporate defence evasion into its own loss function.

---

## Paper 5 — [FL Survey Resource-Constrained IoT]

| Field | Content |
|---|---|
| **Full title** | A Survey on Federated Learning for Resource-Constrained IoT Devices |
| **Authors** | Ahmed Imteaj, Urmish Thakker, Shiqiang Wang, Jian Li, M. Hadi Amini |
| **Year** | 2022 |
| **Venue** | IEEE Internet of Things Journal, Vol. 9, No. 1, pp. 1–24 |
| **URL / DOI** | https://doi.org/10.1109/JIOT.2021.3095077 |
| **Method** | Systematic survey covering FL deployment challenges on constrained devices: limited compute, communication overhead, Non-IID data distribution, and client dropout handling. |
| **Dataset** | N/A (survey paper) |
| **Key result** | Identifies client selection, model compression, and asynchronous aggregation as the three most important unsolved problems when running FL on low-power IoT hardware. |
| **Limitation** | Covers literature up to 2021; does not address more recent techniques like personalized FL or cross-silo settings. |
| **Relevance to our project** | Helps us make smart decisions about how to simulate edge clients in Python without hitting memory or bandwidth limits. |

### Notes / Quotes:
> Proposes a comprehensive taxonomy of FL challenges specific to IoT — compute constraints, communication bottlenecks, data heterogeneity, and security threats — all of which directly affect our prototype design.

---

## Paper 6 — [FLDetector: Poisoning Defense]

| Field | Content |
|---|---|
| **Full title** | FLDetector: Defending Federated Learning Against Model Poisoning Attacks via Detecting Malicious Clients |
| **Authors** | Zaixi Zhang, Xiaoyu Cao, Jinyuan Jia, Neil Zhenqiang Gong |
| **Year** | 2022 |
| **Venue** | Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '22), pp. 2545–2555 |
| **URL / DOI** | https://doi.org/10.1145/3534678.3539231 |
| **Method** | Server tracks each client's update history across rounds and flags clients whose updates deviate inconsistently from their own past behaviour; detected malicious clients are removed before aggregation. |
| **Dataset** | MNIST, CIFAR-10, and FEMNIST |
| **Key result** | Successfully detected and removed the majority of malicious clients under multiple state-of-the-art poisoning attacks, allowing existing Byzantine-robust methods to achieve accurate global models. |
| **Limitation** | Requires the server to store historical model updates for several rounds, increasing memory demands on the aggregator. |
| **Relevance to our project** | Gives us a practical way to spot a client that suddenly starts sending suspicious updates — the defence-side complement to Paper 4's backdoor attack. Useful for when we harden our Flower server. |

### Notes / Quotes:
> "It is still an open challenge how to defend against model poisoning attacks with a large number of malicious clients." — FLDetector addresses this by detecting rather than tolerating.

---

## Paper 7 — [Krum & Multi-Krum]

| Field | Content |
|---|---|
| **Full title** | Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent |
| **Authors** | Peva Blanchard, El Mahdi El Mhamdi, Rachid Guerraoui, Julien Stainer |
| **Year** | 2017 |
| **Venue** | Advances in Neural Information Processing Systems (NeurIPS), Vol. 30 |
| **URL / DOI** | https://papers.nips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html |
| **Method** | Introduces Krum and Multi-Krum aggregation operators, instead of averaging all updates, the server selects the update(s) with the smallest total Euclidean distance to their nearest neighbours, naturally filtering outliers. |
| **Dataset** | Standard distributed validation sets |
| **Key result** | Provides a formal convergence guarantee for up to f Byzantine workers given n total workers (requiring n > 2f + 2). Experimentally, Multi Krum maintains near-baseline accuracy even with 33% Byzantine workers, demonstrating practical resilience far beyond the theoretical minimum. |
| **Limitation** | Krum can slow convergence when benign clients have very different data distributions (Non-IID), because legitimate updates can look like outliers. |
| **Relevance to our project** | This is the foundational paper for all robust FL aggregation — every other defence method in this review compares against Krum. We will implement it as a baseline aggregator to defend against attacks including those described in Paper 4. |

### Notes / Quotes:
> "A single Byzantine omniscient worker can completely corrupt standard linear combination strategies like Federated Averaging."

---

## Paper 8 — [Communication-Efficient FL / FedAvg]

| Field | Content |
|---|---|
| **Full title** | Communication-Efficient Learning of Deep Networks from Decentralized Data |
| **Authors** | H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agüera y Arcas |
| **Year** | 2017 |
| **Venue** | Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS), PMLR Vol. 54 |
| **URL / DOI** | https://proceedings.mlr.press/v54/mcmahan17a.html |
| **Method** | Proposes the Federated Averaging (FedAvg) algorithm — clients perform multiple local SGD steps before sending weight updates to a central server, which averages them into the global model. Introduces the client-server round-based FL architecture used by virtually all subsequent work. |
| **Dataset** | MNIST, CIFAR-10, Shakespeare (next-character prediction) |
| **Key result** | FedAvg achieves convergence 10–100x faster in communication rounds compared to synchronous SGD baselines, while keeping all training data on-device. |
| **Limitation** | Does not address Byzantine robustness, adversarial clients, or privacy guarantees — these are addressed by subsequent work (Papers 7, 6, and 10 in this review). |
| **Relevance to our project** | FedAvg is the aggregation algorithm at the core of our Flower server. This is the paper that defined the standard we are building on and comparing against. |

### Notes / Quotes:
> Introduces the core FL protocol of local computation followed by server-side model averaging — the foundation upon which all papers in this review build.

---

## Paper 9 — [FL Security & Privacy Survey]

| Field | Content |
|---|---|
| **Full title** | A Survey on Security and Privacy of Federated Learning |
| **Authors** | Viraaji Mothukuri, Reza M. Parizi, Seyedamin Pouriyeh, Yan Huang, Ali Dehghantanha, Gautam Srivastava |
| **Year** | 2021 |
| **Venue** | Future Generation Computer Systems, Vol. 115, pp. 619–640 |
| **URL / DOI** | https://doi.org/10.1016/j.future.2020.10.007 |
| **Method** | Systematic survey covering the full FL attack surface — data poisoning, model poisoning, inference attacks, and backdoor attacks — alongside defences including differential privacy and secure aggregation. |
| **Dataset** | N/A (survey paper) |
| **Key result** | Produces a comprehensive threat taxonomy for FL systems and maps each attack class to known defences; highlights IoT as a particularly vulnerable deployment environment. |
| **Limitation** | Published in 2021; does not cover more recent attacks such as gradient inversion at scale. |
| **Relevance to our project** | Tells us exactly what types of attacks our IDS framework needs to be able to handle — essential for writing the threat model in docs/proposal.md. |

### Notes / Quotes:
> "Federated learning does not fully address data privacy concerns — model updates themselves can leak information about local training data through inference and inversion attacks."

---

## Paper 10 — [FedSMOTE-DP Framework]

| Field | Content |
|---|---|
| **Full title** | FedSMOTE-DP: Privacy-Aware Federated Ensemble Learning for Intrusion Detection in IoMT Networks |
| **Authors** | T. Alsolami |
| **Year** | 2026 |
| **Venue** | Sensors (Basel) / PubMed Central, Vol. 26, No. 5, p. 1592 |
| **URL / DOI** | https://doi.org/10.3390/s26051592 |
| **Method** | Combines an FL pipeline with DP-SGD (using the Gaussian mechanism via a Rényi Differential Privacy accountant) and Secure Aggregation. It explicitly benchmarks varying privacy configurations (ε = 3.0, ε = 10.0, and a non-private baseline ε = ∞) alongside local data balancing (SMOTE). |
| **Dataset** | WUSTL-EHMS-2020 and CIC-IoMT-2024 (Internet of Medical Things telemetry) |
| **Key result** | Confirmed that injecting strict Differential Privacy noise (ε = 3.0) still achieved 94.60% NIDS accuracy and a 0.9598 AUC when combined with Local SMOTE preprocessing — proving robust privacy protections can coexist with high network anomaly detection rates. |
| **Limitation** | The introduced mathematical noise causes minor validation instability and objective function fluctuations during the middle rounds of training. |
| **Relevance to our project** | Provides the missing mathematical blueprint for our pipeline, allowing us to evaluate weight-clipping thresholds and Gaussian noise adjustments to defend against model-inversion/privacy attacks. |

### Notes / Quotes:
> "With the co-construction of imbalance recovery and controlled noise injection, it is possible to create an IDS that does not violate strong data privacy policies and ensures that the high detection rate needed to protect important infrastructure is achieved."

---

## Tools and Datasets Identified

| Name | Type | URL | Notes |
|---|---|---|---|
| **Edge-IIoTset** | Dataset | https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot | Primary benchmark dataset — IoT/IIoT cybersecurity traffic with 14 labelled attack categories. |
| **N-BaIoT** | Dataset | https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT | Network traffic from 9 real IoT devices under Mirai and BASHLITE malware — referenced in threat modelling. |
| **CIC-IoMT-2024** | Dataset | https://www.unb.ca/cic/datasets/index.html | Modern medical IoT cybersecurity dataset used to test privacy constraints under Paper 10. |
| **Flower (flwr)** | Library / Tool | https://github.com/adap/flower | Core FL orchestration framework for our project — handles client-server coordination. |
| **PyTorch (torch)** | Library / Tool | https://pytorch.org/ | Used to build, train, and update local neural network models on each simulated edge client. |
| **TensorFlow Privacy** | Library / Tool | https://github.com/tensorflow/privacy | Computes privacy budgets and hooks into FL systems for DP-SGD implementation as highlighted in Paper 10. |

