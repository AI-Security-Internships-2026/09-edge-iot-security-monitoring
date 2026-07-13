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
| 8 | FedProx | Li et al. | 2020 | Proximal Term Regularisation on Local Objective | MNIST, Shakespeare, Synthetic | Addresses non-IID client drift directly — the aggregation baseline we use in our pipeline. |
| 9 | FL Security & Privacy Survey | Mothukuri et al. | 2021 | Survey | N/A | Maps every attack type our IDS needs to defend against. |
| 10 | FedSMOTE-DP Framework | Alsolami | 2026 | DP-SGD + Secure Aggregation + Local SMOTE | WUSTL-EHMS-2020 & CIC-IoMT-2024 | Establishes the exact math and accuracy trade-offs for Differential Privacy in an FL-IDS pipeline. |
| 11 | VARS-FL | Lakas & Ferrag | 2026 | Validation-Aligned Reputation Scoring | Edge-IIoTset (100 clients) | Most recent non-IID FL-IDS benchmark on our exact dataset — direct comparison point. |
| 12 | Heterogeneity-Aware Semi-Decentralised IDS | Alsaleh et al. | 2025 | BiLSTM + WGAN + Hierarchical FL | Edge-IIoTset & others | Shows WGAN-based local balancing as an alternative to SMOTE under non-IID conditions. |

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
| **Method** | Introduces Krum and Multi-Krum aggregation operators — instead of averaging all updates, the server selects the update(s) with the smallest total Euclidean distance to their nearest neighbours, naturally filtering outliers. |
| **Dataset** | Standard distributed validation sets |
| **Key result** | Provides a formal convergence guarantee for up to f Byzantine workers given n total workers (requiring n > 2f + 2). Experimentally, Multi-Krum maintains near-baseline accuracy even with 33% Byzantine workers, demonstrating practical resilience far beyond the theoretical minimum. |
| **Limitation** | Krum can slow convergence when benign clients have very different data distributions (Non-IID), because legitimate updates can look like outliers. |
| **Relevance to our project** | This is the foundational paper for all robust FL aggregation — every other defence method in this review compares against Krum. We will implement it as a baseline aggregator to defend against attacks including those described in Paper 4. |

### Notes / Quotes:
> "A single Byzantine omniscient worker can completely corrupt standard linear combination strategies like Federated Averaging."

---

## Paper 8 — [FedProx]

| Field | Content |
|---|---|
| **Full title** | Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization |
| **Authors** | Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, Virginia Smith |
| **Year** | 2020 |
| **Venue** | Advances in Neural Information Processing Systems (NeurIPS), Vol. 33 |
| **URL / DOI** | https://proceedings.neurips.cc/paper/2020/hash/f4f1f13c8289ac1b1ee0ff176b56fc60-Abstract.html |
| **Method** | Adds a proximal regularisation term to each client's local objective — `(μ/2)||w − w_global||²` — which penalises local models that drift too far from the current global model during local training. This bounds client drift under both statistical heterogeneity (non-IID data) and system heterogeneity (variable local compute). The proximal term is a generalisation of FedAvg: setting μ=0 recovers standard FedAvg exactly. |
| **Dataset** | MNIST, Shakespeare (next-character prediction), Synthetic non-IID benchmark |
| **Key result** | FedProx consistently improves convergence stability and final accuracy over FedAvg under non-IID and partial participation conditions — in some non-IID settings achieving up to 22% better accuracy than FedAvg at the same communication budget. |
| **Limitation** | Introduces a single hyperparameter μ that must be tuned — too high a value over-constrains clients and prevents local specialisation; too low approaches vanilla FedAvg without benefit. |
| **Relevance to our project** | FedProx is the aggregation algorithm at the core of our pipeline, chosen specifically because our Dirichlet α=0.7 non-IID partitioning causes significant client drift that plain FedAvg cannot handle stably. The proximal term directly addresses the round-to-round oscillations observed in our FedAvg baseline runs. |

### Notes / Quotes:
> "FedAvg is a special case of FedProx with μ=0. By tuning μ, practitioners can continuously interpolate between full local specialisation and strict global consistency."

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
| **Relevance to our project** | Tells us exactly what types of attacks our IDS framework needs to be able to handle — essential for writing the threat model in `docs/proposal.md`. |

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
| **Method** | Combines an FL pipeline with DP-SGD (using the Gaussian mechanism via a Rényi Differential Privacy accountant) and Secure Aggregation. Explicitly benchmarks varying privacy configurations (ε = 3.0, ε = 10.0, and a non-private baseline ε = ∞) alongside local data balancing (SMOTE). |
| **Dataset** | WUSTL-EHMS-2020 and CIC-IoMT-2024 (Internet of Medical Things telemetry) |
| **Key result** | Confirmed that injecting strict Differential Privacy noise (ε = 3.0) still achieved 94.60% NIDS accuracy and a 0.9598 AUC when combined with Local SMOTE preprocessing — proving robust privacy protections can coexist with high network anomaly detection rates. |
| **Limitation** | The introduced mathematical noise causes minor validation instability and objective function fluctuations during the middle rounds of training. |
| **Relevance to our project** | Provides the missing mathematical blueprint for our DP-SGD implementation, allowing us to evaluate weight-clipping thresholds and Gaussian noise adjustments to defend against model-inversion and privacy attacks. |

### Notes / Quotes:
> "With the co-construction of imbalance recovery and controlled noise injection, it is possible to create an IDS that does not violate strong data privacy policies and ensures that the high detection rate needed to protect important infrastructure is achieved."

---

## Paper 11 — [VARS-FL]

| Field | Content |
|---|---|
| **Full title** | VARS-FL: Validation-Aligned Client Selection for Non-IID Federated Learning in IoT Systems |
| **Authors** | Mohamed Lakas, Mohamed Amine Ferrag |
| **Year** | 2026 |
| **Venue** | arXiv preprint arXiv:2605.05896 |
| **URL / DOI** | https://doi.org/10.48550/arXiv.2605.05896 |
| **Method** | Replaces random or loss-based client selection with a server-side reputation scoring mechanism. After each round, the server measures how much each client's update actually reduced validation loss — this contribution signal is aggregated into a sliding-window reputation score via a non-stationary multi-armed bandit formulation. Clients with higher reputation scores are preferentially selected in subsequent rounds. |
| **Dataset** | Edge-IIoTset (100-client non-IID setup, 15-class multi-label classification) |
| **Key result** | Compared to a FedAvg baseline (F1-Macro ≈ 0.556, accuracy ≈ 0.767), VARS-FL achieves F1-Macro 0.6422 and accuracy 0.8185 across 100 clients and 100 rounds — requiring up to 36% fewer communication rounds to reach the same accuracy milestones. Minority classes (Ransomware, Backdoor) improved from F1 60–72% under FedAvg to 85–91.5% under VARS-FL. |
| **Limitation** | Requires a held-out server-side validation set representative of the full class distribution — difficult to obtain in practice under strict non-IID conditions. Per-class F1 breakdown is not published, limiting direct comparison. |
| **Relevance to our project** | The most directly comparable prior work: same dataset, same 15-class problem, Dirichlet non-IID partitioning. Their FedAvg baseline (F1-Macro 0.556) serves as a reference floor for our own FedProx baseline, and their best result (F1-Macro 0.64) is a realistic ceiling for what we can target before adding Byzantine defences. |

### Notes / Quotes:
> "Stateless client selection policies ignore historical contribution patterns, biasing the global model towards dominant classes in non-IID environments — VARS-FL addresses this by rewarding clients whose updates demonstrably improve global validation performance."

---

## Paper 12 — [Heterogeneity-Aware Semi-Decentralised IDS]

| Field | Content |
|---|---|
| **Full title** | A Heterogeneity-Aware Semi-Decentralized Model for a Lightweight Intrusion Detection System for IoT Networks Based on Federated Learning and BiLSTM |
| **Authors** | Shuroog Alsaleh, Mohamed El Bachir Menai, Saad A. Al-Ahmadi |
| **Year** | 2025 |
| **Venue** | Sensors (MDPI), Vol. 25, No. 4, p. 1039 |
| **URL / DOI** | https://doi.org/10.3390/s25041039 |
| **Method** | Proposes a multi-tier hierarchical FL framework. Local edge devices train a lightweight Bidirectional LSTM (BiLSTM). A Wasserstein GAN (WGAN) generates synthetic minority-class samples locally to balance each client's skewed data distribution before training. Aggregation is handled through intermediate cluster heads that profile client data distributions before pushing updates to the central cloud server. |
| **Dataset** | Edge-IIoTset (15-class) and additional IoT security benchmarks |
| **Key result** | High-volume classes (Normal, DDoS variants) achieved near-perfect F1 of 98.5–99.9%. Mid-volume network attacks (Port_Scanning, SQL_injection, Password) improved from a FedAvg plateau of 91–94% F1 up to 97–98.5% under the WGAN-augmented hierarchical framework. |
| **Limitation** | WGAN training adds significant computational overhead on constrained IoT devices; the semi-decentralised cluster-head topology introduces a new single-point-of-failure risk not present in flat FL architectures. |
| **Relevance to our project** | Demonstrates that local generative data balancing (WGAN) combined with structural clustering can push mid-volume attack class F1 above 97% — an alternative approach to the class-weighting and Focal Loss strategy we use. Their BiLSTM architecture is also directly comparable to our CNN-LSTM model. |

### Notes / Quotes:
> "The combination of localized generative augmentation and cluster-based aggregation effectively decouples the performance of minority attack classes from the statistical dominance of high-volume traffic classes in non-IID IoT deployments."

## Paper 13 — [Dataset-Centric Evaluation]

| Field | Content |
|---|---|
| **Full title** | Dataset-centric evaluation of federated intrusion detection models in IoT networks |
| **Authors** | J. L. Hernandez-Ramos et al. |
| **Year** | 2026 |
| **Venue** | Scientific Reports |
| **URL / DOI** | https://doi.org/10.1038/s41598-025-32567-w |
| **Method** | Benchmarks FedAvg, FedProx and FedNova using LSTM and Transformer models on Edge-IIoTset under multiple label configurations. |
| **Dataset** | Edge-IIoTset |
| **Key result** | Reports macro-F1 approaching 98% only after harmonising the dataset into a simplified 6-class problem rather than the original 15-class task. |
| **Limitation** | Results are not directly comparable to full 15-class IDS evaluation because minority attack categories are merged. |
| **Relevance to our project** | Justifies retaining the original 15-class formulation, providing a fairer and more challenging benchmark. |

## Paper 14 — [HADA-FL]

| Field | Content |
|---|---|
| **Full title** | HADA-FL |
| **Authors** | Chandu Gutti et al. |
| **Year** | 2025 |
| **Method** | Federated learning with differential privacy on 100 simulated edge devices. |
| **Dataset** | Edge-IIoTset |
| **Key result** | Tightening ε from 5.0 to 0.5 reduced accuracy by only about 1.4 percentage points. |
| **Limitation** | Focuses on privacy trade-offs rather than Byzantine robustness. |
| **Relevance to our project** | Provides the closest comparison for evaluating our RQ3 privacy experiments with DP-SGD. |




## Tools and Datasets Identified

| Name | Type | URL | Notes |
|---|---|---|---|
| **Edge-IIoTset** | Dataset | https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot | Primary benchmark dataset — IoT/IIoT cybersecurity traffic with 14 labelled attack categories. |
| **N-BaIoT** | Dataset | https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT | Network traffic from 9 real IoT devices under Mirai and BASHLITE malware — referenced in threat modelling. |
| **CIC-IoMT-2024** | Dataset | https://www.unb.ca/cic/datasets/index.html | Modern medical IoT cybersecurity dataset used to test privacy constraints under Paper 10. |
| **Flower (flwr)** | Library / Tool | https://github.com/adap/flower | Core FL orchestration framework for our project — handles client-server coordination. |
| **PyTorch (torch)** | Library / Tool | https://pytorch.org/ | Used to build, train, and update local neural network models on each simulated edge client. |
| **TensorFlow Privacy** | Library / Tool | https://github.com/tensorflow/privacy | Computes privacy budgets and hooks into FL systems for DP-SGD implementation as highlighted in Paper 10. |