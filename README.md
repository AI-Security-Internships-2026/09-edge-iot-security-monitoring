# Edge IoT Security Monitoring with Federated Learning

> **CNIT/PNTLab Pisa · TECIP · Scuola Superiore Sant'Anna — AI Security Internship 2026**

---

## Research Problem

Design a federated-learning framework for IoT edge nodes that collaboratively trains an intrusion detection model without transmitting raw sensor data to a central server.

---

## Objectives

1. Conduct a systematic literature review on the topic.
2. Design and implement a proof-of-concept prototype.
3. Evaluate the prototype on real or benchmark datasets.
4. Document findings in a final technical report.
5. Present results to the research group.

---

## Expected Deliverables

| Deliverable | Due |
|---|---|
| Literature review (`docs/literature-review.md`) | Week 2 |
| Architecture design document (`docs/proposal.md`) | Week 3 |
| Working prototype (`src/`) | Week 6 |
| Evaluation results (`experiments/results/`) | Week 7 |
| Final report (`docs/final-report.md`) | Week 8 |

---

## Recommended Technology Stack

```
Python, Flower (Flwr), PyTorch, MQTT, Pandas, Matplotlib
```

See `requirements.txt` for pinned dependencies.

---

## Weekly Workflow

```
Monday     – Review weekly tasks in tasks/week-XX.md
Tue–Thu    – Implementation / experiments
Friday     – Document progress in docs/weekly-progress.md
Friday     – Open weekly Pull Request from your branch → dev
```

---

## Branching Policy

| Branch | Purpose |
|---|---|
| `main` | Stable, supervisor-reviewed code only |
| `dev` | Integration branch — merge weekly PRs here |
| `<your-name>-week-XX` | Your working branch for each week |

**Students must never push directly to `main`.**

---

## Pull Request Policy

- One PR per week, targeting the `dev` branch.
- PR title format: `[Week XX] Brief description`
- PR description must reference the weekly task file and summarise what was done.
- A supervisor or co-student must review before merging.

---

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/AI-Security-Internships-2026/09-edge-iot-security-monitoring.git
cd 09-edge-iot-security-monitoring

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your weekly branch
git checkout dev
git pull origin dev
git checkout -b your-name-week-01

# 5. Run the starter script
python src/main.py
```

---

## Roadmap to September 8, 2026

**Current state:** the most technically ambitious pipeline in the cohort — Multi-Krum Byzantine-robust aggregation, DP-SGD, cryptographic commitment, and partial homomorphic encryption, all implemented (PR #8). Needs consolidation of duplicated code between `src/` and `docker_fl/src/` (issue #9) before merge.

**Novel contribution target:** the plan in issue #10 — test whether privacy (DP/HE) and robustness (Krum) actually work *together*, not just side by side. This is a real open tension in FL-security research.

| Date | Milestone |
|---|---|
| Aug 2 | Consolidate duplicated defense code (issue #9); get PR #8 merged |
| Aug 9 | Experiment 1 (issue #10): DP vs. Krum detection rate across epsilon values |
| Aug 16 | Experiment 2 (issue #10): does partial HE create a blind spot for Krum in the encrypted classifier head? |
| Aug 23 | Experiment 3 (issue #10): checkpoint every privacy configuration; build the comparison table |
| Aug 30 | Full analysis and write-up of the privacy-robustness tradeoff findings |
| Sep 6 | Paper draft |
| **Sep 8** | **Final submission** |

---

## Supervisor Note

This repository is managed by **CNIT/PNTLab Pisa, TECIP, Scuola Superiore Sant'Anna**.
Please contact your supervisor before making architectural changes.
All code must be original or properly attributed.
Do **not** commit API keys, passwords, or large datasets — see `.gitignore`.
