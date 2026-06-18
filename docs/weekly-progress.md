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
**PR link:** [Add link after opening PR]

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
- Begin implementing Krum aggregation as the first defence baseline