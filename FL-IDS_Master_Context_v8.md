# FL-IDS Complete Master Context Prompt — v8 (Weeks 1–10 + Code/Data Audit + Second-Pass Code Review Integrated)

**This revision merges the Weekly Progress Log (Weeks 1–10) into the v5 Master Context Prompt (v6→v7), then adds: (a) a direct code trace of `main.py`, (b) a direct verification of Experiment 2's real result CSVs/manifests against their console-log-reconstructed claims, and (c) a second, independent code-audit pass across all 11 defence/data-loading/model files (`krum.py`, `zkp.py`, `data_loader.py`, `main.py`, `byzantine.py`, `byzantine_fixed.py`, `he_aggregation.py`, `he_local.py`, `local_dp.py`, `model_defs.py`, `task.py`), plus a fully independent re-verification of the CSV data.** Its purpose is to (a) fold in everything from Weeks 9 and 10 that v5 did not yet contain, (b) surface every place the weekly log and the master doc disagree, (c) report what actually held up versus what didn't when checked against real code/data, and (d) surface a **critical, previously-undetected bug**: `multi_krum()` (fixed-m) does not exist anywhere in the uploaded codebase, despite being called by `main.py` and despite this doc's own "never delete" constraint assuming it exists. Read the "Contradictions & Corrections" section first. **New in v8: items 13–17**, most importantly #15 (missing `multi_krum()`) and a new "Code Audit — Dead Code vs. Keep" section.

---

## ⚠️ CONTRADICTIONS & CORRECTIONS (READ FIRST)

### 1. ZKP / Layer 2 — status was stale in v5

- **v5 said:** extending the Layer 2 HMAC commitment scheme to cover the encrypted classifier-head slice is *"not yet designed in detail; a follow-on question, not a solved problem"*, and that `USE_ZKP=False` throughout all experiments, including Experiments 2/3 "as currently scoped."
- **What actually happened (Week 9):** this was designed and built. `defences/zkp.py` was extended with a ciphertext-bound head-norm guard: each client computes the L2 norm of its classifier-head delta, signs it bound to a hash of the actual ciphertext bytes (so a client can't swap ciphertexts after signing), and the server runs a MAD-threshold outlier check over the committed norms as a pre-filter before Krum.
- **Confirmed working:** five runs, 100% detection in every one, across both models and three different attacked-client configurations. This is no longer a future-work item — it is the current, validated Layer 2 mitigation for the HE+Krum blind spot.
- **Resolution:** v5's Layer 2 section and Experiment 2 discussion are rewritten below to reflect this as done, not proposed.

### 2. Two unrelated "ZKP" problems — do not conflate them

- **Week 9's mitigation** (above) is a *new, working* extension of `defences/zkp.py`, purpose-built for the HE+Krum hybrid.
- **Week 10 separately found** that `main.py`'s pre-existing standalone `USE_ZKP` ablation flag **never called `defences/zkp.py` at all** — it ran a bare `‖params‖ ≤ 10.0` check on the full trained weight vector (not a clipped gradient), almost certainly rejecting every legitimate client every round. This was a dead/broken ablation path, unrelated to Week 9's mitigation, now fixed (`pure_zkp` routes through the real HMAC head-norm guard).
- **v5's existing terminology warning still applies and is now more important, not less:** "ZKP" in this codebase has meant at least three different things over the project's life — (1) the original plain norm-threshold check (broken, now fixed as `pure_zkp`), (2) `defences/zkp.py`'s HMAC commitment (the real mechanism), (3) Week 9's ciphertext-bound head-norm guard (an extension of #2). None of these are a real zero-knowledge proof system. Pick one term ("norm-bound commitment") for the write-up and be explicit about which of the three you mean wherever it matters.
- **New caveat from Week 10's deeper review, not previously documented:** the real `zkp.py` mechanism does not prove the claimed norm matches what's actually inside the ciphertext — a client holding the shared key could sign a false norm for a real ciphertext with nothing to catch it. HMAC provides authenticity/binding, not confidentiality-linked correctness proof. State this limitation honestly.

### 3. Experiment 2's blocker is resolved, not open

- v5 listed `main.py`'s `assert sum([USE_KRUM, USE_ADAPTIVE_KRUM, USE_HE]) <= 1` as the active blocker preventing Experiment 2 from running at all, and treated the hybrid aggregation design as unimplemented.
- **Week 9 built it:** `USE_HE_KRUM_HYBRID`, splitting parameters into a plaintext "bulk" slice (Krum-scored) and an encrypted "sensitive" slice (`classifier.*`, HE-aggregated, restricted to whichever clients Krum selected from plaintext evidence). This also replaced the old full-model `USE_HE` path's two known bugs (no decrypt-before-return, unweighted averaging) with a correct implementation, without touching the old path.
- Experiment 2 is **complete**, not blocked. See the rewritten Experiment 2 section below.

### 4. The stealthy-attack call site had the same bug twice, in two different attacks

- **Week 9:** `classifier_head_flip_attack()`'s call site skipped local training entirely for Byzantine clients — it returned last round's unmodified global model for the backbone, then flipped the head. This made Krum's "detection" trivial for the wrong reason (catching "never trained," not "poisoned head"). Fixed: Byzantine clients now train normally first, then only the trained head is flipped/scaled.
- **Week 10:** the *same class of bug* was found in `sign_flip_attack()` — used for the original, already-"COMPLETED" 6-condition Experiment 1 (ε∈{3,9,15}). Byzantine clients never trained and were bitwise-identical to each other every round, which is non-standard (confirmed via literature: RSA, SpectralKrum, FedSV all define the attack as "train first, then negate"). Fixed via `sign_flip_attack_trained()`.
- **This is a material contradiction with v5's claim that Experiment 1 is "DONE."** The original six runs used an untrained, bitwise-identical Byzantine attack.
- **⚠️ UPDATE — the "corrected" results record is itself unverified as corrected.** `Experiment1_DP_vs_Krum_Analysis_v2.md` was checked directly against this contradiction and does **not** appear to reflect `sign_flip_attack_trained()` at all:
  - Its own changelog states, verbatim: *"Changelog v1 → v2: Added Section 0 (code review confirmation)... Added Section 4b... **No changes to the underlying data or any other numeric result from v1.**"*
  - It analyzes the same six files (`results_network_dp3/dp9/dp15.csv`, `results_application_dp3/dp09/dp15.csv`) at the same three epsilon points — not Week 10's extended 13-point grid (ε ∈ {0.5,1,2,4,5,6,7,8,10,11,12,13,14}).
  - Its `krum_score_ratio` table (Network 310.6/313.2/315.8; Application 102.5/103.8/104.8; +1.7%/+2.2%) is digit-for-digit identical to the pre-fix numbers already on record.
  - Section 0 is a careful, line-cited code review confirming the DP-safe-FedProx patch — but contains **zero mention** of `sign_flip_attack_trained()` or the untrained-attacker bug, despite auditing the file end-to-end. A review that thorough would very likely have flagged the attack fix too, if it had actually been re-run under it.
  - **Conclusion:** treat `Experiment1_DP_vs_Krum_Analysis_v2.md` as the **same pre-fix run**, re-described with an added (valid, but unrelated) code-review note — not confirmation that Week 10's corrected sweep reproduces the original headline findings. As of this revision, **no verified analysis document reflecting the corrected, trained-attacker sweep has been produced.** Both "Headline Result 1" (Krum robust to DP noise) and "Headline Result 2" (DP delays rare-class discovery) below still rest on the old, bugged-attack data and need re-confirmation once a genuinely corrected-sweep write-up exists. Do not cite the "v2" label as evidence of correction — it is a documentation-revision number, not a data-revision number.

### 5. "ε=9 sweet spot" — retracted

- v5 flagged ε=9 outperforming ε=15 on the application model as "needing repeat-seed confirmation" but leaned on it as a plausible real effect (backed by two independent metrics).
- Week 10's extended 13-point sweep (ε ∈ {0.5,1,2,4,5,6,7,8,10,11,12,13,14}) found **the real best point across all values is ε=14 on both models**, not ε=9. The earlier claim is explicitly walked back in `FL-IDS_Epsilon_Sweep_Analysis.md`. Do not cite ε=9 as a sweet spot going forward.

### 6. DP calibration accuracy claim was narrower than stated

- v5 states, unqualified: "achieved epsilon within 0.25% of target in every case."
- This is only true across the original ε∈{3,9,15} range. Week 10's extended low-epsilon sweep shows calibration error growing to **~1.16% at ε=0.5**. The 0.25% figure should be scoped to the mid/high-epsilon range, not stated as universal.

### 7. Krum's "zero collateral exclusion" claim is scope-limited

- v5 states Experiment 1's Krum selection was "textbook-clean... zero persistent-exclusion anomaly anywhere in Experiment 1's six runs."
- This remains true **for Experiment 1 specifically**. But Week 9's HE+Krum hybrid run (a different experiment, different attack) surfaced a persistent honest-client exclusion anomaly on the network model — clients 4/5/10 dropped every round regardless of attack — that connects to and has *grown* from the earlier, still-unresolved Condition 5 anomaly (1 excluded client → 3). Week 9 also traced this directly: clients 4/10 hold 3–6× the fleet-median sample count and ~73% of the entire `Vulnerability_scanner` class between them — likely a partition-composition effect, not an attack effect. Keep v5's Experiment-1-scoped claim, but do not generalize it project-wide; the anomaly is real, recurring, and now partially root-caused.

### 8. Repository path drift

- v5 corrected the working directory to `experiments/Current model/` (retracting the weekly log's earlier `experiments/Current tests/`). Nothing in Weeks 9–10 changes this; `experiments/Current model/` remains current ground truth.

### 9. `ADAPTIVE_KRUM_HYBRID_ASSUMED_F` is not actually a CLI-tunable knob

- Week 9's log claims *"added a new `ADAPTIVE_KRUM_HYBRID_ASSUMED_F` knob (default lowered from `NUM_BYZANTINE` to 1)"*, implying it's independently configurable like `--krum-k`.
- **Direct code trace (the `main.py` reviewed in this session) shows it is hardcoded:** `ADAPTIVE_KRUM_HYBRID_ASSUMED_F = min(1, NUM_BYZANTINE)`. There is no `--assumed-f` (or equivalent) argparse entry anywhere in the file. Since `NUM_BYZANTINE ≥ 1` in every configuration this codebase supports, this formula **always evaluates to 1** — it isn't a "lowered default," it's a fixed constant with no way to set it back to `NUM_BYZANTINE` without editing the source.
- **Resolution:** the manifest's `network_he_krum_hybrid_norm_guard_v1` run recorded `"assumed_f": "NUM_BYZANTINE (pre-tuning defaults)"` — this is consistent with that specific run having used an *older* version of `main.py`, before the `min(1, ...)` cap was added, not with the knob being adjustable in the current code. Treat "pre-tuning defaults" in that manifest entry as a code-version marker, not a currently-reachable configuration.

### 10. `HEAD_NORM_GUARD_K` and `ADAPTIVE_KRUM_K` share one CLI flag

- Both are set as `_args.krum_k if _args.krum_k is not None else 2.5` — the same `--krum-k` flag drives the MAD sensitivity for *both* the ciphertext-bound norm guard (encrypted-slice gate) and adaptive Krum (plaintext-slice gate) in the hybrid branch. These are conceptually independent knobs gating different slices of the model, but the code currently has no way to tune them separately. Any run description that reports a single "k=3.5" for a hybrid run means both thresholds moved together, not that they were tuned independently.

### 11. Experiment 2's Week 9 numbers verified against real (non-reconstructed) CSVs — mostly confirmed, two corrections found

Direct analysis of the five uploaded CSVs (`results_application.csv`, `results_application_byz4_10.csv`, `results_network.csv`, `results_network_byz2_7.csv`, `results_network_byz4_10.csv`) against `COMBINED_MANIFEST.md`/`manifest.json`'s console-log-reconstructed claims:

- **Confirmed exactly, to the reported precision, for four of five runs:** best F1-Macro and round (application default 0.7286@24; application_byz4_10 0.6954@23; network_byz4_10 0.8360@20; network_byz2_7 0.8237@21), the round-25 drop/no-drop pattern for each, the network_byz4_10 round-19–25 volatility sequence (0.836→0.536→0.829→0.620→0.782→0.591), the application-default per-class recovery numbers at round 24 (Normal 0.94, Backdoor 0.95, XSS 0.45, Password 0.43, Fingerprinting 0.80), and 100% Byzantine detection every round in every file (`krum_detected_byzantine == 1.0` on every mean row).
- **Confirmed as expected code behavior, not a data gap:** the `zkp_rejected` column is 0 in every row of all five files, even in rounds with real exclusions — because `USE_HE_KRUM_HYBRID`'s norm-guard rejections are folded into `krum_discarded_ids`/`krum_selected`, not the separate `zkp_rejected_this_round` list (that list is only populated by the standalone `USE_ZKP` branch). This is consistent with, and further confirms, these being genuine hybrid-branch runs.
- **Correction 1 — a misattributed client ID:** `COMBINED_MANIFEST.md`'s narrative for the `network_he_krum_hybrid_norm_guard_byz2_7_v1` round-24 anomaly says a "THIRD client (index 9, previously always kept)" was transiently norm-guard-rejected. The real CSV shows this was actually **client 5** (0-indexed 4) — client 10 (index 9) was *already* one of the two permanent extra exclusions every round in this run, not "previously always kept." The rest of that anomaly's description (F1 0.8096→0.6603, honest_mean_score ~2.18M→1.66M, self-corrected by round 25) is accurate; only the client identity is wrong.
- **Correction 2 — an undocumented anomaly:** in `results_application.csv` (the default-Byzantine run), client 9 was transiently excluded at **rounds 3 and 4** — not mentioned anywhere in the manifest. Minor and self-correcting, but should be folded into the record for completeness.
- **Still unverified — `results_network.csv` (the default network run, clients 1,2) is completely empty**, header only, zero data rows. `network_he_krum_hybrid_norm_guard_v1`'s "complete_with_anomalies" status (SSH disconnect, rounds 9–13 missing from console log) is therefore **not verified by real data at all** — not "missing five rounds," but missing the entire run. The manifest's own "Still Outstanding" checklist already flagged needing the real CSV for this run; that item remains open, not resolved, based on what's been uploaded so far.

### 12. `USE_HE_KRUM_HYBRID` (Experiment 2) and `USE_ZKP` (`pure_zkp` ablation) are not interchangeable — confirmed by direct code trace

- They share the same underlying guard-verification calls (`zkp.verify_head_norm_proof()`, `zkp.mad_threshold_head_norms()`) and the same encryption call (`he_local.encrypt_params_with_norm_guard()`), which makes it easy to assume one validates the other.
- **They are structurally different pipelines.** `USE_HE_KRUM_HYBRID` runs the guard as a pre-filter, then runs `adaptive_multi_krum()` on the plaintext "bulk" slice of whoever survives the guard — two sequential defence stages. `USE_ZKP` runs the same guard and then goes **straight to `he_local.aggregate_encrypted()`** — no Krum call anywhere in that branch, one stage only.
- **The code's own "Known Open Items" comment says this explicitly:** *"The pure_zkp ablation's detection rate is NOT directly comparable to Experiment 2's mitigated hybrid runs (which had Krum as a second, redundant layer behind the guard)."*
- **Resolution:** a strong-100%-detection `pure_zkp` result does **not** mean Experiment 2 can be skipped or re-derived from it, and it does not mean the two experiments are "the same logic." They deliberately differ so that comparing them answers a real question (does Krum do any independent work behind the guard, or was the guard alone always sufficient?) — a question that only means something if the two pipelines are genuinely different, which they are. Experiment 2's own results stand on their own (see Contradiction #11 above for their independent CSV verification), not on equivalence with `pure_zkp`.

### 13. CSV structural facts, independently re-derived — mostly confirm the code trace, one important caveat

A second, independent pass over the same five CSVs (this session) re-derived the file structure from scratch rather than trusting the earlier read:

- 11 rows/round (10 clients + MEAN), 25 rounds → 275 data rows + header — confirms `NUM_ROUNDS=25`, `NUM_CLIENTS=10`.
- **`dp_epsilon_spent`/`dp_epsilon_target`/`dp_noise_multiplier` are `N/A` in every row of all four non-empty files** — this is a clean, independent confirmation that Experiment 2's `USE_HE_KRUM_HYBRID` runs are DP-free (`USE_DP=False`), consistent with the documented config but not previously verified from the data itself.
- `krum_scores_byzantine_mean`/`krum_scores_honest_mean`/`krum_score_ratio` are populated only on the MEAN row — per-client rows leave them `N/A`. Matches the CSV-writing code's `append_log_row()` design (these are round-aggregate diagnostics, not per-client).
- **Nuance on `krum_detected_byzantine`:** on a per-client row this is a 0/1 flag, but it's described in this pass as pinned to the *known attacker identity* every round (e.g., clients 2 and 7 read `1` throughout `results_network_byz2_7.csv`) rather than a genuine per-round re-detection signal at the individual-client level — worth distinguishing from `krum_selected`, which *does* vary per round per client and is the actual exclusion indicator.
- **A real point of friction with Contradiction #11's "Correction 1":** this pass argues there is *no per-round exclusion-**identity** column* (no `krum_discarded_ids`-style list) in these CSVs — only the aggregate `krum_selected` count on the MEAN row, which does drop 6→5 at round 24 and recover to 6 at round 25 for `network_byz2_7` (confirming *something* extra was excluded that round), but not *which* client, in this pass's view. Loss/accuracy are described as too noisy across multiple clients that round to back out identity from those columns either.
  - **This is only a partial disagreement, not a full contradiction — flagged transparently rather than resolved:** the same CSV file was analyzed *directly* earlier in this conversation, filtering to `client == 5`'s own row across all 25 rounds using the per-client `krum_selected` column (0/1), and that isolated exactly one round (24) where `krum_selected == 0`, with `krum_selected == 1` in every other round — the kind of evidence that *does* identify the specific excluded client, reproducibly, from a column this second pass does not appear to have queried at the per-client level (it treats `krum_selected` as an aggregate-only field, which is true for `krum_scores_*` but not for `krum_selected` itself, since that field is written for both per-client and MEAN rows per the CSV-writing code in `main.py`).
  - **Net position:** Contradiction #11's "client 5, not client index 9" finding is not being retracted — it is directly reproducible from the uploaded CSV's per-client `krum_selected` values — but flag in any write-up that a second review pass disputed whether this identity is derivable from these columns at all, and that the CSV format would benefit from an explicit `krum_discarded_ids` (or similar) column so this kind of identity claim doesn't depend on cross-referencing per-client rows by hand.

### 14. The standalone-`USE_ZKP` bug never touched Experiment 2 — independently re-confirmed via a second code trace and via the data itself

- A live question surfaced in this session: did the Week-10-fixed `USE_ZKP` bug (bare `‖params‖≤10.0` check, never calling `defences/zkp.py`) ever contaminate Experiment 2's `USE_HE_KRUM_HYBRID` results, given both are "ZKP" in casual conversation?
- **Answer: No — confirmed twice, independently.**
  1. **Code trace:** `USE_HE_KRUM_HYBRID` and `USE_ZKP` are two separate `elif` branches in `main.py`'s aggregation logic. The broken bare-norm check lived *only* in the `USE_ZKP` branch. `USE_HE_KRUM_HYBRID`'s branch has called the real `zkp.verify_head_norm_proof()` / `zkp.mad_threshold_head_norms()` (Part 2, the ciphertext-bound mechanism) since the moment it was built in Week 9 — it was never routed through the broken bare-norm path at any point in its history. These are two branches that happen to share the word "ZKP" in conversation, not two versions of the same code path (exactly the terminology trap Contradiction #2 warns about).
  2. **Data-pattern confirmation:** the old bug would almost certainly reject every client every round (the master doc's own words). The actual Experiment 2 CSVs show smooth 25-round F1 curves climbing from ~0.05–0.5 up to 0.7–0.84, `krum_detected_byzantine=1.0` throughout, and normal per-round client counts — behavior only possible under a functioning guard. This is independent, empirical confirmation on top of the code trace.
- **Practical conclusion — Experiment 2 does not need to be re-run because of the ZKP bug.** The only genuine gap remains the missing `results_network.csv` (default-Byzantine network run) plus the general, not-yet-done repeat-seed confirmation for all five mitigated runs — neither is related to the ZKP bug.

### 15. 🔴 CRITICAL — `multi_krum()` (fixed-m) does not exist in the uploaded codebase, but `main.py` calls it

- **This directly contradicts one of this master doc's own stated hard constraints** (previously: *"`multi_krum()` (fixed-m) must never be deleted or replaced by `adaptive_multi_krum()` — both are needed"* and *"`m`-propagation bug confirmed fixed"*). Both of those statements assumed the function exists and works. A second-pass code audit (this session, reading all 11 uploaded files against each other — grepping every function definition against every call site) found it does not.
- **Evidence:** `defences/krum.py` as uploaded contains only `adaptive_multi_krum()`. There is no `multi_krum()` function definition anywhere in any uploaded file. Yet `main.py`:
  - imports it conditionally: `from defences.krum import multi_krum` (when `USE_KRUM`)
  - calls it at (approximately) line 1416: `global_params, selected_positions = multi_krum(accepted_params, accepted_weights, num_byzantine=NUM_BYZANTINE, m=effective_m)`
- **Currently masked, not safe:** none of the ABLATION_MODE presets reviewed (`pure_dp`, `pure_he`, `pure_zkp`, and a fourth preset name — `krum_dp_sweep` — surfaced in this pass but not previously confirmed in the three-mode `ABLATION_MODE` block this doc traced directly; reconcile before relying on either count) ever set `USE_KRUM=True`, so this import/call never fires today. But `main.py`'s own docstring explicitly invites manual flag edits outside the `ABLATION_MODE` block ("set the flags directly and remove/bypass this block") for exactly the kind of fixed-m comparison run this doc's own Open Items list says is still needed — the moment that happens, it's an immediate `ImportError`, not a graceful fallback.
- **Action required:** either `multi_krum()` was lost in a merge/reorg and needs restoring from an earlier revision, or every place this doc claims fixed-m Krum is "confirmed fixed" / "kept as the Condition 3 comparison point" needs correcting to "currently missing, not merely unused." Until this is resolved, **do not attempt a fixed-m Krum run** (including any future "true fixed-m-Krum epsilon sweep," already an open item) — it will crash immediately.

### 16. Dead code and orphaned modules identified (second-pass audit)

A second-pass audit across `krum.py`, `zkp.py`, `data_loader.py`, `main.py`, `byzantine.py`, `byzantine_fixed.py`, `he_aggregation.py`, `he_local.py`, `local_dp.py`, `model_defs.py`, `task.py` — cross-referencing every function definition against every call site project-wide — found (see the full "Code Audit" section below for detail):
- `byzantine_fixed.py` (whole file) has zero references anywhere in the codebase and is a stale, out-of-date snapshot of `byzantine.py` (missing `gaussian_attack_trained()` entirely) — recommended for deletion.
- `local_dp.py` (whole module) is not imported by `main.py` at all — the real DP path goes straight through Opacus's `PrivacyEngine`. `local_dp.py`'s one-shot output-perturbation approach is a coherent, documented *alternative* mechanism, referenced only in its own docstring and one comparison mention inside `zkp.py`. Not broken, but orphaned from the executable pipeline — flag as reference-only unless a genuine "local/edge-gateway DP" ablation is planned.
- The untrained `sign_flip_attack()`/`gaussian_attack()` in `byzantine.py` are still imported into `main.py` even though never called (only the `_trained` variants and `zero_gradient_attack`/`classifier_head_flip_attack` are actually used) — their docstrings already say "DO NOT USE FOR NEW EXPERIMENTS," but the live import is misleading dead weight.
- `get_round_lr()` in `main.py` — confirmed dead, matches the doc (defined, never called, LR decay disabled by design).
- Everything else audited (`adaptive_multi_krum()`, both parts of `zkp.py`, `he_aggregation.py`+`he_local.py`'s full interface, `data_loader.py`'s label/feature handling, `model_defs.py`/`task.py`'s documented GPU/DP fixes, and all four actively-used attack functions) was **confirmed correctly implemented and matching this doc's claims** — no further drift found there.

### 17. Two remaining action items, stated directly by the user (not yet done as of this revision)

1. **Provide the results of "sweep 1"** — read in context as the still-missing, genuinely `sign_flip_attack_trained()`-based Experiment 1 sweep results (see Contradiction #4's unresolved status) — this master doc cannot close that contradiction until this is supplied.
2. **Run "sweep 2"** — not yet disambiguated in this conversation. Candidates, in descending likelihood given the surrounding context: (a) the still-missing `results_network.csv` for the default-Byzantine `network_he_krum_hybrid_norm_guard_v1` Experiment 2 run; (b) the not-yet-confirmed-complete Gaussian-attack sweep from Week 10; (c) a repeat-seed run of any of the above for reproducibility. Flagged here rather than guessed at — confirm which before treating either as done.

---

## Project Identity

**Student:** Zarawar Khan (GitHub: Zarawar5555), BE Electrical Engineering, SEECS NUST, 4.0 CGPA
**Internship:** 12-week AI Security internship at CNIT/PNTLab, TECIP, Scuola Superiore Sant'Anna, Pisa, Italy
**Supervisor:** Rana Abu Bakar — framing: *"End goal is just to make FL more secure and more private."* Research-contribution orientation, evaluated against FL literature, not commercial viability.
**GitHub org / repo:** AI-Security-Internships-2026 / 09-edge-iot-security-monitoring
**Compute:** NVIDIA DGX Spark (Grace Blackwell GB10, ARM64, unified 128GB CPU/GPU memory), accessed via OpenVPN + SSH, training inside an NGC PyTorch container (`nvcr.io/nvidia/pytorch:25.10-py3` — required for GB10's sm_121 compute capability).

---

## Early Project History (Weeks 1–5, condensed)

- **Weeks 1–2:** environment setup, literature review (10→12 papers), full research proposal, Edge-IIoTset dataset download, and a first minimal Flower v1.31 FedAvg server.
- **Week 3:** Docker Compose + Kubernetes deployments of the full Flower stack; scaled to 10 simulated clients; hit Flower's Windows/Python-3.14 simulation-mode incompatibilities and replaced it with a manual FL training loop (functional workaround, never reconciled with native Flower orchestration — Flower is fully abandoned by Week 7's reorg, see Constraints).
- **Week 4:** integrated the real Edge-IIoTset dataset; found and partially fixed the `VarianceThreshold` feature-destruction bug (later fully resolved by the label-encoder fix, see below); split into network/application 8-class models; adopted FedProx; established pre-defence baselines (**later invalidated** by the LabelEncoder bug — see "Critical" section below).
- **Week 5:** built the first working Multi-Krum defence and ran the first Byzantine-attack benchmarks (**also later invalidated** by the LabelEncoder bug); stood up a parallel Kali Linux / Metasploitable 2 penetration-testing lab to study the 15 Edge-IIoTset attack types directly.

These weeks' *numeric* results are superseded by the label-encoder fix (below); the infrastructure and workflow decisions (Docker/K8s abandonment in favor of a manual loop, FedProx adoption, Multi-Krum implementation) remain in place.

---

## CRITICAL: Everything Before the Label Bug Fix Is Invalid

A fundamental bug invalidates all baselines and Byzantine/Krum results prior to its fix. **Do not use any F1 numbers from before this fix.**

### The LabelEncoder Alphabetical Sorting Bug

`sklearn.LabelEncoder.fit()` sorts labels **alphabetically**, ignoring the intended `ALL_CLASSES` order:

```python
_encoder = LabelEncoder()
_encoder.fit(ATTACK_CLASSES)  # intended order ignored
y = _encoder.transform(df['Attack_type'].values)
```

**Concrete damage:**

| What code said | What code meant | What actually happened |
|---|---|---|
| `y == 7` (cap to 18%) | DDoS_TCP | Was actually capping **Normal** |
| `APP_ORIG_IDX` | Normal/SQL_inj/Upload/Backdoor/Port_Scan/XSS/Password/Fingerprint | Was actually Backdoor/Fingerprint/MITM/Password/Ransomware/SQL_inj/Uploading/XSS |
| `NETWORK_ORIG_IDX` | Normal/DDoS_UDP/DDoS_ICMP/Ransomware/DDoS_HTTP/DDoS_TCP/Vuln_scanner/MITM | Was actually Backdoor/DDoS_HTTP/DDoS_ICMP/DDoS_TCP/DDoS_UDP/Normal/Port_Scanning/Vuln_scanner |

**Real class distribution** (previously misread): Normal is the true majority at 1,615,643 rows (previously misidentified as 24,862); real DDoS_TCP is 50,062 (previously misread as 1,615,643); SQL_injection and Uploading are the genuinely rare classes (1,001 and 1,214 rows respectively, previously attributed to Fingerprinting/MITM).

**Invalidated:** old network baseline (F1-Macro 0.839), old application baseline (0.660), all pre-fix Byzantine/Krum results, all pre-fix ablation F1 numbers (timing/RAM numbers from those ablations remain valid — they don't depend on label correctness).

### The Fix

```python
ALL_CLASSES = [
    'Normal', 'DDoS_UDP', 'DDoS_ICMP', 'Ransomware', 'DDoS_HTTP',
    'SQL_injection', 'Uploading', 'DDoS_TCP', 'Backdoor',
    'Vulnerability_scanner', 'Port_Scanning', 'XSS', 'Password',
    'MITM', 'Fingerprinting'
]
_class_to_idx = {name: i for i, name in enumerate(ALL_CLASSES)}

def encode_labels(series):
    return series.map(_class_to_idx).values
```

`LabelEncoder` is permanently removed; never reintroduce it. The preprocessing cache was rebuilt from scratch after this fix.

---

## Second Major Discovery: Text Features Silently Destroyed

`pd.to_numeric(errors='coerce')` zeroed every text-bearing HTTP/DNS/TCP-payload column (`http.file_data`, `http.request.uri.query`, `http.referer`, `http.request.full_uri`, `dns.qry.name`, `tcp.payload`) — precisely the columns that distinguish XSS, SQL_injection, Uploading, Password, and Fingerprinting. The network model was unaffected (relies on packet-level numeric features only).

**Fix:** `engineer_text_features(df)`, run before numeric coercion, extracts length/entropy/keyword-regex/frequency-encoded signal from these columns (informed by `inspect_weak_class_payloads.py`, which found the real distinguishing signal for XSS/Password/Uploading is User-Agent tool-fingerprints and HTTP method, not injected exploit syntax), then drops the raw text columns.

**Confirmed feature counts (directly measured, current ground truth):**

| Model | Features | Source |
|---|---|---|
| Network | **39** | Measured from Experiment 1 run logs |
| Application | **90** | Measured from Experiment 1 run logs |

Any reference to 40/38/35/52/"~80–91" is stale.

---

## NEW Locked Baselines (post label-fix + text-feature-engineering)

LR decay and EMA were **not** used in the runs that produced these baselines.

### Application Model — Round 20 (LOCKED)
Accuracy **0.8504**, F1-Macro **0.7293**.
Per-class F1: Normal 0.8776, SQL_injection 0.8184, Uploading 0.6110, Backdoor 0.9399, Port_Scanning 0.8268, XSS 0.4518, Password 0.5052, Fingerprinting 0.8035.
**Open anomaly — Client 6:** Password/XSS F1 stuck at 0.03–0.08 vs 0.35–0.60+ for all other clients across all 25 rounds. Hypothesis: near-zero real samples for these classes under Dirichlet(α=0.7). **Still unresolved** — `per_client_audit.py` proposed, never run.

### Network Model — Round 22 (LOCKED)
Accuracy **0.9697**, F1-Macro **0.8289** (round 21: 0.9543/0.8278; round 23: 0.9340/0.7970; round 24: 0.9549/0.8217; round 25: 0.8090/0.6800).
Per-class F1 at round 22: Normal 0.9364, DDoS_UDP 0.9992, DDoS_ICMP 0.9996, Ransomware 0.7387, DDoS_HTTP 0.7902, DDoS_TCP 0.9955, Vulnerability_scanner 0.7198, MITM 0.4514.

**Round-25 instability — now confirmed structural, 5 independent occurrences:** clean baseline (round 25), Condition 5 network (round 23), Condition 5 application (round 21), Experiment 1's `network_dp15` (round 23→25), and Week 9's HE+Krum hybrid run (a 5th occurrence, recommending round 24 as the headline number there instead of the final round). Root cause: non-IID client-sampling variance interacting with mid-sized classes — treat as structural to this FedProx/non-IID setup, not noise.

**MITM sample scarcity:** only 1,214 samples; F1 flat around 0.41–0.48 — a hard data ceiling, confirmed again in every subsequent MITM-involving run.

---

## Literature Comparison

| System | F1-Macro | Accuracy | Clients | Rounds | Notes |
|---|---|---|---|---|---|
| This work — Network | 0.8289 | 0.9697 | 10 | 25 | Correct labels + FedProx |
| This work — Application | 0.7293 | 0.8504 | 10 | 25 | Correct labels + text features |
| VARS-FL (2025) | 0.6422 | 0.8185 | 100 | 100 | Best published on same dataset |
| Rashid et al. | ~0.92 acc only | 0.9249 | N/A | N/A | Near-IID, majority-class bias |

---

## Dataset

**File:** `datasets/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv`
**Cache:** split per model (`dnn_preprocessed_cache_network.npz`, `dnn_preprocessed_cache_application.npz`).

**Corrected class distribution (before per-model subsetting):** Normal 1,615,643; DDoS_TCP 50,062; DDoS_ICMP 116,436; Ransomware 50,062; DDoS_HTTP 121,568; SQL_injection 1,001; Uploading 1,214; Backdoor 50,153; Vulnerability_scanner 22,564; Port_Scanning 10,925; XSS 51,203; Password 37,634; MITM 50,110; Fingerprinting 15,915.

**Cap methodology (confirmed):** caps **Normal** to 18% of total post-cap (VARS-FL methodology; corrected from the mistaken pre-fix "cap DDoS_TCP" framing) — confirmed via logs: "Capping Normal to 18%... Samples after capping: 736,046, Normal 132,488 (18.00%)."

---

## Dual-Model Architecture

### Network-Layer Model — 39 features, 8 classes
Normal(0→0), DDoS_UDP(1→1), DDoS_ICMP(2→2), Ransomware(3→3), DDoS_HTTP(4→4), DDoS_TCP(7→5), Vulnerability_scanner(9→6), MITM(13→7).

### Application-Layer Model — 90 features, 8 classes
Normal(0→0), SQL_injection(5→1), Uploading(6→2), Backdoor(8→3), Port_Scanning(10→4), XSS(11→5), Password(12→6), Fingerprinting(14→7).

### CNN-LSTM Architecture

```python
class CNN_LSTM(nn.Module):
    def __init__(self, num_features, num_classes=8, dp_safe=False):
        Conv1d(1, 64, kernel_size=3, padding=1)
        GroupNorm(8, 64) if dp_safe else BatchNorm1d(64)
        ReLU → MaxPool1d(2)
        Conv1d(64, 128, kernel_size=3, padding=1)
        GroupNorm(8, 128) if dp_safe else BatchNorm1d(128)
        ReLU → MaxPool1d(2)
        DPLSTM(input=128, hidden=64) if dp_safe else LSTM(input=128, hidden=64)
        Linear(64→64) → ReLU → Dropout(0.3) → Linear(64→num_classes)
```

`dp_safe=True` (GroupNorm + DPLSTM) used for every client, honest and Byzantine alike, in every DP-active run — confirmed code-reviewed.

**Fixes locked in:** `FocalLoss.weight` is a registered buffer (was a plain attribute invisible to `.to(device)`); `train()`/`test()` take an explicit `device=` kwarg at every call site; FedProx's non-DP proximal term is built on `param.device` (was hardcoded CPU); `save_best_checkpoint()` implemented and confirmed working, firing automatically whenever a round beats the best F1-Macro so far.

---

## DP-Safe FedProx

**Problem:** Opacus's `DPOptimizer.step()` builds its update entirely from `.grad_sample`; a loss-added proximal term never populates it and is silently discarded — every DP+FedProx run would silently degrade to DP+FedAvg.

**Fix (`_apply_dp_safe_prox_step`):** applies the proximal pull as a separate, deterministic, unnoised parameter update immediately after `optimizer.step()`, once per batch — privacy-safe since it depends only on current params + last round's public global model, never client data.

**Confirmed:** all completed DP runs (Experiment 1's original six, plus Week 10's extended sweep) are genuine DP-SGD+FedProx (mu=0.02), not DP-SGD+FedAvg. **Constraint:** any DP-active run with `PROX_MU != 0` must use this decoupled step, never a loss-based proximal term.

---

## GPU Migration & Infrastructure

- **Hardware:** DGX Spark, Grace Blackwell GB10, ARM64/aarch64, unified 128GB CPU/GPU memory (`nvidia-smi`'s per-process memory table reports "Not Supported" on this hardware — use `free -h` or `torch.cuda.mem_get_info()` instead).
- **Container:** `nvcr.io/nvidia/pytorch:25.10-py3` (older NGC tags don't support GB10's sm_121 compute capability).
- **Working directory:** `experiments/Current model/` (corrected from an earlier assumed `experiments/Current tests/`).
- **Fork+CUDA hang fix:** on GPU, no `ProcessPoolExecutor` is created — client training/eval runs sequentially in-process (forking after CUDA is initialized in the parent hands the child a broken CUDA context). CPU runs unchanged (4-way pool).
- **vLLM memory contention incident (resolved):** a colleague's vLLM server reserved ~101GB of the 119GB unified pool even when idle, causing hangs and small-allocation OOMs; root-caused via host-level `free -h`/cgroup inspection (not container-scoped `ps`); resolved by stopping the process.
- **GPU timing:** ~132–153s/round for all 10 clients combined — full conditions finish in under 65 minutes, versus ~2–6 hour CPU estimates.
- **CLI args (Experiment 1 era):** `--epsilon`, `--tag`. **CLI args (Week 9–10 additions):** `--byzantine <clients>`, `--krum-k <float>`, `--attack-type`, `--gaussian-std`.
- **Known cosmetic issue:** application's ε=9 run was tagged `dp09` vs. network's `dp9` — standardize before further sweeps.

---

## FL Configuration (Confirmed Values)

```
NUM_CLIENTS=10, NUM_ROUNDS=25, LOCAL_EPOCHS=5, LEARNING_RATE=0.001
PROX_MU=0.02          # confirmed 0.02, not the 0.01 in earlier doc revisions
DIRICHLET_ALPHA=0.7
BATCH_SIZE=512        # DP_BATCH_SIZE; confirmed no OOM post-vLLM-kill
```
No LR decay, no EMA (confirmed for all baseline and Experiment 1 runs — `get_round_lr()` present but dead/unused).

---

## Repository Structure (current)

```
09-edge-iot-security-monitoring/
├── experiments/
│   ├── Current model/                 ← working directory (corrected from "Current tests/")
│   │   ├── main.py                    ← unified: ABLATION_MODE, USE_HE_KRUM_HYBRID,
│   │   │                                 argparse (--epsilon/--tag/--byzantine/--krum-k/
│   │   │                                 --attack-type/--gaussian-std), DP-safe FedProx,
│   │   │                                 save-best-checkpoint, GPU sequential-mode fork fix
│   │   ├── model_defs.py              ← dependency-free model (dp_safe flag)
│   │   ├── task.py                    ← corrected weight tables/gamma; live class-count calc
│   │   ├── data_loader.py             ← corrected labels + text features; model_type= param
│   │   ├── build_partitions.py        ← offline per-client .npz partitions, hard-errors
│   │   │                                 instead of silently falling back to synthetic data
│   │   ├── measure_param_scale.py     ← calibrates Gaussian-attack std per model
│   │   ├── check_features.py / verify_label_bug.py
│   │   ├── confusion_matrix.py        ← still never run, now low-effort (checkpoints recoverable)
│   │   ├── plot_epsilon_sweep.py      ← still not written
│   │   └── defences/
│   │       ├── byzantine.py           ← sign_flip_attack / sign_flip_attack_trained (Week 10 fix),
│   │       │                             gaussian_attack_trained, zero_gradient_attack,
│   │       │                             classifier_head_flip_attack (Week 9 call-site fix)
│   │       ├── krum.py                ← multi_krum() (fixed-m); adaptive_multi_krum() (MAD)
│   │       ├── zkp.py                 ← HMAC-SHA256 commitment + Week 9 ciphertext-bound
│   │       │                             head-norm guard (the real, validated mechanism)
│   │       ├── local_dp.py
│   │       └── homomorphic.py         ← still dead code
│   └── Docker tests for RAM and Latency/
│       ├── he_aggregation.py / he_local.py   ← real HE implementation
│       └── ...
├── scripts/build_manifest.py
├── RESULTS AND MANIFESTS/
└── datasets/dnn_preprocessed_cache_{network,application}.npz
```

Reorg history (Week 7): `src/`→`experiments/Current tests/` (later corrected to `Current model/`), `docker_fl/`→`experiments/Docker tests for RAM and Latency/`; removed confirmed-dead weight (`k8s/`, old `docker-compose.yml`, stale root configs, accidental terminal-capture files). Flower is fully abandoned as of this reorg — training is a custom manual FedProx loop, never Flower's native orchestration.

---

## Three-Layer Privacy Stack

Architecture design and implementation are correct across all three layers as of Week 9. F1 numbers from pre-label-fix experiments are invalid; timing/RAM numbers from those ablations remain valid.

### Layer 1 — Opacus DP-SGD

**Mechanism:** `PrivacyEngine.make_private_with_epsilon()`, per-sample gradient clipping (`max_grad_norm=1.5`, confirmed increased from an earlier documented 1.0).
**Quantum safety:** information-theoretic — holds against any adversary regardless of computational power, unlike Layer 3's lattice-hardness-dependent guarantee.
**Known limitation:** per-round ε only, no cross-round composition accountant — an explicit "Option A" decision, stated as a caveat in the write-up rather than fixed.
**Calibration accuracy:** within 0.25% of target across ε∈{3,9,15}; **degrades to ~1.16% at ε=0.5** (Week 10 correction to the earlier blanket 0.25% claim).
**Resolved question:** does not support restricting DP noise to the classifier head as a Krum-preservation measure — Krum showed no meaningful degradation across ε∈{3,9,15} to begin with.

### Layer 2 — Norm-Bound Commitment ("ZKP" in codebase/docs — not a real ZKP)

**File:** `defences/zkp.py`.
**What it is:** an HMAC-SHA256-signed norm bound, not a zero-knowledge proof system (a real ZKP would need Bulletproofs/a STARK; cost estimate for that: low seconds–~10s/client/round, hundreds of MB–low GB RAM, mostly Rust tooling — not worth converting to given the current mechanism's proven track record).
**Original mechanism:** `C = HMAC-SHA256(key, params_bytes || salt)`; server verifies `‖w‖₂ ≤ clip_norm + 1.15·σ·√n_params`. A previously-fixed `×0.01` scaling bug had made this 100× too strict.
**Week 9 extension — now the operative Layer 2 mitigation, confirmed working, not future work:** a ciphertext-bound head-norm guard. Each client computes the L2 norm of its classifier-head *delta* (trained head minus round-starting global head, confirmed correct against `main.py`'s passed `global_params`), signs it bound to a hash of the actual ciphertext bytes submitted, and the server runs a MAD-threshold outlier check over all verified clients' committed norms — a magnitude-only analogue of Krum, applied as a pre-filter before Krum, to the one number the encrypted slice reveals. Explicitly documented limitation: catches magnitude attacks, not a bounded-magnitude directional attack under threshold.
**Confirmed via 5 runs:** 100% detection, both models, three attacked-client configurations; Vulnerability_scanner F1 recovered from 0.0000 to 0.7864; Best F1-Macro within ~0.001 of clean baseline on default-client runs.
**Week 10's separate finding — a different, broken "ZKP" path, now fixed:** `main.py`'s standalone `USE_ZKP` ablation flag never called any of the above — it ran a bare, uncalibrated `‖params‖≤10.0` check on the full weight vector, almost certainly rejecting every client every round. Fixed: `pure_zkp` (under the new `ABLATION_MODE` refactor) now genuinely runs the real HMAC head-norm guard, standalone, with the correct MAD threshold. **Confirmed via ablation:** the standalone guard hits 100% detection, 0% false positives, on both models, all 25 rounds, with no Krum involved at all — and fully absorbs the attack's utility cost (network `pure_zkp` beat its own clean baseline) while running faster than `pure_he` (fewer clients to aggregate).
**Known caveat (Week 10 deeper review, not previously documented):** the real mechanism doesn't prove the claimed norm matches what's actually inside the ciphertext — a client holding the shared key could sign a false norm for a real ciphertext with nothing to catch it. HMAC provides authenticity/binding only, not confidentiality-linked correctness.
**Terminology:** pick one consistent term for the write-up ("norm-bound commitment" recommended over "ZKP") and be explicit about which of the (at least) three historically-conflated mechanisms is meant.
**Confirmed via direct code trace (this session):** `USE_HE_KRUM_HYBRID` and `USE_ZKP` both call this same guard logic, but are otherwise different pipelines — the hybrid runs the guard *then* Krum on survivors; standalone `USE_ZKP` runs the guard *then aggregates directly, no Krum at all*. The code's own comments explicitly warn against comparing their detection rates as if interchangeable (see Contradiction #12). `HEAD_NORM_GUARD_K` (the guard's threshold) and `ADAPTIVE_KRUM_K` (Krum's threshold) currently share a single `--krum-k` CLI value in the hybrid branch — not independently tunable despite gating different slices (Contradiction #10).
**Independently verified against real CSVs (this session):** four of Experiment 2's five mitigated runs check out almost exactly against non-reconstructed data — 100% detection every round, matching F1-Macro/round numbers, matching exclusion patterns. Two small corrections found (a misattributed client ID in one anomaly writeup, an undocumented transient exclusion); one run (`network_he_krum_hybrid_norm_guard_v1`, the default-Byzantine network run) still has **no real CSV at all** — see Contradiction #11.

### Layer 3 — Partial CKKS HE

**Source of truth:** `experiments/Docker tests for RAM and Latency/he_aggregation.py`, `he_local.py`. `defences/homomorphic.py` remains dead code.
**Parameters:** `poly_modulus_degree=4096` → **64-bit** post-quantum security (not 128-bit — state honestly).
**Scope:** partial — only the classifier head encrypted (~5.8% of network-model parameters).
**Timing/RAM:** ~0.2s/round HE ops; ~400MB RAM floor for pure HE (not the bottleneck — training dominates).
**Fixed architecture bug:** clients previously generated independent HE keypairs (breaks homomorphic addition mathematically); now a single shared public context is server-generated and distributed.
**`he_aggregate()`'s two known bugs — fixed as of Week 9 for the hybrid path:** no decrypt-before-return, unweighted averaging. The Week 9 `USE_HE_KRUM_HYBRID` implementation replaced both with a correct decrypt+weighted-average path; the *old* full-model `USE_HE` path was left untouched (still has the bugs) until Week 10's `ABLATION_MODE` refactor routed `pure_he` through the same validated `he_local.py` pipeline.

---

## Multi-Krum (Fixed-m)

**🔴 STATUS CORRECTED (this session) — the function does not currently exist in the codebase.** `defences/krum.py::multi_krum()` was previously described here (per v4/v5) as "Blanchard et al., NeurIPS 2017... `m`-propagation bug confirmed fixed." A second-pass code audit found `krum.py` as uploaded contains **only** `adaptive_multi_krum()` — no `multi_krum()` definition exists anywhere in the reviewed codebase. Yet `main.py` still conditionally imports it (`from defences.krum import multi_krum`, gated on `USE_KRUM`) and calls it with `m=effective_m` in the `USE_KRUM` branch. This never fires today because no reviewed `ABLATION_MODE` preset sets `USE_KRUM=True` — but it is live code waiting to throw `ImportError` the moment anyone hand-edits the flags for a fixed-m comparison run, which this doc's own Open Items list already calls for. See Contradiction #15. **Until this is restored (or the `USE_KRUM` branch is formally retired), treat every prior claim of "m-propagation confirmed fixed" or "kept as the Condition 3 comparison point, ready to use" as false — the function is missing, not merely unused.** Not the method used in Experiment 1's completed runs regardless (see Adaptive Krum below).

## Adaptive Multi-Krum (MAD-Threshold)

`defences/krum.py::adaptive_multi_krum()`: threshold = `median(scores) + k·1.4826·MAD(scores)`. Default `k=2.5` for Experiment 1 and Condition 5; **Week 9 raised the default to `k=3.5`** for the HE-hybrid experiment specifically (via the shared `--krum-k` CLI override — see Contradiction #10, it also moves the norm guard's threshold), justified by the data-split evidence and the norm guard's proven track record — this is a hybrid-experiment-specific default, not a retroactive change to Experiment 1's `k=2.5`.

**Correction (this session, via direct code trace):** `ADAPTIVE_KRUM_HYBRID_ASSUMED_F` is **not** a configurable knob despite Week 9's log describing it as one. The actual code hardcodes `ADAPTIVE_KRUM_HYBRID_ASSUMED_F = min(1, NUM_BYZANTINE)`, with no CLI argument to change it — since `NUM_BYZANTINE` is always ≥1, this constant always evaluates to 1. See Contradiction #9.

**Condition 5 (historical, pre-Experiment-1, pre-GPU):** both models numerically beat their locked clean baselines under active Byzantine attack — **not evidence of a real improvement**, confounded by an unmatched `PROX_MU=0.1` and active LR decay vs. the baseline recipe. The recipe-matched rerun needed to resolve this has still never been done, through Week 10.

---

## ⭐ EXPERIMENT 1 — DP vs. KRUM (Epsilon Sweep)

**Status: UNSETTLED, not complete and not confirmed-corrected.** v4/v5 reported this "COMPLETE." Week 10 then found the original six runs used an untrained, bitwise-identical sign-flip attack (Contradiction #4) and fixed it via `sign_flip_attack_trained()`. But direct inspection of `Experiment1_DP_vs_Krum_Analysis_v2.md` (the document both this doc and the weekly log cite as the corrected results record) shows it is very likely **the same pre-fix data, not a re-run** — same six files, same three epsilon points, identical `krum_score_ratio` numbers to the original, an explicit "no changes to the underlying data" note in its own changelog, and zero mention of the attack fix despite an otherwise thorough code review. **As of this revision, no verified analysis of a `sign_flip_attack_trained()`-based sweep exists.** Everything below through "Headline Result 2" should be read as **the original, unfixed-attack findings**, not confirmed-post-fix findings — the labels below have been corrected accordingly. See Contradiction #4 for the full evidence trail.

### Original Configuration (six runs, ε∈{3,9,15} × {network, application})
```
NUM_CLIENTS=10, NUM_ROUNDS=25, LOCAL_EPOCHS=5, LEARNING_RATE=0.001, PROX_MU=0.02
BYZANTINE_CLIENTS=[0,1], ATTACK_SCALE=5.0 (network) / 2.0 (application)
USE_ADAPTIVE_KRUM=True, method="mad", k=2.5, min_keep_fraction=0.5
USE_DP=True, DP_EPSILON∈{3.0,9.0,15.0}, DP_DELTA=1e-5, DP_MAX_GRAD_NORM=1.5, DP_BATCH_SIZE=512
USE_ZKP=False, USE_HE=False
```
**Deviation from spec (retained, not a bug):** used adaptive (MAD-threshold) Krum throughout, not the originally-specified fixed-m Krum — a deliberate, documented user decision. A true fixed-m sweep, if needed for an apples-to-apples Condition 3 comparison, is still a separate, not-yet-done experiment.

### Week 10 Correction — Attack Bug, and the Re-Run's Verification Status

**Bug:** `sign_flip_attack()` operated on the untouched global model — Byzantine clients never trained, were bitwise-identical every round. Non-standard per literature (RSA/SpectralKrum/FedSV all train-then-negate). Attack **scale** (5.0/2.0) was confirmed fine; only the missing training step was wrong.
**Fix:** `sign_flip_attack_trained()` — mirrors the already-corrected `classifier_head_flip_attack` pattern (train normally, then corrupt the result).
**Claimed re-run:** the weekly log describes extending the original 3-point (ε=3,9,15) sweep to a full 1-increment grid (ε ∈ {0.5,1,2,4,5,6,7,8,10,11,12,13,14}), described as "the corrected, 32-condition sweep" once combined with the concurrent Gaussian-attack sweep (below).
**⚠️ Not independently verified as of this revision.** `Experiment1_DP_vs_Krum_Analysis_v2.md` — the document that should be this re-run's results record — turns out on direct inspection to be the *original* 3-point, pre-fix data (see Contradiction #4). No document reviewed so far confirms the 13-point `sign_flip_attack_trained()` grid was actually analyzed under the fixed attack. Treat the claimed re-run as **logged in the weekly notes but not yet corroborated by a results artifact.**

### Headline Result 1 — Original hypothesis does NOT hold (⚠️ pre-fix data only — see caveat above)

Krum's Byzantine-detection separation (`krum_score_ratio`) changed by under 2.3% across ε∈{3,9,15} on both models; detection stayed at **100.00% in every round of every run**. Sign-flip's parameter deviation is orders of magnitude larger than anything DP-SGD's noise injects at these ε values. **Provisionally resolved, pending re-verification:** does not justify restricting DP noise to the classifier head as a Krum-preservation measure. This is very likely robust to the attack-training fix too (a 5×/2× scaled sign-flip is a huge deviation whether or not the client trained first), but no document currently confirms it under the corrected attack — flag as "highly likely, not yet re-confirmed" rather than settled.

### Headline Result 2 — DP noise delays rare-class discovery (⚠️ pre-fix data only — see caveat above)

Clearest on Fingerprinting (application model, 1,001 raw samples): F1=0.000 for 23/25 rounds at ε=3, reaches ~0.6 by round 25 at ε=9/15. Not a Krum effect (identical exclusion set across ε conditions) — DP noise directly overwhelming gradient signal for an already data-starved class. This finding is about DP noise and class rarity, largely orthogonal to the sign-flip attack's correctness, so it's also likely to hold — but, as with Headline Result 1, it comes from the same pre-fix run and has not been independently re-confirmed under `sign_flip_attack_trained()`.

### Claims from the weekly log that supersede v4/v5's original-sweep numbers (status: logged, not verified against a results document)

- **"ε=9 sweet spot" — retracted per the weekly log.** The log states the real best point across all 13 (extended) values is **ε=14 on both models** — but this claim comes from `FL-IDS_Epsilon_Sweep_Analysis.md`, which has not itself been produced/reviewed in this conversation. Treat as reported, not independently verified.
- **DP calibration — narrower claim than originally stated, per the weekly log.** Within 0.25% of target across ε∈{3,9,15}; log claims this grows to **~1.16% at ε=0.5** — again, sourced from the same not-yet-reviewed document.
- Network model's F1-Macro vs. epsilon is reported as clean and monotonic in the original (pre-fix) data; the application model's non-monotonicity (ε=9 vs ε=15, discussed in `Experiment1_DP_vs_Krum_Analysis_v2.md` Section 3/4b with real per-run numbers) is a genuine finding **from the pre-fix data** — whether it persists under the corrected attack is unknown.
- Round-25 instability recurred a 4th time in the original sweep (`network_dp15`, round 23→25) — consistent with the project-wide structural pattern (5 occurrences total as of Week 9). This is a training-dynamics finding, not attack-dependent, and likely holds regardless of the sign-flip fix.

### Byzantine Attack Diversity (new, Week 10)

- Added `gaussian_attack_trained()` (same train-first pattern as the sign-flip fix) and wired the previously-dead `zero_gradient_attack` into dispatch. New CLI flags: `--attack-type`, `--gaussian-std`.
- `GAUSSIAN_STD` previously had a flat default (10.0) with no model-type split. Recalibrated via `measure_param_scale.py` on the DGX (network delta_std≈4.17, application≈2.64) to model-aware defaults: **network=50.0, application=30.0** — explicitly not ported from RSA's σ=10000, since additive noise doesn't transfer across models the way a multiplicative scale does.
- `run_gaussian_sweep.sh` run concurrently (in a second tmux session) with the sign-flip v2 sweep — GPU-contention risk explicitly flagged and accepted. **Gaussian sweep's completion status was never confirmed** (open item). Unseeded Gaussian noise (reproducibility) also remains an unresolved open decision.

### Ablation Testing — `ABLATION_MODE` Refactor (new, Week 10)

Bugs found in the original single-`main.py` ablation flags, all resolved by the refactor:
1. `USE_HE=True` (standalone) never decrypted its result and used unweighted averaging — fixed, `pure_he` now routes through the validated `he_local.py` pipeline (encrypt→aggregate→decrypt).
2. `BYZANTINE_HEAD_ONLY=True` was hardcoded globally, contaminating a "just HE" ablation with the wrong attack type — fixed, now set explicitly per mode (`False` for pure_dp/pure_he, `True` for pure_zkp).
3. `USE_ZKP=True` never called the real `zkp.py` (see Contradiction #2 above) — fixed, `pure_zkp` now genuinely runs the ciphertext-bound HMAC head-norm guard in isolation with the correct MAD threshold.

**Results (`FL-IDS_Ablation_Analysis.md`):** standalone ZKP guard — 100% detection, 0% false positives, both models, all 25 rounds, no Krum involved; fully absorbs the attack's utility cost (network `pure_zkp` beat its own clean baseline); faster than `pure_he` (fewer clients aggregated).
**Results (`FL-IDS_Epsilon_Sweep_Analysis.md`):** extends Krum-robustness-under-DP down to ε=0.5; retracts the ε=9 sweet-spot claim (real best: ε=14, both models); flags the calibration-error growth at low epsilon (see above).

**Both reports carry explicit caveats:** single-seed noise, recipe differences between ablation conditions, and what the guard-alone result does/doesn't imply about Krum's own role — flagged as not yet citable without a repeat run for anything beyond the stated headline numbers.

---

## ⭐ EXPERIMENT 2 — HE vs. KRUM — COMPLETE (Week 9)

**Status change: no longer blocked/next-step (as v5 described it) — this experiment ran to completion in Week 9.**

### What was built

- **Hybrid aggregation branch (`USE_HE_KRUM_HYBRID`):** splits each client's parameters via `split_sensitive_bulk()` into "sensitive" (`classifier.*`, CKKS-encrypted) and "bulk" (everything else, plaintext). Adaptive Krum scores only the plaintext slice; the encrypted slice is aggregated afterward, restricted to whichever clients Krum selected from plaintext evidence alone. This also fixed the old full-model `USE_HE` path's two known bugs (no decrypt-before-return, unweighted averaging) in the new path, without touching the old one.
- **Attack bug found and fixed first, before trusting any results:** the original `classifier_head_flip_attack` call site skipped local training entirely for Byzantine clients (returning last round's unmodified global backbone). This made detection trivial for the wrong reason. Fixed: Byzantine clients train normally first; only the trained head is flipped/scaled before encryption.

### Headline result — hypothesis confirmed

**0% Byzantine detection, every round, both models, with the corrected attack.** `krum_score_ratio` stayed flat at ≈0.38 (application) / ≈0.237 (network) across all 25 rounds — the attackers didn't just blend in, they scored as *more* trustworthy than the average honest client. Best F1-Macro collapsed from a 0.73/0.83 clean baseline to 0.13 (application) / 0.72 (network); `Normal` F1 (application) and `Vulnerability_scanner` F1 (network) sat at 0.0000 every round while aggregate accuracy still climbed past 90% on the network model — a clean demonstration that aggregate accuracy can hide a fully destroyed class.

This is exactly the blind spot v5 hypothesized: Krum, seeing only the plaintext slice, cannot detect an attack confined to the encrypted classifier head.

### Anomaly surfaced

A persistent honest-client exclusion anomaly on the network model — clients 4/5/10 dropped every round regardless of attack — connecting to (and having grown from) the earlier, still-unresolved Condition 5 anomaly (1 excluded client → 3). Round-25 instability recurred a 5th documented time; round 24 recommended as the headline number instead.

### Mitigation designed, built, and confirmed (see Layer 2 above)

The ciphertext-bound HMAC head-norm guard, run as a MAD-threshold pre-filter before Krum. **Confirmed via five runs:** 100% detection in every one, across both models and three attacked-client configurations. `Vulnerability_scanner` F1 recovered from 0.0000 to 0.7864 (matching/exceeding baseline); application per-class F1 recovered broadly. Best F1-Macro within ~0.001 of the clean baseline for default-client runs.

**Cross-configuration finding:** Krum's *extra* (non-attacker) exclusions are consistently the same clients (network: 4, 10; application: 6, 7) across every configuration tested, including ones that don't target them — strong evidence the exclusion is driven by partition size/composition, not the attack. Confirmed via a new `print_data_split()` diagnostic: clients 4/10 (network) hold 3–6× the fleet-median sample count and ~73% of the entire `Vulnerability_scanner` class between them.

**New CLI/config additions:** `--byzantine <clients>`, `--krum-k <float>`; `ADAPTIVE_KRUM_K` default raised 2.5→3.5. **Correction (this session):** the "new `ADAPTIVE_KRUM_HYBRID_ASSUMED_F` knob" reported in the weekly log is not actually CLI-configurable in the reviewed code — it's hardcoded to `min(1, NUM_BYZANTINE)`, which always equals 1 (see Contradiction #9). Also note `--krum-k` drives *both* the norm guard's and Krum's MAD thresholds together, not independently (Contradiction #10).

Manifests, configs, metrics, and run notes assembled for all five mitigated runs plus a cross-configuration comparison table.

### Independent verification against real CSVs (this session, re-confirmed by a second pass)

Four of the five mitigated runs' uploaded, non-reconstructed CSVs were checked directly against the manifest's console-log-reconstructed numbers and **match almost exactly** — a second, independent pass reproduced this to higher decimal precision: application default 0.72857@24 (manifest: 0.7286@24), application_byz4_10 0.69542@23 (manifest: 0.6954@23), network_byz2_7 0.82372@21 (manifest: 0.8237@21), network_byz4_10 0.83597@20 (manifest: 0.8360@20) — all match to the manifest's stated precision. Same for the round-25 drop/no-drop pattern for each, the `network_byz4_10` volatility sequence (0.836, 0.5357, 0.8288, 0.62, 0.7817, 0.5914 for rounds 20–25 — matches to 3 decimals), the application-default per-class recovery numbers at round 24 (Normal 0.939, Backdoor 0.950, XSS 0.448, Password 0.426, Fingerprinting 0.799), and 100% detection confirmed directly from the `krum_detected_byzantine` column on every mean row of every file. The `network_byz2_7` round-24 anomaly was also confirmed on the `krum_selected` count itself: it drops from 6→5 at round 24 and recovers to 6 at round 25, independently confirming *something* extra was excluded that specific round, on top of the F1 sequence (0.8096→0.6603→0.7706) matching exactly. This is a genuine, twice-independently-reproduced confirmation that the reported Week 9 Experiment 2 numbers are real telemetry, not just self-consistent reconstructions, for four of five runs.

**One nuance surfaced on a second pass, not a retraction (see Contradiction #13 for full detail):** a second review of the same CSVs argued that client identity for the round-24 `network_byz2_7` anomaly ("client 5, not client index 9") isn't derivable from these files, since there's no explicit per-round exclusion-*identity* column, only the aggregate `krum_selected` count. This is only a partial disagreement — the identity claim in Correction 1 below was reproduced directly from the CSV's **per-client** `krum_selected` values (0/1 per client per round; filtering to client 5's own row shows exactly one round, 24, where it reads 0), which the second pass appears not to have queried at that granularity. Both passes agree the *event* (an extra exclusion at round 24) is solidly confirmed by the count drop; only the *identity* claim had a second opinion raised against it, and the per-client column data available in this conversation supports "client 5." Recommendation carried forward from that second pass regardless: **the CSV export should add an explicit `krum_discarded_ids`-style per-round list column** so future identity claims don't require cross-referencing per-client rows by hand.

**A second independent pass also confirmed, from the CSV columns themselves, that these are DP-free runs:** `dp_epsilon_spent`/`dp_epsilon_target`/`dp_noise_multiplier` are `N/A` in every row of all four non-empty files — consistent with, and now verified from the raw data rather than just the documented config, that `USE_HE_KRUM_HYBRID` (Experiment 2) runs with `USE_DP=False`.

**Two corrections surfaced by this check (see Contradiction #11 and #13 for full detail):**
1. The `network_he_krum_hybrid_norm_guard_byz2_7_v1` round-24 anomaly write-up misidentifies the transiently-excluded client — real data shows it was client 5 (0-indexed 4), not "client index 9" as the manifest states (see the nuance above — this remains the better-supported reading, but a second pass flagged the derivation as non-obvious from these columns alone).
2. The application default run has a previously undocumented transient exclusion of client 9 at rounds 3–4.

**One run remains fully unverified:** `network_he_krum_hybrid_norm_guard_v1` (the default-Byzantine network run, clients 1/2) — its uploaded `results_network.csv` is completely empty (header only). Its "complete_with_anomalies" status, including the claimed rounds 9–13 console-log gap, has not been checked against any real data; treat this run's numbers as still resting entirely on console-log reconstruction. **Recovering this specific CSV from the DGX is the one clear, agreed-upon remaining gap in Experiment 2 — not a full re-run of the experiment** (see Contradiction #14: the standalone-`USE_ZKP` bug that motivated re-checking everything never actually touched this code path).

**Distinctness from the `pure_zkp` ablation (Week 10) — do not conflate the two (see Contradiction #12):** `USE_HE_KRUM_HYBRID` (this experiment) and `USE_ZKP` (`pure_zkp`) share the guard-verification functions but are different aggregation pipelines — the hybrid runs the guard *then* Krum on survivors; `pure_zkp` runs the guard *then aggregates directly, no Krum call at all*. The code's own comments explicitly warn their detection rates aren't comparable. A strong `pure_zkp` result is not evidence that Experiment 2 doesn't need its own verification, and vice versa — the two are a deliberate contrast (does Krum do independent work behind the guard?), not a redundant pair. **This has been re-confirmed via a second, independent code trace plus an empirical data-pattern check (Contradiction #14)** — the `USE_ZKP` bug that was fixed in Week 10 never lived in the `USE_HE_KRUM_HYBRID` branch at any point, so Experiment 2 does not need to be re-run on account of that bug. **The only outstanding Experiment 2 gaps are: (1) the missing `results_network.csv`, and (2) repeat-seed confirmation for all five runs — both data-recovery/robustness tasks, not correctness concerns.**

---

## ⭐ EXPERIMENT 3 — Privacy Configuration Checkpoint Manifest

**Status: still not formally built, through Week 10.** Nothing in Weeks 9–10 constructed the proposed `models/manifest.json` system, though Week 9's "assembled manifests, configs, metrics, and run notes for all five mitigated runs plus a cross-configuration comparison table" is a step in this direction, done ad hoc rather than via the systematic indexing structure originally proposed.

### Still needed (unchanged from v5)
A `models/` directory + `manifest.json` indexing every completed run (checkpoint path, config, metrics) so results are reloadable rather than reconstructed by hand from filenames. The "6 combinations" table (baseline/+Krum/+DP/+DP+Krum/+HE/+HE+Krum) still has real gaps: standalone +Krum (no attack, no DP), standalone +DP (no Krum, no attack), and a valid post-label-fix +HE (no Krum, no attack) ablation have not been run as clean rows — Experiment 1's +DP+Krum rows are specifically *under* Byzantine attack, and the ablation-mode runs (Week 10) are each single-mechanism-under-attack, not the clean no-attack ablation this table implies.
**Check first:** whether `scripts/build_manifest.py` (referenced but not reviewed) already covers some of this before writing a redundant tool.

---

## Code Audit — Dead Code vs. Keep (new, this session)

A second-pass, cross-referenced audit of all 11 uploaded source files (`krum.py`, `zkp.py`, `data_loader.py`, `main.py`, `byzantine.py`, `byzantine_fixed.py`, `he_aggregation.py`, `he_local.py`, `local_dp.py`, `model_defs.py`, `task.py`) — every function definition grepped against every call site project-wide, not just checked against this master doc's existing claims.

### 🔴 Critical finding — see Contradiction #15

`multi_krum()` (fixed-m) is imported and called by `main.py` but **does not exist** in the uploaded `krum.py`. Currently masked (no reviewed `ABLATION_MODE` preset sets `USE_KRUM=True`), but a landmine for the first hand-edited fixed-m comparison run. **Do not attempt a fixed-m Krum run until this is resolved** — either restore the function from an earlier revision, or formally retire the `USE_KRUM` branch and correct this doc's constraints accordingly.

### 🟡 Dead code — safe to remove or consolidate

| File / symbol | Status |
|---|---|
| `byzantine.py`: `sign_flip_attack()`, `gaussian_attack()` (untrained versions) | Imported into `main.py` but never called — only the `_trained` variants and `zero_gradient_attack`/`classifier_head_flip_attack` are used. Docstrings already say "DO NOT USE FOR NEW EXPERIMENTS" — keep the functions for reproducing old (pre-fix) runs, but the live unconditional import in `main.py` is misleading; consider a lazy/on-demand import instead so it's clear these aren't part of the active path. |
| `byzantine_fixed.py` (whole file) | Zero references anywhere in the codebase — not imported by `main.py` or anything else. Diffing against `byzantine.py` shows it's a stale earlier snapshot: missing `gaussian_attack_trained()` entirely, and its `gaussian_attack()` docstring is out of date. Looks like an abandoned edit-pass artifact. **Recommendation: delete** — keeping two near-identical attack modules risks someone importing the stale one by mistake. |
| `local_dp.py` (whole module) | Not imported by `main.py` at all — the actual DP path goes straight through Opacus's `PrivacyEngine` (per-sample DP-SGD). This module's one-shot output-perturbation approach (`clip_gradient`/`gaussian_noise`/`apply_local_dp`) is referenced only in its own docstring and one comparison mention inside `zkp.py`. Not broken — a coherent, documented alternative mechanism — but currently orphaned from the executable pipeline. **Keep only if a genuine "local/edge-gateway DP" ablation is planned**; otherwise mark clearly as reference-only so it doesn't get assumed-active. |
| `get_round_lr()` in `main.py` | Confirmed dead — defined, never called (LR decay disabled by design, matches this doc's existing claim). Fine to leave as documented-but-inert, or delete if LR decay definitely won't return. |

### 🟢 Confirmed working, matches this doc's claims — no drift found

- `adaptive_multi_krum()` (`krum.py`) — solid, matches every doc claim, good diagnostics contract, actively called from three places in `main.py`.
- `zkp.py` (Parts 1 + 2) — clean split between the norm-proof (Part 1, used by `local_dp`-adjacent flows) and the ciphertext-bound head-norm guard (Part 2, actively wired through `he_local.encrypt_params_with_norm_guard` → `main.py`'s `USE_HE_KRUM_HYBRID`/`USE_ZKP` branches). The old, broken `zkp_verify_norm()`/`ZKP_MAX_NORM` are genuinely gone, confirmed via grep — not just commented out.
- `he_aggregation.py` + `he_local.py` — consistent interface; every function `he_local.py` calls (`create_server_context`, `encrypt_param_list`, `aggregate_encrypted_param_lists`, `decrypt_param_list`, `plaintext_weighted_sum`) exists and matches signatures. The old buggy `he_aggregate()` is actually deleted, not lingering.
- `data_loader.py` — clean; `LabelEncoder` genuinely absent; `ALL_CLASSES`/`_class_to_idx` mapping matches this doc's corrected indices exactly.
- `model_defs.py` / `task.py` — the documented GPU/DP fixes are actually in the code (`register_buffer` for `FocalLoss.weight`, explicit `device=` kwargs throughout, `param.device`-based proximal term). No drift from the doc here.
- `sign_flip_attack_trained`, `gaussian_attack_trained`, `zero_gradient_attack`, `classifier_head_flip_attack` — all actively called from `_train_one_client`, correctly implemented, train-then-corrupt pattern confirmed in code (not just comments).

### Action items from this audit (see also Open Items below)

1. Restore or formally retire `multi_krum()` / the `USE_KRUM` branch before any fixed-m comparison run is attempted.
2. Delete `byzantine_fixed.py` (stale, unreferenced, out of date).
3. Decide `local_dp.py`'s fate — keep as a documented reference-only module, or wire it into a real ablation.
4. Add a `krum_discarded_ids`-style per-round column to the CSV export (see Contradiction #13) so future exclusion-identity claims don't depend on manual per-client cross-referencing.
5. Reconcile the `ABLATION_MODE` preset count — this doc's own direct code trace found three presets (`pure_dp`, `pure_he`, `pure_zkp`); this session's second audit pass referenced a fourth (`krum_dp_sweep`) without confirming it against the same file. Check which `main.py` revision is authoritative before citing preset names.

---

## Open Items — Current State (v8)

### Resolved since v5
- ~~Experiment 2 blocked on hybrid aggregation code~~ → **built and run, Week 9.**
- ~~`he_aggregate()`'s two known bugs~~ → **fixed for the hybrid path (Week 9) and for `pure_he` (Week 10).**
- ~~Layer 2 extension "not yet designed"~~ → **designed, built, and validated, Week 9.**
- ~~Standalone `USE_ZKP` ablation flag~~ → **found broken and fixed to route through the real mechanism, Week 10.**
- ~~Are Experiment 2's Week 9 numbers real or just self-consistent reconstructions?~~ → **independently verified against real, non-reconstructed CSVs for 4 of 5 runs, twice over (two separate review passes this session) — they check out almost exactly**, modulo the identity-derivation nuance in Contradiction #13.
- ~~Is `pure_zkp` a substitute for or validation of Experiment 2?~~ → **No — confirmed via direct code trace that they are different aggregation pipelines and the code itself warns against comparing them (Contradiction #12).**
- ~~Did the standalone-`USE_ZKP` bug (fixed Week 10) ever contaminate Experiment 2's `USE_HE_KRUM_HYBRID` numbers?~~ → **No — confirmed via a second, independent code trace (two separate `elif` branches, never sharing the broken code path) and via the data's own behavior pattern (Contradiction #14). Experiment 2 does not need re-running on account of this bug.**

### Re-opened / downgraded since v5-v6 (carried from v7)
- **"ε=9 sweet spot retracted, real best is ε=14"** — treated as resolved in v6; **not verified** — sourced from `FL-IDS_Epsilon_Sweep_Analysis.md`, not yet produced/reviewed, and the "v2" analysis document that was supposed to carry the corrected sweep turns out to be the old pre-fix data. Downgrade to "reported in the weekly log, unverified."
- **"Experiment 1 re-run with `sign_flip_attack_trained()`, corrected findings confirmed"** — **downgraded to unverified.** `Experiment1_DP_vs_Krum_Analysis_v2.md` shows every sign of being the unchanged, pre-fix dataset. **No verified document reflecting the actual `sign_flip_attack_trained()` sweep currently exists.** Both of Experiment 1's headline findings most likely still hold but are unconfirmed.

### NEW critical item this session
- **🔴 `multi_krum()` (fixed-m) is missing from the codebase entirely**, despite being imported and called by `main.py`, despite this doc's prior "never delete" constraint, and despite the "m-propagation bug confirmed fixed" claim. This is the single highest-severity open item added in this revision — it silently breaks the moment anyone runs a fixed-m comparison. See Contradiction #15 and the "Code Audit" section above.

### Still unresolved
- **Client 6 (application) Password/XSS anomaly** — `per_client_audit.py` still not run.
- **Client 4 (network) Condition-5-only exclusion anomaly** — did not recur in Experiment 1, but a related/grown version (clients 4/5/10) reappeared in Week 9's hybrid run; root cause (partition composition) is now strongly evidenced but not formally closed.
- **Condition 5's recipe-drift confound** (`PROX_MU=0.1` + LR decay vs. locked-baseline recipe) — the recommended recipe-matched rerun has still never been done.
- **`confusion_matrix.py`** — proposed repeatedly, still never run, now low-effort given recoverable checkpoints.
- **`plot_epsilon_sweep.py`** — still not written.
- **Gaussian-attack sweep's completion status** — never confirmed (Week 10).
- **Unseeded Gaussian noise / reproducibility policy** — undecided.
- **A true fixed-m-Krum epsilon sweep** — still does not exist; blocked further by Contradiction #15 (`multi_krum()` missing) on top of the earlier "entirely adaptive-Krum" gap.
- **`USE_HEAD_NORM_GUARD` fragility in `ABLATION_MODE`** — not reset per-mode; a global flip would silently break `pure_zkp`.
- **Full real-encryption end-to-end ZKP+HE test** — a stub `he_aggregation.py` combining the real `he_local.py` + real `zkp.py` (vs. stand-in fake modules) was never finished.
- **`ABLATION_MODE` is still not CLI-controllable** — switching modes requires editing the file.
- **Experiment 3's manifest system** — still not built as a systematic structure; only ad hoc per-run manifests exist.
- **Produce and review the actual corrected-attack Experiment 1 sweep.** Highest-priority Experiment-1-side open item: find or re-run the genuine `sign_flip_attack_trained()`-based sweep (3-point or 13-point) and produce a results document that isn't the recycled pre-fix analysis. Until this exists, every Experiment 1 number in this master doc should be treated as provisional. **Per the user directly (this session): this is "sweep 1," and its results are still owed.**
- **Recover `results_network.csv` (default-Byzantine network run) from the DGX.** The uploaded copy is empty; `network_he_krum_hybrid_norm_guard_v1`'s status remains console-log-only. Confirmed (this session) this is unrelated to any ZKP bug — purely a data-recovery task.
- **Correct `COMBINED_MANIFEST.md`'s round-24 anomaly writeup** for `network_he_krum_hybrid_norm_guard_byz2_7_v1` (client 5, not "index 9" — see Contradiction #13 for the nuance on how confidently this is established) and add the undocumented client-9 rounds-3–4 exclusion to the application default run's record.
- **Decide whether `HEAD_NORM_GUARD_K` and `ADAPTIVE_KRUM_K` should be split into independently-tunable CLI flags**, and whether `ADAPTIVE_KRUM_HYBRID_ASSUMED_F` should actually be made CLI-configurable (currently hardcoded to 1) — both are currently coupled/fixed in ways the weekly log describes as tunable.
- **Delete `byzantine_fixed.py`** (stale, unreferenced, out of date — see Code Audit section).
- **Decide `local_dp.py`'s fate** — keep as documented reference-only, or wire into a real ablation (see Code Audit section).
- **Add a `krum_discarded_ids`-style column to the CSV export** so per-round exclusion identity doesn't require manual per-client cross-referencing (see Contradiction #13).
- **Reconcile the `ABLATION_MODE` preset count** — three presets directly confirmed by code trace (`pure_dp`, `pure_he`, `pure_zkp`) vs. a fourth (`krum_dp_sweep`) referenced without direct confirmation in a later audit pass.
- **"Sweep 2" — user has stated this still needs to be run, but the specific target is not yet disambiguated in this conversation.** Most likely candidates: the missing default-Byzantine `network_he_krum_hybrid_norm_guard_v1` CSV, the unconfirmed Gaussian-attack sweep, or a repeat-seed run. Confirm with the user before assuming which.

---

## Constraints — Never Violate (v8)

- **Network model: 39 features. Application model: 90 features.** Directly measured; supersedes 38/40/35/52/"~80–91".
- **`LabelEncoder` is removed — never re-add it.** Use `_class_to_idx` manual mapping.
- **DDoS_TCP is class index 7** in the corrected mapping.
- **The 18% cap targets Normal, not DDoS_TCP.**
- **`multi_krum()`'s documented interface is `(aggregated_params, selected_indices)` with `m` propagated explicitly — but as of this session, the function itself is confirmed MISSING from the codebase (Contradiction #15).** `accepted_client_indices` must still be tracked in parallel wherever it's restored (ZKP compaction makes raw positions wrong) — this constraint describes the required interface for when the function is fixed, not a currently-working guarantee.
- **NaN guard must remain in `adaptive_multi_krum()`** (confirmed present and working there); the equivalent guard in `multi_krum()` cannot currently be verified since the function does not exist.
- **No LR decay, no EMA** in the baseline pipeline, Experiment 1, or the Week 10 extended sweep.
- **No Flower** — custom FedProx, direct Python; sequential in-process on GPU, `ProcessPoolExecutor` on CPU.
- **Save-best-checkpoint is implemented and confirmed working** — do not regress to round-only checkpointing.
- **`dp_safe=True` for every client (honest and Byzantine) in every DP-active run.**
- **DP epsilon is per-round, not cumulative** — log `dp_epsilon_target` and `dp_epsilon_spent` as separate columns; use `dp_epsilon_target` as any sweep's x-axis.
- **Any DP-active run with `PROX_MU != 0` must use `_apply_dp_safe_prox_step`, never a loss-based proximal term.**
- **Experiment 1 was run with adaptive Krum by deliberate decision** — a separate fixed-m sweep remains not-yet-done if ever needed.
- **`multi_krum()` (fixed-m) must never be deleted or replaced by `adaptive_multi_krum()`** — this constraint's premise has been overtaken by events: as of this session, `multi_krum()` is confirmed **already missing** from the uploaded codebase while still being imported/called by `main.py`. The constraint now reads as "restore it, don't let it stay missing," not "don't delete an existing, working function." See Contradiction #15 — this is the single highest-severity open item in this revision.
- **On GPU: never create a `ProcessPoolExecutor` for client training** (fork+CUDA hang).
- **`FocalLoss.weight` must be a registered buffer, never a plain attribute.**
- **Every `train()`/`test()` call site must pass `device=` explicitly.**
- **Training recipe (`PROX_MU`, LR-decay, EMA) must be held identical whenever a delta is reported against another run.**
- **CKKS `poly_modulus_degree=4096`, 64-bit security** — state honestly.
- **Per-round DP guarantee only** — state the composition caveat honestly.
- **Repository working directory is `experiments/Current model/`.**
- **`USE_HE_KRUM_HYBRID` is now implemented and is the correct path for HE+Krum experiments** — the old mutual-exclusion assert on the *original* `USE_HE`/`USE_KRUM`/`USE_ADAPTIVE_KRUM` flags may still apply to the un-hybridized paths; do not conflate the hybrid flag with the legacy assert.
- **Any Byzantine attack function must train the client first, then corrupt the result** — confirmed necessary twice now (`classifier_head_flip_attack`, `sign_flip_attack`); check any *new* attack function against this pattern before trusting its results.
- **"ZKP" in this codebase is not a real zero-knowledge proof** — it is an HMAC-signed norm-bound commitment (three historically-conflated variants: broken standalone flag [now fixed], the base HMAC commitment, and Week 9's ciphertext-bound head-norm-guard extension). Use "norm-bound commitment" in the write-up and specify which variant when it matters.
- **`USE_HE_KRUM_HYBRID` (Experiment 2) and `USE_ZKP` (`pure_zkp`) must never be treated as interchangeable or mutually validating** — confirmed via direct code trace that they are different aggregation pipelines (guard+Krum vs. guard-only), and the code's own comments say their detection rates aren't comparable. A result from one is not evidence about the other.
- **`ADAPTIVE_KRUM_HYBRID_ASSUMED_F` is currently hardcoded to `min(1, NUM_BYZANTINE)` (always 1)** — not a CLI-configurable knob, despite being described as one in the weekly log. Do not report it as tunable without first adding a CLI argument for it.
- **`HEAD_NORM_GUARD_K` and `ADAPTIVE_KRUM_K` currently share a single `--krum-k` CLI value** — any reported "k=X" for a hybrid run moved both thresholds together; they are not independently tunable in the current code.
- **Do not treat `Experiment1_DP_vs_Krum_Analysis_v2.md`'s "v2" label as evidence the sign-flip-attack fix was applied to its data.** Its own changelog states no underlying data changed from v1; it covers the same six files/three epsilon points, not Week 10's extended 13-point grid; and it contains no mention of the attack-training bug despite an otherwise careful code review. Until a genuinely corrected-attack results document is produced and reviewed, cite Experiment 1's headline findings as "from the original, pre-fix run" — likely still true, but not re-confirmed.
- **Before citing any Week 9 Experiment 2 run's numbers, check whether its results CSV has actually been recovered from the DGX.** As of this session, four of five mitigated runs are confirmed against real data (checked independently twice); the default-Byzantine network run (`network_he_krum_hybrid_norm_guard_v1`) is still console-log-only — its uploaded CSV is empty.
- **The standalone-`USE_ZKP` bug (fixed Week 10) never affected `USE_HE_KRUM_HYBRID` (Experiment 2)** — confirmed via two separate `elif` branches in `main.py` that never shared the broken code path, plus the data's own behavior pattern (smooth F1 curves, 100% detection — incompatible with the old bug). Do not use this bug as a reason to re-run Experiment 2; the genuine remaining gaps are the missing CSV and repeat-seed confirmation, nothing else.
- **`byzantine_fixed.py` is stale, unreferenced dead code (confirmed via project-wide grep) and should be deleted**, not treated as a live alternative to `byzantine.py`.
- **`local_dp.py` is not wired into the active DP path (Opacus handles all real DP-SGD)** — treat it as reference-only unless a genuine local/edge-gateway DP ablation is explicitly planned; do not assume it's exercised by any existing run.
- **The CSV export currently lacks a `krum_discarded_ids`-style per-round identity column** — any claim about *which specific client* was excluded in a given round should note this limitation unless independently cross-checked at the per-client `krum_selected` level (see Contradiction #13).
- **Any new checkpoint-producing run should be added to a manifest at creation time**, not backfilled later — Experiment 3's whole point is preventing a repeat of the original round-20/round-22 unrecoverable-checkpoint incident.
- **Supervisor framing:** "make FL more secure and more private" — research contribution orientation, not commercial viability.
