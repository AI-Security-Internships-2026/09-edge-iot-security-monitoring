# Experiment 1 — DP vs. Krum Epsilon Sweep: Analysis Report

Generated from 32 condition file(s).

> ## ⚠ DATA-QUALITY WARNING — READ BEFORE TRUSTING ANY NUMBER BELOW
>
> 32/32 condition file(s) have more than one row per round (likely one row per client, not one row per round). Per-round metrics (Krum detection rate, rare-class wake rounds, score-ratio trend) computed from these files may be **incorrect** until this is resolved. See each file's entry under Data-Quality Notes at the bottom of this report for the exact row count detected and what was auto-applied (if anything).

## Per-Condition Summary

| model | tag | epsilon | best_accuracy | final_accuracy | last_n_acc_std | krum_detection_rate | krum_score_ratio_mean | krum_score_ratio_pct_change_total | dp_calibration_mean_pct_error | dp_calibration_max_pct_error | last_round_f1_drop_from_prior | last_round_instability_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| application | dp0p5 | 0.5000 | 0.6752 | 0.6752 | 0.0292 | 1.0000 | 100.7861 | 23.9680 | 1.0200 | 1.0200 | -0.0796 | False |
| application | dp01 | 1.0000 | 0.6593 | 0.5862 | 0.0353 | 1.0000 | 101.9300 | 26.6145 | 0.5100 | 0.5100 | 0.0731 | True |
| application | dp02 | 2.0000 | 0.7154 | 0.7154 | 0.0444 | 1.0000 | 102.0724 | 28.8097 | 0.2800 | 0.2800 | -0.0376 | False |
| application | dp03 | 3.0000 | 0.7023 | 0.6315 | 0.0475 | 1.0000 | 102.9580 | 30.2171 | 0.2300 | 0.2300 | 0.0264 | False |
| application | dp04 | 4.0000 | 0.7342 | 0.7184 | 0.0750 | 1.0000 | 103.4268 | 31.8864 | 0.1300 | 0.1300 | -0.1562 | False |
| application | dp05 | 5.0000 | 0.7675 | 0.7675 | 0.0172 | 1.0000 | 103.7939 | 32.8404 | 0.1000 | 0.1000 | -0.0018 | False |
| application | dp06 | 6.0000 | 0.7097 | 0.6306 | 0.0989 | 1.0000 | 103.3031 | 31.1319 | 0.0767 | 0.0767 | 0.0791 | True |
| application | dp07 | 7.0000 | 0.7454 | 0.5386 | 0.0854 | 1.0000 | 103.6991 | 34.7361 | 0.0929 | 0.0929 | 0.1620 | True |
| application | dp08 | 8.0000 | 0.7892 | 0.7758 | 0.0319 | 1.0000 | 104.1453 | 33.7512 | 0.0513 | 0.0513 | 0.0134 | False |
| application | dp09 | 9.0000 | 0.7633 | 0.7633 | 0.0146 | 1.0000 | 104.2660 | 34.5020 | 0.0422 | 0.0422 | -0.0156 | False |
| application | dp10 | 10.0000 | 0.7439 | 0.7038 | 0.0277 | 1.0000 | 104.5611 | 36.5397 | 0.0400 | 0.0400 | -0.0465 | False |
| application | dp11 | 11.0000 | 0.7320 | 0.7243 | 0.0287 | 1.0000 | 105.3147 | 36.9132 | 0.0445 | 0.0445 | -0.0086 | False |
| application | dp12 | 12.0000 | 0.8026 | 0.7585 | 0.0678 | 1.0000 | 105.8510 | 37.6203 | 0.0458 | 0.0458 | 0.0441 | True |
| application | dp13 | 13.0000 | 0.7748 | 0.7252 | 0.0353 | 1.0000 | 104.5578 | 36.8604 | 0.0485 | 0.0485 | 0.0451 | True |
| application | dp14 | 14.0000 | 0.8125 | 0.8108 | 0.0209 | 1.0000 | 105.5771 | 38.9120 | 0.0371 | 0.0371 | -0.0481 | False |
| application | dp15 | 15.0000 | 0.8114 | 0.8114 | 0.0134 | 1.0000 | 106.2433 | 35.3845 | 0.0400 | 0.0400 | -0.0056 | False |
| network | dp0p5 | 0.5000 | 0.8826 | 0.8826 | 0.0115 | 1.0000 | 307.1342 | 56.9498 | 1.1600 | 1.1600 | -0.0206 | False |
| network | dp01 | 1.0000 | 0.8692 | 0.8691 | 0.0054 | 1.0000 | 308.4284 | 60.7058 | 0.3600 | 0.3600 | -0.0115 | False |
| network | dp02 | 2.0000 | 0.9068 | 0.9068 | 0.0137 | 1.0000 | 311.9295 | 64.6349 | 0.2600 | 0.2600 | -0.0053 | False |
| network | dp03 | 3.0000 | 0.8929 | 0.8821 | 0.0380 | 1.0000 | 313.1624 | 69.9047 | 0.1533 | 0.1533 | -0.0759 | False |
| network | dp04 | 4.0000 | 0.9359 | 0.9340 | 0.0159 | 1.0000 | 312.7740 | 70.5583 | 0.0950 | 0.0950 | -0.0329 | False |
| network | dp05 | 5.0000 | 0.9280 | 0.9198 | 0.0073 | 1.0000 | 311.8630 | 68.8548 | 0.0840 | 0.0840 | -0.0002 | False |
| network | dp06 | 6.0000 | 0.9190 | 0.9095 | 0.0110 | 1.0000 | 314.2643 | 69.9044 | 0.0300 | 0.0300 | 0.0095 | False |
| network | dp07 | 7.0000 | 0.9347 | 0.9347 | 0.0144 | 1.0000 | 314.4657 | 71.8885 | 0.0457 | 0.0457 | -0.0120 | False |
| network | dp08 | 8.0000 | 0.9256 | 0.9256 | 0.0026 | 1.0000 | 314.6493 | 70.0990 | 0.0250 | 0.0250 | -0.0001 | False |
| network | dp09 | 9.0000 | 0.9361 | 0.9100 | 0.0191 | 1.0000 | 316.4808 | 72.7054 | 0.0278 | 0.0278 | 0.0261 | False |
| network | dp10 | 10.0000 | 0.9360 | 0.9196 | 0.0114 | 1.0000 | 313.5333 | 73.2061 | 0.0360 | 0.0360 | 0.0163 | False |
| network | dp11 | 11.0000 | 0.9304 | 0.9304 | 0.0143 | 1.0000 | 316.7708 | 72.6216 | 0.0173 | 0.0173 | -0.0204 | False |
| network | dp12 | 12.0000 | 0.9192 | 0.9030 | 0.0106 | 1.0000 | 313.1902 | 74.6254 | 0.0192 | 0.0192 | 0.0044 | False |
| network | dp13 | 13.0000 | 0.9527 | 0.9527 | 0.0099 | 1.0000 | 316.0450 | 79.2866 | 0.0085 | 0.0085 | -0.0217 | False |
| network | dp14 | 14.0000 | 0.9476 | 0.9476 | 0.0216 | 1.0000 | 316.7340 | 76.0296 | 0.0050 | 0.0050 | -0.0001 | False |
| network | dp15 | 15.0000 | 0.9436 | 0.9428 | 0.0210 | 1.0000 | 316.9036 | 76.4247 | 0.0033 | 0.0033 | 0.0008 | False |

## Headline Finding 1 — Does DP Noise Erode Krum's Detection?

**Application model:**
- krum_score_ratio at ε=0.5 (most noise): 100.8
- krum_score_ratio at ε=15.0 (least noise): 106.2
- Change, lowest→highest ε: 5.41%
- Detection rate across all ε conditions: ['1.000', '1.000', '1.000', '1.000', '1.000', '1.000', '1.000', '1.000', '1.000', '1.000', '1.000', '1.000', '1.000', '1.000', '1.000', '1.000'] (100% in every condition — Krum unaffected by DP noise in this range)
- **Verdict:** Score ratio moved by more than a marginal amount across the ε range — worth checking whether this is a real DP↔Krum interaction or run-to-run noise.

**Network model:**
- krum_score_ratio at ε=0.5 (most noise): 307.1
- krum_score_ratio at ε=15.0 (least noise): 316.9
- Change, lowest→highest ε: 3.18%
- Detection rate across all ε conditions: ['1.000', '1.000', '1.000', '1.000', '1.000', '1.000', '1.000', '1.000', '1.000', '1.000', '1.000', '1.000', '1.000', '1.000', '1.000', '1.000'] (100% in every condition — Krum unaffected by DP noise in this range)
- **Verdict:** Consistent with the *negative* result documented in the project write-up: DP noise does not measurably erode Krum's separation in this ε range.

## Headline Finding 2 — Does DP Noise Delay/Suppress Rare-Class Learning?

**application / ε=1.0 (dp01):**
- Fingerprinting: wakes at round never, final=0.000, max=0.000, rounds at ~zero=25
- SQL_injection: wakes at round 1, final=0.286, max=0.665, rounds at ~zero=0
- Uploading: wakes at round 4, final=0.258, max=0.293, rounds at ~zero=3
- XSS: wakes at round 7, final=0.161, max=0.229, rounds at ~zero=6

**application / ε=2.0 (dp02):**
- Fingerprinting: wakes at round 20, final=0.506, max=0.506, rounds at ~zero=19
- SQL_injection: wakes at round 1, final=0.756, max=0.756, rounds at ~zero=0
- Uploading: wakes at round 3, final=0.269, max=0.344, rounds at ~zero=2
- XSS: wakes at round 6, final=0.259, max=0.260, rounds at ~zero=5

**application / ε=3.0 (dp03):**
- Fingerprinting: wakes at round 20, final=0.542, max=0.542, rounds at ~zero=19
- SQL_injection: wakes at round 1, final=0.673, max=0.697, rounds at ~zero=0
- Uploading: wakes at round 3, final=0.283, max=0.327, rounds at ~zero=2
- XSS: wakes at round 7, final=0.260, max=0.261, rounds at ~zero=6

**application / ε=4.0 (dp04):**
- Fingerprinting: wakes at round 21, final=0.399, max=0.399, rounds at ~zero=20
- SQL_injection: wakes at round 1, final=0.747, max=0.747, rounds at ~zero=0
- Uploading: wakes at round 3, final=0.347, max=0.357, rounds at ~zero=2
- XSS: wakes at round 5, final=0.258, max=0.261, rounds at ~zero=4

**application / ε=5.0 (dp05):**
- Fingerprinting: wakes at round 18, final=0.567, max=0.567, rounds at ~zero=17
- SQL_injection: wakes at round 1, final=0.789, max=0.796, rounds at ~zero=0
- Uploading: wakes at round 2, final=0.456, max=0.469, rounds at ~zero=1
- XSS: wakes at round 6, final=0.270, max=0.282, rounds at ~zero=5

**application / ε=6.0 (dp06):**
- Fingerprinting: wakes at round 14, final=0.600, max=0.600, rounds at ~zero=13
- SQL_injection: wakes at round 1, final=0.274, max=0.651, rounds at ~zero=0
- Uploading: wakes at round 3, final=0.298, max=0.347, rounds at ~zero=2
- XSS: wakes at round 4, final=0.260, max=0.260, rounds at ~zero=3

**application / ε=7.0 (dp07):**
- Fingerprinting: wakes at round 16, final=0.555, max=0.565, rounds at ~zero=15
- SQL_injection: wakes at round 1, final=0.311, max=0.694, rounds at ~zero=0
- Uploading: wakes at round 3, final=0.287, max=0.378, rounds at ~zero=2
- XSS: wakes at round 4, final=0.260, max=0.260, rounds at ~zero=3

**application / ε=8.0 (dp08):**
- Fingerprinting: wakes at round 15, final=0.605, max=0.607, rounds at ~zero=14
- SQL_injection: wakes at round 1, final=0.762, max=0.776, rounds at ~zero=0
- Uploading: wakes at round 3, final=0.428, max=0.448, rounds at ~zero=2
- XSS: wakes at round 4, final=0.306, max=0.311, rounds at ~zero=3

**application / ε=9.0 (dp09):**
- Fingerprinting: wakes at round 17, final=0.620, max=0.620, rounds at ~zero=16
- SQL_injection: wakes at round 1, final=0.785, max=0.785, rounds at ~zero=0
- Uploading: wakes at round 3, final=0.419, max=0.424, rounds at ~zero=2
- XSS: wakes at round 5, final=0.260, max=0.260, rounds at ~zero=4

**application / ε=0.5 (dp0p5):**
- Fingerprinting: wakes at round never, final=0.000, max=0.000, rounds at ~zero=25
- SQL_injection: wakes at round 1, final=0.644, max=0.648, rounds at ~zero=0
- Uploading: wakes at round 4, final=0.177, max=0.221, rounds at ~zero=3
- XSS: wakes at round 10, final=0.243, max=0.243, rounds at ~zero=9

**application / ε=10.0 (dp10):**
- Fingerprinting: wakes at round 16, final=0.587, max=0.587, rounds at ~zero=15
- SQL_injection: wakes at round 1, final=0.672, max=0.734, rounds at ~zero=0
- Uploading: wakes at round 2, final=0.393, max=0.407, rounds at ~zero=1
- XSS: wakes at round 5, final=0.259, max=0.260, rounds at ~zero=4

**application / ε=11.0 (dp11):**
- Fingerprinting: wakes at round 13, final=0.596, max=0.596, rounds at ~zero=13
- SQL_injection: wakes at round 1, final=0.588, max=0.717, rounds at ~zero=0
- Uploading: wakes at round 3, final=0.389, max=0.396, rounds at ~zero=2
- XSS: wakes at round 4, final=0.260, max=0.260, rounds at ~zero=3

**application / ε=12.0 (dp12):**
- Fingerprinting: wakes at round 11, final=0.649, max=0.649, rounds at ~zero=10
- SQL_injection: wakes at round 1, final=0.792, max=0.792, rounds at ~zero=0
- Uploading: wakes at round 3, final=0.372, max=0.445, rounds at ~zero=2
- XSS: wakes at round 3, final=0.234, max=0.278, rounds at ~zero=2

**application / ε=13.0 (dp13):**
- Fingerprinting: wakes at round 14, final=0.665, max=0.665, rounds at ~zero=13
- SQL_injection: wakes at round 1, final=0.738, max=0.805, rounds at ~zero=0
- Uploading: wakes at round 3, final=0.401, max=0.474, rounds at ~zero=2
- XSS: wakes at round 5, final=0.260, max=0.261, rounds at ~zero=4

**application / ε=14.0 (dp14):**
- Fingerprinting: wakes at round 12, final=0.615, max=0.615, rounds at ~zero=11
- SQL_injection: wakes at round 1, final=0.804, max=0.804, rounds at ~zero=0
- Uploading: wakes at round 2, final=0.487, max=0.487, rounds at ~zero=1
- XSS: wakes at round 3, final=0.352, max=0.431, rounds at ~zero=2

**application / ε=15.0 (dp15):**
- Fingerprinting: wakes at round 13, final=0.617, max=0.617, rounds at ~zero=12
- SQL_injection: wakes at round 1, final=0.796, max=0.813, rounds at ~zero=0
- Uploading: wakes at round 2, final=0.464, max=0.505, rounds at ~zero=1
- XSS: wakes at round 3, final=0.294, max=0.294, rounds at ~zero=2

**network / ε=1.0 (dp01):**
- MITM: wakes at round 13, final=0.408, max=0.408, rounds at ~zero=12

**network / ε=2.0 (dp02):**
- MITM: wakes at round 10, final=0.417, max=0.432, rounds at ~zero=9

**network / ε=3.0 (dp03):**
- MITM: wakes at round 8, final=0.403, max=0.446, rounds at ~zero=7

**network / ε=4.0 (dp04):**
- MITM: wakes at round 11, final=0.355, max=0.390, rounds at ~zero=10

**network / ε=5.0 (dp05):**
- MITM: wakes at round 8, final=0.441, max=0.458, rounds at ~zero=7

**network / ε=6.0 (dp06):**
- MITM: wakes at round 7, final=0.298, max=0.458, rounds at ~zero=6

**network / ε=7.0 (dp07):**
- MITM: wakes at round 16, final=0.429, max=0.429, rounds at ~zero=15

**network / ε=8.0 (dp08):**
- MITM: wakes at round 6, final=0.431, max=0.466, rounds at ~zero=5

**network / ε=9.0 (dp09):**
- MITM: wakes at round 6, final=0.404, max=0.428, rounds at ~zero=5

**network / ε=0.5 (dp0p5):**
- MITM: wakes at round never, final=0.000, max=0.000, rounds at ~zero=25

**network / ε=10.0 (dp10):**
- MITM: wakes at round 7, final=0.392, max=0.428, rounds at ~zero=6

**network / ε=11.0 (dp11):**
- MITM: wakes at round 9, final=0.271, max=0.277, rounds at ~zero=8

**network / ε=12.0 (dp12):**
- MITM: wakes at round 5, final=0.310, max=0.431, rounds at ~zero=4

**network / ε=13.0 (dp13):**
- MITM: wakes at round 6, final=0.453, max=0.456, rounds at ~zero=5

**network / ε=14.0 (dp14):**
- MITM: wakes at round 6, final=0.447, max=0.471, rounds at ~zero=5

**network / ε=15.0 (dp15):**
- MITM: wakes at round 7, final=0.451, max=0.462, rounds at ~zero=6

Compare `wake_round` and `rounds at ~zero` across ε for the same model/class: if lower ε consistently wakes later / stays at zero longer, that reproduces the documented dose-dependent rare-class suppression effect.

## F1-Macro / Accuracy vs. Epsilon — Monotonicity Check

**Application model** (using Accuracy — f1_macro unavailable, falling back to accuracy — if this says Accuracy, fix the f1_macro column alias first):
- Epsilons (ascending): [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
- Best Accuracy per epsilon: ['0.6752', '0.6593', '0.7154', '0.7023', '0.7342', '0.7675', '0.7097', '0.7454', '0.7892', '0.7633', '0.7439', '0.7320', '0.8026', '0.7748', '0.8125', '0.8114']
- **NOT monotonic** — flagged violation(s):
  - ε=0.5 (Accuracy=0.6752) → ε=1.0 (Accuracy=0.6593): Accuracy dropped despite higher ε. Treat as needing repeat-seed confirmation before calling it a genuine sweet spot.
  - ε=2.0 (Accuracy=0.7154) → ε=3.0 (Accuracy=0.7023): Accuracy dropped despite higher ε. Treat as needing repeat-seed confirmation before calling it a genuine sweet spot.
  - ε=5.0 (Accuracy=0.7675) → ε=6.0 (Accuracy=0.7097): Accuracy dropped despite higher ε. Treat as needing repeat-seed confirmation before calling it a genuine sweet spot.
  - ε=8.0 (Accuracy=0.7892) → ε=9.0 (Accuracy=0.7633): Accuracy dropped despite higher ε. Treat as needing repeat-seed confirmation before calling it a genuine sweet spot.
  - ε=9.0 (Accuracy=0.7633) → ε=10.0 (Accuracy=0.7439): Accuracy dropped despite higher ε. Treat as needing repeat-seed confirmation before calling it a genuine sweet spot.
  - ε=10.0 (Accuracy=0.7439) → ε=11.0 (Accuracy=0.7320): Accuracy dropped despite higher ε. Treat as needing repeat-seed confirmation before calling it a genuine sweet spot.
  - ε=12.0 (Accuracy=0.8026) → ε=13.0 (Accuracy=0.7748): Accuracy dropped despite higher ε. Treat as needing repeat-seed confirmation before calling it a genuine sweet spot.
  - ε=14.0 (Accuracy=0.8125) → ε=15.0 (Accuracy=0.8114): Accuracy dropped despite higher ε. Treat as needing repeat-seed confirmation before calling it a genuine sweet spot.

**Network model** (using Accuracy — f1_macro unavailable, falling back to accuracy — if this says Accuracy, fix the f1_macro column alias first):
- Epsilons (ascending): [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
- Best Accuracy per epsilon: ['0.8826', '0.8692', '0.9068', '0.8929', '0.9359', '0.9280', '0.9190', '0.9347', '0.9256', '0.9361', '0.9360', '0.9304', '0.9192', '0.9527', '0.9476', '0.9436']
- **NOT monotonic** — flagged violation(s):
  - ε=0.5 (Accuracy=0.8826) → ε=1.0 (Accuracy=0.8692): Accuracy dropped despite higher ε. Treat as needing repeat-seed confirmation before calling it a genuine sweet spot.
  - ε=2.0 (Accuracy=0.9068) → ε=3.0 (Accuracy=0.8929): Accuracy dropped despite higher ε. Treat as needing repeat-seed confirmation before calling it a genuine sweet spot.
  - ε=4.0 (Accuracy=0.9359) → ε=5.0 (Accuracy=0.9280): Accuracy dropped despite higher ε. Treat as needing repeat-seed confirmation before calling it a genuine sweet spot.
  - ε=5.0 (Accuracy=0.9280) → ε=6.0 (Accuracy=0.9190): Accuracy dropped despite higher ε. Treat as needing repeat-seed confirmation before calling it a genuine sweet spot.
  - ε=7.0 (Accuracy=0.9347) → ε=8.0 (Accuracy=0.9256): Accuracy dropped despite higher ε. Treat as needing repeat-seed confirmation before calling it a genuine sweet spot.
  - ε=9.0 (Accuracy=0.9361) → ε=10.0 (Accuracy=0.9360): Accuracy dropped despite higher ε. Treat as needing repeat-seed confirmation before calling it a genuine sweet spot.
  - ε=10.0 (Accuracy=0.9360) → ε=11.0 (Accuracy=0.9304): Accuracy dropped despite higher ε. Treat as needing repeat-seed confirmation before calling it a genuine sweet spot.
  - ε=11.0 (Accuracy=0.9304) → ε=12.0 (Accuracy=0.9192): Accuracy dropped despite higher ε. Treat as needing repeat-seed confirmation before calling it a genuine sweet spot.
  - ε=13.0 (Accuracy=0.9527) → ε=14.0 (Accuracy=0.9476): Accuracy dropped despite higher ε. Treat as needing repeat-seed confirmation before calling it a genuine sweet spot.
  - ε=14.0 (Accuracy=0.9476) → ε=15.0 (Accuracy=0.9436): Accuracy dropped despite higher ε. Treat as needing repeat-seed confirmation before calling it a genuine sweet spot.

## Final-Round Instability Check

- **application / ε=1.0 (dp01)**: accuracy dropped by 0.0731 into the checked round — consider reporting best-round rather than final-round numbers for this run.
- **application / ε=6.0 (dp06)**: accuracy dropped by 0.0791 into the checked round — consider reporting best-round rather than final-round numbers for this run.
- **application / ε=7.0 (dp07)**: accuracy dropped by 0.1620 into the checked round — consider reporting best-round rather than final-round numbers for this run.
- **application / ε=12.0 (dp12)**: accuracy dropped by 0.0441 into the checked round — consider reporting best-round rather than final-round numbers for this run.
- **application / ε=13.0 (dp13)**: accuracy dropped by 0.0451 into the checked round — consider reporting best-round rather than final-round numbers for this run.

## DP Calibration Accuracy

- application / ε=1.0: mean error 0.510%, max error 0.510%
- application / ε=2.0: mean error 0.280%, max error 0.280%
- application / ε=3.0: mean error 0.230%, max error 0.230%
- application / ε=4.0: mean error 0.130%, max error 0.130%
- application / ε=5.0: mean error 0.100%, max error 0.100%
- application / ε=6.0: mean error 0.077%, max error 0.077%
- application / ε=7.0: mean error 0.093%, max error 0.093%
- application / ε=8.0: mean error 0.051%, max error 0.051%
- application / ε=9.0: mean error 0.042%, max error 0.042%
- application / ε=0.5: mean error 1.020%, max error 1.020%
- application / ε=10.0: mean error 0.040%, max error 0.040%
- application / ε=11.0: mean error 0.045%, max error 0.045%
- application / ε=12.0: mean error 0.046%, max error 0.046%
- application / ε=13.0: mean error 0.048%, max error 0.048%
- application / ε=14.0: mean error 0.037%, max error 0.037%
- application / ε=15.0: mean error 0.040%, max error 0.040%
- network / ε=1.0: mean error 0.360%, max error 0.360%
- network / ε=2.0: mean error 0.260%, max error 0.260%
- network / ε=3.0: mean error 0.153%, max error 0.153%
- network / ε=4.0: mean error 0.095%, max error 0.095%
- network / ε=5.0: mean error 0.084%, max error 0.084%
- network / ε=6.0: mean error 0.030%, max error 0.030%
- network / ε=7.0: mean error 0.046%, max error 0.046%
- network / ε=8.0: mean error 0.025%, max error 0.025%
- network / ε=9.0: mean error 0.028%, max error 0.028%
- network / ε=0.5: mean error 1.160%, max error 1.160%
- network / ε=10.0: mean error 0.036%, max error 0.036%
- network / ε=11.0: mean error 0.017%, max error 0.017%
- network / ε=12.0: mean error 0.019%, max error 0.019%
- network / ε=13.0: mean error 0.008%, max error 0.008%
- network / ε=14.0: mean error 0.005%, max error 0.005%
- network / ε=15.0: mean error 0.003%, max error 0.003%

## Data-Quality Notes

**results_application_dp01.csv:**
- Detected 11 rows/round via 'client' — filtered down to the summary/mean row per round. Verify this is the correct aggregate row for your metrics.
- No f1_macro column found.
**results_application_dp02.csv:**
- Detected 11 rows/round via 'client' — filtered down to the summary/mean row per round. Verify this is the correct aggregate row for your metrics.
- No f1_macro column found.
**results_application_dp03.csv:**
- Detected 11 rows/round via 'client' — filtered down to the summary/mean row per round. Verify this is the correct aggregate row for your metrics.
- No f1_macro column found.
**results_application_dp04.csv:**
- Detected 11 rows/round via 'client' — filtered down to the summary/mean row per round. Verify this is the correct aggregate row for your metrics.
- No f1_macro column found.
**results_application_dp05.csv:**
- Detected 11 rows/round via 'client' — filtered down to the summary/mean row per round. Verify this is the correct aggregate row for your metrics.
- No f1_macro column found.
**results_application_dp06.csv:**
- Detected 11 rows/round via 'client' — filtered down to the summary/mean row per round. Verify this is the correct aggregate row for your metrics.
- No f1_macro column found.
**results_application_dp07.csv:**
- Detected 11 rows/round via 'client' — filtered down to the summary/mean row per round. Verify this is the correct aggregate row for your metrics.
- No f1_macro column found.
**results_application_dp08.csv:**
- Detected 11 rows/round via 'client' — filtered down to the summary/mean row per round. Verify this is the correct aggregate row for your metrics.
- No f1_macro column found.
**results_application_dp09.csv:**
- Detected 11 rows/round via 'client' — filtered down to the summary/mean row per round. Verify this is the correct aggregate row for your metrics.
- No f1_macro column found.
**results_application_dp0p5.csv:**
- Detected 11 rows/round via 'client' — filtered down to the summary/mean row per round. Verify this is the correct aggregate row for your metrics.
- No f1_macro column found.
**results_application_dp10.csv:**
- Detected 11 rows/round via 'client' — filtered down to the summary/mean row per round. Verify this is the correct aggregate row for your metrics.
- No f1_macro column found.
**results_application_dp11.csv:**
- Detected 11 rows/round via 'client' — filtered down to the summary/mean row per round. Verify this is the correct aggregate row for your metrics.
- No f1_macro column found.
**results_application_dp12.csv:**
- Detected 11 rows/round via 'client' — filtered down to the summary/mean row per round. Verify this is the correct aggregate row for your metrics.
- No f1_macro column found.
**results_application_dp13.csv:**
- Detected 11 rows/round via 'client' — filtered down to the summary/mean row per round. Verify this is the correct aggregate row for your metrics.
- No f1_macro column found.
**results_application_dp14.csv:**
- Detected 11 rows/round via 'client' — filtered down to the summary/mean row per round. Verify this is the correct aggregate row for your metrics.
- No f1_macro column found.
**results_application_dp15.csv:**
- Detected 11 rows/round via 'client' — filtered down to the summary/mean row per round. Verify this is the correct aggregate row for your metrics.
- No f1_macro column found.
**results_network_dp01.csv:**
- Detected 11 rows/round via 'client' — filtered down to the summary/mean row per round. Verify this is the correct aggregate row for your metrics.
- No f1_macro column found.
**results_network_dp02.csv:**
- Detected 11 rows/round via 'client' — filtered down to the summary/mean row per round. Verify this is the correct aggregate row for your metrics.
- No f1_macro column found.
**results_network_dp03.csv:**
- Detected 11 rows/round via 'client' — filtered down to the summary/mean row per round. Verify this is the correct aggregate row for your metrics.
- No f1_macro column found.
**results_network_dp04.csv:**
- Detected 11 rows/round via 'client' — filtered down to the summary/mean row per round. Verify this is the correct aggregate row for your metrics.
- No f1_macro column found.
**results_network_dp05.csv:**
- Detected 11 rows/round via 'client' — filtered down to the summary/mean row per round. Verify this is the correct aggregate row for your metrics.
- No f1_macro column found.
**results_network_dp06.csv:**
- Detected 11 rows/round via 'client' — filtered down to the summary/mean row per round. Verify this is the correct aggregate row for your metrics.
- No f1_macro column found.
**results_network_dp07.csv:**
- Detected 11 rows/round via 'client' — filtered down to the summary/mean row per round. Verify this is the correct aggregate row for your metrics.
- No f1_macro column found.
**results_network_dp08.csv:**
- Detected 11 rows/round via 'client' — filtered down to the summary/mean row per round. Verify this is the correct aggregate row for your metrics.
- No f1_macro column found.
**results_network_dp09.csv:**
- Detected 11 rows/round via 'client' — filtered down to the summary/mean row per round. Verify this is the correct aggregate row for your metrics.
- No f1_macro column found.
**results_network_dp0p5.csv:**
- Detected 11 rows/round via 'client' — filtered down to the summary/mean row per round. Verify this is the correct aggregate row for your metrics.
- No f1_macro column found.
**results_network_dp10.csv:**
- Detected 11 rows/round via 'client' — filtered down to the summary/mean row per round. Verify this is the correct aggregate row for your metrics.
- No f1_macro column found.
**results_network_dp11.csv:**
- Detected 11 rows/round via 'client' — filtered down to the summary/mean row per round. Verify this is the correct aggregate row for your metrics.
- No f1_macro column found.
**results_network_dp12.csv:**
- Detected 11 rows/round via 'client' — filtered down to the summary/mean row per round. Verify this is the correct aggregate row for your metrics.
- No f1_macro column found.
**results_network_dp13.csv:**
- Detected 11 rows/round via 'client' — filtered down to the summary/mean row per round. Verify this is the correct aggregate row for your metrics.
- No f1_macro column found.
**results_network_dp14.csv:**
- Detected 11 rows/round via 'client' — filtered down to the summary/mean row per round. Verify this is the correct aggregate row for your metrics.
- No f1_macro column found.
**results_network_dp15.csv:**
- Detected 11 rows/round via 'client' — filtered down to the summary/mean row per round. Verify this is the correct aggregate row for your metrics.
- No f1_macro column found.
