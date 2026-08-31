# Docker RAM & Latency Benchmark Results

**Model:** Network-layer CNN-LSTM | **Features:** 37 (measured, 100k-row stratified subsample) | **Classes:** 8 | **Total parameters:** 79,688

**Partitioning:** 2 clients, Dirichlet(α=0.7, seed=42) — client0 = 31,332 train / 7,834 test rows, client1 = 26,566 train / 6,642 test rows

**Training:** 3 simulated FL rounds, 2 local epochs/round, batch_size=256 (512 for DP-SGD)

**Resource profiles:** both capped at 2048MB RAM (headroom, not a tight constraint) — only vCPU allocation differs: 1.0 vCPU ("unthrottled") vs 0.5 vCPU ("throttled")

**Encryption (HE modes):** CKKS via TenSEAL, poly_modulus_degree=4096, depth-1 coefficient chain (64-bit security, Docker-tuned config)

**DP-SGD:** Opacus PrivacyEngine, target ε=3.0, δ=1e-5, max_grad_norm=1.0, DP batch_size=512


> ⚠️ **Read the [Known Issues](#known-issues) section before citing anything from this document** — two columns below have documented data-quality caveats.


---
## 1. Client-Side Training & Mechanism Timing (per round)

Every timed stage on the client: local training, the mode-specific mechanism (HE encryption / DP-SGD setup / ZKP proof generation), and communication.

| Mode | Profile | Client | Round | DP Setup (s) | Train (s) | HE Encrypt (s) | ZKP Proof (s) | ZKP Norm* | N Chunks | % Encrypted | Serialize (s) | Comm. Send (s) | Payload (bytes) | Round Wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline (no defense) | 1.0 vCPU | C0 | 1 | — | 126.9766 | — | — | — | — | — | 0.1208 | 0.0911 | 1756605 | 128.0944 |
| Baseline (no defense) | 1.0 vCPU | C0 | 2 | — | 135.4661 | — | — | — | — | — | 0.1211 | 0.0814 | 1755544 | 136.6903 |
| Baseline (no defense) | 1.0 vCPU | C0 | 3 | — | 141.4477 | — | — | — | — | — | 0.1205 | 0.0814 | 1754576 | 142.6148 |
| Baseline (no defense) | 1.0 vCPU | C1 | 1 | — | 105.9923 | — | — | — | — | — | 0.1245 | 0.0990 | 1756663 | 107.2019 |
| Baseline (no defense) | 1.0 vCPU | C1 | 2 | — | 101.9660 | — | — | — | — | — | 0.1302 | 0.0771 | 1755840 | 103.3142 |
| Baseline (no defense) | 1.0 vCPU | C1 | 3 | — | 100.6436 | — | — | — | — | — | 0.1192 | 0.0749 | 1755113 | 102.2179 |
| Baseline (no defense) | 0.5 vCPU | C0 | 1 | — | 399.1857 | — | — | — | — | — | 0.2277 | 0.1103 | 1756649 | 400.7938 |
| Baseline (no defense) | 0.5 vCPU | C0 | 2 | — | 448.2425 | — | — | — | — | — | 0.2707 | 0.3679 | 1755566 | 450.0351 |
| Baseline (no defense) | 0.5 vCPU | C0 | 3 | — | 451.3578 | — | — | — | — | — | 0.1957 | 0.0796 | 1754799 | 452.6623 |
| Baseline (no defense) | 0.5 vCPU | C1 | 1 | — | 458.6799 | — | — | — | — | — | 0.2410 | 0.2336 | 1756879 | 460.2347 |
| Baseline (no defense) | 0.5 vCPU | C1 | 2 | — | 512.1855 | — | — | — | — | — | 0.1752 | 0.1273 | 1755833 | 513.4538 |
| Baseline (no defense) | 0.5 vCPU | C1 | 3 | — | 521.9777 | — | — | — | — | — | 0.1958 | 0.1258 | 1754936 | 524.0408 |
| Full-Model HE (CKKS) | 1.0 vCPU | C0 | 1 | — | 133.2797 | 0.2940 | — | — | 40 | 100.00 | 0.0304 | 0.0607 | 4562330 | 134.0500 |
| Full-Model HE (CKKS) | 1.0 vCPU | C0 | 2 | — | 132.3062 | 0.4178 | — | — | 40 | 100.00 | 0.0318 | 0.0357 | 4563330 | 132.9805 |
| Full-Model HE (CKKS) | 1.0 vCPU | C0 | 3 | — | 139.3151 | 0.2918 | — | — | 40 | 100.00 | 0.0272 | 0.0392 | 4562874 | 140.0311 |
| Full-Model HE (CKKS) | 1.0 vCPU | C1 | 1 | — | 73.8445 | 0.2791 | — | — | 40 | 100.00 | 0.0306 | 0.0550 | 4563210 | 74.5978 |
| Full-Model HE (CKKS) | 1.0 vCPU | C1 | 2 | — | 86.3320 | 0.3676 | — | — | 40 | 100.00 | 0.0407 | 0.0390 | 4562842 | 87.0644 |
| Full-Model HE (CKKS) | 1.0 vCPU | C1 | 3 | — | 86.5603 | 0.3664 | — | — | 40 | 100.00 | 0.0276 | 0.0341 | 4562738 | 87.2364 |
| Full-Model HE (CKKS) | 0.5 vCPU | C0 | 1 | — | 629.0293 | 0.6028 | — | — | 40 | 100.00 | 0.0230 | 0.0482 | 4563010 | 630.3041 |
| Full-Model HE (CKKS) | 0.5 vCPU | C0 | 2 | — | 634.3335 | 0.6311 | — | — | 40 | 100.00 | 0.0597 | 0.1830 | 4564778 | 635.7150 |
| Full-Model HE (CKKS) | 0.5 vCPU | C0 | 3 | — | 627.6369 | 0.5384 | — | — | 40 | 100.00 | 0.0272 | 0.0498 | 4562958 | 628.8790 |
| Full-Model HE (CKKS) | 0.5 vCPU | C1 | 1 | — | 204.1705 | 0.5844 | — | — | 40 | 100.00 | 0.0431 | 0.0419 | 4562642 | 205.4116 |
| Full-Model HE (CKKS) | 0.5 vCPU | C1 | 2 | — | 183.9433 | 0.6003 | — | — | 40 | 100.00 | 0.0228 | 0.0361 | 4563074 | 185.1751 |
| Full-Model HE (CKKS) | 0.5 vCPU | C1 | 3 | — | 193.3411 | 0.6072 | — | — | 40 | 100.00 | 0.0276 | 0.1344 | 4563214 | 194.5973 |
| Partial HE (classifier head only) | 1.0 vCPU | C0 | 1 | — | 130.2740 | 0.0205 | — | — | 3 | 5.84 | 0.1187 | 0.0973 | 1998512 | 131.6872 |
| Partial HE (classifier head only) | 1.0 vCPU | C0 | 2 | — | 146.6643 | 0.0260 | — | — | 3 | 5.84 | 0.1164 | 0.0806 | 1997626 | 147.8796 |
| Partial HE (classifier head only) | 1.0 vCPU | C0 | 3 | — | 145.6709 | 0.0279 | — | — | 3 | 5.84 | 0.1104 | 0.0746 | 1996471 | 146.8149 |
| Partial HE (classifier head only) | 1.0 vCPU | C1 | 1 | — | 108.9521 | 0.0215 | — | — | 3 | 5.84 | 0.1321 | 0.0922 | 1998056 | 110.3157 |
| Partial HE (classifier head only) | 1.0 vCPU | C1 | 2 | — | 111.9212 | 0.0261 | — | — | 3 | 5.84 | 0.1323 | 0.0796 | 1997682 | 113.1977 |
| Partial HE (classifier head only) | 1.0 vCPU | C1 | 3 | — | 107.8058 | 0.0973 | — | — | 3 | 5.84 | 0.1256 | 0.0790 | 1996695 | 109.0029 |
| Partial HE (classifier head only) | 0.5 vCPU | C0 | 1 | — | 613.4982 | 0.0791 | — | — | 3 | 5.84 | 0.2052 | 0.1121 | 1998283 | 615.0366 |
| Partial HE (classifier head only) | 0.5 vCPU | C0 | 2 | — | 599.3866 | 0.0280 | — | — | 3 | 5.84 | 0.2053 | 0.1082 | 1997206 | 600.8171 |
| Partial HE (classifier head only) | 0.5 vCPU | C0 | 3 | — | 601.2976 | 0.1214 | — | — | 3 | 5.84 | 0.2031 | 0.3020 | 1996683 | 603.1254 |
| Partial HE (classifier head only) | 0.5 vCPU | C1 | 1 | — | 340.3549 | 0.0287 | — | — | 3 | 5.84 | 0.2175 | 0.0977 | 1998995 | 341.9674 |
| Partial HE (classifier head only) | 0.5 vCPU | C1 | 2 | — | 362.6413 | 0.1137 | — | — | 3 | 5.84 | 0.2213 | 0.0972 | 1997664 | 364.2506 |
| Partial HE (classifier head only) | 0.5 vCPU | C1 | 3 | — | 357.1464 | 0.0286 | — | — | 3 | 5.84 | 0.1899 | 0.0807 | 1997290 | 358.6517 |
| Partial HE + Norm-Bound ZKP Guard | 1.0 vCPU | C0 | 1 | — | 157.1768 | 0.0347 | 0.0061 | 0.0000 | 3 | 5.84 | 0.1240 | 0.0857 | 1998246 | 158.5163 |
| Partial HE + Norm-Bound ZKP Guard | 1.0 vCPU | C0 | 2 | — | 150.4446 | 0.0313 | 0.0017 | 0.0000 | 3 | 5.84 | 0.1252 | 0.0730 | 1997792 | 151.7096 |
| Partial HE + Norm-Bound ZKP Guard | 1.0 vCPU | C0 | 3 | — | 157.2123 | 0.1091 | 0.0015 | 0.0000 | 3 | 5.84 | 0.1137 | 0.0764 | 1997011 | 158.3896 |
| Partial HE + Norm-Bound ZKP Guard | 1.0 vCPU | C1 | 1 | — | 109.5917 | 0.0217 | 0.0219 | 0.0000 | 3 | 5.84 | 0.1180 | 0.1108 | 1998254 | 110.9437 |
| Partial HE + Norm-Bound ZKP Guard | 1.0 vCPU | C1 | 2 | — | 122.7300 | 0.0999 | 0.0011 | 0.0000 | 3 | 5.84 | 0.1364 | 0.0805 | 1997709 | 124.0385 |
| Partial HE + Norm-Bound ZKP Guard | 1.0 vCPU | C1 | 3 | — | 125.1770 | 0.0230 | 0.0010 | 0.0000 | 3 | 5.84 | 0.1233 | 0.0772 | 1996488 | 126.4181 |
| Partial HE + Norm-Bound ZKP Guard | 0.5 vCPU | C0 | 1 | — | 292.7581 | 0.0728 | 0.0065 | 0.0000 | 3 | 5.84 | 0.1891 | 0.1854 | 1998079 | 294.3798 |
| Partial HE + Norm-Bound ZKP Guard | 0.5 vCPU | C0 | 2 | — | 261.1524 | 0.1168 | 0.0010 | 0.0000 | 3 | 5.84 | 0.1813 | 0.0954 | 1996729 | 262.6857 |
| Partial HE + Norm-Bound ZKP Guard | 0.5 vCPU | C0 | 3 | — | 274.8383 | 0.1018 | 0.0012 | 0.0000 | 3 | 5.84 | 0.1839 | 0.0881 | 1996494 | 276.3498 |
| Partial HE + Norm-Bound ZKP Guard | 0.5 vCPU | C1 | 1 | — | 209.9805 | 0.0225 | 0.0074 | 0.0000 | 3 | 5.84 | 0.2244 | 0.1260 | 1998239 | 211.7253 |
| Partial HE + Norm-Bound ZKP Guard | 0.5 vCPU | C1 | 2 | — | 236.2241 | 0.1011 | 0.0020 | 0.0000 | 3 | 5.84 | 0.1856 | 0.1883 | 1997622 | 237.9462 |
| Partial HE + Norm-Bound ZKP Guard | 0.5 vCPU | C1 | 3 | — | 233.6479 | 0.1096 | 0.0010 | 0.0000 | 3 | 5.84 | 0.1687 | 0.1065 | 1996404 | 235.1887 |
| DP-SGD (Opacus, eps=3.0) | 1.0 vCPU | C0 | 1 | 0.9665 | 702.0823 | — | — | — | — | — | 0.2404 | 0.1723 | 2829057 | 706.0779 |
| DP-SGD (Opacus, eps=3.0) | 1.0 vCPU | C0 | 2 | 1.1334 | 683.5806 | — | — | — | — | — | 0.1887 | 0.1183 | 2828080 | 686.3514 |
| DP-SGD (Opacus, eps=3.0) | 1.0 vCPU | C0 | 3 | 1.1107 | 680.4072 | — | — | — | — | — | 0.1895 | 0.1136 | 2828110 | 683.1523 |
| DP-SGD (Opacus, eps=3.0) | 1.0 vCPU | C1 | 1 | 1.2232 | 686.3960 | — | — | — | — | — | 0.1972 | 0.1477 | 2829704 | 690.5665 |
| DP-SGD (Opacus, eps=3.0) | 1.0 vCPU | C1 | 2 | 1.3640 | 677.6404 | — | — | — | — | — | 0.1975 | 0.1259 | 2829193 | 681.1296 |
| DP-SGD (Opacus, eps=3.0) | 1.0 vCPU | C1 | 3 | 1.3832 | 665.1348 | — | — | — | — | — | 0.1915 | 0.1182 | 2828752 | 668.1882 |
| DP-SGD (Opacus, eps=3.0) | 0.5 vCPU | C0 | 1 | 1.9194 | 2615.3876 | — | — | — | — | — | 0.4179 | 0.7884 | 2829752 | 2623.1978 |
| DP-SGD (Opacus, eps=3.0) | 0.5 vCPU | C0 | 2 | 2.3986 | 2620.1119 | — | — | — | — | — | 0.4100 | 0.2732 | 2829294 | 2625.9841 |
| DP-SGD (Opacus, eps=3.0) | 0.5 vCPU | C0 | 3 | 2.3972 | 2654.7875 | — | — | — | — | — | 0.4198 | 0.5677 | 2828639 | 2660.7805 |
| DP-SGD (Opacus, eps=3.0) | 0.5 vCPU | C1 | 1 | 2.5564 | 2083.7946 | — | — | — | — | — | 0.3689 | 0.4282 | 2828911 | 2091.2276 |
| DP-SGD (Opacus, eps=3.0) | 0.5 vCPU | C1 | 2 | 2.9406 | 2159.9506 | — | — | — | — | — | 0.3707 | 0.4937 | 2828295 | 2166.2572 |
| DP-SGD (Opacus, eps=3.0) | 0.5 vCPU | C1 | 3 | 2.7996 | 2141.1446 | — | — | — | — | — | 0.3772 | 0.4353 | 2827765 | 2147.3600 |

---
## 2. DP-SGD Privacy Accounting (per round)

| Profile | Client | Round | ε Target | ε Achieved | Noise Multiplier |
|---|---|---|---|---|---|
| 1.0 vCPU | C0 | 1 | 3.00 | 2.996223 | 0.798340 |
| 1.0 vCPU | C0 | 2 | 3.00 | 2.996223 | 0.798340 |
| 1.0 vCPU | C0 | 3 | 3.00 | 2.996223 | 0.798340 |
| 1.0 vCPU | C1 | 1 | 3.00 | 2.994767 | 0.822144 |
| 1.0 vCPU | C1 | 2 | 3.00 | 2.994767 | 0.822144 |
| 1.0 vCPU | C1 | 3 | 3.00 | 2.994767 | 0.822144 |
| 0.5 vCPU | C0 | 1 | 3.00 | 2.996223 | 0.798340 |
| 0.5 vCPU | C0 | 2 | 3.00 | 2.996223 | 0.798340 |
| 0.5 vCPU | C0 | 3 | 3.00 | 2.996223 | 0.798340 |
| 0.5 vCPU | C1 | 1 | 3.00 | 2.994767 | 0.822144 |
| 0.5 vCPU | C1 | 2 | 3.00 | 2.994767 | 0.822144 |
| 0.5 vCPU | C1 | 3 | 3.00 | 2.994767 | 0.822144 |

---
## 3. Server-Side Communication Receipt (daemon view, per submission)

Independent server-side measurement of the same 60 client submissions — compare against "Comm. Send (s)" in Table 1 for client-vs-server timing asymmetry.

| Mode | Profile | Client | Round | Daemon Recv (s) | Payload (bytes) | Wall Time Since Daemon Start (s) |
|---|---|---|---|---|---|---|
| Baseline (no defense) | 1.0 vCPU | C0 | 1 | 0.07056 | 1756605 | 138.652 |
| Baseline (no defense) | 1.0 vCPU | C1 | 1 | 0.07906 | 1756663 | 117.787 |
| Baseline (no defense) | 1.0 vCPU | C0 | 2 | 0.07015 | 1755544 | 275.340 |
| Baseline (no defense) | 1.0 vCPU | C1 | 2 | 0.06785 | 1755840 | 221.100 |
| Baseline (no defense) | 1.0 vCPU | C0 | 3 | 0.07122 | 1754576 | 417.956 |
| Baseline (no defense) | 1.0 vCPU | C1 | 3 | 0.06734 | 1755113 | 323.321 |
| Baseline (no defense) | 0.5 vCPU | C0 | 1 | 0.09290 | 1756649 | 412.398 |
| Baseline (no defense) | 0.5 vCPU | C1 | 1 | 0.21647 | 1756879 | 471.876 |
| Baseline (no defense) | 0.5 vCPU | C0 | 2 | 0.30616 | 1755566 | 862.432 |
| Baseline (no defense) | 0.5 vCPU | C1 | 2 | 0.11925 | 1755833 | 985.330 |
| Baseline (no defense) | 0.5 vCPU | C0 | 3 | 0.07214 | 1754799 | 1315.097 |
| Baseline (no defense) | 0.5 vCPU | C1 | 3 | 0.11768 | 1754936 | 1509.371 |
| Full-Model HE (CKKS) | 1.0 vCPU | C0 | 1 | 0.04098 | 4562330 | 142.014 |
| Full-Model HE (CKKS) | 1.0 vCPU | C1 | 1 | 0.04127 | 4563210 | 82.525 |
| Full-Model HE (CKKS) | 1.0 vCPU | C0 | 2 | 0.02688 | 4563330 | 274.996 |
| Full-Model HE (CKKS) | 1.0 vCPU | C1 | 2 | 0.02927 | 4562842 | 169.589 |
| Full-Model HE (CKKS) | 1.0 vCPU | C0 | 3 | 0.03293 | 4562874 | 415.029 |
| Full-Model HE (CKKS) | 1.0 vCPU | C1 | 3 | 0.02654 | 4562738 | 256.826 |
| Full-Model HE (CKKS) | 0.5 vCPU | C0 | 1 | 0.03408 | 4563010 | 641.977 |
| Full-Model HE (CKKS) | 0.5 vCPU | C1 | 1 | 0.02904 | 4562642 | 216.924 |
| Full-Model HE (CKKS) | 0.5 vCPU | C0 | 2 | 0.09525 | 4564778 | 1277.613 |
| Full-Model HE (CKKS) | 0.5 vCPU | C1 | 2 | 0.02879 | 4563074 | 402.100 |
| Full-Model HE (CKKS) | 0.5 vCPU | C0 | 3 | 0.04102 | 4562958 | 1906.574 |
| Full-Model HE (CKKS) | 0.5 vCPU | C1 | 3 | 0.10468 | 4563214 | 596.696 |
| Partial HE (classifier head only) | 1.0 vCPU | C0 | 1 | 0.08529 | 1998512 | 140.722 |
| Partial HE (classifier head only) | 1.0 vCPU | C1 | 1 | 0.07587 | 1998056 | 119.343 |
| Partial HE (classifier head only) | 1.0 vCPU | C0 | 2 | 0.07221 | 1997626 | 288.602 |
| Partial HE (classifier head only) | 1.0 vCPU | C1 | 2 | 0.07149 | 1997682 | 232.541 |
| Partial HE (classifier head only) | 1.0 vCPU | C0 | 3 | 0.06566 | 1996471 | 435.416 |
| Partial HE (classifier head only) | 1.0 vCPU | C1 | 3 | 0.07144 | 1996695 | 341.546 |
| Partial HE (classifier head only) | 0.5 vCPU | C0 | 1 | 0.09699 | 1998283 | 626.502 |
| Partial HE (classifier head only) | 0.5 vCPU | C1 | 1 | 0.07806 | 1998995 | 353.452 |
| Partial HE (classifier head only) | 0.5 vCPU | C0 | 2 | 0.09893 | 1997206 | 1227.321 |
| Partial HE (classifier head only) | 0.5 vCPU | C1 | 2 | 0.08822 | 1997664 | 717.705 |
| Partial HE (classifier head only) | 0.5 vCPU | C0 | 3 | 0.29148 | 1996683 | 1830.444 |
| Partial HE (classifier head only) | 0.5 vCPU | C1 | 3 | 0.07310 | 1997290 | 1076.358 |
| Partial HE + Norm-Bound ZKP Guard | 1.0 vCPU | C0 | 1 | 0.07256 | 1998246 | 167.001 |
| Partial HE + Norm-Bound ZKP Guard | 1.0 vCPU | C1 | 1 | 0.09562 | 1998254 | 119.446 |
| Partial HE + Norm-Bound ZKP Guard | 1.0 vCPU | C0 | 2 | 0.06608 | 1997792 | 318.712 |
| Partial HE + Norm-Bound ZKP Guard | 1.0 vCPU | C1 | 2 | 0.07118 | 1997709 | 243.485 |
| Partial HE + Norm-Bound ZKP Guard | 1.0 vCPU | C0 | 3 | 0.06746 | 1997011 | 477.099 |
| Partial HE + Norm-Bound ZKP Guard | 1.0 vCPU | C1 | 3 | 0.06836 | 1996488 | 369.904 |
| Partial HE + Norm-Bound ZKP Guard | 0.5 vCPU | C0 | 1 | 0.16837 | 1998079 | 306.975 |
| Partial HE + Norm-Bound ZKP Guard | 0.5 vCPU | C1 | 1 | 0.10892 | 1998239 | 224.491 |
| Partial HE + Norm-Bound ZKP Guard | 0.5 vCPU | C0 | 2 | 0.08646 | 1996729 | 569.660 |
| Partial HE + Norm-Bound ZKP Guard | 0.5 vCPU | C1 | 2 | 0.17902 | 1997622 | 462.438 |
| Partial HE + Norm-Bound ZKP Guard | 0.5 vCPU | C0 | 3 | 0.07873 | 1996494 | 846.009 |
| Partial HE + Norm-Bound ZKP Guard | 0.5 vCPU | C1 | 3 | 0.09786 | 1996404 | 697.627 |
| DP-SGD (Opacus, eps=3.0) | 1.0 vCPU | C0 | 1 | 0.15085 | 2829057 | 720.880 |
| DP-SGD (Opacus, eps=3.0) | 1.0 vCPU | C1 | 1 | 0.12294 | 2829704 | 705.367 |
| DP-SGD (Opacus, eps=3.0) | 1.0 vCPU | C0 | 2 | 0.11147 | 2828080 | 1407.232 |
| DP-SGD (Opacus, eps=3.0) | 1.0 vCPU | C1 | 2 | 0.11793 | 2829193 | 1386.499 |
| DP-SGD (Opacus, eps=3.0) | 1.0 vCPU | C0 | 3 | 0.10538 | 2828110 | 2090.384 |
| DP-SGD (Opacus, eps=3.0) | 1.0 vCPU | C1 | 3 | 0.11144 | 2828752 | 2054.688 |
| DP-SGD (Opacus, eps=3.0) | 0.5 vCPU | C0 | 1 | 0.30212 | 2829752 | 2642.791 |
| DP-SGD (Opacus, eps=3.0) | 0.5 vCPU | C1 | 1 | 0.29077 | 2828911 | 2110.849 |
| DP-SGD (Opacus, eps=3.0) | 0.5 vCPU | C0 | 2 | 0.24979 | 2829294 | 5268.811 |
| DP-SGD (Opacus, eps=3.0) | 0.5 vCPU | C1 | 2 | 0.34254 | 2828295 | 4277.058 |
| DP-SGD (Opacus, eps=3.0) | 0.5 vCPU | C0 | 3 | 0.48524 | 2828639 | 7929.573 |
| DP-SGD (Opacus, eps=3.0) | 0.5 vCPU | C1 | 3 | 0.38260 | 2827765 | 6424.467 |

---
## 4. Server-Side Aggregation Timing (per round)

| Mode | Profile | Round | Aggregate (s) | Decrypt (s) | ZKP Verify (s) | ZKP MAD Threshold (s) | Adaptive Krum (s)** | N Clients (Synth.) | N Kept | N Dropped |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline (no defense) | 1.0 vCPU | 1 | — | — | — | — | 0.04352 | 10 | 10 | 0 |
| Baseline (no defense) | 1.0 vCPU | 2 | — | — | — | — | 0.01614 | 10 | 10 | 0 |
| Baseline (no defense) | 1.0 vCPU | 3 | — | — | — | — | 0.01095 | 10 | 10 | 0 |
| Baseline (no defense) | 0.5 vCPU | 1 | — | — | — | — | 0.10191 | 10 | 10 | 0 |
| Baseline (no defense) | 0.5 vCPU | 2 | — | — | — | — | 0.01883 | 10 | 10 | 0 |
| Baseline (no defense) | 0.5 vCPU | 3 | — | — | — | — | 0.06459 | 10 | 10 | 0 |
| Full-Model HE (CKKS) | 1.0 vCPU | 1 | 0.11704 | 0.02986 | — | — | — | — | — | — |
| Full-Model HE (CKKS) | 1.0 vCPU | 2 | 0.11847 | 0.02475 | — | — | — | — | — | — |
| Full-Model HE (CKKS) | 1.0 vCPU | 3 | 0.11473 | 0.02753 | — | — | — | — | — | — |
| Full-Model HE (CKKS) | 0.5 vCPU | 1 | 0.28172 | 0.02980 | — | — | — | — | — | — |
| Full-Model HE (CKKS) | 0.5 vCPU | 2 | 0.22805 | 0.07780 | — | — | — | — | — | — |
| Full-Model HE (CKKS) | 0.5 vCPU | 3 | 0.25723 | 0.02687 | — | — | — | — | — | — |
| Partial HE (classifier head only) | 1.0 vCPU | 1 | 0.01251 | 0.00168 | — | — | — | — | — | — |
| Partial HE (classifier head only) | 1.0 vCPU | 2 | 0.01435 | 0.00179 | — | — | — | — | — | — |
| Partial HE (classifier head only) | 1.0 vCPU | 3 | 0.01082 | 0.00164 | — | — | — | — | — | — |
| Partial HE (classifier head only) | 0.5 vCPU | 1 | 0.07151 | 0.00230 | — | — | — | — | — | — |
| Partial HE (classifier head only) | 0.5 vCPU | 2 | 0.01662 | 0.00194 | — | — | — | — | — | — |
| Partial HE (classifier head only) | 0.5 vCPU | 3 | 0.01382 | 0.00176 | — | — | — | — | — | — |
| Partial HE + Norm-Bound ZKP Guard | 1.0 vCPU | 1 | 0.01337 | 0.00177 | 0.00114 | 0.03281 | — | — | — | — |
| Partial HE + Norm-Bound ZKP Guard | 1.0 vCPU | 2 | 0.01269 | 0.00162 | 0.00116 | 0.00027 | — | — | — | — |
| Partial HE + Norm-Bound ZKP Guard | 1.0 vCPU | 3 | 0.01172 | 0.00169 | 0.00128 | 0.00025 | — | — | — | — |
| Partial HE + Norm-Bound ZKP Guard | 0.5 vCPU | 1 | 0.01475 | 0.00190 | 0.00134 | 0.07506 | — | — | — | — |
| Partial HE + Norm-Bound ZKP Guard | 0.5 vCPU | 2 | 0.01613 | 0.00236 | 0.00149 | 0.00035 | — | — | — | — |
| Partial HE + Norm-Bound ZKP Guard | 0.5 vCPU | 3 | 0.01289 | 0.00211 | 0.00183 | 0.00036 | — | — | — | — |
| DP-SGD (Opacus, eps=3.0) | 1.0 vCPU | 1 | — | — | — | — | 0.06558 | 10 | 10 | 0 |
| DP-SGD (Opacus, eps=3.0) | 1.0 vCPU | 2 | — | — | — | — | 0.02132 | 10 | 10 | 0 |
| DP-SGD (Opacus, eps=3.0) | 1.0 vCPU | 3 | — | — | — | — | 0.01564 | 10 | 10 | 0 |
| DP-SGD (Opacus, eps=3.0) | 0.5 vCPU | 1 | — | — | — | — | 0.22270 | 10 | 10 | 0 |
| DP-SGD (Opacus, eps=3.0) | 0.5 vCPU | 2 | — | — | — | — | 0.02433 | 10 | 10 | 0 |
| DP-SGD (Opacus, eps=3.0) | 0.5 vCPU | 3 | — | — | — | — | 0.01725 | 10 | 9 | 1 |

---
## 5. RAM — Peak & Average (all processes)

| Mode | Profile | Process | Peak RAM (MB) | Avg RAM (MB) | N Samples | cgroup Mem Limit (MB) | cgroup CPU Limit (cores) |
|---|---|---|---|---|---|---|---|
| Baseline (no defense) | 1.0 vCPU | Client 0 | 382.9 | 366.9 | 1571 | 2048 | 1.0 |
| Baseline (no defense) | 1.0 vCPU | Client 1 | 382.4 | 375.4 | 1199 | 2048 | 1.0 |
| Baseline (no defense) | 1.0 vCPU | Server (aggregation) | 63.3 | 49.5 | — | — | — |
| Baseline (no defense) | 0.5 vCPU | Client 0 | 385.9 | 373.3 | 4904 | 2048 | 0.5 |
| Baseline (no defense) | 0.5 vCPU | Client 1 | 382.4 | 367.0 | 5590 | 2048 | 0.5 |
| Baseline (no defense) | 0.5 vCPU | Server (aggregation) | 63.1 | 50.8 | — | — | — |
| Full-Model HE (CKKS) | 1.0 vCPU | Client 0 | 401.1 | 379.3 | 1571 | 2048 | 1.0 |
| Full-Model HE (CKKS) | 1.0 vCPU | Client 1 | 396.2 | 376.3 | 960 | 2048 | 1.0 |
| Full-Model HE (CKKS) | 1.0 vCPU | Server (aggregation) | 72.9 | 56.0 | — | — | — |
| Full-Model HE (CKKS) | 0.5 vCPU | Client 0 | 399.5 | 378.3 | 7144 | 2048 | 0.5 |
| Full-Model HE (CKKS) | 0.5 vCPU | Client 1 | 399.5 | 375.5 | 2233 | 2048 | 0.5 |
| Full-Model HE (CKKS) | 0.5 vCPU | Server (aggregation) | 72.9 | 56.9 | — | — | — |
| Partial HE (classifier head only) | 1.0 vCPU | Client 0 | 394.7 | 378.9 | 1641 | 2048 | 1.0 |
| Partial HE (classifier head only) | 1.0 vCPU | Client 1 | 392.8 | 376.5 | 1288 | 2048 | 1.0 |
| Partial HE (classifier head only) | 1.0 vCPU | Server (aggregation) | 51.5 | 44.8 | — | — | — |
| Partial HE (classifier head only) | 0.5 vCPU | Client 0 | 395.2 | 377.0 | 6894 | 2048 | 0.5 |
| Partial HE (classifier head only) | 0.5 vCPU | Client 1 | 392.6 | 375.4 | 3970 | 2048 | 0.5 |
| Partial HE (classifier head only) | 0.5 vCPU | Server (aggregation) | 53.9 | 44.1 | — | — | — |
| Partial HE + Norm-Bound ZKP Guard | 1.0 vCPU | Client 0 | 394.4 | 376.7 | 1811 | 2048 | 1.0 |
| Partial HE + Norm-Bound ZKP Guard | 1.0 vCPU | Client 1 | 393.5 | 378.0 | 1398 | 2048 | 1.0 |
| Partial HE + Norm-Bound ZKP Guard | 1.0 vCPU | Server (aggregation) | 60.6 | 50.1 | — | — | — |
| Partial HE + Norm-Bound ZKP Guard | 0.5 vCPU | Client 0 | 397.3 | 375.6 | 3146 | 2048 | 0.5 |
| Partial HE + Norm-Bound ZKP Guard | 0.5 vCPU | Client 1 | 392.6 | 375.3 | 2583 | 2048 | 0.5 |
| Partial HE + Norm-Bound ZKP Guard | 0.5 vCPU | Server (aggregation) | 60.0 | 49.9 | — | — | — |
| DP-SGD (Opacus, eps=3.0) | 1.0 vCPU | Client 0 | 799.4 | 673.2 | 8110 | 2048 | 1.0 |
| DP-SGD (Opacus, eps=3.0) | 1.0 vCPU | Client 1 | 784.1 | 660.8 | 7807 | 2048 | 1.0 |
| DP-SGD (Opacus, eps=3.0) | 1.0 vCPU | Server (aggregation) | 76.7 | 57.6 | — | — | — |
| DP-SGD (Opacus, eps=3.0) | 0.5 vCPU | Client 0 | 798.8 | 652.4 | 29665 | 2048 | 0.5 |
| DP-SGD (Opacus, eps=3.0) | 0.5 vCPU | Client 1 | 792.6 | 652.0 | 24083 | 2048 | 0.5 |
| DP-SGD (Opacus, eps=3.0) | 0.5 vCPU | Server (aggregation) | 76.7 | 58.3 | — | — | — |

---
## 6. Averaged Summary (per mode × profile)

| Mode | Profile | Avg Train C0 (s) | Avg Train C1 (s) | Avg DP Setup (s) | Avg HE Encrypt (s) | Avg ZKP Proof (s) | Avg Comm. Send (s) | Avg Daemon Recv (s) | Avg Payload (bytes) | Peak RAM C0 (MB) | Peak RAM C1 (MB) | Peak RAM Server (MB) | Avg Aggregate (s) | Avg Decrypt (s) | Avg ZKP Verify (s) | Avg ZKP Threshold (s) | Avg Adaptive Krum (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline (no defense) | 1.0 vCPU | 134.63 | 102.87 | — | — | — | 0.0842 | 0.07103 | 1755724 | 382.9 | 382.4 | 63.3 | — | — | — | — | 0.02354 |
| Baseline (no defense) | 0.5 vCPU | 432.93 | 497.61 | — | — | — | 0.1741 | 0.15410 | 1755777 | 385.9 | 382.4 | 63.1 | — | — | — | — | 0.06178 |
| Full-Model HE (CKKS) | 1.0 vCPU | 134.97 | 82.25 | — | 0.3361 | — | 0.0439 | 0.03298 | 4562887 | 401.1 | 396.2 | 72.9 | 0.11675 | 0.02738 | — | — | — |
| Full-Model HE (CKKS) | 0.5 vCPU | 630.33 | 193.82 | — | 0.5940 | — | 0.0822 | 0.05548 | 4563279 | 399.5 | 399.5 | 72.9 | 0.25567 | 0.04482 | — | — | — |
| Partial HE (classifier head only) | 1.0 vCPU | 140.87 | 109.56 | — | 0.0365 | — | 0.0839 | 0.07366 | 1997507 | 394.7 | 392.8 | 51.5 | 0.01256 | 0.00170 | — | — | — |
| Partial HE (classifier head only) | 0.5 vCPU | 604.73 | 353.38 | — | 0.0666 | — | 0.1330 | 0.12113 | 1997687 | 395.2 | 392.6 | 53.9 | 0.03398 | 0.00200 | — | — | — |
| Partial HE + Norm-Bound ZKP Guard | 1.0 vCPU | 154.94 | 119.17 | — | 0.0533 | 0.00555 | 0.0839 | 0.07354 | 1997583 | 394.4 | 393.5 | 60.6 | 0.01259 | 0.00169 | 0.00119 | 0.01111 | — |
| Partial HE + Norm-Bound ZKP Guard | 0.5 vCPU | 276.25 | 226.62 | — | 0.0874 | 0.00318 | 0.1316 | 0.11989 | 1997261 | 397.3 | 392.6 | 60.0 | 0.01459 | 0.00212 | 0.00155 | 0.02526 | — |
| DP-SGD (Opacus, eps=3.0) | 1.0 vCPU | 688.69 | 676.39 | 1.197 | — | — | 0.1327 | 0.12000 | 2828816 | 799.4 | 784.1 | 76.7 | — | — | — | — | 0.03418 |
| DP-SGD (Opacus, eps=3.0) | 0.5 vCPU | 2630.10 | 2128.30 | 2.502 | — | — | 0.4977 | 0.34218 | 2828776 | 798.8 | 792.6 | 76.7 | — | — | — | — | 0.08809 |

---
## Known Issues

**\* ZKP Norm column (Table 1) is always 0.0 — known measurement bug, does NOT affect timing.**

The head-norm delta is computed as `trained_head - head_before_training`. The "before" snapshot was taken via `tensor.cpu().numpy()`, which for an already-CPU tensor returns a *view* sharing memory with the tensor, not a copy. When the optimizer later mutates that tensor in place, the earlier snapshot is silently overwritten too — so the delta is trivially zero every round, for every client, in both profiles. This affects **only** the displayed norm value. It does **not** affect ZKP Proof Time, HE Encrypt Time, Train Time, ZKP Verify Time, or ZKP MAD Threshold Time — those all time the algorithm itself, which runs identically regardless of whether the input value is real or zero. **Do not cite the ZKP Norm column as a real result.**


**\*\* Adaptive Krum / ZKP MAD-Threshold timing uses 10 synthetic clients, not 10 real ones.**

Only 2 real Docker clients run in this suite. Adaptive Krum's distance computation is O(n²×d) in client count, and the project's main FL pipeline uses NUM_CLIENTS=10 — a 2-client timing wouldn't be representative, and a 2-point MAD threshold is statistically degenerate. 8 additional "clients" are synthesized by adding small i.i.d. Gaussian jitter to copies of the 2 real clients' actual trained parameters/norms, to keep client count and model dimensionality realistic. **This is a TIMING measurement only** — "N Kept"/"N Dropped" in Table 4 describe an artifact of the synthetic jitter, not real attack detection. Do not cite these as evidence of defense effectiveness.


**ZKP pipeline scope note:** the `he_partial_zkp` mode in this suite mirrors the project's `pure_zkp` ablation (guard → aggregate, no Krum call in that flow) — it does **not** reproduce `USE_HE_KRUM_HYBRID` (guard → Krum on survivors → aggregate), which runs Krum as a second layer behind the guard. Adaptive Krum timing in this workbook (Table 4/6, `baseline`/`dp` rows) is a separate, standalone benchmark, never layered behind the ZKP guard in this test suite. Same underlying `zkp.py` mechanism (ciphertext-bound HMAC head-norm guard) either way — only the surrounding pipeline shape differs.


**Resource profile note:** both "unthrottled" and "throttled" profiles use the same 2048MB RAM cap — a "don't crash, just measure" ceiling, not a tight simulated constraint. Only CPU allocation (1.0 vs 0.5 vCPU) differs between profiles. Peak RAM stayed well under this ceiling in every run (Table 5), confirming headroom was never the limiting factor.
