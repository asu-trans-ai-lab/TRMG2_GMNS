# Computational-graph scaling benchmark

Calibration target = base-year LOADED link volumes (kernel UE). Prior = neutral theta (the 'original coefficients'). Rails = trust region lam=0.5 + bounds +/-1.5.

| case | zones | links | od | paths | Pi R2 | t_kernel | t_extract | t_fwd | t_fwd+bwd | %RMSE start | no-rails %RMSE (drift_max) | rails %RMSE (drift_max) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sioux_falls | 24 | 76 | 528 | 1,400 | 1.0 | 0.4s | 0.0s | 0.001s | 0.0s | 24.16 | 10.89 (5.0) | 13.13 (0.397) |
| chicago_sketch | 386 | 2950 | 93,135 | 255,881 | 1.0 | 17.3s | 3.3s | 0.022s | 0.036s | 44.15 | 41.65 (5.0) | 41.98 (1.5) |
| chicago_regional | 1769 | 39018 | 1,881,643 | 17,493,269 | 0.99967 | 1030.0s | 803.0s | 1.08s | 2.187s | 50.11 | 34.65 (5.0) | 34.57 (1.5) |

TRMG2 (3,147 zones) row: pending 4-period operator extraction (kernel patched; AM in flight). Improvement plans: see PERFORMANCE.md P0-P2 (f32, DemandLite fused C++ fwd+bwd, sparse pi export, concurrent periods).
