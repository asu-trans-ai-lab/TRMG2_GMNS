# tensor_ftt/ — Flow-Through-Tensor on the RELOADED assignment snapshot

`ftt_pipeline.py` operates on the **real tensors from the previous assignment**,
reloaded as a fixed snapshot — following `../../flow_through_tensor.tex`. It does
**not** re-run the network assignment and does **not** synthesize anything.

## The point of this design (per the revision)
"Flow through tensors" assumes the assigned **path-flow matrix**, the **OD→path
matrix**, and the **skim** from the last assignment are already available. We
**reload the entire snapshot** so we can:
1. **converge better** — warm-start from the real path flows `f_P`, iterate in
   matrix form instead of all-or-nothing from scratch;
2. **verify** — `Π · f_OD` must reproduce the kernel's link volumes (R²);
3. **go to ONNX** — the operators are fixed sparse matrices, i.e. a clean
   differentiable graph.

## What it loads (from `../matrices/` + `../scenario_<per>/`)
| tensor | file | shape |
|---|---|---|
| Δ = B_OD,P (OD→path proportions) | `delta_<per>.npz` | \|P\| × \|OD\| |
| A = A_P,L (path→link incidence) | `A_<per>.npz` | \|L\| × \|P\| |
| Π = A·Δ (OD→link operator) | `pi_<per>.npz` | \|L\| × \|OD\| |
| f_OD (assigned OD demand) | `scenario_<per>/demand_*.csv` | \|OD\| |
| t_L, v_kernel (previous skim + volumes) | `scenario_<per>/link_performance.csv` | \|L\| |

AM snapshot = **1,039,117 OD × 1,557,815 paths × 75,939 links**.

## Forward / backward (FTT)
```
forward :  f_P = Δ f_OD ;  f_L = A f_P  (= Π f_OD) ;  t_L = φ(f_L)
backward:  t_P = Aᵀ t_L ;  t_OD = Δᵀ t_P ;  dL/df_OD = Πᵀ (f_L − counts)
```
The backward gives the ODME / calibration gradient directly — gradient descent on
`f_OD` (or, upstream, on the demand-model θ) with the operators held fixed.

## Snapshot consistency (important)
`Π · f_OD == kernel volumes` (R²=1.0) **only when the snapshot is internally
consistent** — Δ, A, Π, the OD, and the volumes all from the *same* assignment.
If you change the demand and re-assign, refresh the operators before verifying:
```
python ../matrix_ops.py AM        # re-extract Δ/A/Π from the latest routes
python ftt_pipeline.py AM         # -> R^2 1.000000  [CONSISTENT]
```
The script prints `[CONSISTENT]` or `[STALE]` so you always know. (A stale mix of
old Π with new demand reads ~0.98 — that is the whole reason we reload the
*entire* snapshot together, not piecemeal.)

## Run
```
python ftt_pipeline.py            # AM
python ftt_pipeline.py MD PM NT   # if those periods are extracted
```
Pure numpy + scipy.sparse (loads the real ~100 MB operators in ~15 s). The torch/
ONNX export path reuses the same fixed Π — see `../tcg/` for the packaged version.
