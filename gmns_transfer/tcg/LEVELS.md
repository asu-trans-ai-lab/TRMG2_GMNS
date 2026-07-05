# The 3-level operator design (expert-reviewed)

Reviewed by an ONNX/onnxruntime operator specialist and a differentiable-
computational-graph operator specialist. Their verdicts shaped this design and
are summarized below; the code implements it and `run_levels.py` proves it
(6/6 consistency checks PASS).

## The levels

| level | map | representation | ONNX? |
|---|---|---|---|
| **1 Demand** | theta -> OD [Z,Z] | dense torch graph (softmax/LSE/gather) | **YES — real ONNX artifact** |
| **2 Assignment** | theta -> v = Pi·vec(OD) [A] | Level 1 + sparse SpMV (scipy/torch) | **NO — runtime sparse op** |
| **3 Factored** | v = A_inc·(Delta·vec(OD)) | Pi un-fused; column-generation form | **NO — runtime sparse op** |

One math core (`common/demand_graph.py`), three views: Level 2 = `Pi @` glued to
Level 1; Level 3 = the same Pi un-fused into `A_inc @ Delta`. No level
reimplements the demand math.

## ONNX-operator expert — verdict: Pi does NOT go in ONNX
- onnxruntime has **no executing sparse matmul** on CPU or the CoreML/MPS
  providers. `SparseTensorProto` is storage-only; there is no `SparseMatMul`.
- Dense Pi is catastrophic: Z=386 -> 5.4 GB, Z=1769 -> 438 TB, Z=3147 -> 3 PB.
- Gather/Multiply/ScatterND-add emulation works in principle but bakes
  ~1.36 GB of int64 COO indices at TRMG2 scale, and `ScatterND`-add is
  unsupported on CoreML -> CPU fallback anyway. Net: worse than a host SpMV.
- **Plan:** ship `demand.onnx` (Level 1) + `pi.npz` (CSR, ~540 MB at Z=3147);
  apply `v = Pi_csr @ od` in the host. Gradients stay in torch
  (`torch.sparse.mm` is differentiable in the dense OD) — ONNX is inference-only.

## Differentiable-operator expert — verdict: calibrate at Level 2, exact adjoint
- Calibration operates at Level 2. Adjoint chain:
  `dL/dv --Pi^T--> dL/dOD --demand_backward--> dL/dtheta`.
- `dL/dOD = Pi^T dL/dv` is **exact for fixed columns** (Danskin/envelope at UE;
  the R²=1.0 reproduction is the witness). It is NOT exact across active-set
  changes -> refresh Pi in the **outer bilevel loop** (rerun kernel), don't
  differentiate through the equilibrium; trust-region the theta step by Pi drift.
- **ADMM stacks AFTER behavioral calibration, not interleaved** — interleaving
  lets per-OD-cell multipliers steal behavioral signal (identifiability
  collapse). ADMM = a bounded stage-2 polish; `mu` is the drift rail (x -> 1).
  Its active column set is fed by influence restriction: drop zero-norm columns
  of `Pi_obs = A_inc[obs] @ Delta` (OD cells touching no observed link are
  unidentifiable). Column generation (`append_paths`) grows this set monotonically.
- **DemandLite** ships fwd+bwd behind a `torch.autograd.Function`, guaranteeing
  the same segment reduction (the 1/C-sparse `[Z,Z,C]` LSE) and the same analytic
  VJP + cache blob as the numpy reference (the C6/C7 checks).

## The `[Z,Z,C]` bottleneck (both briefs)
The masked log-sum-exp tensor is `[Z,Z,C]` but 1/C sparse (only a destination's
own cluster is finite). A **segmented/scatter** implementation materializes `Z²`
not `Z²C` — a C-fold reduction (12x at TRMG2 scale). This is the DemandLite C++
target and a GPU `scatter_reduce` target; the dense form is the reference it
must match cell-for-cell (check C5).

## Consistency checks (run_levels.py, 6/6 PASS)
| id | check | proves |
|---|---|---|
| C1 | Level3 `A_inc@Delta` matvec == Level2 `Pi@od` | factored == fused |
| C2 | Level2 `theta->v` == Pi @ Level1 `theta->OD` | composition |
| C4 | Level3 `Delta^T A_inc^T` == `Pi^T` | adjoints agree |
| grad | torch.sparse.mm -> theta.grad | Level 2 differentiable |
| C8 | ADMM reduces count %RMSE, x->1 under mu | stage-2 polish + drift rail |
| C9 | drop zero-norm Pi_obs cols lossless | influence restriction |

Pending (need DemandLite / real UE refresh): C5 (segmented==dense LSE),
C6/C7 (DemandLite fwd+bwd == reference via shared cache), C10/C11 (column-gen
+ bilevel). Specs in the expert briefs; stubs in `modules/`.

## Folder map
```
tcg/
  common/demand_graph.py     the one math core (DemandLayers, np_forward, Pi utils)
  level1_demand/             theta -> OD ; ONNX export lives in common/export_*.py
  level2_assignment/level2.py  theta -> v = Pi@OD (runtime sparse, differentiable)
  level3_sparse/level3.py     AssignmentOp(A_inc,Delta): matvec/rmatvec/append_paths
  modules/admm/              column_tools.py (DynODME influence-restrict + ADMM)
  modules/demandlite/        demandlite.h (fused C++ fwd/bwd API) + integration notes
  modules/lowrank/           low-rank (LoRA) utility-adapter design
  run_levels.py              the 6-check consistency runner
  run_onnx.py / gpu_bench.py / consistency_check.py   Level-1 ONNX + efficiency
  artifacts/                 pre-built ONNX for all networks (Level 1)
```
