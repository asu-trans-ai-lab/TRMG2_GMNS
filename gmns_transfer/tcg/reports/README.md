# Expert reports

Verbatim briefs from the two specialists consulted on the 3-level operator
design. LEVELS.md (parent folder) is the distilled design that implements them.

- **ONNX_operator_expert.md** — ONNX/onnxruntime operator specialist.
  Verdict: Pi does NOT go in ONNX (no runtime sparse matmul on CPU/CoreML/MPS;
  dense = 3 PB; index emulation = 1.36 GB + CoreML CPU-fallback). Ship
  demand.onnx (Level 1) + pi.npz (CSR); calibrate in torch. 6 questions answered
  with opset/provider specifics and size tables.

- **differentiable_operator_expert.md** — differentiable computational-graph
  operator specialist. The 3-level API (one math core, three views), the exact
  adjoint chain dL/dOD = Pi^T dL/dv (Danskin at UE, fixed columns), the bilevel
  outer loop, ADMM stacked-after-behavioral (not interleaved) with influence
  restriction, DemandLite as a torch.autograd.Function, and the C1-C11
  consistency-check plan.
