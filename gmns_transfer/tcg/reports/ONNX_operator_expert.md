# ONNX / onnxruntime for a Travel-Demand Graph: Sparse Pi at Levels 2-3
### Expert brief — ONNX/onnxruntime operator specialist

## Bottom line up front

**Do not put Pi in ONNX.** onnxruntime does not execute sparse matmul on any
provider you care about (CPU sparse is a dead stub; CoreML/MPS have no sparse
path at all). The dense fallback is catastrophic at your scale, and the
index-gather emulation is large, slow, and non-differentiable. Keep the ONNX
artifact = Level-1 demand only, and apply Pi as a scipy/torch sparse op in the
host process. Details and numbers below.

---

## Q1 — Can sparse Pi be embedded in an ONNX graph, and does onnxruntime execute it?

**Storage: yes. Execution: effectively no.**

- **SparseTensorProto exists** (added in IR v7 / opset 11, `onnx.TensorProto`
  companion `SparseTensorProto` with COO `indices` + `values`). You can attach a
  `sparse_initializer` to a graph. So the *file format* supports a sparse constant.
- **Operators that declare sparse tensor inputs in the ONNX standard:**
  essentially only `Constant`/`ConstantOfShape` (produce sparse) and the generic
  type system. **There is no standard `MatMul`, `Gemm`, or `SpMM` op with a
  sparse-tensor input in the ONNX opset.** `MatMul`/`Gemm` are typed for dense
  tensors only. There is no `SparseMatMul` in ai.onnx.
- **onnxruntime runtime reality (~1.20):**
  - ORT has a `SparseTensor` C++/Python API and can *hold* a sparse initializer,
    but the set of kernels that consume it is tiny and CPU-only. There is a
    contrib/experimental path historically exposed for sparse, but **`MatMul`
    with a sparse operand is not a supported executing kernel on the default CPU
    EP** in any way you should build on. In practice ORT will either reject the
    sparse-typed edge into MatMul at graph-partition time or require you to
    `Cast`/densify.
  - **CoreML EP and the Apple GPU path (MPS via CoreML): zero sparse support.**
    These providers compile a fixed dense subgraph; any node they don't support
    falls back to CPU, and sparse-operand MatMul isn't a supported CPU kernel
    either. So on Apple targets, a sparse Pi node simply cannot be placed — it
    would force a densify or fail partitioning.

**Plain statement:** Sparse MatMul is **not runtime-executable** in onnxruntime
for your providers. SparseTensor is a storage feature, not a compute feature.
Anything that "works" will have silently densified.

---

## Q2 — Best practical Level-2 representation

### (a) Dense Pi as a MatMul initializer — quantify the catastrophe

`Pi` is `[A x Z*Z]`, fp32 = 4 bytes:

| Z | A (approx) | Z*Z | Dense Pi bytes | Verdict |
|---|---|---|---|---|
| 386 | ~9k | 148,996 | 9,000 x 148,996 x 4 = **5.4 GB** | infeasible |
| 1769 | ~35k | 3.13M | 35,000 x 3.13M x 4 = **438 TB** | absurd |
| 3147 (TRMG2) | ~76k | 9.90M | 76,000 x 9.9M x 4 = **3.0 PB** | absurd |

Dense is dead past toy sizes. Even Z=386 (5 GB) blows past CoreML model-size and
any reasonable initializer budget. Rule it out.

### (b) Gather / Multiply / ScatterAdd emulation of SpMM — viable but costly

This is the *only* way to express the sparse contraction with standard,
provider-portable ops. Represent Pi in **COO** as three initializers/inputs:
- `row` : int64 `[nnz]` — link index a
- `col` : int64 `[nnz]` — OD flat index (i*Z + j)
- `val` : fp32 `[nnz]` — Pi entry

Op sequence (given `OD` produced by Level-1, shape `[Z,Z]`):

```
odflat   = Reshape(OD, [Z*Z])                 # dense demand vector
picked   = Gather(odflat, col, axis=0)         # [nnz]  demand at each nnz's OD pair
contrib  = Mul(picked, val)                    # [nnz]  weighted contribution
v        = ScatterND(zeros[A], row_expanded, contrib, reduction='add')
                                               # segment-sum into A link bins
```

Notes:
- Use **`ScatterND` with `reduction='add'`** (opset 16+; reduction attr added at
  opset 16, extended for ScatterND at 16/18). Confirm your target opset >= 18 for
  robust add-reduction. On CoreML EP, `ScatterND` with reduction is **not
  supported** -> falls back to CPU (the whole scatter runs on CPU, killing the
  point of the GPU provider).
- `Gather` is well-supported everywhere including CoreML.
- Alternative to ScatterND: sort by row and use `SegmentSum` — but ai.onnx has
  **no SegmentSum** (it's a TF/contrib op), so ScatterND-add is your standard route.

**Size of the index tensors (COO, this is what dominates):** two int64 + one
fp32 per nnz = 8 + 8 + 4 = **20 bytes/nnz**.

| Z | nnz | 20 B x nnz |
|---|---|---|
| 386 | ~ (386/3147)^2 x 67.8M = **1.0M** | ~20 MB |
| 1769 | ~ (1769/3147)^2 x 67.8M = **21.4M** | ~430 MB |
| 3147 | **67.8M** | **~1.36 GB** (540 MB just the int64 `col`) |

CSR trims `row` from `[nnz]` to `[A+1]` row-pointers, but ai.onnx has no
CSR-aware kernel, so you'd rebuild row indices anyway — COO is the honest cost.
**~1.36 GB of frozen index tensors baked into the model** at Z=3147, all of
which must be memory-mapped and, for CoreML, would force CPU fallback on the
scatter.

**Verdict on (b):** technically correct, provider-portable in principle, but (i)
1.36 GB of frozen index tensors baked into the model, (ii) `ScatterND`-add not on
CoreML -> CPU fallback anyway, (iii) no compression benefit over just doing the
sparse op in the host. It buys you nothing the runtime sparse op wouldn't, at
higher cost.

### (c) Keep Pi OUT of ONNX — the pragmatic and correct answer

- ONNX artifact = Level-1 `DemandLayers` -> `OD` (already built, verified).
- Host code: `v = Pi_csr @ OD.reshape(-1)` via `scipy.sparse.csr_matrix @` or
  `torch.sparse.mm`. At 67.8M nnz this is a sub-second SpMV on CPU and trivially
  memory-efficient (CSR = 67.8M x (4 val + 4 col-idx) + A row-ptr = **~540 MB**,
  half the COO-in-ONNX cost, and you keep it as a native sparse object, not a
  densifiable tensor).

**Recommendation: (c).** (b) only if you have a hard requirement for a single
self-contained graph AND accept CPU-only execution AND the 1.36 GB. Otherwise
never.

---

## Q3 — Level-3 factored `Pi = A_inc @ Delta`: two sparse matmuls, better or worse for ONNX?

**Worse for ONNX/onnxruntime, neutral-to-better for the host.**

- Same runtime wall: both `A_inc` `[A x n_paths]` and `Delta` `[n_paths x Z*Z]`
  are sparse, so you now need **two** sparse contractions instead of one. ONNX
  has no sparse matmul for either, so you'd need the Gather/Scatter emulation
  **twice**, doubling the ScatterND-add CPU-fallback nodes and adding an
  intermediate dense (or COO) path-flow vector `f = Delta @ vec(OD)` of length
  `n_paths` (which at column-gen scale can be large).
- The factoring is genuinely useful **outside** ONNX for column generation:
  `f = Delta @ vec(OD)` (OD->path shares) then `v = A_inc @ f` (path->link) are
  two clean SpMVs, and `A_inc` is a stable 0/1 incidence you compute once. That's
  a host-side win (cache `A_inc`, regenerate only `Delta` columns as you generate
  columns).

**Verdict:** Keep the factorization — but as **two scipy/torch sparse SpMVs in
the host**, not in ONNX. In ONNX it strictly doubles the cost and CPU-fallback
surface.

---

## Q4 — Differentiability / gradients through Pi (calibration)

**Gradients do not belong in ONNX. Keep training in torch; ONNX is inference-only.**

- onnxruntime is an inference runtime. There is `onnxruntime-training` (ORTModule)
  but it's for training *torch* models via ORT kernels, not for differentiating an
  arbitrary exported `.onnx` — and it has **no sparse-matmul autograd** either.
- You already have the right structure: `DemandLayers` is a `torch.nn.Module`.
  For calibration, define the **full** pipeline in torch:
  - `OD = DemandLayers(theta)` (differentiable, dense)
  - `v = torch.sparse.mm(Pi, OD.reshape(-1,1))` — **`torch.sparse.mm` is
    differentiable w.r.t. the dense operand** (`OD`), which is exactly what you
    need. Pi's *values* are constants (extracted from a solved UE), so you don't
    need grad through Pi's structure; you need grad through `OD`, and torch gives
    you that for free.
  - Loss on `v` vs observed counts -> `theta.grad`. Pure torch, GPU/MPS-capable,
    autograd end-to-end.
- **The split:** torch owns calibration (needs gradients, needs the sparse-dense
  product in the graph). ONNX owns deployment inference of the demand stage only.
  Export the *calibrated* `DemandLayers` to ONNX after training; apply Pi as a
  runtime sparse op at inference. Never round-trip gradients through ONNX.

---

## Q5 — External-data at 67.8M int64 (~540 MB `col`, ~1.36 GB full COO)

- **ONNX external-data (`.onnx.data` sidecar):** any initializer over the 2 GB
  single-protobuf limit *must* be external anyway.
  `onnx.save_model(..., save_as_external_data=True, all_tensors_to_one_file=True,
  size_threshold=1024)` writes the big index tensors to a sidecar; the `.onnx`
  keeps only offset/length refs. The 2 GB protobuf cap is on the model proto, but
  individual external tensors and the sidecar file can exceed 2 GB, so 1.36 GB is
  fine mechanically.
- **Load time:** ORT `mmap`s external data by default, so cold load is
  page-fault-on-touch, not a 1.36 GB upfront copy — fast to open, but the first
  inference pays the page-in. Still, you're shipping a 1.36 GB sidecar with every
  model.
- **CoreML provider limits:** CoreML compiles the model into an `.mlmodelc`; it
  has practical model-size ceilings and **does not consume ONNX external-data as
  memory-mapped weights the way CPU EP does** — weights get baked/copied into the
  CoreML model. A 1.36 GB (or even 540 MB) index blob pushed through CoreML is a
  hard problem: large compile times, memory blowup, and the sparse-scatter nodes
  fall back to CPU anyway. **CoreML + big index tensors is a non-starter.**

This external-data pain is another concrete reason to keep Pi out: as a host-side
scipy/torch CSR it's a plain 540 MB `.npz`/`.pt` you load once, not a
model-embedded sidecar that fights the provider.

---

## Q6 — Bottom-line 3-level plan

| Level | What it is | ONNX artifact or runtime op? | Reason |
|---|---|---|---|
| **1 — Demand** `theta -> OD` | dense, softmax/logsumexp/gather/einsum | **Real ONNX artifact** (keep as built) | Dense, small (42 KB-41 MB), all ops well-supported on CPU + CoreML/MPS; verified 1e-4 vs torch. This is the sweet spot for ONNX. |
| **2 — Link volumes** `v = Pi*vec(OD)` | one sparse SpMV, Pi 67.8M nnz | **Runtime op (scipy/torch sparse), OUT of ONNX** | No executing sparse MatMul in ORT on any provider; dense is 3 PB; Gather/Scatter emulation = 1.36 GB baked indices + CoreML CPU-fallback for ScatterND-add. Host CSR SpMV is sub-second, 540 MB, and composes cleanly with the ONNX demand output. |
| **3 — Factored** `Pi = A_inc*Delta` | two sparse SpMVs, column-gen form | **Runtime op (two scipy/torch SpMVs), OUT of ONNX** | Doubles the sparse contractions ONNX can't do; but the factorization is a real host-side win for column generation — cache `A_inc`, regenerate `Delta` columns. Keep it in the solver, not the graph. |
| **Calibration (cross-cutting)** | gradients through `OD` (and optionally `theta`) | **torch autograd, not ONNX** | `torch.sparse.mm` is differentiable w.r.t. the dense `OD`; ONNX/ORT have no sparse-matmul autograd. Train in torch, export the calibrated `DemandLayers` to ONNX for inference only. |

**Concrete implementation shape:**

1. Ship `demand.onnx` (+ external weights for large Z) — Level 1, unchanged.
2. Ship `pi.npz` (CSR: `data`, `indices`, `indptr`) or `pi.pt`
   (`torch.sparse_csr_tensor`) — ~540 MB at Z=3147.
3. Inference host:
   ```python
   od = ort_session.run(None, feed)[0].reshape(-1)      # ONNX demand
   v  = Pi_csr @ od                                       # scipy SpMV -> link volumes
   ```
4. Calibration host (torch, MPS/CUDA):
   ```python
   od = demand_layers(theta).reshape(-1, 1)
   v  = torch.sparse.mm(Pi, od)                            # differentiable in od
   loss = mse(v, counts); loss.backward()                 # theta.grad
   ```
5. For column generation (Level 3): keep `A_inc` (0/1, static) and `Delta`
   (regenerated) as two torch/scipy sparse tensors; `f = Delta @ od; v = A_inc @ f`.

**The one-line answer:** Pi should not go in ONNX. onnxruntime cannot execute a
sparse matmul on CPU or on CoreML/MPS; the dense fallback is petabytes and the
index-emulation is a 1.36 GB CPU-fallback node that beats no runtime sparse op.
Put Level 1 (demand) in ONNX, apply Pi as a scipy/torch sparse SpMV in the host,
and do all gradient-based calibration in torch.

### Two caveats to verify against your exact ORT build
- **`ScatterND` reduction='add' opset floor:** reduction attr on ScatterND landed
  at opset 16 and was firmed up by 18 — pin your exporter `opset_version>=18` if
  you ever build path (b).
- **CoreML EP op coverage** shifts release to release; confirm `ScatterND`-with-
  reduction is still unsupported in your ORT (it has been). Even if it becomes
  supported, the 540 MB-1.36 GB index blob through CoreML remains the blocker, so
  the recommendation is unchanged.

These two matter only if someone insists on path (b); for the recommended plan
(Pi out of ONNX) neither matters.
