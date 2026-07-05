# The 4-step graph as a neural network: standard formats, GPU training,
# low-rank weights, and highway/DEQ tricks

Companion to TCG_MATH.pdf. Position: tcg_tensor.py already IS a neural
network in PyTorch — layers (gen, nested-softmax DC, linear TOD/mode maps,
sparse assignment) with parameters theta and a loss. Everything below is
about exploiting the deep-learning toolchain on top of that fact.

## 1. Standard NN format export — yes, and what each buys

| route | status | what it buys |
|---|---|---|
| native torch (`state_dict` + module) | works today (tcg_tensor.py) | GPU via `.to('cuda')`, torch.compile fusion, AMP/bf16, DataLoader batching, DDP multi-GPU, TensorBoard, checkpointing — the entire training stack for free |
| `torch.export` / TorchScript | works (graph is static; masked LSE/softmax/einsum all supported) | deployable artifact, C++ runtime (libtorch) — an alternative road to "DemandLite" without hand-writing kernels |
| ONNX | partial | the sparse Pi matmul is the blocker (ONNX sparse support is weak). Workarounds: (a) export the DEMAND layers L1–L4 to ONNX and keep L5 as a runtime sparse op; (b) gather–scatter reformulation of Pi; (c) block-dense Pi for subnetworks. Recommended: (a) — the demand layers are the compute-heavy differentiable part anyway |
| TensorFlow SavedModel | via ONNX or reimplementation (mapping table in TCG_MATH sec 7) | only if a TF-based platform is mandated |

Practical recommendation: treat **torch + torch.compile on a GPU box** as the
production target ("existing technology" leverage, zero custom code), keep
ONNX for the L1–L4 demand subgraph as the interchange artifact. On this
CPU-only machine we develop/validate; the same file runs on GPU unchanged.

Expected GPU effect (straightforward estimates, to be measured on a GPU box):
the Z^2 softmax stack at TRMG2 scale (8 purposes x 2 skims x 3147^2) is
~10-50x on a consumer GPU; the Pi spmm (67.8M nnz) maps to cuSPARSE. The
whole inner calibration solve drops from minutes (numpy f32) to seconds —
this SUPERSEDES parts of the DemandLite C++ plan: torch.compile does the
kernel fusion we were going to hand-write.

## 2. Beyond ADMM: the optimizer menu the NN framing unlocks

- **Adam/AdamW with minibatched observations**: sample count subsets per step
  (SGD over links!) — scales to arbitrarily many observations and adds
  implicit regularization; epochs replace outer iterations.
- **L-BFGS on GPU** for the final polish (torch.optim.LBFGS).
- **Mixed precision (bf16)** for the softmax stack — memory-bandwidth-bound,
  so near-2x; keep f32 accumulations in LSE (already max-shifted).
- **Distributed**: periods/purposes are data-parallel; per-origin blocks
  (the ADMM sharing structure) are also a natural sharding — ADMM and DDP
  are complementary, not competing: ADMM for the OD-cell stage-2 layer
  (million-dim, decomposable, rails built-in), SGD/Adam for the behavioral
  theta and any low-rank adapter weights.
- **Implicit differentiation / DEQ** (see sec 4) for the feedback fixed point
  instead of unrolling.

## 3. Low-rank weight approximation — three concrete instances

(1) **Pi is already factored**: Pi = A·Delta is a structured factorization
    (paths are the factors). Column generation = growing the factor basis.
    Do NOT SVD Pi; exploit the path factorization directly (sec 7 of
    TCG_MATH: pricing c_r = A^T t).

(2) **Zone-space compression (the superzone encoder-decoder)**: with
    aggregation encoders E in {0,1}^{Z x Z'} (superzone_hier / demand_kmeans
    from the od-compression framework), run the demand layers in compressed
    space and decode: OD ~= E' ODhat E'^T, Pi_hat = Pi (E (x) E). This is a
    *linear autoencoder around the demand layers* — measured before:
    12-28x speedups at R^2 ~0.93-0.98 on Chicago/AZTDM. In NN terms: a
    bottleneck layer with frozen, interpretable weights.

(3) **LoRA-style low-rank utility adapters** (the new idea worth a paper):
    instead of calibrating only cluster-level dASC (a rank-1-per-cluster
    pattern), allow a rank-r correction to the DC utility matrix:
        U_p  <-  U_p + a_p b_p^T,   a_p, b_p in R^{Z x r},  r <= 4
    with trust region ||a b^T||_F penalized. This is exactly LoRA: frozen
    pretrained weights (TRMG2 coefficients) + tiny trainable low-rank
    adapters. Gauge/identifiability discipline from TCG_MATH sec 5 applies
    (column-center b, pin scales). It generalizes dASC (r=1 with b = cluster
    indicator) and gives the calibration a controlled capacity dial:
    r is the model-quality vs drift knob, reportable like lambda.

## 4. "Highway network tricks" — the mapping

| NN trick | travel-model counterpart | status |
|---|---|---|
| residual/skip connection | incremental assignment around base year: v = v_base + Pi (OD - OD_base) — calibrate the RESIDUAL demand, not the total; base year is the identity path | direct fit with the engine's warm-start machinery; recommended default for forecast-year runs |
| highway gating T·H + (1-T)·x | MSA damping of operator/skim refresh (gate = 1/n), and converged-period freezing in TRMG2's own feedback | already present; can be made learnable |
| deep equilibrium model (DEQ) | the skim-demand-assignment feedback fixed point; differentiate implicitly (one linear solve) instead of unrolling 5 iterations | the principled upgrade of our Danskin argument; implicit-diff through the fixed point is the right way to backprop the FEEDBACK loop, task #9's gradient story |
| layer norm / max-shift | LSE stabilization (already in the masked softmax) | done |
| dropout / stochastic depth | count-subset minibatching; random period/purpose sampling | free with DataLoader |
| weight sharing | same theta across periods (already); same adapters across forecast years | design choice, keeps dof low |

## 5. Sequencing (no new hardware required to start)

1. Package tcg_tensor as a proper `nn.Module` with `state_dict` (theta +
   optional LoRA adapters) — small refactor.
2. `torch.export` the demand subgraph; attempt ONNX for L1–L4; record the
   artifact in the repo (interchange format request fulfilled).
3. On a GPU box: torch.compile benchmark vs the CPU numbers in
   review/tcg_scaling.md — one number decides how much of DemandLite
   survives.
4. LoRA adapter experiment at Chicago-sketch scale: r in {1,2,4}, report
   %RMSE vs drift (Frobenius) curve alongside the lambda trust-region curve.
5. DEQ/implicit-diff prototype on the feedback loop (after task #9 builds
   the loop itself).

Guard-rail invariant throughout: TRMG2 coefficients stay frozen; adapters
and theta carry explicit norm penalties and bounds; every experiment reports
the drift table (the sec 5 identifiability discipline is unchanged by the
NN tooling).
