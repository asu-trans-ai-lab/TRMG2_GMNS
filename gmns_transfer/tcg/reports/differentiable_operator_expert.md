# 3-Level Operator Design Brief — Travel-Demand Computational Graph
### Expert brief — differentiable computational-graph operator specialist

## 0. Design invariant (the spine)

One math core, three views. Everything is `v = Pi @ vec(OD)`, `OD = D(theta)`.
The three "levels" are not three implementations — they are three **entry points
into one cached forward tape**. Duplication is avoided by making Level 2 =
`compose(Pi_op, Level1)` and Level 3 = the same `Pi_op` with its two factors
`A_inc, Delta` exposed instead of fused. A single `Ctx` (context/cache) dataclass
carries everything backward needs.

---

## 1. The clean 3-level operator API

Common context object (what each forward caches for backward):

```python
@dataclass
class DemandCtx:
    # Level 1 intermediates (all needed by the analytic adjoints)
    P: np.ndarray          # [Z] or [P,Z] generation, post-balancing
    logits2: np.ndarray    # [Z,Z] within-cluster utilities (finite entries only)
    Y_upper: np.ndarray    # [Z,C] upper-nest logsumexp (cluster inclusive values)
    q_lower: np.ndarray    # segmented lower softmax shares (CSR-packed, 1/C sparse)
    cluster_of: np.ndarray # [Z] dest->cluster id (the mask)
    seg_ptr: np.ndarray    # CSR row pointers for the [Z,Z,C] segment reduction
    mode_tod_W: np.ndarray # linear map coeffs used in OD assembly
    OD: np.ndarray         # [Z,Z] output (also the Level-1 return)
```

### Level 1 — demand operator `D(theta) -> OD`

```python
def demand_forward(theta, net, *, cache=True) -> tuple[OD, DemandCtx]: ...
def demand_backward(dL_dOD, ctx) -> dtheta: ...     # analytic VJP
```

- Shapes: `theta` = flat param vector (generation coeffs, nest params lambda_c,
  mode/TOD betas); `OD` = `[Z,Z]`.
- The `[Z,Z,C]` masked LSE is **never materialized dense**. Forward computes the
  upper nest via a **segmented log-sum-exp** over `seg_ptr` (CSR), giving the 12x
  at TRMG2 scale. `ctx.q_lower` stores only the finite entries.
- Backward is the verified chain: cluster-softmax VJP `Vbar = y - Q*Y` applied
  segment-wise, then upper-nest, then generation. No autograd needed here — but
  it must equal autograd to 1e-14 (your existing check).

### Level 2 — assignment-composed forward `theta -> v`

```python
def full_forward(theta, net, Pi, *, cache=True) -> tuple[v, FullCtx]:
    OD, dctx = demand_forward(theta, net)
    v = Pi @ OD.reshape(-1)          # Pi is [A, Z*Z], fixed within inner solve
    return v, FullCtx(dctx=dctx, Pi=Pi)

def full_backward(dL_dv, fctx) -> dtheta:
    dL_dOD = (fctx.Pi.T @ dL_dv).reshape(Z, Z)   # exact for fixed columns
    return demand_backward(dL_dOD, fctx.dctx)
```

Level 2 owns **no new math** — it is literally `Pi @` glued to Level 1, and its
backward is `Pi^T` glued to Level 1's backward. That is the anti-duplication
guarantee: `full_backward` calls `demand_backward`.

### Level 3 — factored sparse form (`A_inc`, `Delta` separate)

```python
class AssignmentOp:
    def __init__(self, A_inc, Delta):   # [A, n_paths] csr, [n_paths, |OD|] csc
        self.A_inc, self.Delta = A_inc, Delta
    def matvec(self, od_vec):           # v = A_inc @ (Delta @ od_vec)
        self.path_flow = self.Delta @ od_vec        # cache path flows
        return self.A_inc @ self.path_flow
    def rmatvec(self, dL_dv):           # Pi^T dL_dv = Delta^T @ (A_inc^T @ dL_dv)
        return self.Delta.T @ (self.A_inc.T @ dL_dv)
    def append_paths(self, new_rows_A, new_rows_Delta):
        # column generation: vstack path rows, keep everything else valid
        ...
```

Level 3 is Level 2 with `Pi` **un-fused** into `A_inc @ Delta`, exposing the path
axis so **column generation appends rows** without rebuilding `Pi`. `Pi @` in
Level 2 is the fused fast path; `AssignmentOp` is the same operator when you need
the path structure. Provide `AssignmentOp.as_dense_Pi()` so Level 2 and Level 3
are provably the same map (consistency check C1 below).

**One codebase:** `demand_forward/backward` is written once. `Pi` is one operator
object with two representations (dense-fused `.matvec` vs factored `A_inc@Delta`).
Levels 2 and 3 differ only in which representation they hold; both call the
identical `demand_*`. Nothing is reimplemented.

---

## 2. Differentiability across levels & the bilevel loop

**Where calibration operates:** Level 2 (behavioral theta-calibration) — you need
`dL/dtheta`, and `v` is where the loss lives. The adjoint chain is:

```
dL/dv  --Pi^T-->  dL/dOD  --demand_backward-->  dL/dtheta
```

**`dL/dOD = Pi^T dL/dv` is exact when columns are fixed.** This is Danskin/
envelope at UE: with the path set frozen and `Delta` treated as constant, the
assignment map is linear (`v = Pi @ od`), so its Jacobian *is* `Pi` and the VJP is
exactly `Pi^T`. Your `R^2 = 1.0` reproduction is the numerical witness. This is
the inner-loop gradient.

**When it is NOT exact:** the moment the equilibrium path set / shares must change
with `od` — i.e., `dDelta/dod != 0`. Two regimes:
- **Active-set changes** (a new path becomes attractive, or an existing path's
  share moves off the boundary): `Pi` itself depends on `od`, so
  `dL/dOD = Pi^T dL/dv` drops the `(d Pi/d od)^T` term. Fine to ignore *within*
  an inner solve (Danskin holds at the equilibrium point for the frozen active
  set), wrong *across* one.
- **Large `theta` steps** that push flows far enough to invalidate UE. Then `Pi`
  is stale.

**Bilevel outer loop:**
```
repeat:
  (outer) rerun kernel at current OD  ->  refresh A_inc, Delta  ->  rebuild Pi        # UE resolve
  (inner) fix Pi; gradient calibration on theta via Level 2 adjoint until inner-gap small
  check outer relative gap (Pi drift / flow drift)
until outer gap < tol
```
The inner loop uses the exact fixed-column adjoint (cheap, exact by Danskin). The
outer loop absorbs `dDelta/dod` by *re-solving* rather than differentiating
through the kernel — this is the standard MPEC-with-implicit-UE treatment and
avoids differentiating the equilibrium. Trust-region on the theta step keeps the
frozen-`Pi` assumption valid (tie the TR radius to observed `Pi` drift between
outer iterations).

---

## 3. ADMM (module A) as an operator

**Ordering: SEPARATE layer, stacked AFTER behavioral theta-calibration — not
interleaved.** Argument:
- The two stages estimate **different things**. Theta-calibration fits
  *behavioral parameters* (a low-dim, structured, transferable object). ADMM
  stage-2 fits *per-OD-cell multipliers* `x` (a high-dim, unstructured
  correction, `|OD|` free variables) to close the residual counts gap.
  Interleaving lets the flexible cell-multipliers absorb signal that should update
  behavior, causing **identifiability collapse** — theta stops moving because `x`
  already explained the counts. Stack them: behavior first (bias the structure
  right), then a *bounded* cell correction with `mu||x-1||^2` as a trust-region
  drift rail keeping `x` near 1 (near the behavioral prior).
- Practically: theta-calibration is the outer/slow layer; ADMM is a fast post-fit
  polish that you can re-run cheaply whenever `Pi` refreshes. Keeping it separate
  means the ADMM block never contaminates the behavioral gradient.

**Operator signature:**
```python
def admm_odme(Pi_obs, od_seed, counts, *, W, mu,
              rho=1.0, max_iter, tol) -> tuple[x, AdmmCtx]:
    # solves  min ||W (Pi_obs @ (od_seed * x) - counts)||^2 + mu ||x - 1||^2
    # per-origin block split (Boyd 7.3 sharing form)
    # returns x [|OD|] per-OD multipliers, ctx (for optional VJP / warm start)
```
- Inputs: `Pi_obs` = `Pi` restricted to rows of **observed/counted links**
  `[A_obs, |OD|]`; `od_seed` = post-theta OD (the seed the multipliers scale);
  `counts`; diagonal weight `W`; `mu` drift rail.
- Output: `x` per-OD multipliers; adjusted OD is `od_seed * x`.
- Blocks: partition columns by origin; each origin-block is a small least-squares
  solved in parallel; the sharing consensus variable is the **link-flow
  contribution** `z_a = sum_blocks Pi_obs^(k) x^(k)`, coupled by the counts. This
  is exactly Boyd 7.3 sharing.

**Influence restriction + Level-3 form combine cleanly:**
- **Drop zero-norm columns of `Pi_obs`**: any OD cell whose column in `Pi_obs` is
  all-zero touches no observed link -> its multiplier is *unidentifiable*, pinned
  to 1 by the `mu` rail. Remove those columns from the ADMM problem entirely (they
  only add mu-regularized noise). This is the "influence restriction": ADMM
  operates on the observable subspace only.
- **Use Level 3 to build `Pi_obs` without densifying**:
  `Pi_obs = A_inc[obs_links, :] @ Delta`. The zero-norm test is `column j of
  Pi_obs == 0` <=> `no observed link carries any path serving OD j` <=> cheap
  check on `A_inc[obs] @ Delta` sparsity pattern. Column generation appends paths
  -> `Delta` gains rows -> some previously zero-norm OD columns may become
  influenced -> the ADMM active column set grows monotonically. So Level 3's
  `append_paths` is what *feeds* the influence-restriction set.

**Differentiability:** ADMM's `x` is an argmin; if you ever need `dx/dtheta`, use
implicit differentiation of the KKT of the (strongly convex, thanks to `mu`)
inner LS. But in the recommended stacked design you typically **don't** backprop
through ADMM — it is a terminal polish. Expose the implicit-diff VJP only if you
want end-to-end gradients through the correction.

---

## 4. DemandLite (module B) as a drop-in Level-1/2 operator

DemandLite replaces `demand_forward/backward` with a fused OpenMP kernel. It must
be a **drop-in behind the same signature**, wrapped in a
`torch.autograd.Function`.

**Recommendation: expose BOTH `dl_forward` and `dl_backward`, and wrap in a
custom autograd.Function.** Do not rely on autograd tracing through the C++ (you
can't) and do not ship forward-only (you'd lose the whole point — the fused
backward is where the 12x C-fold reduction pays off symmetrically). The wrapper:

```python
class DemandLiteFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, net_handle):
        OD, blob = dl_forward(theta, net_handle)   # blob = opaque cached intermediates
        ctx.blob = blob                            # C-side cache, not Python tensors
        return torch.as_tensor(OD)
    @staticmethod
    def backward(ctx, dL_dOD):
        dtheta = dl_backward(dL_dOD.contiguous(), ctx.blob)
        return torch.as_tensor(dtheta), None
```

**API guarantees for gradient consistency:**
1. **Same segment reduction.** `dl_forward` must implement the identical masked
   segmented LSE over `seg_ptr` (dest's own cluster finite only). The `[Z,Z,C]`
   tensor stays implicit; the C++ loops the CSR segments (the 12x). Cluster
   assignment `cluster_of` and `seg_ptr` are passed in / shared, not recomputed
   with a different convention.
2. **Exact adjoint, not autodiff-of-C.** `dl_backward` must implement the verified
   analytic VJP `Vbar = y - Q*Y` segment-wise + the upper-nest + generation chain
   — bit-for-bit the same formula as the numpy reference. The regression is:
   `dl_backward` output == `demand_backward` output to ~1e-12 (looser than 1e-14
   only for float accumulation order under OpenMP; use deterministic reduction
   order or Kahan if you want tight).
3. **Cache contract.** `blob` must hold exactly
   `{P, Y_upper, q_lower(CSR), cluster_of, seg_ptr}` — the same set the numpy
   `DemandCtx` caches. This makes the reference and DemandLite *interchangeable*:
   a test can run `dl_forward` then `demand_backward` (cross-adjoint) and get the
   same gradient.
4. **Determinism flag.** OpenMP reduction order affects the last ~2 ULP. Provide
   `DL_DETERMINISTIC=1` (fixed chunking / ordered reduction) for the consistency
   tests; allow fast nondeterministic mode in production.
5. **Numerical guards match.** Same LSE max-subtraction, same `lambda_c` clamping,
   same handling of empty clusters — otherwise adjoints diverge at boundaries.

DemandLite slots into Level 2 unchanged: `full_forward` just calls
`DemandLiteFn.apply(theta, net)` instead of `demand_forward`. `Pi @` and `Pi^T`
are untouched (they're already fast/sparse and don't need C++).

---

## 5. Folder layout & consistency-check plan

```
demandgraph/
  core/
    segments.py        # CSR seg_ptr build, segmented LSE, cluster mask  (THE shared kernel math)
    adjoints.py        # Vbar = y - Q*Y, upper-nest VJP, generation VJP  (analytic, once)
  level1/demand.py     # demand_forward/backward  -> uses core/
  level2/full.py       # full_forward/backward = Pi @ (Level1)           (no new math)
  level3/assignment.py # AssignmentOp(A_inc, Delta), append_paths, as_dense_Pi
  admm/sharing.py      # admm_odme, per-origin blocks, influence restriction
  demandlite/
    _kernel.cpp/.pyx   # dl_forward/dl_backward (OpenMP)
    wrapper.py         # DemandLiteFn(autograd.Function)
  lowrank/adapters.py  # low-rank theta / OD adapters (optional demand parameterization)
  tests/consistency/   # the cross-checks below
```

**Minimal cross-checks that prove mutual consistency:**

| ID | Check | Proves |
|----|-------|--------|
| C1 | `AssignmentOp(A_inc,Delta).as_dense_Pi() == Pi` (and `matvec == Pi@od`) | Level 3 factored == Level 2 fused |
| C2 | `full_forward(theta) == (Pi @ demand_forward(theta).reshape(-1))` | Level 2 == Level 1 then Pi |
| C3 | `torch.autograd.gradcheck` vs `demand_backward` and vs `full_backward` (finite-diff) to 1e-6; analytic vs autograd to 1e-14 | adjoint chain correct end-to-end |
| C4 | `Pi^T dL/dv` (Level2 backward) == `Delta^T (A_inc^T dL/dv)` (Level3 rmatvec) | fused vs factored adjoint agree |
| C5 | segmented-LSE forward == dense `[Z,Z,C]` masked LSE on a small Z (reference brute force) | the 12x sparse path is exact, not approximate |
| C6 | `dl_forward == demand_forward` (OD, 1e-12) and `dl_backward == demand_backward` (dtheta, 1e-12) under `DL_DETERMINISTIC` | DemandLite == reference, fwd AND bwd |
| C7 | cross-adjoint: `dl_forward` then `demand_backward` == full DemandLite gradient | cache contract (blob == DemandCtx) is honored |
| C8 | ADMM monotone: `||W(Pi_obs (od*x) - counts)||` decreases vs `x=1`, and `x->1` as `mu->inf` | ADMM reduces count loss; drift rail behaves |
| C9 | influence restriction: dropping zero-norm `Pi_obs` columns leaves `x` on kept columns unchanged (to solver tol) | restriction is lossless |
| C10 | column-gen invariance: after `append_paths`, `as_dense_Pi()` still reproduces kernel volumes (R^2=1.0) and previously-zero-norm columns update correctly | Level 3 + ADMM active-set growth is consistent |
| C11 | bilevel sanity: one outer refresh reduces outer relative gap; inner (fixed-Pi) calibration reduces count loss monotonically under TR | inner Danskin exactness + outer loop wiring |

C5 is the load-bearing one for the sparse claim; C6/C7 for DemandLite; C1/C2/C4
tie the three levels; C8/C9 for ADMM; C11 for the bilevel.

---

## Key recommendations, distilled

- **Write the kernel math once** in `core/` (segmented LSE + analytic adjoints).
  Levels 1/2/3 are thin compositions, never reimplementations.
- **Calibrate at Level 2**; adjoint is `Pi^T` then `demand_backward`. Exact by
  Danskin **only for fixed columns** — refresh `Pi` in the outer loop, don't
  differentiate through the kernel; gate theta steps with a trust region tied to
  `Pi` drift.
- **Stack ADMM after behavioral calibration**, don't interleave — otherwise
  cell-multipliers steal behavioral signal. Feed its active column set from
  Level 3's `A_inc[obs] @ Delta` sparsity; drop zero-norm (unidentifiable) columns.
- **DemandLite ships fwd + bwd behind a `torch.autograd.Function`**, guaranteeing
  the same segment reduction, the same analytic VJP, and the same cache blob as
  the numpy reference (`DL_DETERMINISTIC` for the 1e-12 regression).
- **Prove it with C1-C11** — the two anchors are C5 (sparse == dense LSE) and
  C6/C7 (DemandLite == reference, fwd and bwd, via the shared cache contract).
