# Problem statement — GMNS skims & assignment transfers

This is the contract for how level-of-service (skims) and assignment move
through the reproduction, so another MPO can plug in their own network.

## Principle: skims come from the GMNS network via the TAPLite kernel

The travel model is **supply-consistent**: every congested travel time used by
the demand model is produced by assigning demand to the **GMNS network** with the
**TAPLite (DTALite) kernel** — not read from a black-box matrix. The loop is:

```
 SE + params ─▶ demand OD ─▶ [TAPLite assign on GMNS link.csv] ─▶ link times
        ▲                                                              │
        └──────────────── skim (zone→zone congested time) ◀───────────┘
```

### Auto skims (implemented, GMNS-native)
- Input: `scenario/link.csv` (GMNS) with `vdf_fftt` free-flow times; per-period
  congested times seeded from `params/init_cong_time_2020.csv`.
- Skim engine: shortest-path over the GMNS link graph, one matrix per period
  (AM/MD/PM/NT), intrazonal diagonal = `√area·√2/3·(60/30)` min.
- After assignment, `link_performance.csv` travel_time replaces the seed → the
  next outer iteration re-skims. This is the feedback (step 9).
- **This is fully reproducible from bundled GMNS files. No proprietary software, no external
  skim matrix.**

### Transit skims (assumed-supplied for now)
- The mode-choice logsums (`mc_logsums.TransitComposite`, etc.) that feed
  destination choice need transit LOS (in-vehicle time, wait, fare, access).
  GMNS + TAPLite produce the **auto** side; the **transit** side needs a transit
  pathfinder we do not rewrite here.
- **Assumption (current):** transit skims / mode-choice logsums are *supplied as
  input tensors*. The pipeline exposes a plug-in point: drop a per-OD transit-LOS
  (or precomputed `mc_logsum`) matrix and the DC utility consumes it directly.
  Until supplied, those utility rows are deferred (documented in FIDELITY_STATUS).
- When ITRE provides the frozen transit skims, no code changes are needed beyond
  pointing the loader at the matrix.

## Assignment transfer = the Flow-Through-Tensor mapping

The assignment step is a Flow-Through-Tensor forward/backward pair. Demand and
supply meet through two sparse incidence tensors, extracted once from the kernel's
route output:

| tensor | shape | meaning |
|---|---|---|
| `B_OD,P` (Δ) | \|OD\| × \|P\| | OD → path choice proportions |
| `A_P,L`      | \|P\| × \|L\|  | path → link incidence |
| `Π = Aᵀ Bᵀ`  | \|L\| × \|OD\| | OD → link (the transfer operator) |

- **Forward (flow):** `f_P = Bᵀ f_OD` → `f_L = Aᵀ f_P` → `t_L = φ(f_L)` (BPR).
- **Backward (time/gradient):** `t_P = A t_L` → `t_OD = B t_P`; and the
  count-loss gradient `∂L/∂f_OD = Π ᵀ ∂L/∂f_L`.
- These matrices are **loaded from the pre-assignment result** (`route_assignment.csv`),
  never recomputed in Python — that is what makes calibration matrix-form and
  differentiable (see `matrix_ops.py`, `matrices/`).

## What another MPO must supply
1. A GMNS `node.csv` + `link.csv` (centroids first, sorted by from_node — see README).
2. Zonal SE data (`se_data/`) and the choice parameters (or their own).
3. **Optionally** a transit-LOS / mc_logsum matrix (else transit terms deferred).
Everything else — auto skims, assignment, the Π transfer operator, the
gradients — is produced from those by the GMNS + TAPLite pipeline.
