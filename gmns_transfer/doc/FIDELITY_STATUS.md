# Full-compatibility status — 3-step demand vs original TRMG2

Tracking the fixes from REVIEW2_FINDINGS.md (N1–N9) + the earlier veteran findings.

## Landed (fully reproducible with bundled + public inputs)
| fix | what it does | source of truth |
|---|---|---|
| N4 | zone-level demographic generation (Pct_Worker/Child), total = survey wTrips | 05 - Resident Productions.rsc |
| N2 | auto-fold by `other_shares_hb.csv` (HOV-skewed), not survey sov/hov shares | 16…rsc:144-177 |
| per-period skims | AM/MD/PM/NT each use their own congested time | 13…rsc:216-225 |
| intrazonal times | skim diagonal = √area·√2/3·(60/30), not blocked | 10 - Skimming.rsc:78-80 |
| IZ diagonal | `intra_cluster.IZ` on the i==j diagonal, not same-cluster | 13…rsc + intrazonals.csv |
| unscaled conditional | P(j\|c) uses raw U; θ only in the logsum | NestedDC.rsc CalcFinalProbs:395-405 |
| log1p | size term = Log(1+size), not Log(size) | 13…rsc:149-153 |
| N9 | HBW shadow-price loop (3 passes, step 0.85, balance ΣA=ΣP) | 13…rsc:275-330 |

That is the **entire DC-internal math** made faithful — nesting, conditional,
intrazonal handling, per-period skims, double constraint.

## Blocked — each remaining gap traces to an UPSTREAM step we have not built
| fix | needs | which step |
|---|---|---|
| **N8** MC logsums into DC (largest spatial term, transit coef up to 1.1) | **transit skims** (IVT/wait/fare/access) — not in GMNS | mode choice (12) + transit skimming; transit LOS = ITRE export |
| **N1** motorized-only productions into DC | `se.access_walk.O` (a computed gamma accessibility) + person `is_senior/HHKids/veh_per_adult` | accessibility (03) + population synthesis / auto-ownership (04) |
| **N3/N7** per-segment DC (v0/ilvi/ihvi/ilvs/ihvs) | segment shares per zone = auto-availability × income cross-class | auto-ownership model (04) |

**Key insight:** the DC-internal reproduction is now essentially complete. Full
compatibility with the original is now gated on **three upstream inputs**, not on
the distribution code:
1. transit skims (ITRE) → unlocks N8, the biggest spatial term;
2. a gamma-accessibility step (03) → unlocks the walk-accessibility terms + N1;
3. population synthesis + auto ownership (04) → unlocks the market segments (N3/N7)
   and the person-level generation/NM split.

Each has a clean plug-in point: N8 reads a frozen transit-LOS / mc_logsum matrix
when available; N1/N3 read per-zone accessibility and segment-share vectors. The
math that consumes them is already in place.

## Measured effect of the DC-math batch (v1.6)
Running the 4 DC-math fixes (per-period skims, intrazonal times, IZ-diagonal,
unscaled conditional) moved the outputs in the expected fidelity direction:

| purpose | avg AM time before → after | reading |
|---|---|---|
| W_HB_W | 13.0 → **16.5 min** | removing the spurious same-cluster boost lets work trips reach real job sites, not pile up locally |
| N_HB_OD_Long | 11.0 → **12.7 min** | ditto for long discretionary |
| N_HB_OD_Short | 10.9 → **12.9 min** | ditto |

Generation totals unchanged (fixed by N4, match survey wTrips exactly). Captured
share vs counts 0.273 → **0.296**. Trip times are now longer and more realistic —
the old low values were an artifact of the same-cluster boost + intrazonal
exclusion, not real behavior. `log1p` + the N9 shadow-price loop are coded and in
the next verification run.

## Skims & the tensor view
- How skims and assignment transfer through the pipeline (GMNS + TAPLite for auto;
  transit assumed-supplied): `SKIM_AND_ASSIGNMENT.md`.
- The whole 4-step as one differentiable Flow-Through-Tensor graph, everything
  supplied as tensors: `../tensor_ftt/`.
