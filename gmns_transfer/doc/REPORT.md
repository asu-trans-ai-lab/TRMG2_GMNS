# TRMG2 Open Reproduction — Status Report, CPU Profile, Expert Panel, and
# Auto-Calibration Plan

Date: 2026-07-04. Package: `gmns/` inside the TRMG2 repo copy.
Principle in force: **reproduce, don't invent — every deviation named, gated,
and tied to the task that closes it.**

---

## 1. Executive summary

- The TRMG2 master network is fully reproduced in GMNS and **proven** against
  the source model's own export (topology 135,922/135,922 exact; gmns-ready 0 errors).
- A first-3-steps demand chain (v1.5) runs end-to-end using TRMG2's own
  destination-choice tables — no invented parameters — feeding 4-period,
  3-class TAPLite assignments that converge to ≤0.0005% gaps in ~10.5 min
  total wall time.
- A four-expert panel review (sections 5–6) confirmed the architecture,
  found concrete fidelity defects to fix (one silent purpose-coverage bug,
  one utility-term misreading), and graded the gate suite: strong taxonomy,
  needs the assignment-side gates (screenlines, VMT identity, per-link R²)
  that TRMG2's published data already supports.
- The auto-calibration design (CG-ODME/TCG-lite, section 7) calibrates
  **exactly TRMG2's own calibration knobs** (~200 dof: Calibrated_Delta*
  columns + calibration_factors.csv) by analytic forward/backward gradients
  in the same pipeline — with a synthetic-recovery experiment as the fidelity
  proof and a counts experiment gated on demand completeness.

## 2. TRMG2's exact step sequence — what runs, what gates it

| # | TRMG2 step (rsc) | reproduction | gated against | status |
|---|---|---|---|---|
| 01 | CreateScenario | build_gmns.py (base 2020) | geometry export | DONE |
| 02 | Network Calculations | build_gmns.py | capacity/speed lookups; gmns-ready; counts coverage | DONE (A1/A2 named) |
| 03 | Accessibility | — | their per-TAZ access_* values (nhb_generation.csv) | PENDING (G6, task #2) |
| 04 | Population synthesis + AO | — | 731,913 HH / 1,783,548 persons; ao_calib_targets | PENDING (G10/G11, task #3) |
| 05 | Resident productions | aggregate person rates (stand-in) | survey wTrips per purpose | **G3 FAIL** — HBW +49%, EK12 −44% → drives task #4/#5 |
| 06 | NonMotorized | — | NM calibration_targets | PENDING (G12) |
| 07 | Aggregation (segments) | none (aggregate) | — | DEVIATION D4 (segment averaging) |
| 08 | Time of Day | exact master CSVs | factors sum to 1 (G16 planned) | DONE |
| 10 | Skimming | init_cong_time dijkstra | — | PARTIAL — period-mapping fix queued (v1.6-3) |
| 12 | Mode choice | survey shares (stand-in) | Target_HB_MCShares | DEVIATION D2 (task #5) |
| 13 | Destination choice | **their zone+cluster tables verbatim** | IZ shares; TLD; avg time diagnostic | v1.5 DONE; fidelity fixes queued (v1.6-1/2/4/5) |
| 13b | Apportionment | P × P(dest) × P(mode) | conservation identities (G16 planned) | DONE |
| 14 | NHB | — | their per-TAZ NHB values | PENDING (G13, task #7) |
| — | CV/Univ/Airport/External | — | component totals; freeway counts | PENDING (task #8) |
| 16 | Assignment matrices | exact TOD/dir/occ CSVs | occupancy conservation | DONE (other_shares fix queued) |
| 17 | Roadway assignment | TAPLite kernel | gap ≤0.01% (G8 PASS ≤0.0005%) | DONE (3 of 6 classes) |
| — | Feedback loop | — | skim %RMSE ≤0.1, ≤5 iters | PENDING (G14, task #9) |

Current gate tally (v1.5): **PASS G1, G2, G8; FAIL G3 (informative — quantifies
the generation stand-in); DEVIATION G4/G7/G9 (named D1–D3); PENDING G6,
G10–G14.** Auditor-mandated additions G15–G19 (screenlines, conservation
identities, TLD coincidence ratio, VMT + volume-group %RMSE + per-link R²,
sensitivity/reproducibility) are specced in section 6.

## 3. CPU time per step and improvement plan

Measured (3,147 zones × 33,963 nodes × 75,939 links; Windows, Python 3.11,
kernel 8-threads):

| stage | v1 (gravity) | v1.5 (their nested DC) | improvement path |
|---|---|---|---|
| load inputs | 0.05 s | ~0.1 s | — |
| skims (dijkstra) | 41.6 s (2 graphs) | ~40 s (2) → 4 graphs in v1.6 | thread/chunk sources → ~8 s; or kernel-side skims |
| generation + size terms | 0.04 s | 0.04 s | — |
| distribution | 143.9 s (320 β-bisection evals — **removed in v1.5**) | ~30–60 s (16 nested-DC evals) | f32 + fused C++ (DemandLite) → <5 s |
| OD assembly + demand write | 15 s + ~70 s CSV | same | DTAB binary (0.11 s/matrix, 52×) → ~2 s |
| 4 × assignment | 593 s | 524 s (118/168/120/138) | concurrent pairs → ~330 s; warm starts → 3–5× on later feedback iters |
| **total** | ~870 s | **628 s** | **first iteration ~6 min; warm feedback iters ~2 min** |

C++ priority order (revised by the calibration review): **(1) sparse π/select-
link export in the kernel** (unblocks CG-ODME and select-link analysis — was
already a panel P0), **(2) DemandLite fused nested-DC forward+backward**
(8–15× on the distribution stage and the calibration inner loop), (3)
bi-conjugate FW. The gradient work does NOT wait on C++: numpy float32 runs
the calibration inner solve in 8–25 min, acceptable for the demonstrations.

## 4. v1.5 results snapshot

All 4 periods converge ≤0.00045%; daily 2.27M vehicles; captured share vs
4,659 counts ≈ 24% — **as designed** (resident HB only) and further reduced by
two now-identified defects (OD purposes dropped by a lookup bug — fixed in
v1.5b, rerunning; IZ term misapplied — v1.6). Freeways worst (−83%): externals
and trucks live there. These numbers are the honest baseline the component
build-out and the calibration experiments will be measured against.

## 5. Expert panel — verdicts and consolidated actions

Four independent reviews were run against the actual package and GISDK source.

**R1 — MPO program manager**: "a network and assignment engine reproduction
with a demand scaffold — and the package itself agrees, which is the most
trustworthy thing about it." Would certify nothing until components close the
count gap, BUT: topology validation = "best network QA artifact I've seen";
demands an **engine-parity certificate** (TRMG2's own OD through our kernel,
link-by-link scatter vs reference volumes) as the single most convincing
deliverable; component order by count volume: externals → CV → NHB.

**R2 — GISDK veteran** (fidelity audit vs NestedDC.rsc line-by-line):
found 5 substantive items, priority-ordered:
1. eda-key bug silently dropping both OD purposes (~0.5–1M person trips/day)
   — *found independently by gate G3 the same hour; fixed, rerunning*;
2. `intra_cluster.IZ` is the **intrazonal diagonal** dummy (utils.rsc
   2428–2431), not same-cluster — misapplication gave home clusters a
   spurious e^1.415 ≈ 4.1× HBW boost; couple the fix with restoring
   intrazonal alternatives (area-based diagonal times, 10:75–89);
3. each period uses its OWN skim; the AM/PM-vs-MD/NT switch selects the
   directionality-averaged variant (pa·t + ap·tᵀ), not the period;
4. TRMG2's within-cluster conditional uses UNSCALED utilities (θ only in the
   logsum — CalcFinalProbs 395–405); our 1/θ sharpening distorts low-θ
   suburban clusters by 1.45–1.6×;
5. shadow-price 3-pass update, other_shares_hb.csv for auto_pay collapse,
   per-segment DC (share TVD 24% at segment level, ~1–2% aggregate), log1p
   size terms.

**R3 — validation auditor**: gate taxonomy praised; assignment-side gates
missing though reference data is on disk; tolerances must cite a basis
(NCHRP 716/765, TMIP, or "TRMG2's own achieved value"); the gate report must
exist as a committed artifact with a failing exit code; reference-file hashes
and a per-step "no invented parameters" boundary table required. Ranked
additions → section 6.

**R4 — calibration scientist**: integration design adopted in section 7.

**Consolidated P0 actions** (in order): (1) v1.5b rerun + gates rerun with
committed GATES_REPORT (in progress); (2) v1.6 fidelity fixes = R2 items 2–4 +
conservation identities gate G16; (3) staged component build-out externals →
CV → NHB with staged G9 tolerances (±25% → ±10%); (4) engine-parity
certificate when ITRE OD matrices arrive; (5) hygiene: exit codes, pinned
environment, kernel hash, config not hardcoded paths, regenerate stale
review docs.

## 6. Gate suite v2 (auditor-mandated additions)

| new gate | reference (on disk) | criterion |
|---|---|---|
| G15 screenlines/cutlines | count_comparison_by_{screenline,cutline}.csv | ±10%/±15% each + within ±5 pts of TRMG2's PctDiff |
| G16 conservation identities | none needed | OD row sums = P; TOD factors sum to 1; PA→OD and occupancy conserve; nonnegativity — exact PASS |
| G17 TLD per purpose | wAvgTrpLen + dc est tables | coincidence ratio ≥0.70; mean ±5% (replaces auto-PASS G5) |
| G18 VMT + volume-group + per-link | HwyCapacityImpacts base VMT 57,609,957; vol-group table | daily VMT ±5%; %RMSE ≤ published+3/stratum; R² ≥0.88, slope 0.95–1.05 |
| G19 sensitivity + reproducibility | HwyCapacityImpacts/HwyCapModeShift | reproduce ΔVMT +0.20%, Δdelay −4.60% within ±50% rel; rerun drift ≤0.1% |

Plus: G3 tolerance → ±5% regional / ±10% per purpose (NCHRP 716); G5 →
DIAGNOSTIC status; G8 → also compute TRMG2's %RMSE convergence measure;
gates exit non-zero on FAIL; SHA-256 manifest of all reference files.

## 7. Auto-calibration in the SAME pipeline (CG-ODME / TCG-lite)

**Position:** TRMG2 is itself a calibrated model with named knobs —
`Calibrated_DeltaASC`/`Calibrated_DeltaIC` (dc/*_cluster.csv) and
`calibration_factors.csv`. CG-ODME turns **exactly those ~200 knobs and no
others**, trust-region-anchored to TRMG2's published values; every `*_zone.csv`
behavioral coefficient stays frozen. The calibrated artifact is written back
as TRMG2-format CSVs — a calibrated run is the same pipeline with updated
tables.

**Identifiability by construction:** generation factors set totals (anchored
by survey wTrips), delta-ASCs set spatial pattern (anchored by counts +
screenlines) — block-diagonal, cleaner than OD-cell ODME.

**Loss:** counts (normalized per link, stratified 80/20 calibrate/holdout) +
screenlines + survey wTrips (heavy weight) + TLD quantiles + trust region
λ‖θ−θ_TRMG2‖²; mode/IZ shares as guard gates, not loss terms. GradNorm-style
weight init; λ by L-curve, reported.

**Bi-level loop = TRMG2's own feedback loop:** forward(θ) → DTAB demand →
4 warm-started kernel runs → new skims + sparse π on observed links (kernel
select-link export = C++ item 1; MSA-damped across refreshes) → L-BFGS-B inner
solve (~60–120 evals; 8–25 min numpy f32, <2 min with DemandLite). End-to-end
≈ 45–75 min. Convergence: θ step norm + the same skim %RMSE ≤ 0.1 as G14.

**Experiments:**
- **A (fidelity proof, runnable after v1.6):** zero TRMG2's published deltas,
  generate pseudo-counts with them, recover them (target: HBW ΔIC vector
  within ±0.05; gradcheck ≤1e-3). Becomes gate "CG-recovery".
- **B (quality improvement, precondition: captured share ≥ 0.9 after
  components land):** baseline vs CG-calibrated on identical pipeline;
  deliverable tables: PctDiff/%RMSE by facility and volume group
  (calibration + holdout) vs TRMG2's published 34.58; guard metrics
  (wTrips ±15%, TLD ±10%); parameter-drift table; outer-loop trace.
  **Expected honest gain: All-links %RMSE 34.6 → ~29–32**, most of it on
  arterials/collectors; freeway accuracy is a component-completeness problem,
  not a calibration problem. Anything promising much more from behavioral
  knobs is OD-cell ODME in disguise — if desired, run it as an explicitly
  separate stage-2 with tight trust region.

## 8. Roadmap (tasks) — updated with panel findings

1. v1.5b + gates rerun (task #1, in progress) → commit GATES_REPORT.md
2. **v1.6 fidelity release** (task #6 expanded): IZ-diagonal fix + intrazonal
   alternatives, per-period skims with pa/ap averaging, unscaled within-nest
   conditional, shadow-price update loop, other_shares_hb collapse, log1p,
   segment-aware DC — each item cites its rsc lines from the R2 audit
3. Gate suite v2 (G15–G19 + tolerances + exit codes + manifest)
4. Components by count volume: externals (NCSTM OMX) → CV → NHB (task #7/#8)
5. Steps 03/04/05: accessibility → synthesis+AO → decision-tree rules
   (tasks #2/#3/#4), unlocking segment-level DC and the deferred utility rows
6. Step 12 nested MC (task #5) → logsum composites into DC
7. Feedback loop (task #9) + C++ items: π export → DemandLite fwd+bwd
8. CG-ODME Experiments A then B (section 7)
9. Engine-parity certificate when ITRE matrices arrive
