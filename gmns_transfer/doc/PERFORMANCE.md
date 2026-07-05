# Performance analysis — TRMG2 GMNS 4-period pipeline (2026-07-04)

Machine: Windows 11, Python 3.11, numpy/scipy (single-threaded BLAS ops for
these shapes), TAPLite kernel DTALite.exe (8 threads). Problem size: 3,147
internal zones x 33,963 nodes x 75,939 directed links; 8 HB purposes;
3 vehicle classes x 4 periods; ~1.8M OD rows written per class-period set.

## Measured wall times (bench_demand.py + make_od_4period.py run)

| stage | time | notes |
|---|---|---|
| S1 load inputs (SE, rates, size terms, factors) | 0.05 s | negligible |
| S2 skims: 2x dijkstra (time + distance) | **41.6 s** | scipy, single-threaded |
| S3 trip generation (8 purposes) | 0.003 s | pure vector ops |
| S4 attraction size terms | 0.03 s | |
| S5 one gravity evaluation (3147^2 softmax) | 0.358 s (f64) / 0.191 s (f32) | inner-loop unit cost |
| S6 beta calibration: 8 purposes x 40 bisection evals | **143.9 s** | 320 gravity evals + distance dot |
| S7 OD assembly (8p x 4per x 3 classes, PA->OD) | 15.2 s | transpose-add accumulation |
| S8 demand output, per matrix: CSV vs .npy | 5.7 s vs **0.11 s** | 52x gap; 12 matrices ~70 s total |
| Assignment: TAPLite x 4 periods (30 FW iters each) | **593 s** (130/130/145/188) | all gaps <= 0.007% |
| **Total pipeline** | **~870 s (~14.5 min)** | kernel 68% / demand-Python 32% |

Bottleneck ranking: (1) 4x kernel assignment 593 s, (2) beta calibration 144 s,
(3) demand CSV writes ~70 s, (4) skims 42 s, (5) OD assembly 15 s.

## Improvement plan (staged, with expected effect)

### P0 — cheap Python wins (hours of work, ~3.5x demand-side)
- float32 gravity: 1.9x measured on S5 -> S6 144 -> ~76 s.
- Brent/secant root-finding instead of 40-step bisection (avg trip length is
  smooth & monotone in beta): 40 -> ~7 evals, additional ~5x -> S6 ~15 s.
- Binary demand (.npy or the kernel's DTAB `demand_format=1` via
  dtalite_qa.demandbin): 70 s -> ~2 s writes AND faster kernel-side parsing.
- Thread the two dijkstras + chunk sources across processes: 42 -> ~8 s.
- Demand side total: ~270 s -> ~45 s.

### P1 — assignment-side wins (existing kernel features)
- Run the 4 period kernels CONCURRENTLY (2 at a time on 8 cores): 593 -> ~330 s wall.
- Warm starts between global feedback iterations (skip-Seed +
  `warm_start_flows`, already in the kernel: SCAG restart hit 0.07% gap at
  iter 1): later feedback loops ~3-5x cheaper than the first.
- In-process pybind11 binding (pytaplite native) removes exe spawn + CSV
  round-trips (~10-15 s/period).

### P2 — C++ demand kernel ("DemandLite" companion to TAPLite)
The gravity/DC layer is a perfect C++ target: the numpy version materializes
4 temporaries per evaluation (u, exp, rowsum, q) at 3147^2 x 8B = 79 MB each —
it is memory-bandwidth-bound. A fused OpenMP C++ loop (one pass per origin row:
exp, accumulate rowsum, avg-length, write OD) with float32:
- expected 8-15x vs current f64 numpy on 8 cores -> full calibration < 10 s,
  OD assembly < 2 s;
- natural packaging: `demandlite` pybind11 module beside pytaplite (same
  static-link MinGW + GIL-release pattern), API
  `calibrate_gravity(P, lnA, T, D, target_len) -> (beta, OD)` and
  `assemble_od(purpose_mats, tod, dir, occ) -> class-period OD` writing DTAB
  binary directly for the kernel;
- keeps the architecture rule: C++ = compute kernels, Python = orchestration.

### Projected end state
| | today | after P0-P2 |
|---|---|---|
| demand steps (gen+distribution+assembly+IO) | ~270 s | **~15 s** |
| 4-period assignment (first global iteration) | 593 s | ~330 s |
| subsequent feedback iterations (warm start) | ~870 s | **~100-150 s** |
| full 5-iteration feedback roadway model | ~70 min (est.) | **~12-15 min** |

Reference point: the original TRMG2 full model is a multi-hour run (all
components). Our roadway-only replication already fits in 15 min/iteration
in pure Python + kernel, and the plan above brings a full feedback-converged
roadway model under ~15 minutes total.

## v1 demand accuracy snapshot (context for the numbers above)

Daily assigned vs 4,659 counts: captured share 25.4% (expected low: v1 is
resident HB only — no NHB, CV/trucks, externals, university, airport; freeways
worst at -79% because externals+trucks concentrate there). Per-period
convergence 0.006% / 0.000002% / 0.002% / 0.0002%. v2 accuracy work (xgboost
production-model port for exact person-level rates, NHB regression layer,
CV gravity + NCSTM OMX externals) is tracked separately from performance.
