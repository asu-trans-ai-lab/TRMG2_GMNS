# OD validation framework — from seed demand to loaded network

A learning framework for a common situation: we have an OD demand produced by the
first three demand-model steps, but **not** yet the model owner's official
pre-assignment OD matrix. How do we know the OD is reasonable?

**Do not start naively from assignment.** A bad loaded-network result can come from
several upstream causes — wrong socioeconomic inputs, wrong trip-generation rates,
missing household/employment markets, short/long trip-length bias, missing
commercial/external trips — and *only then* assignment or network problems. So the
validation follows an ordered QA ladder, and each rung is made visible before the
next is read:

> **① Seed data & trip generation first · ② Trip distribution & trip-length second ·
> ③ Assignment / count validation third · ④ Direct OD-to-OD only when the official
> matrix is shared.**

This prevents the common mistake of jumping to R², %RMSE, tensors, or ODME while the
seed demand is still incomplete — i.e. fixing the wrong problem.

---

## ① Seed data & trip generation — the first demand-size gate
Before assigning anything, check whether trip-production and attraction totals are
in the right range, by purpose, household group, income, auto ownership, workers,
employment category, and district/area type.
> **Does the synthetic demand have the right *number of trips* before we ask the
> network to carry them?**
```text
production_bias_purpose = (modeled_prod − benchmark_prod) / benchmark_prod
attraction_bias_purpose = (modeled_attr − benchmark_attr) / benchmark_attr
```
Benchmarks: the model report, NCHRP-style trip-rate ranges, survey summaries, or the
owner's published control totals. A large bias is **not** necessarily a criticism of
the original model — it often means incomplete SE inputs, missing households/jobs, or
an incomplete translation of the generation equations. If generation is 20–30% off,
no tensor / ODME / assignment calibration can fix it: **the seed demand is not yet
online.**

**RAG review**

| Gate | Green | Amber | Red |
|---|--:|--:|--:|
| Total productions by purpose | ±5–10% | ±10–20% | > ±20% |
| Total attractions by purpose | ±5–10% | ±10–20% | > ±20% |
| District-level productions | mostly aligned | several outliers | systematic spatial bias |
| Household / employment controls | consistent | partial mismatch | missing / misclassified |
| Top-zone review | largest zones plausible | several suspicious | major missing generators |

---

## ② Trip distribution — the trip-length & market-structure gate
> **Are trips going to reasonable destinations, with reasonable trip lengths,
> *before* assignment?**
Check against the model report and published summaries: average trip length by
purpose, trip-length frequency distribution, intrazonal vs interzonal share,
district-to-district movement, HBW vs HBO vs NHB patterns, long-distance/cross-region
flows, and external / through / airport / university / freight markets if the official
model has them.
```text
avg_trip_length_bias_purpose = (modeled_ATL − benchmark_ATL) / benchmark_ATL
```
A model can have the right *number* of trips but make them too short or too local. Then
totals look fine while freeway VMT, screenline volumes, and regional through-movements
come out too low — which is why average trip length and district-pair review come
**before** loaded-network statistics.

| Check | What it tells us |
|---|---|
| Average trip length by purpose | trips too short or too long |
| Trip-length distribution | whether the long-trip tail is missing |
| District-to-district matrix | whether regional markets are structurally right |
| Intrazonal share | whether too many trips stay local |
| Screenline OD movement | whether major corridor demand is represented |
| Purpose-specific patterns | whether HBW/HBO/NHB/CV/external are mixed correctly |

---

## ③ Assignment / count validation — the loaded-network gate
Only now assign the OD and compare loaded volumes to observed counts (no official OD
needed). The **first mark is still magnitude bias**, read before any pattern metric:
```text
bias = ( Σ modeled_volume − Σ observed_count ) / Σ observed_count
```
If magnitude bias is large, do **not** celebrate correlation — go back upstream and
identify whether the missing volume is generation, distribution, missing purposes,
missing external / commercial trips, time-of-day factoring, or period aggregation.
Only after magnitude is in range read pattern: scale-adjusted %RMSE, correlation R²,
GEH<5%, and errors by facility type, area type, screenline, and volume group.
> **Magnitude first · Pattern second · Diagnosis third.**

---

## ④ VMT & VHT — system-level reasonableness
If generation and distribution are reasonable, VMT should be too. VMT right but VHT
wrong points to congestion / capacity / speed / VDF issues rather than demand.

| VMT | VHT | Likely issue |
|---|---|---|
| Low | Low | missing trips or trips too short |
| Low | High | too much congestion on too little demand — network/capacity |
| Correct | Low | speeds too high / congestion underrepresented |
| Correct | High | speeds too low, capacity too low, bottleneck overloading |
| Correct | Correct | system-level demand & congestion broadly plausible |

---

## ⑤ Published model targets are the reference
Without the official OD, the benchmark is the model owner's published validation.
TRMG2's published count-validation %RMSE targets:

| Facility type | published %RMSE |
|---|--:|
| Freeway | 10.4 |
| ML Highway | 17.1 |
| TL Highway | 23.2 |
| Major Arterial | 30.0 |
| Arterial | 44.4 |
| Major Collector | 48.8 |
| Collector | 60.4 |
| Local | 75.9 |
| **All** | **34.6** |

These teach the expected *shape*: freeways validate tightest (high-volume, well-
counted); collectors/local are noisy and naturally higher. A **freeway shortfall
often signals missing long-distance / through / external / commercial / NHB markets.**

---

## ⑥ When the official OD matrix arrives
Add a direct OD-to-OD comparison *on top of* count validation: matrix %RMSE,
coincidence ratio, district-to-district cell agreement, purpose-level matrix, trip-
length distribution, production/attraction marginals. This does **not** replace the
loaded-network check — the final test of an OD is whether it loads the network
correctly.

---

## The correct learning sequence
1. **Seed data & trip generation first** — are the trip totals by purpose reasonable?
2. **Trip distribution second** — are trip lengths, district flows, and markets reasonable?
3. **Assign & check magnitude bias third** — does the network carry the right *total* traffic?
4. **Only then read pattern metrics** — *where* does it load correctly or not?
5. **Use published targets as the benchmark** — approach the owner's own validation quality.
6. **Add direct OD-to-OD later** if the official matrix is shared — an addition, not a replacement.

Implementation: `od_validation.py` (gate ③, magnitude-first). Gates ① and ② read the
model report's control totals and trip-length summaries when available; hooks are in
place. Interim reproduction numbers are reviewed with the model owners; this repo
publishes the framework and the targets.
