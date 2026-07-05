# OD validation framework — verifying a demand OD without the reference matrix

A learning framework for a common situation: you have an origin–destination
demand produced by the first three steps, but **not** the model owner's official
pre-assignment OD matrix. How do you know your OD is reasonable?

**You assign it and check the loaded network against observed traffic counts.**
This needs no extra input, and it stays valid later if the official OD is shared
(a direct OD-to-OD check is then added on top).

`od_validation.py` implements this. Results are reviewed with the model owners;
this page documents the *method* and the published *targets*.

## The order matters — magnitude bias first, always

> **① Check magnitude bias first. ② Only then read pattern.**

**① Magnitude bias** — the first and decisive mark:
```
bias = ( Σ modeled_volume − Σ observed_count ) / Σ observed_count
```
If the bias is large, the OD is not yet the right *size* — it is missing trips or
markets — and **no pattern metric matters yet.** A model at a fraction of observed
volume cannot be rescued by a good correlation. Fix magnitude first.

**② Pattern** — only after magnitude is in the ballpark. With the magnitude gap set
aside (model scaled to the count total), does the OD load the network in the right
*proportions*?
- **scale-adjusted %RMSE** — shape error, independent of the total
- **correlation R²** — do high-count links carry high modeled volume?
- **GEH < 5 (%)** — the traffic-engineering closeness measure
- read **by facility type, area type, screenline, volume group** — this localizes
  *where* any shortfall sits (freeways, for instance, carry through / commercial /
  long-distance travel that a resident-only demand omits).

Reporting pattern before magnitude teaches the wrong habit; this framework refuses to.

## The target — TRMG2's own published validation

The bar to aim for is the model owner's reported accuracy. TRMG2's published
count validation (from its documentation) sets the reference:

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

Freeways validate tightest (few, high-volume, well-counted); local streets loosest
(many, low-volume, noisy) — a universal shape worth teaching on its own.

## When the official OD arrives
`validate_vs_matrix()` compares the official pre-assignment OD to
`matrices/f_od_*.npy` directly — matrix %RMSE, coincidence ratio, and 12-district
cell agreement — *in addition to* the count validation above, which never depended
on it.

## Run
```
python od_validation.py     # -> review/od_validation.md (local; shared with owners)
```
Requires the per-period assignment outputs (regenerate with `make_od_4period.py`).
