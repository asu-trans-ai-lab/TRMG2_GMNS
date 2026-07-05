# Second expert review of the 3-step demand code (make_od_4period.py)
## What the first review missed

The first (veteran-practitioner) review found 7 fidelity issues (IZ-as-same-cluster,
theta-scaled conditional, intrazonal exclusion, period-skim mapping, size
averaging, no shadow-price loop, log1p/other_shares). A second adversarial audit
against the GISDK source found **9 more defects (N1–N9)** the first review missed,
plus verified several suspected problems as NOT bugs. Ordered by impact on the
output OD.

### Top new findings

**N8 — the omitted mode-choice logsums are the DOMINANT term, not a minor one.**
The `nz(mc_logsums.*)` rows are deferred (known), but their *magnitude* was
missed: for N_HB_OD_Long the transit-composite coefficient is **1.1 — about 6×
the CongTime coefficient (−0.171)**. So the single strongest driver of
destination choice for OD_Long/OD_Short/HBO is currently absent, flattening those
distributions. This, not the IZ bug, is the largest spatial error. (K12 has no
mc_logsum row, so its DC is complete — correct.)

**N2 — the auto-mode fold uses the wrong shares (materially wrong SOV/HOV split).**
The code folds auto_pay/other_auto proportionally to the survey pct_sov/hov2/hov3
(line 286–290). TRMG2 uses a *separate* empirical distribution in
`other_shares_hb.csv`:

| purpose | TRMG2 other_shares (sov/hov2/hov3) | code's basis |
|---|---|---|
| W_HB_W | **0.482 / 0.276 / 0.242** | 0.855 / 0.048 / 0.018 |
| W_HB_EK12 | **0 / 1 / 0** | 0.492 / 0.484 / 0 |
| N_HB_OMED | 0.065 / 0.760 / 0.174 | 0.42 / 0.303 / 0.135 |

Paid/other-auto trips are disproportionately *shared rides*; the code puts ~85%
of them in SOV for work, TRMG2 puts 48%. Wrong SOV-vs-HOV vehicle counts (and
wrong occupancy division). Fix: read `other_shares_hb.csv`, fold with its factors.

**N1 — non-motorized trips are never removed before destination choice.**
TRMG2 splits productions into `_m` (motorized) / `_nm` at the person level in
`06 - NonMotorized.rsc`, and **only `_m` is aggregated and fed to DC**
(`07 - Aggregation.rsc:31-33`). The code feeds *total* productions to DC.
The line-289 renormalization approximately drops the NM share from the *assigned
volume*, but **the DC destinations are still chosen using motorized+NM
productions**, distorting the spatial pattern (walk-heavy purposes: OD_Short 25.5%
walkbike, K12 21%) and the HBW shadow-price balancing.

**N7 — the `Segment` column is entirely unread.** `parse_zone_table` never reads
`r["Segment"]`, so every segment-scoped coefficient (e.g. HBW `se.access_walk.D`
= 0.353 restricted to `v0`) is either dropped (deferred) or mis-scoped. Beyond
the known "size averaging," the *whole segmentation* (v0/ilvi/ihvi/ilvs/ihvs) is
absent — TRMG2 runs DC per segment.

**N3 — H/L size averaging contaminates HBW with the wrong segment's attractors.**
`w_hbw_size_L` includes `Office_EL`, `Retail_EL` that `_H` lacks; averaging (line
280) mixes low-income retail/office attractors into the high-income work size,
shifting HBW destinations for the wrong segment — a spatial error on the single
largest purpose. (Useful sub-finding: the `A /= len(size_cols)` halving is
*benign* — a constant `log(2)` cancels in the DC softmax.)

**N4 — flat per-capita productions for work purposes.** `P = HH_POP × constant`
discards the decision tree; a worker-rich zone and a retiree zone get the same
work productions. The dimensional base is OK (per-capita mean × total pop), but
the spatial flattening is the concrete cause of gate G3's purpose-level bias.
Fixable at zone level via `Pct_Worker/Pct_Child/Pct_Senior` (in se_2020.csv)
without full population synthesis.

**N9 — stale HBW shadow price, no attraction target.** The warm-start
`shadow_prices` is the converged value from a *prior* run; with the current (flat,
N4) productions it balances to the wrong marginal, and there is no `w_hbw_a`
attraction target or 3-iteration update loop.

**Sign-flip nuance on the known IZ bug:** for most purposes the mis-applied
`intra_cluster.IZ` is a ~4× home-cluster *boost*, but for W_HB_EK12 the summed
coefficient is negative (−0.825), so it's a *suppression* — the bug's direction
is purpose-dependent.

### Verified NOT bugs (don't re-chase)
- **Trip conservation** — TOD factors sum to exactly 1.0 for all 8 purposes;
  directionality `f·PA+(1−f)·PA.T` conserves; the two-skim-then-TOD reuse does
  not double-count.
- **Piecewise/cutoff parsing** — all `max(0, CongTime−30)` and `> XX` thresholds
  parse correctly, including the quoted rows.
- **HB occupancy** `/2`, `/o3` matches `16…rsc:209-210`.
- **PURPOSES → size_cols and eda keys** — match the CSV headers exactly.
- **`A /= len(size_cols)`** — benign (cancels in softmax).

### Priority to fix (by OD impact)
1. N8 MC logsums (needs step 12) — largest spatial error
2. N2 other_shares_hb fold — quick fix, wrong SOV/HOV split
3. N3/N7 segmentation (per-segment DC + read Segment column)
4. N4 zone-level demographic generation (Pct_Worker/Child/Senior)
5. N9 shadow-price update loop + attraction target
6. N1 motorized-only productions into DC (needs step 06)

Full per-finding detail (line numbers, rsc citations, magnitudes) captured from
the audit; the two quick wins (N2, N4-zone-level) need no other steps built.
