# Reproduction gates report

Principle: reproduce, don't invent. Every stage is checked against a
TRMG2-published reference; deviations are named (D1, D2, ...) with the
task that closes them — never silently absorbed.

Status counts: DEVIATION 3, FAIL 1, PASS 3, PENDING 6, SKIP 1

| gate | step | check | status | result | reference | note |
|---|---|---|---|---|---|---|
| G1 | 02 Network | GMNS build completeness | **PASS** | links 75939, zones 3247, skipped 0, no-coord 0 | capacity.csv/ff_speed lookups + TransCAD export |  |
| G2 | net.net decode | topology vs TransCAD FROM_ID/TO_ID | **SKIP** | 135,922 links in topology.csv; export absent (was 135,922/135,922 exact at build time) | 2026-07-04 TransCAD export (not bundled) |  |
| G3 | 05 Generation | daily person trips vs survey wTrips | **FAIL** | W_HB_W_All +49%; W_HB_O_All +8%; W_HB_EK12_All -44%; N_HB_OME_All +10%; N_HB_OMED_All +14%; N_HB_OD_Short +3%; N_HB_OD_Long +15%; N_HB_K12_All -21% | eda_scheme6.csv wTrips | stand-in aggregate rates; exact reproduction = task #4/#5 (population synthesis + decision-tree rule application) |
| G4 | 13 DC | intrazonal share vs intrazonals.csv | **DEVIATION** | ours 0% for all purposes vs reference N_HB_K12_All 2.9%, N_HB_OD_Long 3.5%, N_HB_OD_Short 10.7%, N_HB_OME_All 3.5% ... | docs/data/output/resident_dc/intrazonals.csv | D1: v1.5 excludes intrazonal trips (no IZ times yet); fix = 10:76-88 intrazonal time formula, task #6 |
| G5 | 13 DC | avg AM congested time per purpose (diagnostic) | **PASS** | W_HB_W_All 13.1min; W_HB_O_All 9.7min; W_HB_EK12_All 9.2min; N_HB_OME_All 11.3min ... | reported only — NOT calibrated (their DC coefficients used verbatim) |  |
| G7 | 12 Mode choice | mode split source | **DEVIATION** | survey shares from eda_scheme6 (auto_pay/other_auto folded into sov/hov) | master/resident/mode/*.csv nested logit | D2: nested-logit MC not yet applied; task #5. Transit LOS frozen pending. |
| G8 | 17 Assignment | relative gap per period | **PASS** | AM 0.000590 %; MD 0.000031 %; PM 0.000541 %; NT 0.000084 % | TRMG2 AssignConvergence 1e-5 (%RMSE-based); ours: kernel rel gap |  |
| G9 | Validation | daily counts: ours vs TRMG2 benchmark | **DEVIATION** | captured share 0.341; Arterial: ours -59.63% / ref -2.42%; Collector: ours -55.24% / ref -2.32%; Freeway: ours -76.74% / ref 0.86%; Local: ours -57.16% / ref -4.56% ... | count_comparison_by_fac_type.csv | D3: v1.5 = resident HB only; NHB/CV/univ/airport/externals pending (tasks #7/#8). Gate flips to PASS/FAIL tolerance once components complete. |
| G6 | 03 Accessibility | access_nearby_sov/transit/walk per zone | **PENDING** | reference ready: nhb_generation.csv access columns (3,062 TAZ) | docs/data/input/nhb/nhb_generation.csv | task #2 |
| G10 | 04 PopSynthesis | 731,913 HH / 1,783,548 persons + marginals | **PENDING** | reference: paper totals + disagg curves (deterministic) | income/size/worker_curves.csv | task #3 |
| G11 | 04 Auto ownership | AO shares uncalibrated -> calibrated | **PENDING** | reference ready: uncalibrated_auto_results.csv + ao_calib_targets.csv | docs/data/output/auto_ownership/ | task #3 |
| G12 | 06 NonMotorized | NM share per purpose | **PENDING** | reference ready: nonmotorized/calibration_targets.csv | docs/data/output/nonmotorized/ | task follows #4 |
| G13 | 14 NHB | per-TAZ NHB gen vs their model values | **PENDING** | reference ready: nhb_generation.csv purpose x mode x period columns | docs/data/input/nhb/nhb_generation.csv | task #7 |
| G14 | Feedback | skim %RMSE <= 0.1 in <= 5 iterations | **PENDING** | loop not built | trmg2.model:227-253 | task #9 |
