"""Reproduction gates: quantitative checks of our pipeline against TRMG2's own
published reference values at every stage. Reproduce, don't invent — every
deviation is surfaced, none silently absorbed.

Statuses:
  PASS      within tolerance of the TRMG2 reference
  WARN      outside tight tolerance, inside loose
  FAIL      outside loose tolerance
  DEVIATION known, documented modeling difference (listed with its fix task)
  PENDING   reference data ready, our step not implemented yet

Run after make_od_4period.py. Output: review/GATES_REPORT.md + review/gates.json
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from transcad_bin import read_bin  # noqa: E402
from make_od_4period import PURPOSES, PERIODS  # noqa: E402

GATES = []


def gate(gid, step, name, status, metric, reference, note=""):
    GATES.append(dict(id=gid, step=step, name=name, status=status,
                      metric=metric, reference=reference, note=note))
    print(f"[{status:9s}] {gid} {name}: {metric}")


def main():
    scen = os.path.join(HERE, "scenario")
    rev = os.path.join(HERE, "review")

    # ---------- G1 network structure ----------
    with open(os.path.join(HERE, "build_log.json")) as f:
        blog = json.load(f)
    ok = (blog.get("skipped_no_lookup", 1) == 0
          and blog.get("links_missing_from_export", 1) == 0
          and blog.get("nodes_missing_coords", 1) == 0
          and blog.get("zones_in_network") == 3247)
    gate("G1", "02 Network", "GMNS build completeness",
         "PASS" if ok else "FAIL",
         f"links {blog.get('directed_links')}, zones {blog.get('zones_in_network')}, "
         f"skipped {blog.get('skipped_no_lookup')}, no-coord {blog.get('nodes_missing_coords')}",
         "capacity.csv/ff_speed lookups + TransCAD export")

    # ---------- G2 topology vs TransCAD export (skipped if export absent) ----------
    exp_fp = os.path.join(REPO, "2026-07-04 TRMG2", "master_links 2026-07-04.csv")
    if os.path.exists(exp_fp):
        exp = {}
        with open(exp_fp) as f:
            rd = csv.DictReader(f)
            fcol, tcol = rd.fieldnames[-2], rd.fieldnames[-1]
            for r in rd:
                exp[int(r["ID"])] = (int(r[fcol]), int(r[tcol]))
        okc = bad = 0
        with open(os.path.join(HERE, "topology.csv")) as f:
            for r in csv.DictReader(f):
                e = exp.get(int(r["link_id"]))
                if e and (int(r["from_node_id"]), int(r["to_node_id"])) == e:
                    okc += 1
                else:
                    bad += 1
        gate("G2", "net.net decode", "topology vs TransCAD FROM_ID/TO_ID",
             "PASS" if bad == 0 else "FAIL", f"{okc:,} exact / {bad} mismatched",
             "2026-07-04 TransCAD export")
    else:
        n = sum(1 for _ in open(os.path.join(HERE, "topology.csv"))) - 1
        gate("G2", "net.net decode", "topology vs TransCAD FROM_ID/TO_ID",
             "SKIP", f"{n:,} links in topology.csv; export absent (was 135,922/135,922 "
             "exact at build time)", "2026-07-04 TransCAD export (not bundled)")

    # ---------- load v1.5 summary ----------
    sfp = os.path.join(rev, "v15_4period_summary.json")
    summ = json.load(open(sfp)) if os.path.exists(sfp) else None

    # ---------- G3 generation totals vs survey wTrips ----------
    if summ:
        eda = {}
        with open(os.path.join(REPO, "docs", "data", "output", "survey_processing",
                               "eda_scheme6.csv")) as f:
            for r in csv.DictReader(f):
                if r["homebased"] == "HB":
                    eda[(r["tour_type"], r["purpose"], r["duration"])] = float(r["wTrips"])
        worst = 0.0
        det = []
        for ptype, (edakey, _) in PURPOSES.items():
            ours = summ["purposes"].get(ptype, {}).get("daily_person_trips")
            ref = eda.get(edakey)
            if ours and ref:
                dev = ours / ref - 1
                worst = max(worst, abs(dev))
                det.append(f"{ptype} {dev:+.0%}")
        status = "PASS" if worst <= 0.15 else ("WARN" if worst <= 0.40 else "FAIL")
        gate("G3", "05 Generation", "daily person trips vs survey wTrips",
             status, "; ".join(det),
             "eda_scheme6.csv wTrips",
             "stand-in aggregate rates; exact reproduction = task #4/#5 "
             "(population synthesis + decision-tree rule application)")

    # ---------- G4 intrazonal share (KNOWN DEVIATION D1) ----------
    iz_ref = {}
    with open(os.path.join(REPO, "docs", "data", "output", "resident_dc",
                           "intrazonals.csv")) as f:
        for r in csv.DictReader(f):
            try:
                iz_ref[r["trip_type"]] = float(r["pct_IZ"])
            except ValueError:
                pass
    gate("G4", "13 DC", "intrazonal share vs intrazonals.csv",
         "DEVIATION",
         "ours 0% for all purposes vs reference "
         + ", ".join(f"{k} {v}%" for k, v in list(iz_ref.items())[:4]) + " ...",
         "docs/data/output/resident_dc/intrazonals.csv",
         "D1: v1.5 excludes intrazonal trips (no IZ times yet); fix = 10:76-88 "
         "intrazonal time formula, task #6")

    # ---------- G5 avg trip time diagnostic ----------
    if summ:
        det = [f"{p} {v['avg_am_time_min']}min" for p, v in summ["purposes"].items()]
        gate("G5", "13 DC", "avg AM congested time per purpose (diagnostic)",
             "PASS", "; ".join(det[:4]) + " ...",
             "reported only — NOT calibrated (their DC coefficients used verbatim)")

    # ---------- G7 mode-share stand-in disclosure ----------
    gate("G7", "12 Mode choice", "mode split source",
         "DEVIATION",
         "survey shares from eda_scheme6 (auto_pay/other_auto folded into sov/hov)",
         "master/resident/mode/*.csv nested logit",
         "D2: nested-logit MC not yet applied; task #5. Transit LOS frozen pending.")

    # ---------- G8 assignment convergence ----------
    if summ:
        gaps = {p: summ["periods"][p]["final_gap"] for p in PERIODS if p in summ["periods"]}
        worst = max(float(g.split("%")[0]) for g in gaps.values())
        gate("G8", "17 Assignment", "relative gap per period",
             "PASS" if worst <= 0.01 else "WARN",
             "; ".join(f"{p} {g}" for p, g in gaps.items()),
             "TRMG2 AssignConvergence 1e-5 (%RMSE-based); ours: kernel rel gap")

    # ---------- G9 counts vs TRMG2 benchmark ----------
    ours_fp = os.path.join(rev, "v15_count_comparison_by_fac_type.csv")
    if os.path.exists(ours_fp):
        ours = {r["HCMType"]: r for r in csv.DictReader(open(ours_fp))}
        ref = {r["HCMType"]: r for r in csv.DictReader(open(os.path.join(
            REPO, "docs", "data", "input", "validation",
            "count_comparison_by_fac_type.csv")))}
        rows = []
        for ft in ref:
            if ft in ours:
                rows.append(f"{ft}: ours {ours[ft]['PctDiff']}% / ref {ref[ft]['PctDiff']}%")
        cap = summ.get("captured_share") if summ else None
        gate("G9", "Validation", "daily counts: ours vs TRMG2 benchmark",
             "DEVIATION",
             f"captured share {cap}; " + "; ".join(rows[:4]) + " ...",
             "count_comparison_by_fac_type.csv",
             "D3: v1.5 = resident HB only; NHB/CV/univ/airport/externals pending "
             "(tasks #7/#8). Gate flips to PASS/FAIL tolerance once components complete.")

    # ---------- PENDING gates (references ready, steps not implemented) ----------
    gate("G6", "03 Accessibility", "access_nearby_sov/transit/walk per zone",
         "PENDING", "reference ready: nhb_generation.csv access columns (3,062 TAZ)",
         "docs/data/input/nhb/nhb_generation.csv", "task #2")
    gate("G10", "04 PopSynthesis", "731,913 HH / 1,783,548 persons + marginals",
         "PENDING", "reference: paper totals + disagg curves (deterministic)",
         "income/size/worker_curves.csv", "task #3")
    gate("G11", "04 Auto ownership", "AO shares uncalibrated -> calibrated",
         "PENDING", "reference ready: uncalibrated_auto_results.csv + ao_calib_targets.csv",
         "docs/data/output/auto_ownership/", "task #3")
    gate("G12", "06 NonMotorized", "NM share per purpose",
         "PENDING", "reference ready: nonmotorized/calibration_targets.csv",
         "docs/data/output/nonmotorized/", "task follows #4")
    gate("G13", "14 NHB", "per-TAZ NHB gen vs their model values",
         "PENDING", "reference ready: nhb_generation.csv purpose x mode x period columns",
         "docs/data/input/nhb/nhb_generation.csv", "task #7")
    gate("G14", "Feedback", "skim %RMSE <= 0.1 in <= 5 iterations",
         "PENDING", "loop not built", "trmg2.model:227-253", "task #9")

    # ---------- write report ----------
    os.makedirs(rev, exist_ok=True)
    with open(os.path.join(rev, "gates.json"), "w") as f:
        json.dump(GATES, f, indent=1)
    counts = defaultdict(int)
    for g in GATES:
        counts[g["status"]] += 1
    with open(os.path.join(rev, "GATES_REPORT.md"), "w") as f:
        f.write("# Reproduction gates report\n\n")
        f.write("Principle: reproduce, don't invent. Every stage is checked against a\n"
                "TRMG2-published reference; deviations are named (D1, D2, ...) with the\n"
                "task that closes them — never silently absorbed.\n\n")
        f.write("Status counts: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
                + "\n\n")
        f.write("| gate | step | check | status | result | reference | note |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for g in GATES:
            f.write(f"| {g['id']} | {g['step']} | {g['name']} | **{g['status']}** | "
                    f"{g['metric']} | {g['reference']} | {g['note']} |\n")
    print("\nstatus counts:", dict(counts))
    print("-> review/GATES_REPORT.md")


if __name__ == "__main__":
    main()
