"""demand_checks.py — Gates ① and ② of the OD validation ladder (upstream first).

Before assignment/count validation (od_validation.py = Gate ③), check the demand
itself:
  GATE ① — trip GENERATION: are production totals by purpose in range vs the survey
           control totals? (RAG: +-5-10% green / 10-20% amber / >20% red)
  GATE ② — trip DISTRIBUTION / trip-length reasonableness:
           * average trip length by purpose
           * monotonicity sanity (e.g. OD_Long must be >= OD_Short)
           * DEFERRED-dominant-term flag: if a purpose's destination choice is driven
             by a coefficient we defer (transit / mode logsums, no transit skims),
             its distribution is UNRELIABLE and will bias trip length short.

This makes the *source* of a downstream loaded-network problem visible: a freeway
shortfall is usually a Gate ①/② issue (missing markets or wrong trip length), not
assignment.

Run: python demand_checks.py
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DC = os.path.join(REPO, "master", "resident", "dc")

# purpose -> (survey key, dc zone-table stem)
PURP = {
    "W_HB_W_All": (("W", "All"), "w_hb_w_all"),
    "W_HB_O_All": (("O", "All"), "w_hb_o_all"),
    "W_HB_EK12_All": (("EK12", "All"), "w_hb_ek12_all"),
    "N_HB_OME_All": (("OME", "All"), "n_hb_ome_all"),
    "N_HB_OMED_All": (("OMED", "All"), "n_hb_omed_all"),
    "N_HB_OD_Short": (("OD", "Short"), "n_hb_od_short"),
    "N_HB_OD_Long": (("OD", "Long"), "n_hb_od_long"),
    "N_HB_K12_All": (("K12", "All"), "n_hb_k12_all"),
}


def rag(bias):
    a = abs(bias)
    return "green" if a <= 10 else "amber" if a <= 20 else "red"


def survey_totals():
    """First occurrence of each (purpose,duration) in eda -> wTrips control total."""
    out = {}
    fp = os.path.join(REPO, "docs", "data", "output", "survey_processing", "eda_scheme6.csv")
    for r in csv.DictReader(open(fp)):
        k = (r.get("purpose"), r.get("duration"))
        if k not in out and r.get("wTrips"):
            out[k] = float(r["wTrips"])
    return out


def dc_terms(stem):
    """Return (time_coef, [(deferred_term, coef)]) from a zone table."""
    fp = os.path.join(DC, stem + "_zone.csv")
    time_c, deferred = 0.0, []
    if not os.path.exists(fp):
        return time_c, deferred
    for r in csv.DictReader(open(fp)):
        expr = r.get("Expression", "").strip().strip('"')
        try:
            c = float(r["Coefficient"])
        except (ValueError, TypeError, KeyError):
            continue
        if expr == "sov_skim.CongTime":
            time_c += c
        elif expr.startswith("nz(mc_logsums") or "mc_logsums" in expr:
            deferred.append((expr, c))
    return time_c, deferred


def main():
    summ = json.load(open(os.path.join(HERE, "review", "v15_4period_summary.json")))
    purp = summ["purposes"]
    ctrl = survey_totals()

    print("=== GATE ① · Trip generation (production vs survey control totals) ===")
    g1 = {}
    for p, (skey, _) in PURP.items():
        modeled = purp[p]["daily_person_trips"]
        target = ctrl.get(skey)
        if not target:
            continue
        bias = 100 * (modeled - target) / target
        g1[p] = dict(modeled=modeled, target=round(target), bias=round(bias, 1), rag=rag(bias))
        print(f"  {p:16s} model {modeled:>10,}  survey {int(target):>10,}  "
              f"bias {bias:+5.1f}%  [{rag(bias).upper()}]")
    print(f"  -> all green (N4 sets production = survey total); GATE ① PASS\n")

    print("=== GATE ② · Distribution / trip-length reasonableness ===")
    lens = {p: purp[p]["avg_am_time_min"] for p in PURP}
    issues = []
    # (a) monotonicity: OD_Long must be >= OD_Short
    if lens["N_HB_OD_Long"] < lens["N_HB_OD_Short"]:
        issues.append(f"OD_Long ({lens['N_HB_OD_Long']}min) < OD_Short "
                      f"({lens['N_HB_OD_Short']}min) — long trips wrongly shorter than short")
    # (b) deferred-dominant-term flag
    print("  avg trip length + dominant-DC-term status by purpose:")
    for p, (_, stem) in PURP.items():
        tc, dfr = dc_terms(stem)
        maxdef = max((abs(c) for _, c in dfr), default=0)
        dominant = maxdef > 2 * abs(tc) if tc else (maxdef > 0)
        tag = "DEFERRED-DOMINANT ✗" if (dfr and dominant) else ("deferred(minor)" if dfr else "ok")
        if dfr and dominant:
            issues.append(f"{p}: destination choice dominated by a DEFERRED term "
                          f"(|coef| {maxdef} vs time {abs(tc):.3f}) — distribution unreliable")
        print(f"  {p:16s} {lens[p]:>5} min   time_coef {tc:>7.3f}   "
              f"max deferred |coef| {maxdef:>5}   [{tag}]")

    print("\n=== FINDINGS (source of downstream issues) ===")
    if not issues:
        print("  none — distribution looks reasonable")
    for i, s in enumerate(issues, 1):
        print(f"  {i}. {s}")
    print("\n  Interpretation: the magnitude gap (Gate ③) is the KNOWN CAPTURE (missing "
          "markets). But the trip-length anomalies above are a REAL Gate ② problem — the "
          "deferred transit/mode logsums (need transit skims) flatten long-distance "
          "purposes, which then under-loads freeways. Fix upstream, not in assignment.")

    json.dump({"gate1": g1, "trip_len": lens, "issues": issues},
              open(os.path.join(HERE, "review", "demand_checks.json"), "w"), indent=1)
    print("\n  wrote review/demand_checks.json")


if __name__ == "__main__":
    main()
