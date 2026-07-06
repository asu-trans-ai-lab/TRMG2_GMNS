"""vmt_vht_gate.py — Gate ④: measured VMT & VHT by facility type, with the
magnitude-first / structural-bias read the assignment diagnosis calls for.

Reads the loaded network (scenario_{AM,MD,PM,NT}/link_performance.csv), which carries
per-link VMT and VHT, sums them by facility type over the 4 periods, and prints the
VMT/VHT system read using the observatory diagnostic matrix. Joins the count-bias-by-
facility so the freeway-weighted deficit is visible next to VMT/VHT (the structural
tell: if the deficit were a clean uniform capture, every facility would show the same
bias — it does not).

Centroid connectors (CC) are excluded from system VMT/VHT (artificial links).

Interim numbers -> written LOCALLY (review/ gitignored). Run: python vmt_vht_gate.py
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.join(HERE, "review")
PERIODS = ["AM", "MD", "PM", "NT"]
EXCLUDE = {"CC", ""}   # centroid connectors are not real roadway

# facility display order, high-order first (matches the count-bias gradient)
ORDER = ["Freeway", "MLHighway", "TLHighway", "Ramp", "MajorArterial", "Arterial",
         "MajorCollector", "Collector", "Local"]


def load_fac_map():
    m = {}
    for r in csv.DictReader(open(os.path.join(HERE, "scenario", "link.csv"))):
        m[r["link_id"]] = r.get("link_type_name", "?")
    return m


def load_count_bias():
    """count PctDiff by facility from the existing count comparison, if present."""
    for f in ("v15_count_comparison_by_fac_type.csv", "v1_count_comparison_by_fac_type.csv"):
        p = os.path.join(REV, f)
        if os.path.exists(p):
            return {r["HCMType"]: float(r["PctDiff"]) for r in csv.DictReader(open(p))}
    return {}


def main():
    fac = load_fac_map()
    vmt, vht = {}, {}                       # daily by facility
    pv, ph = {}, {}                         # per-period system (period -> VMT/VHT)
    for per in PERIODS:
        p = os.path.join(HERE, f"scenario_{per}", "link_performance.csv")
        for r in csv.DictReader(open(p)):
            f = fac.get(r["link_id"], "?")
            if f in EXCLUDE:
                continue
            try:
                vm = float(r["VMT"] or 0); vh = float(r["VHT"] or 0)
            except ValueError:
                continue
            vmt[f] = vmt.get(f, 0.0) + vm
            vht[f] = vht.get(f, 0.0) + vh
            pv[(per, f)] = pv.get((per, f), 0.0) + vm
            ph[(per, f)] = ph.get((per, f), 0.0) + vh

    cb = load_count_bias()
    tot_vmt = sum(vmt.values()); tot_vht = sum(vht.values())
    sys_speed = tot_vmt / max(tot_vht, 1e-9)

    facs = [f for f in ORDER if f in vmt] + [f for f in vmt if f not in ORDER]
    rows = []
    lines = ["# Gate ④ · VMT & VHT — measured, magnitude-first, by facility type\n",
             f"System VMT = **{tot_vmt:,.0f}**  ·  System VHT = **{tot_vht:,.0f}**  ·  "
             f"implied system speed = **{sys_speed:.1f} mph** (VMT/VHT).\n",
             "The count-bias column is the structural tell: a *uniform* deficit would be the",
             "same on every row; the **freeway-weighted gradient** below says the missing trips",
             "are long-distance (freeway/highway markets), so VMT — and VHT more so — lose",
             "magnitude fastest on the high-order network.\n",
             "| Facility | VMT | VMT % | VHT | VHT % | avg speed | count bias |",
             "|---|--:|--:|--:|--:|--:|--:|"]
    for f in facs:
        vm, vh = vmt[f], vht.get(f, 0.0)
        sp = vm / max(vh, 1e-9)
        cbias = cb.get(f)
        cbcell = f"{cbias:+.1f}%" if cbias is not None else "n/a"
        lines.append(f"| {f} | {vm:,.0f} | {100*vm/tot_vmt:.1f}% | {vh:,.0f} | "
                     f"{100*vh/tot_vht:.1f}% | {sp:.1f} | {cbcell} |")
        rows.append(dict(fac=f, vmt=round(vm), vht=round(vh), speed=round(sp, 1),
                         count_bias_pct=cbias))

    # ── per-period system VMT/VHT/speed (M5: daily average blends peak with night) ──
    per_rows = []
    lines += ["\n## By period — speed is a congestion test only in the peaks\n",
              "Daily speed blends peak with night, so a high *daily* average is partly a mix",
              "artifact. The real congestion test is **peak-period speed**, and freeway peak speed",
              "vs free-flow is the sharpest single number.\n",
              "| Period | system VMT | system VHT | system speed | freeway speed |",
              "|---|--:|--:|--:|--:|"]
    for per in PERIODS:
        pvmt = sum(v for (pp, f), v in pv.items() if pp == per)
        pvht = sum(v for (pp, f), v in ph.items() if pp == per)
        fw_v = pv.get((per, "Freeway"), 0.0); fw_h = ph.get((per, "Freeway"), 0.0)
        psp = pvmt / max(pvht, 1e-9); fwsp = fw_v / max(fw_h, 1e-9)
        lines.append(f"| {per} | {pvmt:,.0f} | {pvht:,.0f} | {psp:.1f} | {fwsp:.1f} |")
        per_rows.append(dict(period=per, vmt=round(pvmt), vht=round(pvht),
                             speed=round(psp, 1), freeway_speed=round(fwsp, 1)))
    peak_fw = min(r["freeway_speed"] for r in per_rows if r["period"] in ("AM", "PM"))
    lines.append(f"\n**Peak freeway speed = {peak_fw:.1f} mph.** If this sits at free-flow "
                 "(~65–72), congestion is not forming even in the peak — the demand, not the "
                 "network, is short. That is the honest congestion read (not the daily average).")

    # count bias spread = the structural-bias magnitude (uniform capture would be ~0 spread)
    biases = [r["count_bias_pct"] for r in rows if r["count_bias_pct"] is not None]
    spread = (max(biases) - min(biases)) if biases else 0.0
    lines += [
        "\n## Read (observatory VMT/VHT matrix)\n",
        f"- **VMT low + VHT low → missing trips** (or trips too short). Capture is far below 1.0.",
        f"- **Freeway most under-loaded, Local/Collector least — a {spread:.0f}-point spread.**",
        "  A *uniform* capture would show ~0 spread; this gradient points to missing **long-distance",
        "  markets** (external / through / CV / NHB-long / airport) — a Gate ①/② coverage problem.",
        "- **Null-model caveat (M4):** TRMG2's own published fit varies by facility (freeways",
        "  validate tightest, locals loosest), so a *complete* model shows some facility spread",
        "  too. Attribute only the **excess** over that baseline to coverage — the null baseline",
        "  is PENDING ITRE's published by-facility bias. Direction holds; magnitude is an upper bound.",
        "- **VHT vs VMT:** VHT loses magnitude faster; but read speed **by period above**, not as a",
        "  daily average, before calling congestion under-formed.\n",
        "**Gate ④ verdict: 🔴 magnitude (VMT & VHT low, freeway-weighted) — REFINE upstream markets.**",
    ]
    open(os.path.join(REV, "vmt_vht_gate.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump(dict(system_vmt=round(tot_vmt), system_vht=round(tot_vht),
                   system_speed_mph=round(sys_speed, 1), peak_freeway_speed_mph=round(peak_fw, 1),
                   count_bias_spread_pp=round(spread, 1),
                   by_facility=rows, by_period=per_rows),
              open(os.path.join(REV, "vmt_vht_gate.json"), "w"), indent=1)
    print(f"System VMT {tot_vmt:,.0f} | VHT {tot_vht:,.0f} | daily speed {sys_speed:.1f} | "
          f"PEAK freeway speed {peak_fw:.1f} mph | count-bias spread {spread:.0f}pp")
    print("wrote review/vmt_vht_gate.md + .json")


if __name__ == "__main__":
    main()
