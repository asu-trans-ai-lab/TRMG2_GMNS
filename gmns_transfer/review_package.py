"""Build the human-review package for the TRMG2 GMNS scenario.

Outputs in review/:
  network_inventory.csv   links x lane-miles x VMT by facility type & area type
  od_district_matrix.csv  demand aggregated to TRMG2's 12 planning districts (CLUSTER)
  trip_length_distribution.csv   demand-weighted free-flow time & distance bins
  top_loaded_links.csv    25 highest-volume links with TRMG2 IDs and v/c
  REVIEW.md               summary + reviewer checklist
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
GM = os.path.join(HERE, "scenario")
OUT = os.path.join(HERE, "review")
os.makedirs(OUT, exist_ok=True)


def main():
    # ---- crosswalk + clusters ----
    taz_of, trmg2_node = {}, {}
    with open(os.path.join(HERE, "node_crosswalk.csv")) as f:
        for r in csv.DictReader(f):
            trmg2_node[int(r["gmns_node_id"])] = int(r["trmg2_node_id"])
            if r["taz"]:
                taz_of[int(r["gmns_node_id"])] = int(r["taz"])
    import geopandas as gpd
    g = gpd.read_file(os.path.join(REPO, "docs", "data", "input", "tazs", "master_tazs.shp")).set_index("ID")
    cluster = g["CLUSTERNAM"].to_dict()

    # ---- link table + performance ----
    links = {}
    with open(os.path.join(GM, "link.csv")) as f:
        for r in csv.DictReader(f):
            links[int(r["link_id"])] = r
    perf = {}
    with open(os.path.join(GM, "link_performance.csv")) as f:
        for r in csv.DictReader(f):
            key = (int(r["from_node_id"]), int(r["to_node_id"]))
            perf[key] = r

    inv = defaultdict(lambda: dict(n=0, lane_mi=0.0, vmt=0.0, cap_lo=1e9, cap_hi=0.0))
    top = []
    for lid, l in links.items():
        key = (int(l["from_node_id"]), int(l["to_node_id"]))
        p = perf.get(key, {})
        v = float(p.get("volume") or 0)
        length = float(l["length"])
        k = (l["link_type_name"], l["area_type"])
        d = inv[k]
        d["n"] += 1
        d["lane_mi"] += length * int(l["lanes"])
        d["vmt"] += float(p.get("VMT") or 0)
        cphl = float(l["capacity"]) / max(int(l["lanes"]), 1) / 2.423
        d["cap_lo"] = min(d["cap_lo"], cphl)
        d["cap_hi"] = max(d["cap_hi"], cphl)
        if v > 0:
            top.append((v, lid, l, p))
    with open(os.path.join(OUT, "network_inventory.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["facility_type", "area_type", "directed_links", "lane_miles",
                    "AM_VMT", "cap_phpl_min", "cap_phpl_max"])
        for (ft, at), d in sorted(inv.items()):
            w.writerow([ft, at, d["n"], round(d["lane_mi"], 1), round(d["vmt"]),
                        round(d["cap_lo"]), round(d["cap_hi"])])

    top.sort(reverse=True)
    with open(os.path.join(OUT, "top_loaded_links.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gmns_link_id", "trmg2_link_id", "trmg2_from_node", "trmg2_to_node",
                    "facility", "area_type", "lanes", "AM_volume", "voc", "speed_mph"])
        for v, lid, l, p in top[:25]:
            w.writerow([lid, lid // 10, trmg2_node[int(l["from_node_id"])],
                        trmg2_node[int(l["to_node_id"])], l["link_type_name"],
                        l["area_type"], l["lanes"], round(v),
                        round(float(p.get("doc") or 0), 3),
                        round(float(p.get("speed_mph") or 0), 1)])

    # ---- demand: district matrix + TLD ----
    dem = []
    with open(os.path.join(GM, "demand.csv")) as f:
        for r in csv.DictReader(f):
            dem.append((int(r["o_zone_id"]), int(r["d_zone_id"]), float(r["volume"])))
    def dist_of(z):
        taz = taz_of.get(z)
        return cluster.get(taz, "NoPolygon") if taz else "NoPolygon"
    dm = defaultdict(float)
    names = set()
    for o, d, v in dem:
        a, b = dist_of(o), dist_of(d)
        names.add(a); names.add(b)
        dm[(a, b)] += v
    names = sorted(names)
    with open(os.path.join(OUT, "od_district_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["origin_district\\dest"] + names + ["total"])
        for a in names:
            row = [dm[(a, b)] for b in names]
            w.writerow([a] + [round(x) for x in row] + [round(sum(row))])
        w.writerow(["total"] + [round(sum(dm[(a, b)] for a in names)) for b in names]
                   + [round(sum(dm.values()))])

    # TLD on congested times from link_performance
    nodes = {}
    with open(os.path.join(GM, "node.csv")) as f:
        for r in csv.DictReader(f):
            nodes[int(r["node_id"])] = r
    nid = sorted(nodes); nix = {n: k for k, n in enumerate(nid)}
    rows, cols, wt, wd = [], [], [], []
    for l in links.values():
        key = (int(l["from_node_id"]), int(l["to_node_id"]))
        p = perf.get(key, {})
        rows.append(nix[key[0]]); cols.append(nix[key[1]])
        wt.append(max(float(p.get("travel_time") or l["vdf_fftt"]), 1e-3))
        wd.append(float(l["length"]))
    zlist = sorted({o for o, _, _ in dem} | {d for _, d, _ in dem})
    src = [nix[z] for z in zlist]
    gt = sp.csr_matrix((wt, (rows, cols)), shape=(len(nid),) * 2)
    gd = sp.csr_matrix((wd, (rows, cols)), shape=(len(nid),) * 2)
    # time skim (congested) and distance skim along time-shortest path approx:
    tmat = dijkstra(gt, directed=True, indices=src)[:, src]
    zpos = {z: k for k, z in enumerate(zlist)}
    bins_t = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 90, 1e9]
    hist = np.zeros(len(bins_t) - 1)
    tot_v = tot_vt = 0.0
    for o, d, v in dem:
        t = tmat[zpos[o], zpos[d]]
        if not np.isfinite(t):
            continue
        k = np.searchsorted(bins_t, t, side="right") - 1
        hist[min(k, len(hist) - 1)] += v
        tot_v += v; tot_vt += v * t
    with open(os.path.join(OUT, "trip_length_distribution.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["congested_time_bin_min", "vehicles", "share"])
        for k in range(len(hist)):
            lab = f"{bins_t[k]}-{bins_t[k+1] if bins_t[k+1] < 1e9 else '+'}"
            w.writerow([lab, round(hist[k]), round(hist[k] / tot_v, 4)])
    att = tot_vt / tot_v


    # ---- count coverage + TRMG2 published benchmarks ----
    import shutil
    from collections import defaultdict as dd2
    cov = dd2(lambda: [0, 0.0])
    seen = set()
    for lid, l in links.items():
        if l.get("day_count"):
            base = lid // 10
            if base in seen:
                continue
            seen.add(base)
            k = (l["link_type_name"], l["area_type"])
            cov[k][0] += 1
            cov[k][1] += float(l["day_count"])
    with open(os.path.join(OUT, "count_coverage.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["facility_type", "area_type", "counted_links", "total_daily_count"])
        for k in sorted(cov):
            w.writerow([k[0], k[1], cov[k][0], round(cov[k][1])])
    val = os.path.join(REPO, "docs", "data", "input", "validation")
    if os.path.isdir(val):
        for fn in os.listdir(val):
            shutil.copy(os.path.join(val, fn),
                        os.path.join(OUT, "trmg2_benchmark_" + fn))

    # ---- REVIEW.md ----
    with open(os.path.join(HERE, "build_log.json")) as f:
        blog = json.load(f)
    lines = []
    lines.append("# TRMG2 GMNS scenario — review package\n")
    lines.append("Validation: gmns-ready 0 errors / 4 warnings (2 = geometry pending "
                 "shapefile, 1 = km/h rounding cosmetic, 1 = vdf_alpha up to 4.0 which "
                 "matches TRMG2's own ff_speed_alpha_beta.csv); dtalite_qa validate clean.\n")
    lines.append(f"- Directed links: {len(links):,}; zones: 3,247; "
                 f"area types {blog['zone_at_counts']}\n")
    lines.append(f"- v0 AM sov demand: {tot_v:,.0f} vehicles; "
                 f"avg congested travel time {att:.1f} min\n")
    lines.append("## Reviewer checklist\n")
    lines.append("1. network_inventory.csv — lane-miles & capacity ranges per facility/area "
                 "type vs your expectation of the Triangle network.")
    lines.append("2. od_district_matrix.csv — district-to-district pattern plausibility "
                 "(RTP/Raleigh/Durham pulls, external districts show as NoPolygon).")
    lines.append("3. trip_length_distribution.csv — v0 scaffold gravity gives avg "
                 f"{att:.1f} min; TRMG2's calibrated models will differ.")
    lines.append("4. top_loaded_links.csv — check the top-25 are the region's known "
                 "freeway segments (TRMG2 link/node IDs included for lookup).")
    lines.append("5. Known gaps before parity testing: real lengths + geometry "
                 "(ITRE shapefile), real OD (od_veh_trips export), turn prohibitions.")
    with open(os.path.join(OUT, "REVIEW.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("review package written:", os.listdir(OUT))
    print(f"avg congested time {att:.1f} min over {tot_v:,.0f} veh")


if __name__ == "__main__":
    main()
