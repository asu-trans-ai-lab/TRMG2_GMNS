"""Build a GMNS (node.csv / link.csv) scenario from the decoded TRMG2 master network.

v2: upgraded with the 2026-07-04 TransCAD export (user-provided):
  - real link LENGTH + authoritative DIR from "2026-07-04 TRMG2/master_links.csv"
  - real node coordinates + link geometry WKT from "2026-07-04 TRMG2_shp/"
  - the export also VALIDATED the net.net topology decode: 135,922/135,922 exact.

Still decoded without TransCAD (from the release binaries):
  - topology.csv (net.net decode), link/node attributes (BIN/DCB), SE data,
    capacity/speed lookups, TAZ polygons (GitHub docs shapefile).

Remaining documented approximations vs TRMG2 (02 - Network Calculations.rsc):
  A1. Area-type smoothing via centroid-in-buffer (not TransCAD Enclosed inclusion)
  A2. Link area type via BFS hops from zone centroids (not spatial buffer tagging)
  A4. Ramps fall back to MajorArterial capacity/speed rows (TRMG2 re-tags by adjacency)
(A3 retired: lengths are now the real TransCAD values, not NT-time estimates.)

Output: scenario/node.csv, scenario/link.csv, node_crosswalk.csv, build_log.json
"""
import csv
import json
import os
import sys
from collections import defaultdict, deque

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))          # TRMG2 repo root
NET = os.path.join(REPO, "master", "networks")
EXP = os.path.join(REPO, "2026-07-04 TRMG2")
SHP = os.path.join(REPO, "2026-07-04 TRMG2_shp")
TAZ_SHP = os.path.join(REPO, "docs", "data", "input", "tazs", "master_tazs.shp")
sys.path.insert(0, HERE)
from transcad_bin import read_bin  # noqa: E402

AT_ORDER = ["Rural", "Suburban", "Urban", "Downtown"]
AT_RANK = {a: i for i, a in enumerate(AT_ORDER)}
AT_THRESH = [("Downtown", 25000, 0.25), ("Urban", 10000, 0.5), ("Suburban", 1000, 0.5)]
CC_SPEED = {"Rural": 45, "Suburban": 35, "Urban": 30, "Downtown": 20}
AM_HOURS = 2.423   # capacity_period_factors.csv


def load_lookup_csv(path, keycols, valcols):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[tuple(r[k].strip() for k in keycols)] = {v: r[v] for v in valcols}
    return out


def main():
    import geopandas as gpd
    log = {"approximations": ["A1 buffer-smoothing via centroid-in-buffer",
                              "A2 link area type via BFS from centroids",
                              "A4 ramps fall back to MajorArterial rows"]}

    # ---------- TransCAD export: real lengths, DIR, coordinates, geometry ----------
    exp = {}
    with open(os.path.join(EXP, "master_links 2026-07-04.csv")) as f:
        for r in csv.DictReader(f):
            exp[int(r["ID"])] = (int(r["DIR"]), float(r["LENGTH"] or 0))
    gn = gpd.read_file(os.path.join(SHP, "master_nodes 2026-07-04.shp")).to_crs(epsg=4326)
    ncoord = {int(i): (round(p.x, 6), round(p.y, 6))
              for i, p in zip(gn["ID"], gn.geometry)}
    gl = gpd.read_file(os.path.join(SHP, "master_links 2026-07-04.shp")).to_crs(epsg=4326)
    lwkt = {}
    for i, geom in zip(gl["ID"], gl.geometry):
        try:
            lwkt[int(i)] = geom.wkt if geom is not None else ""
        except Exception:
            lwkt[int(i)] = ""
    log["export_links"] = len(exp)
    log["export_nodes_with_coords"] = len(ncoord)

    # ---------- zones: area type (A1) ----------
    _, se_rows = read_bin(os.path.join(REPO, "master", "sedata", "se_2020.bin"))
    se = {r["TAZ"]: r for r in se_rows}
    g = gpd.read_file(TAZ_SHP).set_index("ID")
    g = g.to_crs(epsg=32617)  # UTM 17N, meters
    tot_pop = sum(r["HH_POP"] or 0 for r in se_rows)
    tot_emp_all = sum((r["Industry"] or 0) + (r["Office"] or 0) + (r["Service_RateLow"] or 0)
                      + (r["Service_RateHigh"] or 0) + (r["Retail"] or 0) for r in se_rows)
    factor = tot_pop / tot_emp_all
    zone_at = {}
    for taz, r in se.items():
        if taz in g.index:
            area = g.loc[taz].geometry.area / 2_589_988.0
            emp = (r["Industry"] or 0) + (r["Office"] or 0) + (r["Service_RateLow"] or 0) \
                + (r["Service_RateHigh"] or 0) + (r["Retail"] or 0)
            dens = ((r["HH_POP"] or 0) + emp * factor) / max(area, 1e-6)
            at = "Rural"
            for name, cutoff, _ in AT_THRESH:
                if dens >= cutoff:
                    at = name
                    break
            zone_at[taz] = at
        else:
            zone_at[taz] = "Rural"
    cent_pts = g.geometry.centroid
    for name, _, buf_mi in AT_THRESH:
        ids = [z for z, a in zone_at.items() if a == name and z in g.index]
        if not ids or buf_mi <= 0:
            continue
        buf = g.loc[ids].geometry.buffer(buf_mi * 1609.34).union_all()
        for z in g.index:
            if AT_RANK[zone_at.get(z, "Rural")] < AT_RANK[name] and buf.contains(cent_pts.loc[z]):
                zone_at[z] = name
    log["zone_at_counts"] = {a: sum(1 for v in zone_at.values() if v == a) for a in AT_ORDER}
    log["zones_without_polygon"] = len([z for z in se if z not in g.index])

    # ---------- topology + node area type (A2: BFS) ----------
    topo = []
    with open(os.path.join(HERE, "topology.csv")) as f:
        for r in csv.DictReader(f):
            topo.append((int(r["link_id"]), int(r["from_node_id"]), int(r["to_node_id"])))
    _, node_rows = read_bin(os.path.join(NET, "master_links_.BIN"))
    centroids = set(r["ID"] for r in node_rows if r.get("Centroid") == 1)
    adj = defaultdict(list)
    for _, a, b in topo:
        adj[a].append(b)
        adj[b].append(a)
    node_at = {}
    q = deque()
    for z in centroids:
        node_at[z] = zone_at.get(z, "Rural")
        q.append(z)
    while q:
        n = q.popleft()
        for m in adj[n]:
            if m not in node_at:
                node_at[m] = node_at[n]
                q.append(m)

    # ---------- link attributes ----------
    _, link_rows = read_bin(os.path.join(NET, "master_links.BIN"))
    attrs = {r["ID"]: r for r in link_rows}
    _, init_rows = read_bin(os.path.join(NET, "init_cong_time_2020.bin"))
    init = {r["ID"]: r for r in init_rows}

    cap_lut = load_lookup_csv(os.path.join(NET, "capacity.csv"),
                              ["HCMType", "AreaType", "HCMMedian"], ["cape_phpl"])
    spd_lut = load_lookup_csv(os.path.join(NET, "ff_speed_alpha_beta.csv"),
                              ["HCMType", "AreaType"], ["ModifyPosted", "Alpha", "Beta"])

    def median_for(hcm, med):
        if hcm in ("Freeway", "MLHighway", "Superstreet"):
            return "Restrictive"
        if hcm in ("TLHighway", "CC"):
            return "None"
        return med if med in ("Restrictive", "NonRestrictive") else "None"

    links_out, skipped, no_export = [], 0, 0
    len_pairs = []   # (real, NT-estimate) to grade the retired A3 approximation
    for lid, fn, tn in topo:
        r = attrs.get(lid)
        if r is None or not r.get("DTWB") or "D" not in r["DTWB"]:
            continue
        hcm = r["HCMType"] or ""
        if not hcm:
            continue
        e = exp.get(lid)
        if e is None:
            no_export += 1
            continue
        dir_code, length = e
        if length <= 0:
            length = 0.05
        at = max((node_at.get(fn, "Rural"), node_at.get(tn, "Rural")), key=lambda a: AT_RANK[a])
        med = median_for(hcm, r.get("HCMMedian") or "")
        if hcm == "CC":
            ffs, alpha, beta, cap_phpl = CC_SPEED[at], 0.15, 4.0, 10000.0
        else:
            s = spd_lut.get((hcm, at))
            c = cap_lut.get((hcm, at, med)) or cap_lut.get((hcm, at, "None"))
            if s is None or c is None:
                s = s or spd_lut.get(("MajorArterial", at))
                c = c or cap_lut.get(("MajorArterial", at, med)) \
                      or cap_lut.get(("MajorArterial", at, "None"))
            if s is None or c is None:
                skipped += 1
                continue
            ffs = max((r["PostedSpeed"] or 25) + float(s["ModifyPosted"]), 10.0)
            alpha, beta = float(s["Alpha"]), float(s["Beta"])
            cap_phpl = float(c["cape_phpl"])
        it = init.get(lid)
        if it and (it.get("ABInitCongTimeNT") or 0) > 0 and hcm != "CC":
            len_pairs.append((length, it["ABInitCongTimeNT"] / 60.0 * ffs))
        wkt = lwkt.get(lid, "")
        directions = []
        if dir_code >= 0:
            directions.append((1, "ABLanes"))
        if dir_code <= 0:
            directions.append((-1, "BALanes"))
        for direction, lanes_f in directions:
            lanes = r.get(lanes_f) or 0
            if lanes <= 0:
                lanes = 1   # DIR is authoritative; repair missing lane count
            fftt = length / ffs * 60.0
            a, b = (fn, tn) if direction == 1 else (tn, fn)
            links_out.append(dict(
                link_id=lid * 10 + (1 if direction == 1 else 2),
                from_node_id=a, to_node_id=b, dir_flag=1,
                length=round(length, 5), vdf_length_mi=round(length, 5),
                lanes=lanes, capacity=round(cap_phpl * lanes * AM_HOURS, 1),
                free_speed=round(ffs * 1.609, 1), vdf_free_speed_mph=ffs,
                vdf_toll=0, link_type=1 if hcm == "CC" else 2,
                link_type_name=hcm, area_type=at, allowed_uses="",
                day_count=(r.get("DailyCount") or ""),
                screenline=(r.get("Screenline") or ""),
                vdf_alpha=alpha, vdf_beta=beta, vdf_plf=1,
                vdf_fftt=round(fftt, 4), geometry=wkt))

    log["directed_links"] = len(links_out)
    log["skipped_no_lookup"] = skipped
    log["links_missing_from_export"] = no_export
    if len_pairs:
        import numpy as np
        real = np.array([p[0] for p in len_pairs])
        est = np.array([p[1] for p in len_pairs])
        log["retired_A3_length_check"] = dict(
            n=len(len_pairs),
            corr=round(float(np.corrcoef(real, est)[0, 1]), 4),
            mean_abs_rel_err=round(float(np.mean(np.abs(est - real) / np.maximum(real, 0.01))), 4))

    # ---------- nodes: renumber for the kernel zone model ----------
    used = set()
    for l in links_out:
        used.add(l["from_node_id"]); used.add(l["to_node_id"])
    zones_used = sorted(n for n in used if n in centroids)
    thru = sorted(n for n in used if n not in centroids)
    newid = {}
    for k, n in enumerate(zones_used, start=1):
        newid[n] = k
    for k, n in enumerate(thru, start=len(zones_used) + 1):
        newid[n] = k
    out_dir = os.path.join(HERE, "scenario")
    os.makedirs(out_dir, exist_ok=True)
    missing_coord = 0
    with open(os.path.join(out_dir, "node.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "zone_id", "x_coord", "y_coord"])
        for n in zones_used + thru:
            xy = ncoord.get(n)
            if xy is None:
                missing_coord += 1
                xy = (0, 0)
            w.writerow([newid[n], newid[n] if n in centroids else "", xy[0], xy[1]])
    log["nodes_missing_coords"] = missing_coord
    for l in links_out:
        l["from_node_id"] = newid[l["from_node_id"]]
        l["to_node_id"] = newid[l["to_node_id"]]
    # kernel CSR builder requires links sorted ascending by from_node_id
    links_out.sort(key=lambda l: (l["from_node_id"], l["to_node_id"]))
    with open(os.path.join(out_dir, "link.csv"), "w", newline="") as f:
        cols = list(links_out[0].keys())
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for l in links_out:
            w.writerow(l)
    with open(os.path.join(HERE, "node_crosswalk.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trmg2_node_id", "gmns_node_id", "taz"])
        for n in zones_used + thru:
            w.writerow([n, newid[n], n if n in centroids else ""])
    log["nodes"] = len(used)
    log["zones_in_network"] = len(zones_used)
    with open(os.path.join(HERE, "build_log.json"), "w") as f:
        json.dump(log, f, indent=1)
    print(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
