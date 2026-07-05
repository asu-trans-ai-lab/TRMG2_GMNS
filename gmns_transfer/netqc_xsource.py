"""NetQC cross-source verification for MAG 2024 (test run of the TRMG2 P7
approach on the MAG/ADOT data): MPO network vs TWO independent sources on a
5x5-mile window, then a per-link suspicion ranking that says *which specific
information is most suspicious and why*.

Sources:
  1. OSM (Overpass, motorway..secondary + _link) — supplies lanes / oneway /
     maxspeed / name evidence via the P7 matcher (stage A pairwise + stage B
     chain aggregation + dual-carriageway + class-affinity refinements,
     reusing netqc_osm helpers).
  2. ADOT AllRoadsNetwork 2023 (authoritative state route inventory,
     RouteId/RouteName/RouteSubtype only — no lanes) — supplies *coverage +
     route designation* evidence: a freeway-class model link with no ADOT
     route geometry nearby is suspicious; RouteSubtype 98 = Non-ADOT route.

Outputs under <net out dir> (netqc_mag/):
  osm_win/*.osm + adot_window.geojson    cached extracts
  netqc_osm_match.csv / netqc_osm_issues.csv   (same schema as TRMG2 P7,
      + ADOT_NO_ROUTE / adot columns)
  suspects.csv + gis/suspects.gpkg|.shp|.gml   ranked, QGIS-ready
  XSOURCE.md                                    summary + top suspects table

Usage: python netqc_xsource.py [mag2024]
"""
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict

import geopandas as gpd
from shapely.geometry import LineString

import netqc_osm as base
from netqc_generic import CONFIGS, load

HERE = os.path.dirname(os.path.abspath(__file__))
NEWF = os.path.abspath(os.path.join(HERE, "..", "..", "..", "New folder"))
ADOT_GJ = os.path.join(NEWF, "ADOT_AllRoadsNetwork_2023_5661467596376727376.geojson")

XCFG = {
    "mag2024": dict(
        center=(33.46, -112.10),   # I-10/I-17 central Phoenix
        size_mi=5.0,
        # MAG ft -> class-consistent OSM highway classes: used as affinity
        # tie-break AND as the gate for lane evidence (a 1L frontage road
        # snapping onto the I-10 mainline must not become a LANE_MISMATCH)
        pref={"FT1": {"motorway", "trunk"}, "FT2": {"motorway", "trunk"},
              "FT8": {"motorway_link", "trunk_link", "primary_link",
                      "secondary_link"},
              "FT9": {"motorway_link", "trunk_link", "primary_link"},
              "FT16": {"motorway_link", "trunk_link"},
              "FT6": {"primary", "secondary", "trunk"},
              "FT7": {"secondary", "tertiary", "unclassified", "residential",
                      "primary"},
              "FT3": {"secondary", "tertiary", "primary"},
              "FT4": {"secondary", "tertiary", "primary"}},
        matchable={"FT1", "FT2", "FT8", "FT9", "FT16", "FT6", "FT7",
                   "FT3", "FT4"},
        high={"FT1", "FT2", "FT8", "FT9", "FT16"},
        class_gap={"FT5", "FT11", "FT12"},
        # widen the window extract so frontage roads/service streets have
        # their true OSM counterpart available to match
        hwy=["motorway", "motorway_link", "trunk", "trunk_link",
             "primary", "primary_link", "secondary", "secondary_link",
             "tertiary", "tertiary_link", "unclassified", "residential"],
    ),
}


def stream_adot_window(south, west, north, east, cache):
    """One pass over the 299 MB AllRoads geojson; keep features whose any
    vertex falls in the window. Cached."""
    if os.path.exists(cache):
        return json.load(open(cache, encoding="utf-8"))
    feats = []
    marker = '{"type":"Feature"'
    buf = ""
    with open(ADOT_GJ, encoding="utf-8") as f:
        while True:
            chunk = f.read(8 << 20)
            if not chunk:
                break
            buf += chunk
            parts = buf.split(marker)
            buf = marker + parts[-1]
            for frag in parts[1:-1]:
                s = marker + frag.rstrip(",")
                try:
                    ft = json.loads(s)
                except json.JSONDecodeError:
                    continue
                g = ft.get("geometry") or {}
                if g.get("type") != "LineString":
                    continue
                if any(west <= c[0] <= east and south <= c[1] <= north
                       for c in g["coordinates"][::5] or g["coordinates"]):
                    feats.append(dict(
                        route_id=(ft["properties"].get("RouteId") or "").strip(),
                        name=(ft["properties"].get("RouteName") or "").strip(),
                        subtype=ft["properties"].get("RouteSubtype"),
                        subtype_v=(ft["properties"].get("RouteSubtype_Value")
                                   or "").strip(),
                        pts=[(c[0], c[1]) for c in g["coordinates"]]))
    # tail fragment
    s = buf.rstrip("]}").rstrip(",")
    if s.startswith(marker):
        try:
            ft = json.loads(s)
            g = ft.get("geometry") or {}
            if g.get("type") == "LineString" and any(
                    west <= c[0] <= east and south <= c[1] <= north
                    for c in g["coordinates"][::5]):
                feats.append(dict(
                    route_id=(ft["properties"].get("RouteId") or "").strip(),
                    name=(ft["properties"].get("RouteName") or "").strip(),
                    subtype=ft["properties"].get("RouteSubtype"),
                    subtype_v=(ft["properties"].get("RouteSubtype_Value") or "").strip(),
                    pts=[(c[0], c[1]) for c in g["coordinates"]]))
        except json.JSONDecodeError:
            pass
    json.dump(feats, open(cache, "w", encoding="utf-8"))
    return feats


def main(net="mag2024"):
    cfg = CONFIGS[net]
    x = XCFG[net]
    out = cfg["out"]
    osmdir = os.path.join(out, "osm_win")
    os.makedirs(osmdir, exist_ok=True)

    cy, cx = x["center"]
    dlat = x["size_mi"] / 2 / 69.17
    dlon = x["size_mi"] / 2 / (69.17 * math.cos(math.radians(cy)))
    south, north, west, east = cy - dlat, cy + dlat, cx - dlon, cx + dlon

    # ---- retarget the netqc_osm helper module at this window
    base.CY, base.CX = cy, cx
    base.M_PER_DEG_LON = 111320.0 * math.cos(math.radians(cy))
    base.SOUTH, base.NORTH, base.WEST, base.EAST = south, north, west, east
    base.OSMFILE = os.path.join(osmdir, f"{net}_window_v2.osm")
    base.OSMDIR = osmdir
    base.HWY = x["hwy"]

    base.download_osm()
    ways = base.parse_osm()
    segs = base.osm_segments(ways)
    print(f"OSM: {len(ways)} ways -> {len(segs)} directional segments")

    adot = stream_adot_window(south, west, north, east,
                              os.path.join(out, "adot_window.geojson"))
    adot_route = [a for a in adot if a["subtype"] != 98]
    print(f"ADOT AllRoads in window: {len(adot)} features "
          f"({len(adot_route)} ADOT-designated routes)")
    for a in adot:
        a["ppts"] = [base.proj(*p) for p in a["pts"]]

    # ---- MPO links in window
    links, nodes = load(cfg)
    mpo = []
    for l in links:
        if l["ft"] not in x["matchable"]:
            continue
        w = l.get("wkt", "")
        if not w or "LINESTRING" not in w:
            continue
        body = w[w.find("(") + 1: w.rfind(")")]
        pts, inside = [], False
        for pair in body.split(","):
            xy = pair.split()
            if len(xy) >= 2:
                lon, lat = float(xy[0]), float(xy[1])
                if west <= lon <= east and south <= lat <= north:
                    inside = True
                pts.append(base.proj(lon, lat))
        if inside and len(pts) >= 2:
            mpo.append(dict(id=str(l["id"]), a=l["a"], b=l["b"], ft=l["ft"],
                            lanes=l["lanes"], ffs=l["ffs"], cap=l["cap"],
                            name="", pts=pts, wkt=w))
    ids = {(l["a"], l["b"]) for l in mpo}
    for l in mpo:
        l["twoway"] = (l["b"], l["a"]) in ids
    print(f"MPO links in window: {len(mpo)} "
          f"(high-class {sum(1 for l in mpo if l['ft'] in x['high'])})")

    for s in segs:
        s["bb"] = base.bbox(s["pts"])
        s["chord"] = base.chord(s["pts"])
        s["len"] = base.length_m(s["pts"])

    CELL = 200.0
    grid = defaultdict(list)
    for i, s in enumerate(segs):
        x0, y0, x1, y1 = s["bb"]
        for gx in range(int(x0 // CELL), int(x1 // CELL) + 1):
            for gy in range(int(y0 // CELL), int(y1 // CELL) + 1):
                grid[(gx, gy)].append(i)

    def candidates(bb):
        o = set()
        for gx in range(int(bb[0] // CELL), int(bb[2] // CELL) + 1):
            for gy in range(int(bb[1] // CELL), int(bb[3] // CELL) + 1):
                o.update(grid.get((gx, gy), []))
        return o

    agrid = defaultdict(list)
    for i, a in enumerate(adot):
        b = base.bbox(a["ppts"], 40.0)
        for gx in range(int(b[0] // CELL), int(b[2] // CELL) + 1):
            for gy in range(int(b[1] // CELL), int(b[3] // CELL) + 1):
                agrid[(gx, gy)].append(i)

    # ---- stage A + B matching (P7 with class affinity)
    matches = []
    for l in mpo:
        mb = base.bbox(l["pts"], 25.0)
        mchord = base.chord(l["pts"])
        mlen = base.length_m(l["pts"])
        pref = x["pref"].get(l["ft"])
        scored = []
        for i in candidates(mb):
            s = segs[i]
            if not base.bbox_isect(mb, s["bb"]):
                continue
            dc = mchord[0] * s["chord"][0] + mchord[1] * s["chord"][1]
            if dc <= 0:
                continue
            h = base.hausdorff(l["pts"], s["pts"])
            sc = (max(0.0, dc) + max(0.0, 1.0 - h / 50.0)
                  + min(mlen, s["len"]) / max(mlen, s["len"])) / 3.0
            scored.append((sc, i))
        best, best_s = None, 0.0
        if scored:
            scored.sort(reverse=True)
            near = [i for sc, i in scored if sc >= scored[0][0] - 0.05]
            pick = next((i for i in near if pref and segs[i]["hwy"] in pref),
                        near[0])
            best_s = dict((i, sc) for sc, i in scored)[pick]
            best = segs[pick]
        if best is not None and best_s >= 0.6:
            matches.append(dict(link=l, kind="MATCHED", score=best_s,
                                segs=[best], coverage=1.0))
        else:
            samples = base.densify(l["pts"], 50.0)
            hits = []
            for p in samples:
                bn, bd, bnp, bdp = None, 25.0, None, 25.0
                for i in candidates((p[0] - 30, p[1] - 30, p[0] + 30, p[1] + 30)):
                    s = segs[i]
                    dc = mchord[0] * s["chord"][0] + mchord[1] * s["chord"][1]
                    if dc <= 0.3:
                        continue
                    d = base.min_dist_to_polyline(p, s["pts"])
                    if d < bd:
                        bd, bn = d, i
                    if pref and s["hwy"] in pref and d < bdp:
                        bdp, bnp = d, i
                hits.append(bnp if bnp is not None else bn)
            cov = sum(1 for h in hits if h is not None) / len(hits)
            if cov >= 0.8:
                used = [segs[i] for i in sorted({h for h in hits if h is not None})]
                matches.append(dict(link=l, kind="MATCHED_CHAIN", score=cov,
                                    segs=used, coverage=cov))
            else:
                matches.append(dict(link=l, kind="MPO_UNMATCHED", score=cov,
                                    segs=[], coverage=cov))

    # ---- ADOT coverage per link
    for m in matches:
        l = m["link"]
        samples = base.densify(l["pts"], 100.0)
        best_route, covered = "", 0
        route_hits = Counter()
        for p in samples:
            for i in agrid.get((int(p[0] // CELL), int(p[1] // CELL)), []):
                a = adot[i]
                if base.min_dist_to_polyline(p, a["ppts"]) < 30.0:
                    covered += 1
                    route_hits[(a["route_id"], a["subtype"])] += 1
                    break
        m["adot_cov"] = covered / len(samples)
        if route_hits:
            (rid, st), _ = route_hits.most_common(1)[0]
            m["adot_route"] = rid
            m["adot_subtype"] = st
        else:
            m["adot_route"], m["adot_subtype"] = "", None

    # ---- attribute diff -> typed issues
    def has_antiparallel(l):
        mchord = base.chord(l["pts"])
        samples = base.densify(l["pts"], 50.0)
        near = 0
        for p in samples:
            for i in candidates((p[0] - 40, p[1] - 40, p[0] + 40, p[1] + 40)):
                s = segs[i]
                dc = mchord[0] * s["chord"][0] + mchord[1] * s["chord"][1]
                if dc < -0.3 and base.min_dist_to_polyline(p, s["pts"]) < 35.0:
                    near += 1
                    break
        return near / len(samples) >= 0.5

    issues = []
    for m in matches:
        l = m["link"]
        adot_note = (f"ADOT route {m['adot_route']}" if m["adot_route"]
                     else "no ADOT route nearby")
        if m["kind"] == "MPO_UNMATCHED":
            tag = ("OSM_CLASS_GAP" if l["ft"] in x["class_gap"]
                   else "MPO_UNMATCHED")
            sev = 3.0 if (tag == "MPO_UNMATCHED" and m["adot_cov"] < 0.3) else 1.0
            issues.append(dict(issue=tag, link_id=l["id"], ft=l["ft"],
                               mpo="", osm=f"coverage {m['coverage']:.2f}",
                               adot=adot_note, severity=sev,
                               detail="no OSM match"
                               + ("; ALSO absent from ADOT inventory -> "
                                  "possible phantom/misplaced link"
                                  if m["adot_cov"] < 0.3 else "")))
            continue
        lens = [s["len"] for s in m["segs"]]
        lane_w = Counter()
        for s, w in zip(m["segs"], lens):
            dl = base.dir_lanes(s)
            if dl is not None:
                lane_w[dl] += w
        osm_lanes = lane_w.most_common(1)[0][0] if lane_w else None
        pref = x["pref"].get(l["ft"])
        class_ok = (pref is None
                    or any(s["hwy"] in pref for s in m["segs"]))
        if osm_lanes is not None:
            dL = abs(l["lanes"] - osm_lanes)
            if (l["ft"] in x["high"] and dL >= 1) or dL >= 2:
                if class_ok:
                    issues.append(dict(
                        issue="LANE_MISMATCH", link_id=l["id"], ft=l["ft"],
                        mpo=f"{l['lanes']}L", osm=f"{osm_lanes}L",
                        adot=adot_note, severity=2.0 + dL,
                        detail=f"{m['kind']} score {m['score']:.2f}; osm ways "
                               + ",".join(sorted({s['wid'] for s in m['segs']})[:4])))
                else:
                    issues.append(dict(
                        issue="CLASS_CONFLICT", link_id=l["id"], ft=l["ft"],
                        mpo=f"{l['ft']} {l['lanes']}L",
                        osm="/".join(sorted({s['hwy'] for s in m['segs']})[:3])
                        + f" {osm_lanes}L", adot=adot_note, severity=1.0,
                        detail="matched OSM way class inconsistent with MPO "
                               "facility class — match likely snapped to a "
                               "parallel roadway; lane evidence withheld"))
        ow_yes = sum(w for s, w in zip(m["segs"], lens)
                     if s["oneway"] in ("yes", "1", "true"))
        ow_frac = ow_yes / sum(lens) if sum(lens) else 0
        if l["twoway"] and ow_frac > 0.8 and not has_antiparallel(l):
            issues.append(dict(issue="ONEWAY_CONFLICT", link_id=l["id"],
                               ft=l["ft"], mpo="two-way", osm="oneway",
                               adot=adot_note, severity=4.0,
                               detail="no opposite-direction OSM way nearby"))
        sp = [(base.parse_maxspeed(s["maxspeed"]), w)
              for s, w in zip(m["segs"], lens) if base.parse_maxspeed(s["maxspeed"])]
        if sp and class_ok:
            osm_sp = sum(v * w for v, w in sp) / sum(w for _, w in sp)
            if abs(l["ffs"] - osm_sp) > 15:
                issues.append(dict(issue="SPEED_MISMATCH", link_id=l["id"],
                                   ft=l["ft"], mpo=f"{l['ffs']:.0f}",
                                   osm=f"{osm_sp:.0f} posted", adot=adot_note,
                                   severity=1.5,
                                   detail=f"{m['kind']} (ffs vs posted, "
                                          ">15 mph apart)"))
        if l["ft"] in x["high"] and m["adot_cov"] < 0.3:
            issues.append(dict(issue="ADOT_NO_ROUTE", link_id=l["id"],
                               ft=l["ft"], mpo=l["ft"], osm="",
                               adot=f"coverage {m['adot_cov']:.2f}",
                               severity=3.0,
                               detail="freeway-class model link with no ADOT "
                                      "AllRoads geometry within 30 m"))

    # ---- write match + issues
    with open(os.path.join(out, "netqc_osm_match.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["link_id", "ft", "lanes", "kind", "score", "coverage",
                    "osm_ways", "osm_lanes", "osm_maxspeed",
                    "adot_cov", "adot_route", "adot_subtype"])
        for m in matches:
            l = m["link"]
            lv = sorted({base.dir_lanes(s) for s in m["segs"]
                         if base.dir_lanes(s) is not None})
            w.writerow([l["id"], l["ft"], l["lanes"], m["kind"],
                        f"{m['score']:.3f}", f"{m['coverage']:.2f}",
                        " ".join(sorted({s["wid"] for s in m["segs"]})[:6]),
                        "/".join(map(str, lv)),
                        " ".join(sorted({s["maxspeed"] for s in m["segs"]
                                         if s["maxspeed"]})[:3]),
                        f"{m['adot_cov']:.2f}", m["adot_route"],
                        m["adot_subtype"]])
    with open(os.path.join(out, "netqc_osm_issues.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["issue", "link_id", "ft", "mpo",
                                          "osm", "adot", "severity", "detail"])
        w.writeheader()
        w.writerows(issues)

    # ---- suspicion ranking: generic screens + OSM + ADOT per link
    gen = defaultdict(float)
    gen_scr = defaultdict(set)
    for r in csv.DictReader(open(os.path.join(out, "issues.csv"),
                                 encoding="utf-8")):
        if r["link_id"]:
            gen[str(r["link_id"])] += float(r["severity"]) * 0.5
            gen_scr[str(r["link_id"])].add(r["screen"])
    sus = defaultdict(lambda: dict(score=0.0, why=[]))
    for it in issues:
        d = sus[it["link_id"]]
        d["score"] += float(it["severity"])
        d["why"].append(f"{it['issue']}({it['mpo']} vs {it['osm']}; {it['adot']})")
    for lid, sc in gen.items():
        if lid in sus or sc >= 1.5:
            sus[lid]["score"] += sc
            sus[lid]["why"].append("screens:" + ";".join(sorted(gen_scr[lid])))
    mpo_by_id = {l["id"]: l for l in mpo}
    rows = []
    for lid, d in sorted(sus.items(), key=lambda kv: -kv[1]["score"]):
        l = mpo_by_id.get(lid)
        rows.append(dict(link_id=lid, ft=l["ft"] if l else "",
                         lanes=l["lanes"] if l else "",
                         ffs=l["ffs"] if l else "",
                         score=round(d["score"], 1),
                         why=" | ".join(d["why"])[:800]))
    with open(os.path.join(out, "suspects.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["link_id", "ft", "lanes", "ffs",
                                          "score", "why"])
        w.writeheader()
        w.writerows(rows)

    # suspects GIS layer (window links, ranked)
    feats = []
    for r in rows:
        l = mpo_by_id.get(r["link_id"])
        if not l:
            continue
        inv = 1.0 / base.M_PER_DEG_LON
        pts = [((px * inv) + cx, (py / 110540.0) + cy) for px, py in l["pts"]]
        feats.append(dict(**{k: r[k] for k in
                             ("link_id", "ft", "lanes", "ffs", "score", "why")},
                          geometry=LineString(pts)))
    if feats:
        gdf = gpd.GeoDataFrame(feats, crs="EPSG:4326")
        gisdir = os.path.join(out, "gis")
        os.makedirs(gisdir, exist_ok=True)
        gdf.to_file(os.path.join(gisdir, f"suspects_{net}.gpkg"),
                    layer="suspects", driver="GPKG")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gdf.to_file(os.path.join(gisdir, "shp", f"{net}_suspects.shp"),
                        driver="ESRI Shapefile")
            gdf.to_file(os.path.join(gisdir, "gml", f"{net}_suspects.gml"),
                        driver="GML")

    kinds = Counter(m["kind"] for m in matches)
    hk = Counter(m["kind"] for m in matches if m["link"]["ft"] in x["high"])
    icnt = Counter(i["issue"] for i in issues)
    h_tot = sum(hk.values())
    h_ok = hk["MATCHED"] + hk["MATCHED_CHAIN"]
    with open(os.path.join(out, "XSOURCE.md"), "w", encoding="utf-8") as f:
        f.write(f"# Cross-source verification test run — {net}\n\n")
        f.write(f"Window: {x['size_mi']}x{x['size_mi']} mi centered "
                f"({cy}, {cx}); OSM {len(ways)} ways; ADOT AllRoads "
                f"{len(adot)} features ({len(adot_route)} ADOT routes); "
                f"MPO links {len(mpo)}\n\n")
        f.write("| metric | value |\n|---|---|\n")
        for k, v in sorted(kinds.items()):
            f.write(f"| {k} | {v} |\n")
        f.write(f"| high-class matched | {h_ok}/{h_tot} = "
                f"{100 * h_ok / max(1, h_tot):.1f}% |\n\n")
        f.write("| issue | count |\n|---|---|\n")
        for k, v in sorted(icnt.items()):
            f.write(f"| {k} | {v} |\n")
        f.write("\n## Top 25 suspects (see suspects.csv / gis/suspects_*.gpkg)\n\n")
        f.write("| link | ft | lanes | ffs | score | why |\n|---|---|---|---|---|---|\n")
        for r in rows[:25]:
            f.write(f"| {r['link_id']} | {r['ft']} | {r['lanes']} | "
                    f"{r['ffs']} | {r['score']} | "
                    + r["why"][:220].replace("|", "/") + " |\n")
    print("match kinds:", dict(kinds))
    print(f"high-class matched: {h_ok}/{h_tot}")
    print("issues:", dict(icnt))
    print(f"suspects: {len(rows)} -> suspects.csv, XSOURCE.md, gis/")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mag2024")
