"""P7 cross-source verification: TRMG2 MPO network vs OSM (osm2gmns) on a
5x5-mile RTP-area window.

Pipeline (NETQC_WORKPACKAGE.md section 7):
  1. Download OSM ways (motorway|trunk|primary|secondary + their _link
     classes) for the window via Overpass (osm2gmns.downloadOSMData wants an
     area relation id, not a bbox — so Overpass direct, cached).
  2. Build a GMNS network with osm2gmns -> netqc/osm_rtp/{node,link}.csv
     (deliverable / rendering source). Attribute ground truth (lanes,
     maxspeed, oneway, name) is re-parsed from the OSM XML tags directly,
     because osm2gmns fills defaults where tags are missing and collapses
     the *_link distinction.
  3. Map-match MPO -> OSM, geometry only, local meter projection:
     stage A (pairwise, per spec): candidates = OSM links whose bbox
       intersects the MPO link's 25 m buffer; score = mean of
       (a) direction cosine between chord vectors (clipped at 0),
       (b) 1 - clip(symmetric Hausdorff / 50 m),
       (c) length ratio min(L1,L2)/max(L1,L2);
       accept best score >= 0.6.
     stage B (chain aggregation): MPO links longer than typical OSM
       segmentation never clear stage A's length-ratio/Hausdorff terms, so
       unmatched MPO links are sampled every 50 m; each sample snaps to the
       nearest direction-consistent OSM link within 25 m; >= 80% snapped
       samples => MATCHED_CHAIN with length-weighted attribute aggregation.
     Remaining: MPO_UNMATCHED. Reverse sweep marks OSM_UNMATCHED coverage
     gaps (OSM links with no MPO link within 25 m).
  4. Attribute diff per matched pair -> typed issues:
       LANE_MISMATCH   |dL| >= 1 on H-class, >= 2 elsewhere (lanes tagged)
       ONEWAY_CONFLICT MPO two-way vs OSM oneway, or MPO one-way vs OSM
                       two-way street
       SPEED_MISMATCH  > 10 mph vs maxspeed where tagged (informational:
                       MPO field is free-flow, not posted)
       NAME_MISMATCH   informational
  5. Render the top LANE/ONEWAY conflicts side-by-side (MPO facility colors
     + OSM dashed black with its own labels) -> netqc/site_osm_###.png for
     the P6 verdict loop.

Outputs (all under netqc/): osm_rtp/ GMNS, netqc_osm_match.csv,
netqc_osm_issues.csv, OSM_XCHECK.md, site_osm_###.png
"""
import csv
import math
import os
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
GM = os.path.join(HERE, "scenario")
OUT = os.path.join(HERE, "netqc")
OSMDIR = os.path.join(OUT, "osm_rtp")
os.makedirs(OSMDIR, exist_ok=True)
OSMFILE = os.path.join(OSMDIR, "rtp_window.osm")

# 5x5 mile window centered on I-40 through RTP / SW Durham (covers flagged
# nodes 10980, 10983, 5336 = sites 002/003/011 of the v2 top-50)
CY, CX = 35.905, -78.900
DLAT = 2.5 / 69.17
DLON = 2.5 / (69.17 * math.cos(math.radians(CY)))
SOUTH, NORTH, WEST, EAST = CY - DLAT, CY + DLAT, CX - DLON, CX + DLON

HWY = ["motorway", "motorway_link", "trunk", "trunk_link",
       "primary", "primary_link", "secondary", "secondary_link"]
HIGH = {"Freeway", "MLHighway", "Ramp"}
MPO_FT_KEEP = {"Freeway", "MLHighway", "Ramp", "MajorArterial", "Arterial",
               "Superstreet", "TLHighway", "MajorCollector"}
# class affinity: geometry-only matching snaps short ramps braided within
# 25 m of the mainline onto the motorway way; used only as a tie-breaker
# among near-equal candidates and for chain snapping preference.
CLASS_PREF = {
    "Freeway": {"motorway", "trunk"},
    "MLHighway": {"motorway", "trunk", "primary"},
    "Ramp": {"motorway_link", "trunk_link", "primary_link", "secondary_link"},
    "MajorArterial": {"trunk", "primary", "secondary"},
    "Arterial": {"primary", "secondary"},
    "Superstreet": {"trunk", "primary", "secondary"},
    "TLHighway": {"trunk", "primary", "secondary"},
}
# MPO classes whose OSM counterpart (tertiary/unclassified) is outside the
# section-7 extract: their unmatched links are a class-coverage gap, not
# evidence of geometry missing from OSM.
OSM_CLASS_GAP_FT = {"MajorCollector", "Arterial"}

M_PER_DEG_LAT = 110540.0
M_PER_DEG_LON = 111320.0 * math.cos(math.radians(CY))


def proj(lon, lat):
    return ((lon - CX) * M_PER_DEG_LON, (lat - CY) * M_PER_DEG_LAT)


# ---------------------------------------------------------------- download
def download_osm():
    if os.path.exists(OSMFILE) and os.path.getsize(OSMFILE) > 10000:
        print(f"cached OSM: {OSMFILE} ({os.path.getsize(OSMFILE)/1e6:.1f} MB)")
        return
    clauses = "\n".join(
        f'  way["highway"="{h}"]({SOUTH:.6f},{WEST:.6f},{NORTH:.6f},{EAST:.6f});'
        for h in HWY)
    q = f"[out:xml][timeout:300];\n(\n{clauses}\n);\n(._;>;);\nout;\n"
    mirrors = ["https://overpass-api.de/api/interpreter",
               "https://overpass.kumi.systems/api/interpreter",
               "https://overpass.openstreetmap.ru/api/interpreter"]
    for attempt in range(6):
        url = mirrors[attempt % len(mirrors)]
        try:
            print(f"overpass try {attempt + 1}: {url}")
            req = urllib.request.Request(url, data=q.encode(),
                                         headers={"User-Agent": "netqc-p7/1.0"})
            with urllib.request.urlopen(req, timeout=320) as r:
                data = r.read()
            if len(data) > 10000:
                with open(OSMFILE, "wb") as f:
                    f.write(data)
                print(f"downloaded {len(data)/1e6:.1f} MB -> {OSMFILE}")
                return
            print(f"  response too small ({len(data)} B), retrying")
        except Exception as e:
            print(f"  failed: {e}")
        time.sleep(5 * (attempt + 1))
    raise RuntimeError("all Overpass mirrors failed")


# ---------------------------------------------------- parse OSM XML directly
def parse_osm():
    """Return ways: id -> dict(tags, pts[(lon,lat)...])."""
    nodes = {}
    ways = {}
    for _, el in ET.iterparse(OSMFILE, events=("end",)):
        if el.tag == "node":
            nodes[el.get("id")] = (float(el.get("lon")), float(el.get("lat")))
        elif el.tag == "way":
            tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
            if tags.get("highway") in HWY:
                pts = [nodes[nd.get("ref")] for nd in el.findall("nd")
                       if nd.get("ref") in nodes]
                if len(pts) >= 2:
                    ways[el.get("id")] = dict(tags=tags, pts=pts)
            el.clear()
    return ways


def build_gmns():
    """osm2gmns GMNS build (deliverable); failures are non-fatal."""
    if os.path.exists(os.path.join(OSMDIR, "link.csv")):
        print("osm2gmns GMNS already built")
        return
    try:
        import osm2gmns as og
        net = og.getNetFromFile(OSMFILE, link_types=HWY)
        og.outputNetToCSV(net, output_folder=OSMDIR)
        print(f"osm2gmns: {net.number_of_nodes} nodes, "
              f"{net.number_of_links} links -> {OSMDIR}")
    except Exception as e:
        print(f"osm2gmns build failed (matching uses raw XML anyway): {e}")


# ------------------------------------------------------------ OSM segments
def osm_segments(ways):
    """Directional segments from raw ways. Two-way ways emit both directions
    so the direction-cosine term can pick the right carriageway."""
    segs = []
    for wid, w in ways.items():
        t = w["tags"]
        ow = t.get("oneway", "no")
        pts = [proj(*p) for p in w["pts"]]
        lanes = t.get("lanes")
        fwd = dict(wid=wid, hwy=t["highway"], name=t.get("name", t.get("ref", "")),
                   maxspeed=t.get("maxspeed", ""), oneway=ow, pts=pts,
                   lanes_total=lanes, lanes_dir=t.get("lanes:forward"))
        if ow == "-1":
            fwd["pts"] = pts[::-1]
            fwd["lanes_dir"] = t.get("lanes:backward")
            fwd["oneway"] = "yes"
            segs.append(fwd)
            continue
        segs.append(fwd)
        if ow not in ("yes", "1", "true"):
            segs.append(dict(wid=wid, hwy=t["highway"],
                             name=t.get("name", t.get("ref", "")),
                             maxspeed=t.get("maxspeed", ""), oneway=ow,
                             pts=pts[::-1], lanes_total=lanes,
                             lanes_dir=t.get("lanes:backward")))
    return segs


def dir_lanes(seg):
    """Best-effort directional lane count from OSM tags; None if untagged."""
    if seg["lanes_dir"]:
        try:
            return int(float(seg["lanes_dir"]))
        except ValueError:
            return None
    if seg["lanes_total"] is None:
        return None
    try:
        total = int(float(seg["lanes_total"]))
    except ValueError:
        return None
    ow = seg["oneway"]
    if ow in ("yes", "1", "true"):
        return total
    return max(1, round(total / 2))  # two-way 'lanes' tag counts both dirs


def parse_maxspeed(s):
    if not s:
        return None
    s = s.lower().strip()
    try:
        if "mph" in s:
            return float(s.replace("mph", "").strip())
        return float(s) * 0.621371  # bare number = km/h
    except ValueError:
        return None


# ------------------------------------------------------------------ geometry
def polyline_pts(pl):
    return np.asarray(pl, dtype=float)


def pt_seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def min_dist_to_polyline(p, poly):
    return min(pt_seg_dist(p[0], p[1], poly[i][0], poly[i][1],
                           poly[i + 1][0], poly[i + 1][1])
               for i in range(len(poly) - 1))


def densify(poly, step=25.0):
    out = [tuple(poly[0])]
    for i in range(len(poly) - 1):
        ax, ay = poly[i]
        bx, by = poly[i + 1]
        d = math.hypot(bx - ax, by - ay)
        n = max(1, int(d // step))
        for k in range(1, n + 1):
            out.append((ax + (bx - ax) * k / n, ay + (by - ay) * k / n))
    return out


def hausdorff(pa, pb):
    da = max(min_dist_to_polyline(p, pb) for p in densify(pa))
    db = max(min_dist_to_polyline(p, pa) for p in densify(pb))
    return max(da, db)


def chord(poly):
    v = (poly[-1][0] - poly[0][0], poly[-1][1] - poly[0][1])
    n = math.hypot(*v)
    return (v[0] / n, v[1] / n) if n > 0 else (0.0, 0.0)


def length_m(poly):
    return sum(math.hypot(poly[i + 1][0] - poly[i][0],
                          poly[i + 1][1] - poly[i][1])
               for i in range(len(poly) - 1))


def bbox(poly, buf=0.0):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs) - buf, min(ys) - buf, max(xs) + buf, max(ys) + buf)


def bbox_isect(b1, b2):
    return not (b1[2] < b2[0] or b2[2] < b1[0] or b1[3] < b2[1] or b2[3] < b1[1])


# --------------------------------------------------------------------- main
def main():
    download_osm()
    build_gmns()
    ways = parse_osm()
    segs = osm_segments(ways)
    print(f"OSM: {len(ways)} ways -> {len(segs)} directional segments")

    csv.field_size_limit(10_000_000)
    mpo = []
    with open(os.path.join(GM, "link.csv")) as f:
        for r in csv.DictReader(f):
            ft = r["link_type_name"]
            if ft not in MPO_FT_KEEP:
                continue
            g = r["geometry"]
            body = g[g.find("(") + 1: g.rfind(")")]
            pts = []
            inside = False
            for pair in body.split(","):
                xy = pair.split()
                lon, lat = float(xy[0]), float(xy[1])
                if WEST <= lon <= EAST and SOUTH <= lat <= NORTH:
                    inside = True
                pts.append(proj(lon, lat))
            if not inside or len(pts) < 2:
                continue
            mpo.append(dict(id=int(r["link_id"]), a=int(r["from_node_id"]),
                            b=int(r["to_node_id"]), ft=ft,
                            lanes=int(r["lanes"]),
                            ffs=float(r["vdf_free_speed_mph"]),
                            name=r.get("name", ""), pts=pts))
    ids = {(l["a"], l["b"]) for l in mpo}
    for l in mpo:
        l["twoway"] = (l["b"], l["a"]) in ids
    print(f"MPO links in window: {len(mpo)} "
          f"(H-class {sum(1 for l in mpo if l['ft'] in HIGH)})")

    for s in segs:
        s["bb"] = bbox(s["pts"])
        s["chord"] = chord(s["pts"])
        s["len"] = length_m(s["pts"])

    # spatial hash for OSM segments (200 m cells)
    CELL = 200.0
    grid = defaultdict(list)
    for i, s in enumerate(segs):
        x0, y0, x1, y1 = s["bb"]
        for gx in range(int(x0 // CELL), int(x1 // CELL) + 1):
            for gy in range(int(y0 // CELL), int(y1 // CELL) + 1):
                grid[(gx, gy)].append(i)

    def candidates(bb):
        out = set()
        for gx in range(int(bb[0] // CELL), int(bb[2] // CELL) + 1):
            for gy in range(int(bb[1] // CELL), int(bb[3] // CELL) + 1):
                out.update(grid.get((gx, gy), []))
        return out

    matches = []
    for l in mpo:
        mb = bbox(l["pts"], 25.0)
        mchord = chord(l["pts"])
        mlen = length_m(l["pts"])
        pref = CLASS_PREF.get(l["ft"])
        scored = []
        for i in candidates(mb):
            s = segs[i]
            if not bbox_isect(mb, s["bb"]):
                continue
            dc = mchord[0] * s["chord"][0] + mchord[1] * s["chord"][1]
            if dc <= 0:
                continue
            h = hausdorff(l["pts"], s["pts"])
            sc = (max(0.0, dc) + max(0.0, 1.0 - h / 50.0)
                  + min(mlen, s["len"]) / max(mlen, s["len"])) / 3.0
            scored.append((sc, i))
        best, best_s = None, 0.0
        if scored:
            scored.sort(reverse=True)
            top_s = scored[0][0]
            # among near-equal candidates prefer the class-compatible one
            near = [i for sc, i in scored if sc >= top_s - 0.05]
            pick = next((i for i in near
                         if pref and segs[i]["hwy"] in pref), near[0])
            best_s = dict((i, sc) for sc, i in scored)[pick]
            best = segs[pick]
        if best is not None and best_s >= 0.6:
            matches.append(dict(link=l, kind="MATCHED", score=best_s,
                                segs=[best], coverage=1.0))
            continue
        # stage B: chain aggregation
        samples = densify(l["pts"], 50.0)
        hits = []
        for p in samples:
            bn, bd = None, 25.0
            bn_pref, bd_pref = None, 25.0
            for i in candidates((p[0] - 30, p[1] - 30, p[0] + 30, p[1] + 30)):
                s = segs[i]
                d = min_dist_to_polyline(p, s["pts"])
                # local direction consistency at nearest OSM segment
                dc = mchord[0] * s["chord"][0] + mchord[1] * s["chord"][1]
                if dc <= 0.3:
                    continue
                if d < bd:
                    bd, bn = d, i
                if pref and s["hwy"] in pref and d < bd_pref:
                    bd_pref, bn_pref = d, i
            hits.append(bn_pref if bn_pref is not None else bn)
        cov = sum(1 for h in hits if h is not None) / len(hits)
        if cov >= 0.8:
            used = [segs[i] for i in
                    sorted({h for h in hits if h is not None})]
            matches.append(dict(link=l, kind="MATCHED_CHAIN", score=cov,
                                segs=used, coverage=cov))
        else:
            matches.append(dict(link=l, kind="MPO_UNMATCHED", score=cov,
                                segs=[], coverage=cov))

    # reverse sweep: OSM segments with no MPO link nearby (coverage gaps)
    mgrid = defaultdict(list)
    for i, l in enumerate(mpo):
        b = bbox(l["pts"], 25.0)
        for gx in range(int(b[0] // CELL), int(b[2] // CELL) + 1):
            for gy in range(int(b[1] // CELL), int(b[3] // CELL) + 1):
                mgrid[(gx, gy)].append(i)
    osm_unmatched = []
    for s in segs:
        if s["hwy"].endswith("_link"):
            continue  # ramps often absent from MPO by design
        mid = densify(s["pts"], 50.0)
        cand = set()
        for p in mid:
            cand.update(mgrid.get((int(p[0] // CELL), int(p[1] // CELL)), []))
        near = 0
        for p in mid:
            if any(min_dist_to_polyline(p, mpo[i]["pts"]) < 25.0 for i in cand):
                near += 1
        if near / len(mid) < 0.2 and s["len"] > 100:
            osm_unmatched.append(s)

    # ------------------------------------------------- attribute diff
    def has_antiparallel(l):
        """True if an opposite-direction OSM segment runs near this MPO link
        (i.e., OSM models the road as a dual carriageway: two oneway ways)."""
        mchord = chord(l["pts"])
        samples = densify(l["pts"], 50.0)
        near = 0
        for p in samples:
            for i in candidates((p[0] - 40, p[1] - 40, p[0] + 40, p[1] + 40)):
                s = segs[i]
                dc = mchord[0] * s["chord"][0] + mchord[1] * s["chord"][1]
                if dc < -0.3 and min_dist_to_polyline(p, s["pts"]) < 35.0:
                    near += 1
                    break
        return near / len(samples) >= 0.5

    issues = []
    for m in matches:
        if m["kind"] == "MPO_UNMATCHED":
            tag = ("OSM_CLASS_GAP" if m["link"]["ft"] in OSM_CLASS_GAP_FT
                   else "MPO_UNMATCHED")
            issues.append(dict(issue=tag, link_id=m["link"]["id"],
                               ft=m["link"]["ft"], mpo="", osm="",
                               detail=f"coverage {m['coverage']:.2f}"))
            continue
        l = m["link"]
        lens = [s["len"] for s in m["segs"]]
        # length-weighted modal directional lanes over tagged segs
        lane_w = Counter()
        for s, w in zip(m["segs"], lens):
            dl = dir_lanes(s)
            if dl is not None:
                lane_w[dl] += w
        osm_lanes = lane_w.most_common(1)[0][0] if lane_w else None
        if osm_lanes is not None:
            dL = abs(l["lanes"] - osm_lanes)
            if (l["ft"] in HIGH and dL >= 1) or dL >= 2:
                issues.append(dict(
                    issue="LANE_MISMATCH", link_id=l["id"], ft=l["ft"],
                    mpo=f"{l['lanes']}L", osm=f"{osm_lanes}L",
                    detail=f"{m['kind']} score {m['score']:.2f}; osm ways "
                           + ",".join(sorted({s['wid'] for s in m['segs']}))))
        ow_yes = sum(w for s, w in zip(m["segs"], lens)
                     if s["oneway"] in ("yes", "1", "true"))
        ow_frac = ow_yes / sum(lens) if sum(lens) else 0
        if l["twoway"] and ow_frac > 0.8:
            if has_antiparallel(l):
                issues.append(dict(
                    issue="DUAL_CARRIAGEWAY_REP", link_id=l["id"], ft=l["ft"],
                    mpo="two-way centerline", osm="dual oneway carriageways",
                    detail="representation difference, informational"))
            else:
                issues.append(dict(issue="ONEWAY_CONFLICT", link_id=l["id"],
                                   ft=l["ft"], mpo="two-way", osm="oneway",
                                   detail=f"{m['kind']}; no opposite-direction "
                                          "OSM way nearby"))
        elif not l["twoway"] and ow_frac < 0.2 and l["ft"] in HIGH:
            issues.append(dict(issue="ONEWAY_CONFLICT", link_id=l["id"],
                               ft=l["ft"], mpo="one-way", osm="two-way street",
                               detail=f"{m['kind']}"))
        sp_w = [(parse_maxspeed(s["maxspeed"]), w)
                for s, w in zip(m["segs"], lens) if parse_maxspeed(s["maxspeed"])]
        if sp_w:
            osm_sp = sum(v * w for v, w in sp_w) / sum(w for _, w in sp_w)
            if abs(l["ffs"] - osm_sp) > 10:
                issues.append(dict(issue="SPEED_MISMATCH", link_id=l["id"],
                                   ft=l["ft"], mpo=f"{l['ffs']:.0f} ffs mph",
                                   osm=f"{osm_sp:.0f} posted mph",
                                   detail=f"{m['kind']} (informational: "
                                          "ffs vs posted)"))
        names = {s["name"] for s in m["segs"] if s["name"]}
        if l["name"] and names and not any(
                l["name"].lower() in n.lower() or n.lower() in l["name"].lower()
                for n in names):
            issues.append(dict(issue="NAME_MISMATCH", link_id=l["id"],
                               ft=l["ft"], mpo=l["name"],
                               osm="|".join(sorted(names)[:3]),
                               detail="informational"))
    for s in osm_unmatched:
        issues.append(dict(issue="OSM_UNMATCHED", link_id="",
                           ft=s["hwy"], mpo="",
                           osm=f"way {s['wid']} {s['name']}",
                           detail=f"{s['len']:.0f} m with no MPO link within "
                                  "25 m (coverage gap)"))

    # --------------------------------------------------------- outputs
    with open(os.path.join(OUT, "netqc_osm_match.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["link_id", "ft", "lanes", "kind", "score", "coverage",
                    "osm_ways", "osm_lanes", "osm_maxspeed", "osm_names"])
        for m in matches:
            l = m["link"]
            lane_vals = sorted({dir_lanes(s) for s in m["segs"]
                                if dir_lanes(s) is not None})
            w.writerow([l["id"], l["ft"], l["lanes"], m["kind"],
                        f"{m['score']:.3f}", f"{m['coverage']:.2f}",
                        " ".join(sorted({s["wid"] for s in m["segs"]})),
                        "/".join(map(str, lane_vals)),
                        " ".join(sorted({s["maxspeed"] for s in m["segs"]
                                         if s["maxspeed"]})),
                        "|".join(sorted({s["name"] for s in m["segs"]
                                         if s["name"]})[:3])])
    with open(os.path.join(OUT, "netqc_osm_issues.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["issue", "link_id", "ft", "mpo",
                                          "osm", "detail"])
        w.writeheader()
        w.writerows(issues)

    kinds = Counter(m["kind"] for m in matches)
    hk = Counter(m["kind"] for m in matches if m["link"]["ft"] in HIGH)
    icnt = Counter(i["issue"] for i in issues)
    h_total = sum(hk.values())
    h_matched = hk["MATCHED"] + hk["MATCHED_CHAIN"]
    with open(os.path.join(OUT, "OSM_XCHECK.md"), "w") as f:
        f.write("# P7 cross-source verification: MPO vs OSM (RTP window)\n\n")
        f.write(f"Window: 5x5 mi centered ({CY:.3f}, {CX:.3f}); bbox "
                f"({SOUTH:.4f},{WEST:.4f},{NORTH:.4f},{EAST:.4f})\n\n")
        f.write(f"OSM: {len(ways)} ways / {len(segs)} directional segments; "
                f"MPO links in window: {len(mpo)}\n\n")
        f.write("| metric | value |\n|---|---|\n")
        for k, v in sorted(kinds.items()):
            f.write(f"| {k} (all classes) | {v} |\n")
        f.write(f"| H-class matched | {h_matched}/{h_total} = "
                f"{100 * h_matched / max(1, h_total):.1f}% "
                f"(acceptance >= 80%) |\n\n")
        f.write("| issue | count |\n|---|---|\n")
        for k, v in sorted(icnt.items()):
            f.write(f"| {k} | {v} |\n")
        f.write("\nDetails: netqc_osm_match.csv, netqc_osm_issues.csv; "
                "side-by-side renders: site_osm_###.png\n")
    print("match kinds:", dict(kinds))
    print(f"H-class matched: {h_matched}/{h_total}")
    print("issues:", dict(icnt))

    # ------------------------------------------- side-by-side renders (P6)
    lane_conf = [i for i in issues
                 if i["issue"] in ("LANE_MISMATCH", "ONEWAY_CONFLICT")]
    by_id = {l["id"]: l for l in mpo}
    match_by_id = {m["link"]["id"]: m for m in matches}
    for k, it in enumerate(lane_conf[:8]):
        l = by_id.get(it["link_id"])
        if l is None:
            continue
        m = match_by_id[l["id"]]
        cxm = sum(p[0] for p in l["pts"]) / len(l["pts"])
        cym = sum(p[1] for p in l["pts"]) / len(l["pts"])
        R = 800.0
        fig, ax = plt.subplots(figsize=(9, 9))
        FT_COLOR = {"Freeway": "#d62728", "MLHighway": "#ff7f0e",
                    "Ramp": "#9467bd", "MajorArterial": "#1f77b4",
                    "Arterial": "#17becf", "Superstreet": "#8c564b",
                    "TLHighway": "#e377c2", "MajorCollector": "#2ca02c"}
        for o in mpo:
            ob = bbox(o["pts"])
            if not bbox_isect(ob, (cxm - R, cym - R, cxm + R, cym + R)):
                continue
            xs = [p[0] for p in o["pts"]]
            ys = [p[1] for p in o["pts"]]
            ax.plot(xs, ys, color=FT_COLOR.get(o["ft"], "#333"),
                    lw=0.8 + 1.1 * o["lanes"], alpha=0.8, zorder=2)
        for s in segs:
            if not bbox_isect(s["bb"], (cxm - R, cym - R, cxm + R, cym + R)):
                continue
            xs = [p[0] for p in s["pts"]]
            ys = [p[1] for p in s["pts"]]
            ax.plot(xs, ys, color="black", lw=0.9, ls="--", alpha=0.7, zorder=3)
            dl = dir_lanes(s)
            if dl is not None and s["len"] > 150:
                mi = len(s["pts"]) // 2
                ax.text(s["pts"][mi][0], s["pts"][mi][1],
                        f"osm {dl}L {s['maxspeed']}", fontsize=6,
                        color="black",
                        bbox=dict(fc="yellow", ec="none", alpha=0.5, pad=0.4))
        xs = [p[0] for p in l["pts"]]
        ys = [p[1] for p in l["pts"]]
        ax.plot(xs, ys, color="red", lw=1.0, ls=":", zorder=6)
        ax.plot([xs[0]], [ys[0]], "k*", ms=16, zorder=7)
        ax.text(cxm, cym, f"MPO {l['lanes']}L {l['ffs']:.0f}mph", fontsize=8,
                weight="bold",
                bbox=dict(fc="white", ec="red", alpha=0.8, pad=0.6))
        ax.set_xlim(cxm - R, cxm + R)
        ax.set_ylim(cym - R, cym + R)
        ax.set_aspect(1.0)
        ax.set_title(f"site_osm_{k:03d} | {it['issue']} link {l['id']} "
                     f"({l['ft']}) | MPO {it['mpo']} vs OSM {it['osm']}",
                     fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"site_osm_{k:03d}.png"), dpi=110)
        plt.close(fig)
    print(f"rendered {min(8, len(lane_conf))} side-by-side conflict sites")


if __name__ == "__main__":
    main()
