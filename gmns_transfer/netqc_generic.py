"""NetQC generic screen battery — config-driven, portable across GMNS-ish
networks (TRMG2, MAG 2024, AZTDM baseline, ...).

Broadened issue taxonomy (superset of NETQC_WORKPACKAGE.md S1-S5; each
issue row carries the *reasoning* and the *source* of evidence so a human
can verify it in QGIS via netqc_gis.py):

  Topology
    S1  lane balance at high-class merge/diverge (twin pair-exclusion)
    S2  lane discontinuity along same-class degree-2 corridor
    S3  free-speed discontinuity along same-class degree-2 corridor
    S6  dead-end: non-centroid node with only inbound or only outbound links
    S7a self-loop link (from == to)
    S7b duplicate directed link (same from,to coded twice)
    S7c zero/near-zero coded length
    S7d coded length vs geodesic chord ratio (unit errors, bad geometry)
    S13 small isolated components of the drivable non-connector graph
  Attributes
    S8a lanes <= 0 or implausibly high
    S8b capacity <= 0
    S8c sentinel capacity on a non-connector (>= 90000 = "unrestricted")
    S8d free-speed outlier for a drivable link (< 8 or > 85)
    S8e missing facility class on a non-connector
    S9  capacity-per-lane outlier within facility x area group (< 0.5x / > 2x median)
    S10 unit heuristics per facility group (speed kmh-vs-mph, capacity
        daily-vs-hourly) — group medians, GSATS/MAG lesson
    S12 two-way twin asymmetry (AB vs BA lanes differ >= 2 or capacity ratio > 2)
  Connectors / zonal loading
    S4  centroid connector touching a high-class (freeway) link
    S11a zone with no connector
    S11b connector longer than 3 mi
    S11c node with extreme degree (> 10)

Output per network: <out>/issues.csv (screen,kind,node,link_id,severity,
detail,source), <out>/SCREENS.md summary. GIS export = netqc_gis.py.

Usage: python netqc_generic.py {trmg2|mag2024|aztdm}
"""
import csv
import math
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
NEWF = os.path.abspath(os.path.join(
    HERE, "..", "..", "..", "New folder"))

# --------------------------------------------------------------- configs

def _f(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _i(v, default=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


CONFIGS = {
    "trmg2": dict(
        link_csv=os.path.join(HERE, "scenario", "link.csv"),
        node_csv=os.path.join(HERE, "scenario", "node.csv"),
        out=os.path.join(HERE, "netqc", "generic"),
        link=lambda r: dict(
            id=r["link_id"], a=_i(r["from_node_id"]), b=_i(r["to_node_id"]),
            ft=r["link_type_name"], at=r["area_type"],
            lanes=_i(r["lanes"]), ffs=_f(r["vdf_free_speed_mph"]),
            cap=_f(r["capacity"]), len_mi=_f(r["length"]),
            wkt=r.get("geometry", "")),
        node=lambda r: dict(
            id=_i(r["node_id"]), x=_f(r["x_coord"]), y=_f(r["y_coord"]),
            zone=(r.get("zone_id") or "").strip()),
        high={"Freeway", "MLHighway", "Ramp"},
        is_conn=lambda l: l["ft"] == "CC",
        twin_base=lambda l: _i(l["id"]) // 10,
        speed_units="mph", cap_units="hourly/lane-ish",
    ),
    "mag2024": dict(
        link_csv=os.path.join(NEWF, "MAG 2024 Network", "final_links.csv"),
        node_csv=os.path.join(NEWF, "MAG 2024 Network", "final_nodes.csv"),
        out=os.path.join(NEWF, "netqc_mag"),
        # facility_type codes (MAG): 1 freeway, 2 HOV, 3 expressway, ramps 8/9,
        # arterials 5/6; connector = link_type 0
        link=lambda r: dict(
            id=r["link_id"], a=_i(r["from_node_id"]), b=_i(r["to_node_id"]),
            ft=("connector" if r["link_type"] == "0"
                else "FT" + str(_i(r["facility_type"], -1))),
            at="AT" + str(_i(r["AT"], -1)),
            lanes=_i(r["lanes"]), ffs=_f(r["free_speed"]),
            cap=_f(r["capacity"]), len_mi=_f(r["vdf_length_mi"], 0.0)
            or _f(r["length"]) / 1609.34,   # length col is meters
            wkt=r.get("geometry", "")),
        node=lambda r: dict(
            id=_i(r["node_id"]), x=_f(r["x"]), y=_f(r["y"]),
            zone=(r.get("zone_id") or "").strip()),
        # empirically inferred from speed/lanes medians (FT1 71mph/3ln
        # freeway, FT2 57mph HOV, FT8/9 ramps, FT16 70mph system ramp) —
        # CONFIRM against the MAG facility-type dictionary before edits
        high={"FT1", "FT2", "FT8", "FT9", "FT16"},
        is_conn=lambda l: l["ft"] == "connector",
        twin_base=None,   # no documented twin id scheme; use (b,a) existence
        speed_units="mph?", cap_units="unknown (declare!)",
    ),
    "aztdm": dict(
        link_csv=os.path.join(NEWF, "AZTDM GMNS Network",
                              "final_version_clean_Apr25",
                              "GMNS_Network_baseline", "link.csv"),
        node_csv=os.path.join(NEWF, "AZTDM GMNS Network",
                              "final_version_clean_Apr25",
                              "GMNS_Network_baseline", "node.csv"),
        out=os.path.join(NEWF, "netqc_aztdm"),
        # AZTDM FT codes: 1 local/collector?, 2 freeway?, ... report by code;
        # centroid connectors = FT 9 (volume_group -1, cap 99999) per profile
        link=lambda r: dict(
            id=r["link_id"], a=_i(r["from_node_id"]), b=_i(r["to_node_id"]),
            ft="FT" + str(_i(r["FT"], -1)), at="AT" + str(_i(r["AT"], -1)),
            lanes=_i(r["lanes"]), ffs=_f(r["free_speed"]),
            cap=_f(r["capacity"]), len_mi=_f(r["length"]),
            wkt=r.get("geometry", "")),
        node=lambda r: dict(
            id=_i(r["node_id"]), x=_f(r["x_coord"]), y=_f(r["y_coord"]),
            zone=(r.get("zone_id") or "").strip()),
        # empirically inferred: FT1 = 65mph/2ln highway-freeway class (63k
        # links incl. rural highways), FT9 = connectors (20mph, lanes=9
        # sentinel) — CONFIRM against AZTDM FT dictionary before edits
        high={"FT1"},
        is_conn=lambda l: l["ft"] == "FT9",
        twin_base=None,
        speed_units="mph", cap_units="capacity col = 99999 sentinel; real "
                                     "caps in VDF_cap1/2/3 per period (declare!)",
    ),
}


def geodesic_mi(x1, y1, x2, y2):
    ky = 69.17
    kx = 69.17 * math.cos(math.radians((y1 + y2) / 2))
    return math.hypot((x2 - x1) * kx, (y2 - y1) * ky)


def load(cfg):
    csv.field_size_limit(10_000_000)
    links, nodes = [], {}
    with open(cfg["node_csv"], encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            n = cfg["node"](r)
            nodes[n["id"]] = n
    with open(cfg["link_csv"], encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            links.append(cfg["link"](r))
    return links, nodes


def main(net):
    cfg = CONFIGS[net]
    os.makedirs(cfg["out"], exist_ok=True)
    links, nodes = load(cfg)
    high, is_conn = cfg["high"], cfg["is_conn"]
    inb, outb = defaultdict(list), defaultdict(list)
    ab = {}
    for l in links:
        outb[l["a"]].append(l)
        inb[l["b"]].append(l)
        ab.setdefault((l["a"], l["b"]), []).append(l)
    issues = []

    def add(screen, sev, detail, source, node="", link_id=""):
        issues.append(dict(screen=screen, node=node, link_id=link_id,
                           severity=round(sev, 1), detail=detail,
                           source=source))

    # ---- S1 lane balance (twin pair-exclusion; see NETQC_WORKPACKAGE §1)
    for n in set(list(inb) + list(outb)):
        li = [l for l in inb.get(n, []) if l["ft"] in high]
        lo = [l for l in outb.get(n, []) if l["ft"] in high]
        if not li or not lo:
            continue
        if any(l["ft"] not in high and not is_conn(l)
               for l in inb.get(n, []) + outb.get(n, [])):
            continue
        tb = cfg["twin_base"] or (lambda l: 0)
        ik = {(l["a"], tb(l)) for l in li}
        ok = {(l["b"], tb(l)) for l in lo}
        tw = ik & ok
        li = [l for l in li if (l["a"], tb(l)) not in tw]
        lo = [l for l in lo if (l["b"], tb(l)) not in tw]
        if not li or not lo:
            continue
        imb = sum(l["lanes"] for l in li) - sum(l["lanes"] for l in lo)
        if abs(imb) >= 2:
            kind = ("merge" if len(li) > len(lo)
                    else "diverge" if len(lo) > len(li) else "thru")
            add("S1_lane_balance", abs(imb) * 2,
                f"in {len(li)}legs/{sum(l['lanes'] for l in li)}ln vs out "
                f"{len(lo)}legs/{sum(l['lanes'] for l in lo)}ln ({kind})",
                "topology: lane conservation at controlled-access node",
                node=n)

    # ---- S2/S3 degree-2 discontinuities
    for n in set(list(inb) + list(outb)):
        li, lo = inb.get(n, []), outb.get(n, [])
        if len(li) == 1 and len(lo) == 1:
            i, o = li[0], lo[0]
            if i["ft"] == o["ft"] and i["a"] != o["b"] and not is_conn(i):
                d = abs(i["lanes"] - o["lanes"])
                if i["ft"] in high and d >= 2:
                    add("S2_lane_jump", d * 1.5,
                        f"{i['ft']} {i['lanes']}ln -> {o['lanes']}ln",
                        "topology: same-facility degree-2 node", node=n)
                if abs(i["ffs"] - o["ffs"]) > 15:
                    add("S3_speed_jump", 1.0,
                        f"{i['ft']} {i['ffs']:.0f} -> {o['ffs']:.0f}",
                        "attribute: posted/free speed continuity", node=n)

    # ---- S6 dead ends (non-centroid)
    for n, nd in nodes.items():
        if nd["zone"]:
            continue
        deg_i, deg_o = len(inb.get(n, [])), len(outb.get(n, []))
        if (deg_i == 0) != (deg_o == 0):
            add("S6_dead_end", 2.0,
                f"{deg_i} in / {deg_o} out — traffic {'sink' if deg_o == 0 else 'source'}",
                "topology: directed connectivity", node=n)

    # ---- S7 link geometry/hygiene
    for l in links:
        if l["a"] == l["b"]:
            add("S7a_self_loop", 3.0, f"link {l['id']} from==to=={l['a']}",
                "topology", link_id=l["id"], node=l["a"])
        if l["len_mi"] <= 0.001 and not is_conn(l):
            add("S7c_zero_length", 2.0,
                f"link {l['id']} coded length {l['len_mi']:.4f} mi",
                "attribute: length field", link_id=l["id"], node=l["a"])
        na, nb = nodes.get(l["a"]), nodes.get(l["b"])
        if na and nb and l["len_mi"] > 0.01 and not is_conn(l):
            # geometry length (polyline if WKT present, else chord) — the
            # chord alone false-positives every loop ramp
            g = 0.0
            w = l.get("wkt", "")
            if w and "LINESTRING" in w:
                body = w[w.find("(") + 1: w.rfind(")")]
                pts = []
                for pair in body.split(","):
                    xy = pair.split()
                    if len(xy) >= 2:
                        pts.append((float(xy[0]), float(xy[1])))
                g = sum(geodesic_mi(*pts[i], *pts[i + 1])
                        for i in range(len(pts) - 1))
                basis = "geometry length"
            if g <= 0.02:
                g = geodesic_mi(na["x"], na["y"], nb["x"], nb["y"])
                basis = "node chord"
            if g > 0.02:
                ratio = l["len_mi"] / g
                if ratio < 0.8 or ratio > 1.5:
                    add("S7d_length_ratio", min(4.0, 1.5 * abs(math.log(max(ratio, 1e-9)))),
                        f"link {l['id']} coded {l['len_mi']:.2f} mi vs "
                        f"{basis} {g:.2f} mi (ratio {ratio:.2f}) — unit or "
                        "geometry error",
                        f"geometry: coded length vs {basis}",
                        link_id=l["id"], node=l["a"])
    for (a, b), ls in ab.items():
        if len(ls) > 1:
            add("S7b_duplicate", 3.0,
                f"{len(ls)} parallel links {a}->{b}: "
                + ",".join(str(l["id"]) for l in ls[:5]),
                "topology", node=a, link_id=ls[0]["id"])

    # ---- S8 attribute validity
    for l in links:
        if is_conn(l):
            continue
        if l["lanes"] <= 0 or l["lanes"] > 8:
            add("S8a_lanes", 3.0, f"link {l['id']} lanes={l['lanes']}",
                "attribute", link_id=l["id"], node=l["a"])
        if l["cap"] <= 0:
            add("S8b_capacity", 3.0, f"link {l['id']} capacity={l['cap']}",
                "attribute", link_id=l["id"], node=l["a"])
        elif l["cap"] >= 90000:
            add("S8c_cap_sentinel", 2.5,
                f"link {l['id']} {l['ft']} capacity={l['cap']:.0f} looks like "
                "an 'unrestricted' sentinel on a real link",
                "attribute: sentinel value", link_id=l["id"], node=l["a"])
        if l["ffs"] and (l["ffs"] < 8 or l["ffs"] > 85):
            add("S8d_speed", 2.0,
                f"link {l['id']} {l['ft']} free_speed={l['ffs']:.0f} "
                f"({cfg['speed_units']})",
                "attribute", link_id=l["id"], node=l["a"])
        if l["ft"] in ("FT-1", "") or l["ft"] is None:
            add("S8e_no_class", 1.5, f"link {l['id']} has no facility class",
                "attribute", link_id=l["id"], node=l["a"])

    # ---- S9 cap/lane outliers + S10 group unit heuristics
    grp = defaultdict(list)
    for l in links:
        if not is_conn(l) and l["lanes"] > 0 and 0 < l["cap"] < 90000:
            grp[(l["ft"], l["at"])].append(l)
    for (ft, at), ls in grp.items():
        med = sorted(x["cap"] / x["lanes"] for x in ls)[len(ls) // 2]
        if med <= 0:
            continue
        for x in ls:
            r = (x["cap"] / x["lanes"]) / med
            if r < 0.5 or r > 2.0:
                add("S9_cap_outlier", 1.0 + min(2.0, abs(math.log(r))),
                    f"link {x['id']} {ft}/{at} cap/ln {x['cap']/x['lanes']:.0f} "
                    f"= {r:.2f}x group median {med:.0f}",
                    "attribute: group consistency", link_id=x["id"], node=x["a"])
        if med > 2400:
            add("S10_cap_units", 3.0,
                f"group {ft}/{at}: median cap/lane {med:.0f} > 2400 — "
                "looks DAILY, not hourly (declare capacity convention)",
                "unit heuristic (GSATS AB_CAP lesson)", node="",
                link_id=ls[0]["id"])
        spds = sorted(x["ffs"] for x in ls if x["ffs"] > 0)
        if spds:
            smed = spds[len(spds) // 2]
            if smed > 90:
                add("S10_speed_units", 3.0,
                    f"group {ft}/{at}: median free speed {smed:.0f} — "
                    "looks km/h, not mph",
                    "unit heuristic", node="", link_id=ls[0]["id"])

    # ---- S12 twin asymmetry
    seen = set()
    for (a, b), ls in ab.items():
        if (b, a) in ab and (b, a) not in seen:
            seen.add((a, b))
            l1, l2 = ls[0], ab[(b, a)][0]
            if is_conn(l1) or is_conn(l2):
                continue
            if abs(l1["lanes"] - l2["lanes"]) >= 2:
                add("S12_twin_asym", 2.0,
                    f"{l1['id']} {l1['lanes']}ln vs reverse {l2['id']} "
                    f"{l2['lanes']}ln", "topology: AB/BA consistency",
                    link_id=l1["id"], node=a)
            c1, c2 = l1["cap"], l2["cap"]
            if 0 < c1 < 90000 and 0 < c2 < 90000 and max(c1, c2) / min(c1, c2) > 2:
                add("S12_twin_asym", 1.5,
                    f"{l1['id']} cap {c1:.0f} vs reverse {l2['id']} cap {c2:.0f}",
                    "topology: AB/BA consistency", link_id=l1["id"], node=a)

    # ---- S4 / S11 connectors & zonal loading
    zones_with_conn = set()
    for l in links:
        if not is_conn(l):
            continue
        za = nodes.get(l["a"], {}).get("zone")
        zb = nodes.get(l["b"], {}).get("zone")
        if za:
            zones_with_conn.add(za)
        if zb:
            zones_with_conn.add(zb)
        for other in outb.get(l["b"], []) + inb.get(l["a"], []):
            if other["ft"] in high and not is_conn(other):
                add("S4_cc_to_freeway", 3.0,
                    f"connector {l['id']} touches {other['ft']} {other['id']}",
                    "topology: zonal loading on controlled access",
                    node=l["b"], link_id=l["id"])
                break
        if l["len_mi"] > 3.0:
            add("S11b_long_connector", 2.0,
                f"connector {l['id']} is {l['len_mi']:.1f} mi",
                "zonal loading plausibility", link_id=l["id"], node=l["a"])
    all_zones = {nd["zone"] for nd in nodes.values() if nd["zone"]}
    for z in sorted(all_zones - zones_with_conn)[:200]:
        add("S11a_zone_no_connector", 2.5, f"zone {z} has no connector",
            "zonal coverage", node="")
    for n in set(list(inb) + list(outb)):
        deg = len(inb.get(n, [])) + len(outb.get(n, []))
        if deg > 10 and not nodes.get(n, {}).get("zone"):
            add("S11c_high_degree", 1.5, f"node degree {deg}",
                "topology", node=n)

    # ---- S13 isolated components (undirected, non-connector)
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(x, y):
        parent.setdefault(x, x)
        parent.setdefault(y, y)
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for l in links:
        if not is_conn(l):
            union(l["a"], l["b"])
    comp = defaultdict(int)
    for x in list(parent):
        comp[find(x)] += 1
    main_c = max(comp.values()) if comp else 0
    for root, size in comp.items():
        if size < main_c and size <= 50:
            add("S13_isolated_component", 2.5,
                f"component of {size} nodes detached from main network "
                f"({main_c} nodes), e.g. node {root}",
                "topology: connectivity", node=root)

    # ---- dedupe, rank, write
    seen_k = set()
    ranked = []
    for it in sorted(issues, key=lambda x: -x["severity"]):
        k = (it["screen"], it["node"], it["link_id"], it["detail"][:40])
        if k not in seen_k:
            seen_k.add(k)
            ranked.append(it)

    # systemic findings: a link-level screen firing network-wide is one
    # data-convention defect, not N independent errors — surface it as a
    # single top-severity row (link rows are kept for the GIS layer)
    n_links = max(1, len(links))
    per_screen = Counter(it["screen"] for it in ranked)
    for scr, cnt in per_screen.items():
        if cnt > max(500, 0.05 * n_links):
            ex = [it for it in ranked if it["screen"] == scr][:5]
            ranked.insert(0, dict(
                screen=scr + "_SYSTEMIC", node="", link_id="",
                severity=5.0,
                detail=f"{cnt:,} of {n_links:,} links fire {scr} — systemic "
                       "data convention, fix at the source, e.g. "
                       + ",".join(str(e["link_id"] or e["node"]) for e in ex),
                source="systemic aggregation of " + scr))
    with open(os.path.join(cfg["out"], "issues.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["screen", "node", "link_id",
                                          "severity", "detail", "source"])
        w.writeheader()
        w.writerows(ranked)
    by = Counter(it["screen"] for it in ranked)
    with open(os.path.join(cfg["out"], "SCREENS.md"), "w", encoding="utf-8") as f:
        f.write(f"# NetQC generic screens — {net}\n\n")
        f.write(f"links {len(links):,}, nodes {len(nodes):,}; "
                f"flagged (deduped) {len(ranked):,}\n\n")
        f.write(f"speed units assumed: {cfg['speed_units']}; capacity "
                f"convention: {cfg['cap_units']}\n\n")
        f.write("| screen | count |\n|---|---|\n")
        for s, c in sorted(by.items()):
            f.write(f"| {s} | {c} |\n")
    print(net, "issues:", dict(sorted(by.items())))
    print("->", cfg["out"])
    return ranked


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "trmg2")
