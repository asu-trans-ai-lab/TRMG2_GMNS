"""AI-assisted network QC: heuristic screens -> rendered map snippets -> VLM verdicts.

Pipeline (prototype of the vision-QC loop discussed for MAG-class networks):
  1. SCREENS (pure topology/attributes, ranked by severity):
     S1 lane balance at freeway merge/diverge nodes (in-lanes vs out-lanes)
     S2 lane discontinuity along a facility (same road class, abrupt lane change)
     S3 speed discontinuity (posted+modify jump > 15 mph between adjacent links)
     S4 centroid connector plugged directly into Freeway/Ramp
     S5 capacity-per-lane outliers within facility x area type
  2. RENDER each flagged site as a PNG snippet: links within ~1 km, width =
     lanes, color = facility, direction arrows, lane labels — enough context
     for a vision-language model (or a human) to judge.
  3. VLM VERDICT loop: the snippets are reviewed by an AI reviewer which
     labels each site PLAUSIBLE / SUSPICIOUS / ERROR with a reason, producing
     netqc_verdicts.csv -> the input to network editing.

Outputs: netqc/issues.csv, netqc/site_###.png, netqc/NETQC_REPORT.md
"""
import csv
import json
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# overridable so the P8 edit loop can re-screen a patched network copy
GM = os.environ.get("NETQC_SCEN", os.path.join(HERE, "scenario"))
OUT = os.environ.get("NETQC_OUT", os.path.join(HERE, "netqc"))
os.makedirs(OUT, exist_ok=True)

FT_COLOR = {"Freeway": "#d62728", "MLHighway": "#ff7f0e", "Ramp": "#9467bd",
            "MajorArterial": "#1f77b4", "Arterial": "#17becf",
            "Superstreet": "#8c564b", "TLHighway": "#e377c2",
            "MajorCollector": "#2ca02c", "Collector": "#98df8a",
            "Local": "#7f7f7f", "CC": "#cccccc"}
HIGH = {"Freeway", "MLHighway", "Ramp"}


def parse_wkt(wkt):
    if not wkt or "LINESTRING" not in wkt:
        return None
    body = wkt[wkt.find("(") + 1: wkt.rfind(")")]
    pts = []
    for pair in body.split(","):
        xy = pair.split()
        if len(xy) >= 2:
            pts.append((float(xy[0]), float(xy[1])))
    return pts if len(pts) >= 2 else None


def main():
    nodes = {}
    with open(os.path.join(GM, "node.csv")) as f:
        for r in csv.DictReader(f):
            nodes[int(r["node_id"])] = (float(r["x_coord"]), float(r["y_coord"]),
                                        bool(r["zone_id"]))
    links = []
    with open(os.path.join(GM, "link.csv")) as f:
        for r in csv.DictReader(f):
            links.append(dict(
                id=int(r["link_id"]), a=int(r["from_node_id"]), b=int(r["to_node_id"]),
                ft=r["link_type_name"], at=r["area_type"], lanes=int(r["lanes"]),
                ffs=float(r["vdf_free_speed_mph"]), cap=float(r["capacity"]),
                length=float(r["length"]), pts=parse_wkt(r.get("geometry", ""))))
    inb, outb = defaultdict(list), defaultdict(list)
    for l in links:
        outb[l["a"]].append(l)
        inb[l["b"]].append(l)

    issues = []

    # ---- S1 lane balance at high-class merge/diverge nodes ----
    # P1 twin exclusion: if an inbound (a,n) and an outbound (n,a) share the
    # same base id (link_id // 10), they are the two directions of one two-way
    # carriageway segment. Both members are removed from the accounting —
    # dropping only the outbound side would leave the opposing carriageway's
    # inbound leg in the sum and manufacture fake merges at two-way thru nodes.
    for n in set(list(inb) + list(outb)):
        li, lo = inb.get(n, []), outb.get(n, [])
        hi_i = [l for l in li if l["ft"] in HIGH]
        hi_o = [l for l in lo if l["ft"] in HIGH]
        if not hi_i or not hi_o:
            continue
        # pure freeway junction (no arterial legs) -> lane conservation expected
        if any(l["ft"] not in HIGH and l["ft"] != "CC" for l in li + lo):
            continue
        in_keys = {(l["a"], l["id"] // 10) for l in hi_i}
        out_keys = {(l["b"], l["id"] // 10) for l in hi_o}
        twins = in_keys & out_keys
        hi_i = [l for l in hi_i if (l["a"], l["id"] // 10) not in twins]
        hi_o = [l for l in hi_o if (l["b"], l["id"] // 10) not in twins]
        if not hi_i or not hi_o:
            continue
        lin = sum(l["lanes"] for l in hi_i)
        lout = sum(l["lanes"] for l in hi_o)
        imbalance = lin - lout
        is_merge = len(hi_i) > len(hi_o)
        is_div = len(hi_o) > len(hi_i)
        # expected: merge loses 0-1 lanes, diverge gains 0-1; flag big breaks
        if abs(imbalance) >= 2:
            issues.append(dict(screen="S1_lane_balance", node=n,
                               severity=abs(imbalance) * 2,
                               detail=f"in {len(hi_i)}legs/{lin}ln vs "
                                      f"out {len(hi_o)}legs/{lout}ln "
                                      f"({'merge' if is_merge else 'diverge' if is_div else 'thru'})"))

    # ---- S2 lane discontinuity along same-class through nodes ----
    for n in set(list(inb) + list(outb)):
        li, lo = inb.get(n, []), outb.get(n, [])
        if len(li) == 1 and len(lo) == 1:
            i, o = li[0], lo[0]
            if i["ft"] == o["ft"] and i["ft"] in HIGH and i["a"] != o["b"]:
                d = abs(i["lanes"] - o["lanes"])
                if d >= 2:
                    issues.append(dict(screen="S2_lane_jump", node=n, severity=d * 1.5,
                                       detail=f"{i['ft']} {i['lanes']}ln -> {o['lanes']}ln"))

    # ---- S3 speed discontinuity ----
    for n in set(list(inb) + list(outb)):
        li, lo = inb.get(n, []), outb.get(n, [])
        if len(li) == 1 and len(lo) == 1:
            i, o = li[0], lo[0]
            if i["ft"] == o["ft"] and i["ft"] != "CC" and abs(i["ffs"] - o["ffs"]) > 15:
                issues.append(dict(screen="S3_speed_jump", node=n, severity=1.0,
                                   detail=f"{i['ft']} {i['ffs']:.0f} -> {o['ffs']:.0f} mph"))

    # ---- S4 centroid connector into freeway/ramp ----
    for l in links:
        if l["ft"] == "CC":
            for other in outb.get(l["b"], []) + inb.get(l["a"], []):
                if other["ft"] in ("Freeway", "MLHighway"):
                    issues.append(dict(screen="S4_cc_to_freeway", node=l["b"],
                                       severity=3.0,
                                       detail=f"CC {l['id']} touches {other['ft']} {other['id']}"))

    # ---- S5 capacity-per-lane outliers ----
    grp = defaultdict(list)
    for l in links:
        if l["ft"] != "CC" and l["lanes"] > 0:
            grp[(l["ft"], l["at"])].append(l["cap"] / l["lanes"])
    med = {k: sorted(v)[len(v) // 2] for k, v in grp.items()}
    for l in links:
        if l["ft"] != "CC" and l["lanes"] > 0:
            m = med[(l["ft"], l["at"])]
            r = (l["cap"] / l["lanes"]) / m if m else 1
            if r < 0.5 or r > 2.0:
                issues.append(dict(screen="S5_cap_outlier", node=l["a"], severity=1.0,
                                   detail=f"link {l['id']} {l['ft']}/{l['at']} "
                                          f"cap/ln ratio {r:.2f} vs group median"))

    # dedupe by (screen, node), rank
    seen = set()
    ranked = []
    for it in sorted(issues, key=lambda x: -x["severity"]):
        key = (it["screen"], it["node"])
        if key not in seen:
            seen.add(key)
            ranked.append(it)
    with open(os.path.join(OUT, "issues.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["screen", "node", "severity", "detail"])
        w.writeheader()
        w.writerows(ranked)
    by_screen = defaultdict(int)
    for it in ranked:
        by_screen[it["screen"]] += 1
    print("issues by screen:", dict(by_screen))

    # ---- render top sites ----
    TOP = int(os.environ.get("NETQC_TOP", "50"))  # 0 = screens only
    for k, it in enumerate(ranked[:TOP]):
        n = it["node"]
        if n not in nodes:
            continue
        cx, cy, _ = nodes[n]
        R = 0.008  # ~0.8 km in degrees
        fig, ax = plt.subplots(figsize=(9, 9))
        for l in links:
            if l["pts"] is None:
                continue
            xa, ya, _ = nodes[l["a"]]
            xb, yb, _ = nodes[l["b"]]
            if not (min(xa, xb) < cx + R and max(xa, xb) > cx - R
                    and min(ya, yb) < cy + R and max(ya, yb) > cy - R):
                continue
            xs = [p[0] for p in l["pts"]]
            ys = [p[1] for p in l["pts"]]
            col = FT_COLOR.get(l["ft"], "#333")
            ax.plot(xs, ys, color=col, linewidth=0.8 + 1.1 * l["lanes"],
                    alpha=0.85, zorder=2 if l["ft"] in HIGH else 1)
            # direction arrow + lane label at midpoint for high-class links
            if l["ft"] in HIGH or l["ft"] in ("MajorArterial",):
                mi = len(l["pts"]) // 2
                p0, p1 = l["pts"][max(mi - 1, 0)], l["pts"][min(mi + 1, len(l["pts"]) - 1)]
                ax.annotate("", xy=p1, xytext=p0,
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2))
                ax.text(l["pts"][mi][0], l["pts"][mi][1],
                        f"{l['lanes']}L {l['ffs']:.0f}mph", fontsize=7,
                        color=col, weight="bold",
                        bbox=dict(fc="white", ec="none", alpha=0.6, pad=0.5))
        ax.plot([cx], [cy], "k*", markersize=18, zorder=5)
        ax.set_xlim(cx - R, cx + R)
        ax.set_ylim(cy - R, cy + R)
        ax.set_aspect(1 / math.cos(math.radians(cy)))
        ax.set_title(f"site {k:03d} | {it['screen']} @ node {n} | {it['detail']}",
                     fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        handles = [plt.Line2D([0], [0], color=c, lw=3, label=ft)
                   for ft, c in FT_COLOR.items() if ft != "CC"]
        ax.legend(handles=handles, fontsize=6, loc="lower right", ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"site_{k:03d}.png"), dpi=110)
        plt.close(fig)
    print(f"rendered {min(TOP, len(ranked))} sites -> {OUT}")

    with open(os.path.join(OUT, "NETQC_REPORT.md"), "w") as f:
        f.write("# AI network QC — screening pass\n\n")
        f.write(f"Total flagged (deduped): {len(ranked)}\n\n")
        f.write("| screen | count |\n|---|---|\n")
        for s, c in sorted(by_screen.items()):
            f.write(f"| {s} | {c} |\n")
        f.write("\nTop sites rendered as site_###.png for VLM/human verdict.\n")
        f.write("Verdict loop: each snippet -> PLAUSIBLE / SUSPICIOUS / ERROR "
                "+ reason -> netqc_verdicts.csv -> edit proposals.\n")


if __name__ == "__main__":
    main()
