"""NetQC GIS export — turn every screen/verdict/cross-source finding into
QGIS-loadable layers so a human can verify each flag against the map, with
the *evidence source and reasoning carried as attributes* on every feature.

Layers written per network (EPSG:4326):
  nodes_flagged   Point   one feature per flagged node; fields: screens,
                          max_severity, n_issues, details, sources, and (when
                          available) VLM verdict / reason / action / confidence
  links_flagged   Line    one feature per flagged link; fields as above plus
                          the link's own attributes (ft, lanes, speed, cap)
  osm_issues      Line    (if a cross-source run exists) typed MPO-vs-OSM
                          discrepancies with both values side by side
  patches         Line    (if an edit loop ran) old -> new with evidence
  issues_table    -       attribute-only table of ALL issue rows incl.
                          systemic findings (GPKG only)

Formats: one GeoPackage (primary, holds everything), plus per-layer ESRI
Shapefile and GML in sibling folders (field names may truncate in .shp —
the GPKG is authoritative).

Usage: python netqc_gis.py {trmg2|mag2024|aztdm}
"""
import csv
import os
import sys
import warnings
from collections import defaultdict

import geopandas as gpd
import pandas as pd
from shapely import from_wkt
from shapely.geometry import LineString, Point

from netqc_generic import CONFIGS, load

HERE = os.path.dirname(os.path.abspath(__file__))


def agg_issues(rows, key):
    out = defaultdict(lambda: dict(screens=set(), sev=0.0, n=0,
                                   details=[], sources=set()))
    for r in rows:
        k = r.get(key, "")
        if not k:
            continue
        d = out[k]
        d["screens"].add(r["screen"])
        d["sev"] = max(d["sev"], float(r["severity"]))
        d["n"] += 1
        if len(d["details"]) < 6:
            d["details"].append(f"[{r['screen']}] {r['detail']}")
        d["sources"].add(r["source"])
    return out


def main(net):
    cfg = CONFIGS[net]
    outdir = cfg["out"]
    gisdir = os.path.join(outdir, "gis")
    os.makedirs(gisdir, exist_ok=True)
    links, nodes = load(cfg)
    lk = {str(l["id"]): l for l in links}

    issues = list(csv.DictReader(open(os.path.join(outdir, "issues.csv"),
                                      encoding="utf-8")))
    # optional evidence joins (TRMG2 v2 outputs live one level up in netqc/)
    verd = {}
    vfp = os.path.join(HERE, "netqc", "netqc_verdicts.csv")
    if net == "trmg2" and os.path.exists(vfp):
        for r in csv.DictReader(open(vfp, encoding="utf-8")):
            verd[r["node"]] = r

    layers = {}

    # ---------------- nodes_flagged
    nrows = []
    for nid, d in agg_issues(issues, "node").items():
        nd = nodes.get(int(float(nid))) if nid.replace(".", "").isdigit() else None
        if nd is None:
            continue
        v = verd.get(nid, {})
        nrows.append(dict(
            node_id=nid, screens=";".join(sorted(d["screens"])),
            max_severity=d["sev"], n_issues=d["n"],
            details=" | ".join(d["details"])[:1000],
            sources=" | ".join(sorted(d["sources"]))[:500],
            vlm_verdict=v.get("verdict", ""),
            vlm_reason=v.get("reason", "")[:500],
            vlm_action=v.get("proposed_action", "")[:250],
            vlm_conf=v.get("confidence", ""),
            geometry=Point(nd["x"], nd["y"])))
    if nrows:
        layers["nodes_flagged"] = gpd.GeoDataFrame(nrows, crs="EPSG:4326")

    # ---------------- links_flagged
    lrows = []
    for lid, d in agg_issues(issues, "link_id").items():
        l = lk.get(lid)
        if l is None:
            continue
        geom = None
        if l.get("wkt") and "LINESTRING" in l["wkt"]:
            try:
                geom = from_wkt(l["wkt"])
            except Exception:
                geom = None
        if geom is None:
            na, nb = nodes.get(l["a"]), nodes.get(l["b"])
            if not na or not nb:
                continue
            geom = LineString([(na["x"], na["y"]), (nb["x"], nb["y"])])
        lrows.append(dict(
            link_id=lid, ft=l["ft"], lanes=l["lanes"], free_speed=l["ffs"],
            capacity=l["cap"], length_mi=l["len_mi"],
            screens=";".join(sorted(d["screens"])),
            max_severity=d["sev"], n_issues=d["n"],
            details=" | ".join(d["details"])[:1000],
            sources=" | ".join(sorted(d["sources"]))[:500],
            geometry=geom))
    if lrows:
        layers["links_flagged"] = gpd.GeoDataFrame(lrows, crs="EPSG:4326")

    # ---------------- cross-source + patches layers (if runs exist)
    def link_geom(lid):
        l = lk.get(str(lid))
        if not l:
            return None
        if l.get("wkt") and "LINESTRING" in l["wkt"]:
            try:
                return from_wkt(l["wkt"])
            except Exception:
                pass
        na, nb = nodes.get(l["a"]), nodes.get(l["b"])
        return (LineString([(na["x"], na["y"]), (nb["x"], nb["y"])])
                if na and nb else None)

    for tag, fp in [("osm_issues", os.path.join(outdir, "netqc_osm_issues.csv")),
                    ("osm_issues", os.path.join(HERE, "netqc", "netqc_osm_issues.csv")
                     if net == "trmg2" else "")]:
        if fp and os.path.exists(fp) and tag not in layers:
            rows = []
            for r in csv.DictReader(open(fp, encoding="utf-8")):
                g = link_geom(r["link_id"]) if r["link_id"] else None
                if g is None:
                    continue
                l = lk[str(r["link_id"])]
                rows.append(dict(
                    link_id=r["link_id"], issue=r["issue"], ft=r["ft"],
                    mpo_value=r["mpo"], osm_value=r["osm"],
                    adot_value=r.get("adot", ""),
                    severity=float(r.get("severity", 0) or 0),
                    reasoning=r["detail"][:500],
                    lanes=l["lanes"], free_speed=l["ffs"], geometry=g))
            if rows:
                layers[tag] = gpd.GeoDataFrame(rows, crs="EPSG:4326")

    pfp = os.path.join(HERE, "netqc", "netqc_patches.jsonl") if net == "trmg2" \
        else os.path.join(outdir, "netqc_patches.jsonl")
    if os.path.exists(pfp):
        import json
        rows = []
        for line in open(pfp, encoding="utf-8"):
            p = json.loads(line)
            g = link_geom(p["link_id"])
            if g is None:
                continue
            rows.append(dict(patch_id=p["patch_id"], link_id=str(p["link_id"]),
                             field=p["field"], old=str(p["old"]),
                             new=str(p["new"]), status=p["status"],
                             by=p["by"],
                             evidence=" | ".join(p["evidence"])[:800],
                             geometry=g))
        if rows:
            layers["patches"] = gpd.GeoDataFrame(rows, crs="EPSG:4326")

    # ---------------- write GPKG + SHP + GML
    gpkg = os.path.join(gisdir, f"netqc_{net}.gpkg")
    if os.path.exists(gpkg):
        os.remove(gpkg)
    for name, gdf in layers.items():
        gdf.to_file(gpkg, layer=name, driver="GPKG")
    # attribute-only issues table into the GPKG
    df = pd.DataFrame(issues)
    gpd.GeoDataFrame(df, geometry=[None] * len(df), crs="EPSG:4326") \
        .to_file(gpkg, layer="issues_table", driver="GPKG")

    shpdir = os.path.join(gisdir, "shp")
    gmldir = os.path.join(gisdir, "gml")
    os.makedirs(shpdir, exist_ok=True)
    os.makedirs(gmldir, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # shp field-name truncation warnings
        for name, gdf in layers.items():
            try:
                gdf.to_file(os.path.join(shpdir, f"{net}_{name}.shp"),
                            driver="ESRI Shapefile")
            except Exception as e:
                print(f"  shp {name} skipped: {e}")
            try:
                gdf.to_file(os.path.join(gmldir, f"{net}_{name}.gml"),
                            driver="GML")
            except Exception as e:
                print(f"  gml {name} skipped: {e}")
    print(f"{net}: wrote {gpkg}")
    for name, gdf in layers.items():
        print(f"  layer {name}: {len(gdf):,} features")
    print(f"  + issues_table ({len(df):,} rows), shp/ and gml/ copies")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "trmg2")
