"""Next-gen assignment map: deck.gl (WebGL) interactive HTML.

Upgrades over dtalite_qa.vizmap: GPU rendering of the full network, volume-
scaled widths, v/c color ramp, count-station overlay (assigned/observed ratio),
hover tooltips, layer toggles, dark basemap. Single self-contained HTML
(deck.gl + maplibre from CDN; data embedded as JSON).

Usage:
  python vizmap_deck.py                 -> uses scenario/ (single period)
  python vizmap_deck.py --daily         -> aggregates scenario_{AM,MD,PM,NT}/
Output: assignment_map_deck.html (in the chosen scenario dir or HERE for daily)
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PERIODS = ["AM", "MD", "PM", "NT"]


def parse_wkt(wkt):
    if not wkt or not wkt.startswith("LINESTRING"):
        return None
    body = wkt[wkt.find("(") + 1: wkt.rfind(")")]
    pts = []
    for pair in body.split(","):
        xy = pair.split()
        if len(xy) >= 2:
            pts.append([round(float(xy[0]), 5), round(float(xy[1]), 5)])
    return pts if len(pts) >= 2 else None


def load_perf(sdir):
    perf = {}
    fp = os.path.join(sdir, "link_performance.csv")
    if not os.path.exists(fp):
        return perf
    with open(fp) as f:
        for r in csv.DictReader(f):
            key = (int(r["from_node_id"]), int(r["to_node_id"]))
            perf[key] = (float(r["volume"] or 0), float(r.get("doc") or 0),
                         float(r.get("speed_mph") or 0))
    return perf


def main():
    daily = "--daily" in sys.argv
    if daily:
        perfs = [load_perf(os.path.join(HERE, f"scenario_{p}")) for p in PERIODS]
        title = "TRMG2 GMNS — daily assigned volume (AM+MD+PM+NT)"
        out_html = os.path.join(HERE, "assignment_map_deck.html")
    else:
        perfs = [load_perf(os.path.join(HERE, "scenario"))]
        title = "TRMG2 GMNS — AM assignment"
        out_html = os.path.join(HERE, "scenario", "assignment_map_deck.html")

    links, counts = [], []
    seen_count = set()
    with open(os.path.join(HERE, "scenario", "link.csv")) as f:
        for r in csv.DictReader(f):
            if r["link_type_name"] == "CC":
                continue   # connectors: visual noise
            path = parse_wkt(r.get("geometry", ""))
            if path is None:
                continue
            key = (int(r["from_node_id"]), int(r["to_node_id"]))
            vol = sum(p.get(key, (0, 0, 0))[0] for p in perfs)
            doc = max(p.get(key, (0, 0, 0))[1] for p in perfs)
            links.append(dict(
                p=path, v=round(vol), c=round(doc, 3),
                id=int(r["link_id"]), ft=r["link_type_name"], at=r["area_type"],
                ln=int(r["lanes"]), cap=round(float(r["capacity"]))))
            if r["day_count"]:
                base = int(r["link_id"]) // 10
                if base not in seen_count:
                    seen_count.add(base)
                    mid = path[len(path) // 2]
                    counts.append(dict(pos=mid, obs=round(float(r["day_count"])),
                                       est=0, id=base))
    # second pass: daily two-way assigned at counted links
    if counts:
        twoway = {}
        with open(os.path.join(HERE, "scenario", "link.csv")) as f:
            for r in csv.DictReader(f):
                if not r["day_count"]:
                    continue
                base = int(r["link_id"]) // 10
                key = (int(r["from_node_id"]), int(r["to_node_id"]))
                twoway[base] = twoway.get(base, 0) + sum(
                    p.get(key, (0, 0, 0))[0] for p in perfs)
        for c in counts:
            c["est"] = round(twoway.get(c["id"], 0))

    max_v = max((l["v"] for l in links), default=1) or 1
    stats = dict(links=len(links), max_vol=max_v,
                 vmt=round(sum(l["v"] for l in links) / 1000) * 1000,
                 count_stations=len(counts))

    html = HTML_TEMPLATE
    html = html.replace("__TITLE__", title)
    html = html.replace("__LINKS__", json.dumps(links, separators=(",", ":")))
    html = html.replace("__COUNTS__", json.dumps(counts, separators=(",", ":")))
    html = html.replace("__MAXV__", str(max_v))
    html = html.replace("__STATS__", json.dumps(stats))
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"deck.gl map: {len(links):,} links, {len(counts)} count stations -> {out_html}")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<script src="https://unpkg.com/deck.gl@9.0.36/dist.min.js"></script>
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet"/>
<style>
 html,body{margin:0;height:100%;font-family:system-ui,sans-serif;background:#111}
 #map{position:absolute;inset:0}
 #panel{position:absolute;top:10px;left:10px;z-index:10;background:rgba(20,20,25,.88);
   color:#eee;padding:12px 14px;border-radius:10px;font-size:13px;max-width:290px}
 #panel h1{font-size:15px;margin:0 0 6px}
 .lg{display:flex;align-items:center;gap:6px;margin:2px 0}
 .sw{width:26px;height:8px;border-radius:3px;display:inline-block}
 label{display:block;margin:4px 0;cursor:pointer}
 #tt{position:absolute;pointer-events:none;z-index:11;background:rgba(15,15,20,.92);
   color:#fff;padding:8px 10px;border-radius:8px;font-size:12px;display:none;max-width:260px}
 .muted{color:#9aa;font-size:11px}
</style></head><body>
<div id="map"></div>
<div id="panel">
 <h1>__TITLE__</h1>
 <div class="muted" id="stats"></div>
 <label><input type="checkbox" id="cbFlow" checked> Link flows (width = volume)</label>
 <label><input type="checkbox" id="cbCnt" checked> Count stations (est / obs)</label>
 <div style="margin-top:6px">v/c color:
   <span class="lg"><span class="sw" style="background:#2ecc71"></span>&lt; 0.5</span>
   <span class="lg"><span class="sw" style="background:#f1c40f"></span>0.5 – 0.85</span>
   <span class="lg"><span class="sw" style="background:#e67e22"></span>0.85 – 1.0</span>
   <span class="lg"><span class="sw" style="background:#e74c3c"></span>1.0 – 1.2</span>
   <span class="lg"><span class="sw" style="background:#9b59b6"></span>&gt; 1.2</span></div>
 <div style="margin-top:6px">count ratio (est/obs):
   <span class="lg"><span class="sw" style="background:#3498db"></span>&lt; 0.7 under</span>
   <span class="lg"><span class="sw" style="background:#ecf0f1"></span>0.7 – 1.3</span>
   <span class="lg"><span class="sw" style="background:#e74c3c"></span>&gt; 1.3 over</span></div>
</div>
<div id="tt"></div>
<script>
const LINKS=__LINKS__, COUNTS=__COUNTS__, MAXV=__MAXV__, STATS=__STATS__;
document.getElementById('stats').textContent =
  STATS.links.toLocaleString()+' links · max vol '+STATS.max_vol.toLocaleString()
  +' · '+STATS.count_stations+' count stations';
function vcColor(c){
  if(c>1.2) return [155,89,182,220];
  if(c>1.0) return [231,76,60,220];
  if(c>0.85) return [230,126,34,210];
  if(c>0.5) return [241,196,15,200];
  return [46,204,113,160];
}
function ratColor(r){
  if(r<0.7) return [52,152,219,230];
  if(r>1.3) return [231,76,60,230];
  return [236,240,241,210];
}
const tt=document.getElementById('tt');
function showTT(o,html){
  if(o && o.object){tt.style.display='block';tt.style.left=(o.x+14)+'px';
    tt.style.top=(o.y+14)+'px';tt.innerHTML=html;}
  else tt.style.display='none';
}
function layers(){
  const L=[];
  if(document.getElementById('cbFlow').checked)
    L.push(new deck.PathLayer({id:'flows',data:LINKS,getPath:d=>d.p,
      getWidth:d=>30+570*Math.sqrt(d.v/MAXV),widthUnits:'meters',widthMinPixels:.4,
      getColor:d=>vcColor(d.c),capRounded:true,jointRounded:true,pickable:true,
      onHover:o=>showTT(o,o.object?('<b>link '+o.object.id+'</b> · '+o.object.ft+' · '
        +o.object.at+'<br>vol <b>'+o.object.v.toLocaleString()+'</b> · v/c <b>'
        +o.object.c+'</b><br>'+o.object.ln+' lanes · cap '
        +o.object.cap.toLocaleString()):null)}));
  if(document.getElementById('cbCnt').checked && COUNTS.length)
    L.push(new deck.ScatterplotLayer({id:'cnt',data:COUNTS,getPosition:d=>d.pos,
      getRadius:d=>120+380*Math.sqrt(d.obs/120000),radiusUnits:'meters',
      radiusMinPixels:2,getFillColor:d=>ratColor(d.obs?d.est/d.obs:0),
      stroked:true,getLineColor:[0,0,0,180],lineWidthMinPixels:1,pickable:true,
      onHover:o=>showTT(o,o.object?('<b>count stn (link '+o.object.id+')</b><br>observed '
        +o.object.obs.toLocaleString()+' · assigned '+o.object.est.toLocaleString()
        +'<br>ratio <b>'+(o.object.obs?(o.object.est/o.object.obs).toFixed(2):'-')
        +'</b>'):null)}));
  return L;
}
const dk=new deck.DeckGL({container:'map',
  mapStyle:'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
  initialViewState:{longitude:-78.75,latitude:35.85,zoom:9.3,pitch:0},
  controller:true,layers:layers()});
for(const id of ['cbFlow','cbCnt'])
  document.getElementById(id).onchange=()=>dk.setProps({layers:layers()});
</script></body></html>
"""

if __name__ == "__main__":
    main()
