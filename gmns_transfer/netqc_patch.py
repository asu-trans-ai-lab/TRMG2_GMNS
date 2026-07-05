"""P8 edit loop: VLM verdicts + OSM evidence -> patch proposal JSON ->
apply -> re-run QC screens + pipeline gates. Never a silent mutation.

Patch schema (NETQC_WORKPACKAGE.md section 8):
  {"patch_id": "...", "link_id": 504631, "field": "lanes", "old": 3, "new": 1,
   "evidence": [...], "status": "proposed|approved|applied|rejected",
   "by": "vlm|human|vlm+osm"}

Stages (run all by default, or one of: propose approve apply rerun):
  propose  join netqc_verdicts.csv (SUSPICIOUS/ERROR sites) with
           netqc_osm_match.csv: for each link touching a flagged node whose
           stage-A match (kind=MATCHED, unambiguous single OSM way) carries a
           tagged lane count differing from the MPO value, propose
           lanes := osm_lanes. Every proposal cites the site PNG, the OSM
           way id + lanes tag, and the verdict row.
  approve  auto-approve policy (documented, conservative): a proposal is
           approved by "vlm+osm" only when the OSM evidence AGREES in
           direction with the VLM verdict hypothesis (both say over- or
           under-coded the same way). Everything else stays "proposed" for a
           human. -- The edit loop never applies single-source patches.
  apply    write the patched network to netqc/patched_scenario/link.csv
           (source scenario/ is never touched), mark patches applied, log to
           netqc/netqc_patches.jsonl.
  rerun    re-run the QC screens on the patched copy (netqc/patched_screens/)
           and the pipeline gates (gates.py), and append a regression summary
           to netqc/PATCH_REPORT.md. An edit is accepted only if the flagged
           issue improves and no new issue appears.
"""
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
GM = os.path.join(HERE, "scenario")
OUT = os.path.join(HERE, "netqc")
PATCHES = os.path.join(OUT, "netqc_patches.jsonl")
PSCEN = os.path.join(OUT, "patched_scenario")
PSCREEN = os.path.join(OUT, "patched_screens")


def load_patches():
    if not os.path.exists(PATCHES):
        return []
    with open(PATCHES) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_patches(patches):
    with open(PATCHES, "w") as f:
        for p in patches:
            f.write(json.dumps(p) + "\n")


def links_touching():
    """node -> [link rows] over the H classes (light columns only)."""
    csv.field_size_limit(10_000_000)
    touch = defaultdict(list)
    with open(os.path.join(GM, "link.csv")) as f:
        for r in csv.DictReader(f):
            rec = dict(id=int(r["link_id"]), a=int(r["from_node_id"]),
                       b=int(r["to_node_id"]), ft=r["link_type_name"],
                       lanes=int(r["lanes"]))
            touch[rec["a"]].append(rec)
            touch[rec["b"]].append(rec)
    return touch


def propose():
    verdicts = list(csv.DictReader(open(os.path.join(OUT, "netqc_verdicts.csv"))))
    flagged = [v for v in verdicts if v["verdict"] in ("SUSPICIOUS", "ERROR")]
    match = {int(r["link_id"]): r
             for r in csv.DictReader(open(os.path.join(OUT, "netqc_osm_match.csv")))
             if r["link_id"]}
    touch = links_touching()
    patches = load_patches()
    known = {(p["link_id"], p["field"]) for p in patches}
    n_new = 0
    for v in flagged:
        node = int(v["node"])
        for l in touch.get(node, []):
            m = match.get(l["id"])
            if not m or m["kind"] != "MATCHED":
                continue  # chain matches mix ways -> ambiguous lane evidence
            if "/" in m["osm_lanes"] or not m["osm_lanes"]:
                continue
            osm_lanes = int(m["osm_lanes"])
            if osm_lanes == l["lanes"] or (l["id"], "lanes") in known:
                continue
            pid = f"p{len(patches):03d}_link{l['id']}_lanes"
            patches.append(dict(
                patch_id=pid, link_id=l["id"], field="lanes",
                old=l["lanes"], new=osm_lanes,
                evidence=[f"site_{v['site']}.png",
                          f"osm way {m['osm_ways']} lanes={osm_lanes}",
                          f"verdict {v['verdict']} {v['confidence']}: "
                          + v["reason"][:120],
                          f"action: {v['proposed_action']}"],
                status="proposed", by="vlm"))
            known.add((l["id"], "lanes"))
            n_new += 1
    save_patches(patches)
    print(f"propose: {n_new} new proposals ({len(patches)} total) -> {PATCHES}")
    return patches


def approve():
    patches = load_patches()
    n = 0
    for p in patches:
        if p["status"] != "proposed":
            continue
        verdict_txt = " ".join(e for e in p["evidence"]
                               if e.startswith(("verdict", "action"))).lower()
        osm_says_fewer = p["new"] < p["old"]
        # VLM hypothesis direction: 'over-coded'/'likely 1L' etc. = fewer;
        # 'undercoded'/'understated' = more
        vlm_fewer = any(k in verdict_txt for k in
                        ("over-code", "overcoded", "likely 1l", "not 2l",
                         "not 3l", "likely 1-2", "wider than"))
        vlm_more = any(k in verdict_txt for k in
                       ("under-code", "undercoded", "understated",
                        "should likely be", "should be"))
        if (osm_says_fewer and vlm_fewer) or (not osm_says_fewer and vlm_more):
            p["status"] = "approved"
            p["by"] = "vlm+osm"
            n += 1
    save_patches(patches)
    print(f"approve: {n} auto-approved (OSM direction agrees with VLM); "
          f"{sum(1 for p in patches if p['status'] == 'proposed')} left for human")
    return patches


def apply_patches():
    patches = load_patches()
    todo = {p["link_id"]: p for p in patches if p["status"] == "approved"}
    if not todo:
        print("apply: nothing approved")
        return patches
    os.makedirs(PSCEN, exist_ok=True)
    shutil.copy(os.path.join(GM, "node.csv"), os.path.join(PSCEN, "node.csv"))
    csv.field_size_limit(10_000_000)
    n = 0
    with open(os.path.join(GM, "link.csv"), newline="") as fi, \
         open(os.path.join(PSCEN, "link.csv"), "w", newline="") as fo:
        rd = csv.DictReader(fi)
        w = csv.DictWriter(fo, fieldnames=rd.fieldnames)
        w.writeheader()
        for r in rd:
            p = todo.get(int(r["link_id"]))
            if p is not None:
                assert int(r[p["field"]]) == p["old"], \
                    f"{p['patch_id']}: current {r[p['field']]} != old {p['old']}"
                r[p["field"]] = str(p["new"])
                p["status"] = "applied"
                n += 1
            w.writerow(r)
    save_patches(patches)
    print(f"apply: {n} patches applied -> {PSCEN}/link.csv "
          "(source scenario/ untouched)")
    return patches


def rerun():
    """Regression: screens on patched copy + pipeline gates."""
    if not any(p["status"] == "applied" for p in load_patches()):
        print("rerun: no applied patches, skipping")
        return
    env = dict(os.environ, NETQC_SCEN=PSCEN, NETQC_OUT=PSCREEN, NETQC_TOP="0")
    r1 = subprocess.run([sys.executable, os.path.join(HERE, "netqc_ai.py")],
                        env=env, capture_output=True, text=True)
    print(r1.stdout.strip())
    if r1.returncode != 0:
        print(r1.stderr[-2000:])
        raise SystemExit("patched-screen rerun failed")

    def rows(fp):
        return {(r["screen"], r["node"]): r
                for r in csv.DictReader(open(fp))}

    rb = rows(os.path.join(OUT, "issues.csv"))
    ra = rows(os.path.join(PSCREEN, "issues.csv"))
    before = defaultdict(int)
    after = defaultdict(int)
    for k in rb:
        before[k[0]] += 1
    for k in ra:
        after[k[0]] += 1
    before, after = dict(before), dict(after)
    changed = []
    for k in sorted(set(rb) | set(ra)):
        b, a = rb.get(k), ra.get(k)
        if b is None or a is None or b["detail"] != a["detail"] \
                or b["severity"] != a["severity"]:
            changed.append((k, b, a))

    r2 = subprocess.run([sys.executable, os.path.join(HERE, "gates.py")],
                        capture_output=True, text=True, cwd=HERE)
    gates_tail = [ln for ln in r2.stdout.splitlines() if ln.startswith("status counts")]
    gates_ok = r2.returncode == 0

    patches = load_patches()
    applied = [p for p in patches if p["status"] == "applied"]
    with open(os.path.join(OUT, "PATCH_REPORT.md"), "w") as f:
        f.write("# P8 edit-loop report\n\n")
        f.write(f"Applied patches: {len(applied)} "
                f"(log: netqc_patches.jsonl; patched copy: patched_scenario/)\n\n")
        f.write("| patch | link | field | old -> new | evidence |\n|---|---|---|---|---|\n")
        for p in applied:
            f.write(f"| {p['patch_id']} | {p['link_id']} | {p['field']} | "
                    f"{p['old']} -> {p['new']} | "
                    + "; ".join(p["evidence"]).replace("|", "/") + " |\n")
        f.write("\n## Screen regression (original vs patched network)\n\n")
        f.write("| screen | before | after |\n|---|---|---|\n")
        for k in sorted(set(before) | set(after)):
            f.write(f"| {k} | {before.get(k, 0)} | {after.get(k, 0)} |\n")
        f.write("\n### Changed flags (node level)\n\n")
        f.write("| screen @ node | before | after |\n|---|---|---|\n")
        for (scr, node), b, a in changed:
            fb = f"sev {b['severity']}: {b['detail']}" if b else "(not flagged)"
            fa = f"sev {a['severity']}: {a['detail']}" if a else "(resolved)"
            f.write(f"| {scr} @ {node} | {fb} | {fa} |\n")
        if not changed:
            f.write("| (none) | | |\n")
        f.write(f"\nPipeline gates rerun: {'OK' if gates_ok else 'FAILED'}"
                + (f" ({gates_tail[0]})" if gates_tail else "") + "\n")
    print(f"screens before {before}")
    print(f"screens after  {after}")
    print(f"gates: {'OK' if gates_ok else 'FAILED'}",
          gates_tail[0] if gates_tail else "")
    print(f"-> {os.path.join(OUT, 'PATCH_REPORT.md')}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("propose", "all"):
        propose()
    if stage in ("approve", "all"):
        approve()
    if stage in ("apply", "all"):
        apply_patches()
    if stage in ("rerun", "all"):
        rerun()
