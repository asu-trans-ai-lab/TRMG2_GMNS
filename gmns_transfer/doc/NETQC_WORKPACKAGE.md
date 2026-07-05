# NetQC Work Package — AI-Assisted Network QC & Editing
## Problem statements, solutions, and reproduction guide (standalone)

Status: v1 prototype working on TRMG2 (this repo). Designed to branch into its
own project and port to any GMNS network (MAG is the motivating case: lanes=0,
capacity mismatches, connector placement found during that conversion).

**Portability:** the core module needs ONLY `numpy` + `matplotlib` + two CSVs
(`node.csv`, `link.csv` with WKT geometry). No kernel, no geopandas, no
Windows dependency. See §9 for the MacBook recipe and §10 for the exact
reference numbers your run must reproduce.

---

## 1. Problem P1 — Lane-balance violations at controlled-access nodes

**Statement.** For every node n whose incident links are all in
H = {Freeway, MLHighway, Ramp} (ignoring centroid connectors), lane
conservation should hold approximately: at a merge (2 in, 1 out)
`L_in − L_out ∈ {0, 1}`; at a diverge (1 in, 2 out) `L_out − L_in ∈ {0, 1}`;
at a through node `L_in = L_out`. Violations indicate coded-lane errors,
missing auxiliary lanes, or topology defects.

**Solution.** (a) Build in/out link lists per node. (b) **Exclude the directed
twin** (FIXED in v2): if an inbound (a,n) and an outbound (n,a) share the same
base id (`link_id // 10`), they are the two directions of one two-way
carriageway segment — remove **both members of each twin pair** from the
accounting. (Implementation note: the one-sided rule originally proposed —
drop only the outbound reverse — is itself defective: at a plain two-way thru
node it keeps both inbound directions but only one outbound, manufacturing a
fake 2-in/1-out merge. Pair removal excludes the two-way segment entirely; a
node whose high-class legs are all twinned is skipped, which is correct
because mixed-direction lane sums are meaningless there and S2 still covers
degree-2 lane jumps.) (c) Flag |imbalance| ≥ 2, severity = 2·|imbalance|
(the v1 extra clause flagging thru nodes at |imbalance| = 1 is removed per
this rule — those are S2 territory).

**Output schema.** `issues.csv`: screen, node, severity, detail
(`in {legs}/{lanes}ln vs out {legs}/{lanes}ln (merge|diverge|thru)`).

**Acceptance (v2 measured).** Twin exclusion is active at 275 of 2,213
candidate nodes (233 fully-twinned two-way freeway nodes now skipped); S1
drops 119 → 93 (25 thru-|Δ|=1 flags removed by rule (c), 2 verdict changes
from twin exclusion, 1 overlap). site_000-class findings (3-lane ramp off
2-lane mainline) remain. **Caveat discovered:** site_004 (node 12493) does
NOT vanish — geometry inspection shows its second outbound link (349041, 2L
57 mph) is a parallel carriageway/roadway digitized with *distinct* node
chains, not a shared-node twin, so no (from,to)-reversal exists to exclude.
The v1 VLM verdict correctly identified a screen artifact but guessed the
wrong mechanism. That class needs geometric anti-parallel detection or
cross-source (P7 OSM) evidence, not topology-only twin logic.

## 2. Problem P2 — Lane discontinuity along homogeneous corridors

**Statement.** At a degree-2 node (1 in, 1 out, same facility type, not a
U-turn pair), a lane change |ΔL| ≥ 2 with no intersecting street is almost
always a coding error (legitimate adds/drops are ±1 at ramps/intersections).

**Solution.** Filter degree-2 same-facility nodes; flag |ΔL| ≥ 2,
severity 1.5·|ΔL|. Extension: chain consecutive links by road name/facility
into corridors and flag zig-zag lane sequences (3-2-3, 4-2-4).

## 3. Problem P3 — Speed discontinuity

**Statement.** Same setup as P2; free-flow speed jump > 15 mph within a
facility type signals a wrong PostedSpeed or a facility-type miscode.

**Solution.** As implemented; extension: compare against the facility×area
type FFS from the agency lookup (ff_speed_alpha_beta.csv-style) and flag
3-sigma outliers instead of a fixed threshold.

## 4. Problem P4 — Centroid-connector loading points

**Statement.** A centroid connector whose network end touches a Freeway or
MLHighway injects zone demand directly onto controlled-access lanes —
distorting ramp volumes and corrupting count comparisons at that location.
(TRMG2 run: 161 raw flags; many are connectors landing at interchange
complexes — each needs a verdict, not automatic deletion.)

**Solution.** (a) Flag CC links adjacent to H-class links. (b) For each,
compute the nearest non-H node within radius r (0.5–1 mi) as the candidate
re-anchor. (c) Emit an edit proposal (§8) moving the connector end; leave
approval to the human/VLM loop. Severity 3.0.

## 5. Problem P5 — Capacity-per-lane outliers

**Statement.** Within a (facility type × area type) group, capacity/lane
should be near-constant (it derives from a lookup). Ratios < 0.5× or > 2×
the group median indicate unit errors (daily vs hourly — the GSATS
AB_CAP finding), lane errors, or wrong facility coding.

**Solution.** Group medians + ratio flags, as implemented. Extension for
agency networks: run BEFORE any capacity synthesis, on the raw columns, to
catch source-data unit mixups.

## 6. Problem P6 — VLM visual verification (the AI layer)

**Statement.** Screens produce candidates; deciding real-vs-artifact needs
context a human/VLM reads from a map: interchange geometry, parallel
carriageways, one-way pairs, land use. Automate the review at scale.

**Solution.** (a) Render each site: links within ~0.8 km, width ∝ lanes,
facility colors, direction arrows, `{lanes}L {speed}mph` labels, flagged node
starred (implemented: `netqc_ai.py` rendering block). (b) Verdict protocol —
ask the VLM per image:

```
You are auditing a travel-model network. The starred node was flagged:
<screen>: <detail>.  From the rendered geometry and labels, answer:
VERDICT: PLAUSIBLE | SUSPICIOUS | ERROR | FALSE_POSITIVE
REASON: one sentence, cite the visual evidence
ACTION: none | verify-against-imagery | proposed edit (link, field, old->new)
CONFIDENCE: low | medium | high
```

(c) Append to `netqc_verdicts.csv`. v1 demonstration: 2 sites reviewed —
one SUSPICIOUS with a concrete edit hypothesis, one FALSE_POSITIVE that
exposed the P1 twin defect (the VLM as QC-of-the-QC).
(d) v2: underlay aerial imagery (Esri World Imagery / USGS tiles) so lane
counts are verified against ground truth; add a second-opinion pass with a
different rendering scale before any ERROR verdict is accepted.

## 7. Problem P7 — Cross-source verification vs OSM (osm2gmns)

**Statement.** Independent evidence: compare the MPO network with OSM for the
same region — missing links, one-way conflicts, lane and speed mismatches,
and (the map-matching problem) which MPO link corresponds to which OSM way.

**Solution.**
1. Extract: `osm2gmns` on the region bbox (or county PBF), `link_types =
   motorway|trunk|primary|secondary`, build GMNS.
2. Map-match (geometry only, CRS-projected to meters):
   candidate pairs = OSM links whose bounding boxes intersect the MPO link's
   25 m buffer; score = mean of (a) direction cosine between chord vectors,
   (b) 1 − clipped(Hausdorff distance / 50 m), (c) length ratio penalty;
   accept best score ≥ 0.6; else class `MPO_UNMATCHED` (or reverse:
   `OSM_UNMATCHED` for coverage gaps).
3. Attribute diff per matched pair → typed issues:
   `LANE_MISMATCH` (|ΔL| ≥ 1 on H-class, ≥ 2 elsewhere), `ONEWAY_CONFLICT`,
   `SPEED_MISMATCH` (> 10 mph vs maxspeed where tagged), `NAME_MISMATCH`
   (informational).
4. Feed LANE/ONEWAY conflicts into the P6 verdict loop with both sources
   shown side-by-side in the rendering.

**Acceptance.** On a 5×5 mi TRMG2 test window (RTP area suggested), ≥ 80% of
H-class MPO links matched; the discrepancy table renders and at least one
known site (e.g., site_000's 3L ramp) receives OSM lane evidence.

**Results (v2, implemented in `netqc_osm.py`, window centered 35.905,
−78.900 over I-40/RTP — covers flagged nodes 10980/10983/5336 = sites
002/003/011).** OSM 763 ways → 940 directional segments; 864 MPO links in
window. **H-class matched 140/144 = 97.2%** (acceptance ≥ 80% met).
Issues: LANE_MISMATCH 78, ONEWAY_CONFLICT 6, SPEED_MISMATCH 20 (ffs vs
posted, informational), DUAL_CARRIAGEWAY_REP 222, OSM_CLASS_GAP 242,
MPO_UNMATCHED 8, OSM_UNMATCHED 4. Known-site evidence: at node 10980 OSM
tags link 401301's ramp as **lanes=1 vs MPO 2L** (and 401341 at 10983) —
directly confirming the site_002/003 VLM verdicts; these became the first
P8 patches. Three refinements the naive spec needed (all documented in the
script header):
1. **Chain aggregation fallback**: MPO links are longer than OSM
   segmentation, so pairwise Hausdorff+length-ratio alone leaves long links
   unmatched; unmatched links snap 50 m samples to direction-consistent OSM
   segments (≥ 80% coverage = MATCHED_CHAIN).
2. **Dual-carriageway reclassification**: 222 of the raw 228 "oneway
   conflicts" were MPO two-way centerlines matched to one of OSM's two
   oneway carriageways — detected via anti-parallel OSM segments nearby and
   downgraded to informational.
3. **Class-affinity tie-breaking**: geometry-only matching snaps ramps
   braided within 25 m of the mainline onto the motorway way; near-equal
   candidates prefer the class-compatible OSM way (Ramp→*_link etc.).
Also: unmatched Arterial/MajorCollector links are OSM_CLASS_GAP (their OSM
counterpart `tertiary` is outside the §7 extract), not coverage gaps.
Outputs: `netqc/OSM_XCHECK.md`, `netqc_osm_match.csv`,
`netqc_osm_issues.csv`, `site_osm_###.png`, `netqc/osm_rtp/` (GMNS build).

## 8. Problem P8 — The edit loop (closing the circle)

**Statement.** Verdicts must become auditable network edits, never silent
mutations.

**Solution.** Patch proposal JSON:
```json
{"patch_id": "…", "link_id": 504631, "field": "lanes", "old": 3, "new": 1,
 "evidence": ["site_000.png", "osm way 123456 lanes=1", "verdict ERROR high"],
 "status": "proposed|approved|applied|rejected", "by": "vlm|human"}
```
Apply step rewrites link.csv (never the source data), logs to
`netqc_patches.jsonl`, and **re-runs the QC screens + the pipeline gates** so
every edit is regression-checked. This is the gui-ai integration point.

**Implemented (v2, `netqc_patch.py`: propose → approve → apply → rerun).**
Propose joins SUSPICIOUS/ERROR verdicts with unambiguous stage-A OSM lane
evidence at the flagged node. Approve applies a conservative dual-source
policy: auto-approve (`by: vlm+osm`) only when the OSM tag agrees in
*direction* with the VLM hypothesis; single-source proposals wait for a
human. Apply writes `netqc/patched_scenario/link.csv` (source untouched)
with an old-value assertion per patch. Rerun re-screens the patched copy
and runs `gates.py`. First loop closure (2026-07-04): 2 patches
(links 401301, 401341: ramp lanes 2→1 at nodes 10980/10983), S1 severity
6→4 at both nodes (out 4ln→3ln; still ≥2 because the second branch is
genuinely 2L per OSM), no other flag changed, gates unchanged
(pre-existing G3 FAIL is the documented generation-rate deviation).
Report: `netqc/PATCH_REPORT.md`.

---

## 9. MacBook deployment (reproduce in ~10 minutes)

```bash
# 1. environment (any Python 3.10+)
python3 -m venv qc && source qc/bin/activate
pip install numpy matplotlib          # core; add: osm2gmns geopandas for P7

# 2. copy the bundle (see §11) or these files preserving layout:
#    netqc_ai.py
#    scenario/node.csv        (~1 MB)
#    scenario/link.csv        (~45 MB — includes WKT geometry + day_count)

# 3. run
python netqc_ai.py
#    -> netqc/issues.csv, netqc/site_000..011.png, netqc/NETQC_REPORT.md

# 4. verify against §10, then review site PNGs with your VLM of choice
#    using the §6 verdict protocol.
```
No absolute paths in the module; it resolves `scenario/` relative to itself.
macOS note: matplotlib "Agg" backend is set in-module — no display needed.

## 10. Reference results (your run must reproduce these)

**v2 (current, after P1 twin fix — 2026-07-04):**

| item | expected |
|---|---|
| issues by screen | S1_lane_balance 93, S2_lane_jump 4, S3_speed_jump 3, S4_cc_to_freeway 161 (total deduped 261) |
| S1 delta vs v1 | 119 → 93: −25 thru-nodes with \|Δ\|=1 (rule §1(c)), −2 twin-exclusion verdict changes (1 overlaps both causes); 0 added |
| twin-exclusion coverage | active at 275 / 2,213 S1-candidate nodes; 233 fully-twinned (undivided two-way freeway) nodes skipped |
| site_000 | S1 @ node 8110: "in 1legs/2ln vs out 2legs/5ln (diverge)" — 3L/42 mph ramp off 2L/62 mph mainline → SUSPICIOUS (unchanged) |
| site_004 | S1 @ node 12493: SURVIVES the twin fix — second outbound (link 349041) is a parallel roadway on distinct nodes, not a shared-node twin; needs P7/imagery evidence (see §1 caveat) |
| rendering | top 50 sites rendered (site_000 … site_049) for the §6 batch verdict pass |

**v1 (historical, pre-fix — kept for the bundle's reference outputs):**

| item | expected |
|---|---|
| issues by screen | S1_lane_balance 119, S2_lane_jump 4, S3_speed_jump 3, S4_cc_to_freeway 161 (total deduped 287) |
| site_004 | S1 @ node 12493: flagged; v1 VLM called FALSE_POSITIVE and hypothesized the shared-node twin mechanism (mechanism later disproved — see §1) |

(Renderer output is deterministic; PNG bytes may differ across matplotlib
versions but content must match.)

## 11. Bundle

`netqc_bundle.zip` (same folder): the **v1** snapshot — `netqc_ai.py`,
`scenario/node.csv`, `scenario/link.csv`, this document, and the v1
`netqc/` reference outputs (issues.csv + 12 PNGs + verdicts). Kept as the
historical reference for the v1 row of §10.

`netqc_bundle_v2.zip` (same folder): the current branch state —
`netqc_ai.py` (P1 fix), `netqc_osm.py` (P7), `netqc_patch.py` (P8), this
document, `scenario/{node,link}.csv`, and the full v2 `netqc/` outputs
(issues, 50 site PNGs, verdicts, OSM cross-check, patch log + patched
copy). Core still needs only numpy + matplotlib; P7 additionally used
osm2gmns 1.0.1 for the GMNS build (matching itself reads the raw .osm,
which is included, so P7 re-runs offline too). `netqc_patch.py`'s gates
hook calls the repo's `gates.py`; off-repo it reports gates FAILED and
continues.

## 12. Branch roadmap (status 2026-07-04)

1. ~~P1 twin-exclusion fix + re-reference~~ **DONE** (§1, §10 v2; pair
   removal, S1 119→93; site_004 class re-diagnosed as distinct-node
   parallel carriageway)
2. ~~Batch VLM verdicts on all top-50 sites (§6 protocol)~~ **DONE**
   (`netqc/netqc_verdicts.csv`: 25 PLAUSIBLE / 23 SUSPICIOUS /
   2 FALSE_POSITIVE, all 50 = S1; the two FALSE_POSITIVEs are both the
   parallel-carriageway artifact — next screen refinement target)
3. ~~P7 osm2gmns leg on an RTP-area window~~ **DONE** (`netqc_osm.py`, §7
   results; H-class match 97.2%; OSM lane evidence confirmed sites 002/003)
4. Aerial-imagery underlay for P6 (day) — OPEN
5. ~~P8 edit loop + patch schema + gates hook~~ **DONE** (`netqc_patch.py`,
   §8; first closure: 2 dual-evidence lane patches, regression-checked)
6. ~~Port to MAG network~~ **TEST RUN DONE** (2026-07-04, second pass): the
   screens were generalized into a config-driven battery and run on MAG 2024
   and AZTDM; predictions confirmed — S8c/S10 fire on capacity conventions
   (MAG 17,795 sentinel caps; AZTDM 60,448 × capacity=99999) and S4 on
   connector placement (MAG 711). See §13.
7. (new) Geometric anti-parallel / parallel-roadway detection in S1 to
   retire the site_004/site_046 artifact class topology-side — OPEN
8. (new) S4 verdict batch (161 CC-to-freeway flags, the largest screen) and
   P4 re-anchor proposals through the §8 patch loop — OPEN

## 13. Multi-network generalization + QGIS deliverables (2026-07-04)

Three sibling modules extend the work package beyond TRMG2:

- **`netqc_generic.py`** — config-driven screen battery (S1–S13 superset:
  adds dead ends, self-loops/duplicates/zero-length, coded-length vs
  geometry ratio, attribute validity + sentinel detection, unit heuristics,
  twin asymmetry, connector/zonal checks, isolated components; systemic
  aggregation rows when a screen fires network-wide). Configs: `trmg2`,
  `mag2024`, `aztdm` (facility classes inferred from medians where the
  agency dictionary is missing — marked DECLARE).
- **`netqc_gis.py`** — every finding as QGIS layers (GeoPackage primary +
  Shapefile + GML), with screens/severity/details/sources — and, where they
  exist, VLM verdict fields, MPO-vs-OSM-vs-ADOT values, and patch history —
  carried as feature attributes for human verification on a basemap.
- **`netqc_xsource.py`** — the P7 matcher generalized with a second
  evidence source: ADOT AllRoadsNetwork 2023 (authoritative route
  inventory; contributes coverage/route-designation since it has no lane
  attributes) alongside OSM (lanes/oneway/maxspeed), plus per-link
  suspicion ranking (`suspects.csv` + GIS layer). New matcher rule:
  class-consistency gating (CLASS_CONFLICT instead of false LANE_MISMATCH
  when the match snapped to a parallel roadway).

MAG/AZTDM results + top suspects: `New folder/netqc_mag/MAG_TESTRUN.md`
(headline: FT1 freeway links coded 1L/70mph on I-10/I-17 where OSM says 4L,
corroborated by S9 and ADOT route geometry).
