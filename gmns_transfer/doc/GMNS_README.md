# gmns/ — open GMNS reproduction of TRMG2, runnable in place

This folder sits inside the TRMG2 v2.0 model release and reproduces its roadway
model in open GMNS format + a differentiable computational graph. It is
**runnable here**: the built network, the 4-period demand, and the validated
matrix operators are included, and the scripts resolve their inputs from this
release (`../master`, `../docs`).

## Layout
```
TRMG2_v2.0/
  master/        <- TRMG2 network binaries + 133 parameter CSVs (the model)
  docs/data/     <- minimal survey targets + TAZ shapefile + validation counts
                    (mirrored so the scripts run unchanged: eda_scheme6.csv,
                     master_tazs.shp, intrazonals.csv, validation/, nhb_generation.csv)
  gmns/
    *.py                    the pipeline (decode -> build -> demand -> graph -> gates)
    scenario/               built GMNS base network (node.csv, link.csv)
    scenario_{AM,MD,PM,NT}/  runnable kernel inputs (node, link, demand_*, settings, mode_type)
    matrices/               validated Pi = A.Delta operators (.npz) + od_index_AM.csv
    topology.csv            decoded link from/to nodes (from net.net)
    node_crosswalk.csv      TRMG2 node id <-> gmns id
    review/                 reports (GATES_REPORT, count comparisons, ...)
    *.md, TCG_MATH.pdf      docs + the math derivation
```

## What runs immediately (no rebuild, no external data)
1. **Assignment** — the kernel on any period's built inputs:
   ```
   cd scenario_AM
   cp <path>/bin/DTALite.exe .        # or build it: bash <repo>/build.sh
   ./DTALite.exe                       # loads sov/hov2/hov3, writes link_performance.csv
   ```
   Verified: all 3 modes load (sov 380,722 / hov2 52,824 / hov3 25,251 veh),
   75,939 links loaded.
2. **Matrix operators** — after a route-output run (`route_output=1` in
   settings.csv, env `TAPLITE_ROUTE_VOL_MIN=0.04`):
   `python matrix_ops.py AM`  -> validates Pi @ od == kernel volumes (R²=1.0).
   (AM operators are pre-built in `matrices/`.)
3. **Computational graph** — `python tcg_consistency.py` (numpy==torch==onnx),
   `python tcg_tensor.py` (autodiff == analytic adjoints).
4. **Gates / validation** — `python gates.py` reads the copied review data +
   `../docs/data` references -> `review/GATES_REPORT.md`.
5. **Demand rebuild** — `python make_od_4period.py` reproduces the 4-period OD
   from TRMG2's own DC tables in `../master/resident/dc` + targets in
   `../docs/data` (writes scenario_*/demand_*.csv, then assigns).

## What needs the one-time geometry export (full network rebuild only)
`build_gmns.py` reconstructs the GMNS network from scratch and needs the
geometry export (`../2026-07-04 TRMG2*` folders: node/link coords + FROM/TO).
The already-built `scenario/` network is provided, so this is optional. If you
have the export, drop those two folders in `TRMG2_v2.0/` and run
`python build_gmns.py`. Otherwise decode topology from `../master/networks/net.net`
via `python decode_dln.py` (see DECODE_NOTES.md).

## Full guide
`STUDENT_GUIDE.md` (in this folder) is the complete original-files-to-build
runbook. `REPRODUCTION_MAP.md` maps each TRMG2 GISDK step to its script.
`MANIFEST`-style provenance is in the student guide section 7.

## Not included (regenerable / too large)
The giant per-run outputs are omitted: `route_assignment.csv` (~1.4 GB/period),
`od_performance.csv`, `link_performance.csv`, and the Chicago benchmark dirs.
They regenerate by running the kernel + `matrix_ops.py`.
