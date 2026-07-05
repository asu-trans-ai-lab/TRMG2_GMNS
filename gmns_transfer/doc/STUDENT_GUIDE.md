# Student reproduction guide — TRMG2 computational-graph pipeline
### For: Jinxi.  Goal: reproduce EVERYTHING, from the original files to the build.

This package reproduces a four-step travel demand model (the public Triangle
Regional Model, TRMG2) as a differentiable tensor computational graph, in which
traffic assignment is done by a C++ kernel you build yourself and enters the
math as fixed matrix operators. You will: build the kernel, run a real
assignment, build the demand model from TRMG2's own coefficients, extract the
operators, verify the graph across numpy/PyTorch/ONNX, calibrate under guard
rails, validate with gates, and draw the result.

There are TWO parts. Do Part A first (10 min, everything bundled). Part B is
the full TRMG2 reproduction from the original data.

---

## 0. What you will produce
- a built kernel binary (`bin/DTALite`)
- assigned link volumes + an interactive HTML map
- the matrix operators Pi = A·Delta (validated to R²≈1 vs the kernel)
- a calibrated demand model with a measured drift/fit table
- consistency proof: numpy == PyTorch == ONNX (~1e-7)
- (Part B) the full TRMG2 network, 4-period demand, and the gate report

---

## 1. Environment

### 1a. Python (all parts)
```
python3 -m venv ~/tcg_venv          # keep the venv OUT of any Dropbox folder
source ~/tcg_venv/bin/activate      # Windows: ~\tcg_venv\Scripts\activate
pip install -r requirements.txt     # Part A deps
pip install -r full_pipeline/requirements_full.txt   # Part B adds geopandas, etc.
```
Check: `python -c "import numpy, scipy, torch; print('ok')"`.

### 1b. C++ toolchain (to build the kernel)
- macOS: `xcode-select --install`; then `brew install cmake libomp`
  (libomp = multi-core; without it the build falls back to single-thread).
- Linux: `sudo apt install build-essential cmake` (gcc has OpenMP built in).
- Windows: MSYS2/MinGW-w64 `g++`, or use the prebuilt `bin/DTALite.exe`.

Verify: `clang++ --version` (mac) or `g++ --version` (linux).

---

## 2. PART A — quick reproduction (bundled, ~10 min)

Everything needed is in this package (real Sioux Falls + Chicago Sketch
networks, kernel source, the demand graph). No TRMG2 data required.

```
# 1. build the kernel
bash kernel/build_mac.sh            # or kernel/build_unix.sh on Linux

# 2. representation consistency (must PASS before trusting anything)
python code/consistency_check.py    # -> numpy==torch==onnx, VERDICT PASS

# 3. the REAL pipeline: build->assign->Pi->calibrate->map
python code/run_pipeline.py sioux_falls
python code/run_pipeline.py chicago_sketch
open results/assignment_map_sioux_falls.html     # Linux: xdg-open

# 4. GPU / scaling benchmark (eager + torch.compile; auto-detects MPS/CUDA)
python code/gpu_bench.py
```
Or just `bash run_all.sh` to do all of the above.

**Expected (Sioux Falls):** kernel < 1 s, `pi_R2` = 1.0, %RMSE ≈ 24 → 13 after
calibration, `drift_max` ≈ 0.40, an 8 KB offline map. See `EARLY_RESULTS.md`
for the full reference table (R1–R6). If your numbers match, the pipeline
reproduces.

---

## 3. PART B — full TRMG2 reproduction, from the original files

This rebuilds the actual 3,147-zone regional model. The code is in
`full_pipeline/`; the DATA you must acquire (it is large and public).

### 3.0 Get the original data
1. **TRMG2 public repo** (model source + parameters + validation data):
   `git clone https://github.com/Triangle-Modeling-and-Analytics/TRMG2`
   You need its `master/` (network binaries, 133 parameter CSVs) and
   `docs/data/` (survey targets, validation counts, TAZ shapefile).
2. **The network geometry export** (node/link coordinates + FROM_ID/TO_ID):
   from the TRM Service Bureau (ITRE) as shapefile/CSV, OR decode it yourself
   from `master/networks/net.net` (step 3.2).
Point the scripts at your TRMG2 checkout by editing the `REPO`/`ROOT` path
constants at the top of each `full_pipeline/*.py` (they currently expect the
repo layout used during development).

### 3.1 Read the fixed-format binaries
`full_pipeline/transcad_bin.py` reads the `.bin/.DCB` fixed-format tables
(links, nodes, SE data, PUMS seeds). One gotcha it already handles:  9
writes the record length as `609,codepage=1252`.
```
python -m full_pipeline.transcad_bin master/networks/master_links.BIN --out links.csv
```

### 3.2 Decode the link topology from net.net
The from/to node of each link is NOT in the attribute tables — it is in the
built routable network `net.net` (forward-star CSR + node-ID map). Decode it:
```
python full_pipeline/decode_dln.py      # -> topology.csv (135,922 links)
```
See `full_pipeline/DECODE_NOTES.md` for the format. Validate: 135,922/135,922
links match the one-time geometry export exactly.

### 3.3 Build the GMNS network
`full_pipeline/build_gmns.py` reproduces `02 - Network Calculations.rsc`
(area type from SE density → capacity lookup → free-flow speed → per-link BPR
α/β), joins topology + coordinates, renumbers for the kernel (centroids first),
and writes `scenario/node.csv` + `scenario/link.csv`.
```
python full_pipeline/build_gmns.py      # needs geopandas + the TAZ shapefile
```

### 3.4 Build the demand (TRMG2's own destination-choice tables)
`full_pipeline/make_od_4period.py` runs the first-3-steps: generation → nested
destination choice **using TRMG2's own `dc/*_zone.csv` + `*_cluster.csv`
coefficients** (no invented parameters) → mode/TOD → 4-period vehicle OD, then
assigns each period. See `full_pipeline/REPRODUCTION_MAP.md` for the exact
rsc-step → code mapping.
```
python full_pipeline/make_od_4period.py
```

### 3.5 Build & run the kernel (assignment)
Same kernel as Part A; build once (`kernel/build_*.sh`). The demand script
calls it per period. For the matrix operators you need route output:
```
# in each scenario_{AM,MD,PM,NT}/settings.csv set route_output=1, then run,
# with env TAPLITE_ROUTE_VOL_MIN=0.04 for full OD coverage.
```

### 3.6 Extract the matrix operators
`full_pipeline/matrix_ops.py` turns each period's `route_assignment.csv` into
Delta [paths×od], A [links×paths], Pi = A·Delta [links×od], and validates
`Pi @ od` against the kernel's own loaded volumes (expect R² = 1.0).
```
python full_pipeline/matrix_ops.py AM MD PM NT
```

### 3.7 The computational graph (verify the math)
```
python full_pipeline/tcg_prototype.py     # analytic fwd+bwd, gradcheck, recovery
python full_pipeline/tcg_tensor.py        # PyTorch autodiff == analytic (1e-14)
python full_pipeline/tcg_consistency.py   # numpy==torch==onnx==Pi loop
```

### 3.8 Calibration with guard rails + gates
```
python full_pipeline/tcg_benchmark.py     # scaling + rails-vs-no-rails drift table
python full_pipeline/gates.py             # reproduction gates vs TRMG2 references
python full_pipeline/column_tools.py chicago_sketch   # DynODME + ADMM at scale
```

### 3.9 Visualize
```
python full_pipeline/vizmap_deck.py       # deck.gl WebGL map with count overlay
```

---

## 4. The math and the paper
- `TCG_MATH.pdf` — full tensor-form derivation: forward, exact adjoints,
  gauge/identifiability, per-layer dimensions + graph-cell counts per network,
  and the representation-consistency table.
- `PAPER_OUTLINE.md` — the paper this supports (target: TR-C / TRB).

---

## 5. Verification checkpoints (does it reproduce?)
| stage | check | expected |
|---|---|---|
| kernel build | `bin/DTALite` exists, runs | yes |
| consistency | `code/consistency_check.py` | VERDICT PASS, diffs ~1e-7 |
| Part A pipeline | `pipeline_sioux_falls.json` | pi_R2=1.0, %RMSE 24→13 |
| topology decode | 135,922 links vs export | 100% match |
| GMNS build | `build_log.json` | 0 skipped, all zones |
| Pi (TRMG2 AM) | matrix_ops validation | R²=1.000 |
| graph | tcg_tensor autodiff | == analytic 1e-14 |
| calibration | rails vs no-rails | rails bound drift, ≈ fit |

---

## 6. Troubleshooting
- **kernel won't build (mac):** `brew install libomp cmake`, retry; the
  single-thread fallback (kernel/compat/omp.h) builds even without libomp.
- **onnxruntime missing wheel:** the consistency check still runs numpy-vs-torch;
  skip the onnx line.
- **torch.compile error (MPS):** some ops fall back; the eager timings are still
  valid, and the fallback itself is a finding — record it.
- **Part B path errors:** edit the `REPO`/`ROOT` constants at the top of each
  `full_pipeline/*.py` to point at your TRMG2 checkout.
- **big files fill the disk:** the TRMG2 route CSVs are ~1.4 GB/period; delete
  them after `matrix_ops.py` extracts the compact `.npz` operators.

## 7. Where things came from (provenance)
See `MANIFEST.md` — every input file, its source, and which script consumes it.
Bundled networks (Sioux Falls, Chicago) are the TAPLite kernel's own public
test datasets. TRMG2 data is public (GitHub + ITRE). Nothing here is synthetic
except the tensors inside `consistency_check.py`/`gpu_bench.py`, which exist
only to time/verify the math at scale without needing the full data.
