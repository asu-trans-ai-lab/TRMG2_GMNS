# tcg/ — computational-graph, ONNX & efficiency testing (Mac-ready)

Self-contained kit to test the four-step travel demand model as a differentiable
tensor computational graph: verify it, export it to ONNX, and benchmark its
efficiency. **numpy + torch + onnx only** — no kernel, no network data, no
geopandas. Runs unchanged on macOS (MPS), Linux (CUDA), or Windows (CPU).

## Setup
```
python3 -m venv ~/tcg_venv && source ~/tcg_venv/bin/activate   # keep venv out of Dropbox
pip install -r requirements.txt
```

## The three tests

### 1. Consistency — is the graph the same math everywhere?
```
python consistency_check.py
```
Proves numpy == PyTorch == ONNX for the demand OD, the link volumes v = Pi·OD,
and the gradient (analytic == autograd). Expect `VERDICT: PASS` (numpy vs torch
~1e-16 at f64, torch vs onnx ~1e-7 at f32). Run this FIRST — if it fails, the
torch/onnx install is broken and no timing is trustworthy.

### 2. ONNX — PRE-BUILT for every network, ready to test (no export needed)
`artifacts/` already contains verified ONNX models for all scales (theta =
g, dASC, dIC are graph INPUTS, not folded constants). Just run them with
onnxruntime — no torch, no export:
```
python run_onnx.py            # loads + times all 4 models
python run_onnx.py trmg2      # just one
```
Pre-built models (each is `<net>_demand.onnx` + a `.onnx.data` weights sidecar
— KEEP THE PAIR TOGETHER when copying):
| network | Z | C | P | ONNX size | nodes | CPU infer (Windows) |
|---|---|---|---|---|---|---|
| sioux_falls | 24 | 3 | 2 | 42 KB | 44 | 0.04 ms |
| chicago_sketch | 386 | 6 | 2 | 3.7 MB | 66 | 2.1 ms |
| chicago_regional | 1769 | 10 | 2 | 12.8 MB | 70 | 7.3 ms |
| trmg2 | 3147 | 12 | 8 | 40.9 MB | 274 | 385 ms |

(Note: onnxruntime's graph fusion already does TRMG2-scale in 385 ms vs torch
eager 4.33 s — ~11x on CPU. On the Mac, onnxruntime CoreML / torch MPS goes
faster still — that's the number to capture.)

To (re)generate them all: `python export_all.py`. To make a custom scale:
`python export_onnx.py --Z 3147 --C 12 --P 8` (writes artifacts/demand_layers.onnx).

### 3. Efficiency — how fast, and where does GPU help?
```
python gpu_bench.py
```
Auto-detects the device (cuda / mps / cpu) and times eager forward +
forward/backward AND torch.compile at 3 scales (Sioux Falls -> TRMG2 size),
writing `artifacts/gpu_bench.json`. **This is the number we want from the Mac:**
Windows CPU eager at TRMG2 scale was 4.33 s forward; report the Apple-Silicon
MPS + torch.compile time and speedup. See `GPU_FINDINGS.md` for the Windows
baseline and why torch.compile was blocked there (needs MSVC; Mac clang works).

## The math and design
- `TCG_MATH.pdf` — full tensor derivation: forward, exact adjoints, per-layer
  dimensions + graph-cell counts per network, representation-consistency table.
  Key efficiency insight: the `[Z,Z,C]` masked-LSE tensor is 1/C sparse -> a
  segmented/scatter implementation is a C-fold (12x at TRMG2 scale) reduction.
- `NN_EXPORT_DESIGN.md` — the NN framing: export routes, optimizer menu beyond
  ADMM, low-rank (LoRA) utility adapters, highway/DEQ tricks.

## Notes for the Mac run
- MPS: if a torch.compile op is unsupported it falls back; eager timings stay
  valid and the fallback itself is a finding — record it.
- Force GPU-vs-CPU: `PYTORCH_ENABLE_MPS_FALLBACK=1 python gpu_bench.py`.
- To sync results back to the Windows box, copy `artifacts/*.json` (see the
  DROPBOX_SYNC.md in tcg_turnkey/ for the cross-machine workflow).
- This folder tests the GRAPH in isolation. The full TRMG2 pipeline (real
  assignment + calibration) is the parent `gmns/` folder — see its GMNS_README.
