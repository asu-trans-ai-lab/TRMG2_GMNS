# GPU-readiness findings (2026-07-04)

## Environment
- torch 2.11.0+**cpu** — no CUDA/MPS on this machine. GPU execution impossible here.
- The demand-layer code is **device-portable**: every tensor is created with
  `device=` and ops use `g.device`; `.to('cuda')` runs the same file unchanged.

## Eager CPU profile (gpu_bench.json) — the decision-relevant numbers
| scale | Z | C | P | OD cells | fwd | fwd+bwd |
|---|---|---|---|---|---|---|
| sioux_falls | 24 | 3 | 2 | 576 | 0.48 ms | 0.73 ms |
| chicago_sketch | 386 | 6 | 2 | 149k | 6.9 ms | 9.9 ms |
| **trmg2_scale** | 3147 | 12 | 8 | **9.9M** | **4.33 s** | **5.42 s** |

The Z^2 softmax/LSE stack dominates and scales super-linearly (149k->9.9M cells
= 66x, but 6.9ms->4334ms = 628x) — cache/bandwidth bound, the exact workload a
GPU accelerates. One L-BFGS calibration (~100-300 evals) at TRMG2 scale is
~9-27 min in the demand layer alone on CPU eager. This is the case for GPU +
torch.compile (and the fallback DemandLite C++).

## torch.compile (inductor) — blocked on THIS box, not on the method
Windows inductor CPU backend requires MSVC `cl`; the machine has mingw g++
(for the kernel) but inductor rejects it (passes `/help` MSVC flags to g++).
**Finding: run the GPU benchmark on a Linux GPU box** where inductor works out
of the box — the same `gpu_bench.py` runs unchanged (device auto-detected).

## What to run on a GPU box (unchanged)
    python gpu_bench.py        # device auto -> 'cuda'; eager + torch.compile
Expected: the 4.3 s TRMG2 forward drops to tens of ms (cuSPARSE + fused
softmax); this single number decides how much of the DemandLite C++ plan
(04_demandlite_cpp) is still needed vs superseded by torch.compile.

## ONNX export (01_exports) — fixed
The first export folded to a 0-node constant (theta were baked nn.Parameters,
forward took no args). Now theta = (g, dASC, dIC) are graph INPUTS: sf ONNX =
33.8 KB, **44 compute nodes**, onnxruntime(theta) == torch(theta) at random
theta (4.9e-4). The artifact is a real parameterized demand graph, GPU-ready.
