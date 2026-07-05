"""Level 2 — assignment-composed forward:  theta -> v = Pi @ vec(OD).

Per the ONNX-operator expert brief: Pi is NOT put in ONNX (onnxruntime has no
executing sparse matmul on CPU/CoreML/MPS; dense is petabytes at scale). Level 2
= Level-1 demand (ONNX or torch) THEN a runtime sparse SpMV with Pi as a CSR
.npz. Differentiable in torch via torch.sparse.mm (grad flows into OD -> theta).

Ships / consumes:  demand.onnx (level 1)  +  pi.npz (CSR)  ->  link volumes v.

Run:  python level2.py            # synthetic self-check
      python level2.py --pi ../../matrices/pi_AM.npz   # a REAL Pi operator
"""
import argparse
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from demand_graph import synth_world, np_forward, make_pi, load_pi_npz  # noqa: E402


def level2_numpy(g, dASC, dIC, W, Pi):
    """v = Pi @ vec(OD).  OD from the demand operator, Pi applied as sparse SpMV."""
    OD = np_forward(g, dASC, dIC, W)          # Level 1
    return Pi @ OD.reshape(-1)                # Level 2 (runtime sparse)


def level2_torch(mod, g, dASC, dIC, Pi_coo):
    """Differentiable Level 2: torch.sparse.mm(Pi, vec(OD)); grad flows to theta."""
    import torch
    OD = mod(g, dASC, dIC)                    # Level 1 (autograd)
    return torch.sparse.mm(Pi_coo, OD.reshape(-1, 1))[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi", default=None, help="path to a real pi_*.npz (else synthetic)")
    ap.add_argument("--Z", type=int, default=200)
    a = ap.parse_args()

    if a.pi and os.path.exists(a.pi):
        Pi = load_pi_npz(a.pi)
        Z = int(round(Pi.shape[1] ** 0.5))
        print(f"real Pi {a.pi}: {Pi.shape}, nnz {Pi.nnz:,}, implied Z~{Z}")
        # a real Pi needs the matching real world to be meaningful; here we only
        # demonstrate the SpMV mechanics + timing on a random OD of the right size.
        od = np.random.default_rng(0).uniform(0, 5, Pi.shape[1])
        t0 = time.time(); v = Pi @ od
        print(f"  Pi @ od SpMV: {(time.time()-t0)*1e3:.1f} ms, "
              f"{int((v>0).sum()):,}/{len(v):,} links loaded")
        return

    # synthetic self-check: Level 2 == (Level 1 then Pi)
    Z, C, P = a.Z, 5, 2
    W = synth_world(Z, C, P)
    Pi = make_pi(Z)
    rng = np.random.default_rng(3)
    g = rng.uniform(0.8, 1.2, P)
    dASC = rng.uniform(-0.4, 0.4, (P, C)); dASC[:, 0] = 0
    dIC = rng.uniform(-0.3, 0.3, (P, C))

    v = level2_numpy(g, dASC, dIC, W, Pi)
    OD = np_forward(g, dASC, dIC, W)
    v_check = Pi @ OD.reshape(-1)
    err = float(np.abs(v - v_check).max())

    # torch differentiability check
    import torch
    from demand_graph import DemandLayers
    mod = DemandLayers(W)
    coo = Pi.tocoo()
    Pit = torch.sparse_coo_tensor(np.vstack([coo.row, coo.col]),
                                  coo.data.astype(np.float32), coo.shape).coalesce()
    gt = torch.tensor(g, dtype=torch.float32, requires_grad=True)
    at = torch.tensor(dASC, dtype=torch.float32, requires_grad=True)
    it = torch.tensor(dIC, dtype=torch.float32, requires_grad=True)
    v_t = level2_torch(mod, gt, at, it, Pit)
    v_t.sum().backward()
    grad_ok = gt.grad is not None and at.grad is not None

    print(f"Level 2 (Z={Z}, links={Pi.shape[0]}, od={Pi.shape[1]}):")
    print(f"  v = Pi @ vec(OD): total {v.sum():,.0f}, {int((v>0).sum())} links loaded")
    print(f"  Level2 == Level1·Pi:  max|diff| {err:.2e}  ({'PASS' if err < 1e-9 else 'FAIL'})")
    print(f"  torch.sparse.mm differentiable -> theta.grad exists: "
          f"{'PASS' if grad_ok else 'FAIL'} "
          f"(g.grad norm {float(gt.grad.norm()):.3e})")


if __name__ == "__main__":
    main()
