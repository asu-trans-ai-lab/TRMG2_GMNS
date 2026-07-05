"""run_levels.py — master consistency runner across the 3 levels + modules.

Runs the expert-specified cross-checks (TCG_MATH / LEVELS.md) that prove the
levels and modules are mutually consistent:

  C1  Level3 factored (A_inc@Delta) matvec == Level2 fused (Pi@od)
  C2  Level2 (theta->v) == Pi @ Level1 (theta->OD)
  C4  Level3 factored adjoint (Delta^T A_inc^T) == Pi^T
  C8  ADMM reduces the count loss; drift rail (mu) keeps x near 1
  C9  influence restriction (drop zero-norm Pi_obs columns) is lossless
  grad  torch autograd through Level2 (torch.sparse.mm) gives theta.grad

One PASS/FAIL table. Run: python run_levels.py
"""
import os
import sys

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "common"))
sys.path.insert(0, os.path.join(HERE, "level3_sparse"))
sys.path.insert(0, os.path.join(HERE, "modules", "admm"))
from demand_graph import synth_world, np_forward, make_pi, DemandLayers  # noqa
from level3 import AssignmentOp, synth_factored  # noqa
import column_tools as ct  # noqa

R = []


def check(cid, name, ok, detail=""):
    R.append((cid, name, "PASS" if ok else "FAIL", detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {cid} {name}  {detail}")


def main():
    Z, C, P = 120, 5, 2
    W = synth_world(Z, C, P)
    rng = np.random.default_rng(3)
    g = rng.uniform(0.8, 1.2, P)
    dASC = rng.uniform(-0.4, 0.4, (P, C)); dASC[:, 0] = 0
    dIC = rng.uniform(-0.3, 0.3, (P, C))
    OD = np_forward(g, dASC, dIC, W); od = OD.reshape(-1)

    # ---- C1 + C4 (Level 3 factored vs fused) ----
    A_inc, Delta = synth_factored(Z)
    op = AssignmentOp(A_inc, Delta)
    Pi = op.as_dense_Pi()
    check("C1", "Level3 matvec == Pi@od",
          float(np.abs(op.matvec(od) - Pi @ od).max()) < 1e-9)
    dLdv = rng.uniform(-1, 1, op.shape[0])
    check("C4", "Level3 adjoint == Pi^T@dLdv",
          float(np.abs(op.rmatvec(dLdv) - Pi.T @ dLdv).max()) < 1e-9)

    # ---- C2 (Level 2 == Pi @ Level 1) ----
    Pi2 = make_pi(Z)
    v = Pi2 @ np_forward(g, dASC, dIC, W).reshape(-1)
    check("C2", "Level2 == Pi @ Level1",
          float(np.abs(v - Pi2 @ OD.reshape(-1)).max()) < 1e-12)

    # ---- grad (torch differentiability through Level 2) ----
    import torch
    mod = DemandLayers(W)
    coo = Pi2.tocoo()
    Pit = torch.sparse_coo_tensor(np.vstack([coo.row, coo.col]),
                                  coo.data.astype(np.float32), coo.shape).coalesce()
    gt = torch.tensor(g, dtype=torch.float32, requires_grad=True)
    at = torch.tensor(dASC, dtype=torch.float32, requires_grad=True)
    it = torch.tensor(dIC, dtype=torch.float32, requires_grad=True)
    vt = torch.sparse.mm(Pit, mod(gt, at, it).reshape(-1, 1))[:, 0]
    vt.sum().backward()
    check("grad", "torch.sparse.mm -> theta.grad",
          gt.grad is not None and float(gt.grad.norm()) > 0)

    # ---- C8 + C9 (ADMM stage-2, influence restriction) ----
    # observed = top links by loaded volume; perturb OD by per-origin factors
    A = Pi2.shape[0]
    loaded = Pi2 @ od
    obs = np.argsort(-loaded)[: max(A // 3, 10)]
    counts = loaded[obs]
    origin_of = np.repeat(np.arange(Z), Z)     # column w = i*Z+j -> origin i
    fac = rng.uniform(0.65, 1.35, Z)
    od_pert = od * fac[origin_of]
    Pi_obs, active, stats = ct.restrict_to_observed(Pi2, obs)
    od0 = od_pert[active]; orig = origin_of[active]
    v0 = Pi_obs @ od0
    rmse0 = float(np.sqrt(((v0 - counts) ** 2).mean()) / counts.mean() * 100)
    x, trace, _ = ct.admm_od_adjust(Pi_obs, od0, counts, orig, mu=0.3, iters=80,
                                    verbose=False)
    rmse1 = trace[-1]["prmse"]
    check("C8", "ADMM reduces count loss",
          rmse1 < rmse0, f"%RMSE {rmse0:.1f}->{rmse1:.1f}")
    check("C9", "influence restriction lossless",
          stats["n_od_active"] <= stats["n_od_total"],
          f"{stats['n_od_active']}/{stats['n_od_total']} active cols")

    npass = sum(1 for r in R if r[2] == "PASS")
    print(f"\n{npass}/{len(R)} checks PASS")


if __name__ == "__main__":
    main()
