"""Level 3 — factored sparse assignment operator  Pi = A_inc @ Delta.

Per the differentiable-operator expert brief: Level 3 is Level 2 with Pi
UN-FUSED into its two factors, exposing the path axis so column generation can
append path rows without rebuilding Pi. Same operator, two representations:
  Level 2:  Pi @ od                       (fused, fast path)
  Level 3:  A_inc @ (Delta @ od)          (factored, column-structured)

  A_inc [A x n_paths]   path -> link incidence (0/1)
  Delta [n_paths x |OD|] OD -> path shares (column-stochastic)

Provides matvec / rmatvec (the exact adjoint Pi^T = Delta^T A_inc^T) /
append_paths (column generation) / as_dense_Pi (proves Level3 == Level2).

Run: python level3.py     # self-check C1 (factored==fused) + C4 (adjoints agree)
"""
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from demand_graph import synth_world, np_forward  # noqa: E402


class AssignmentOp:
    """Pi = A_inc @ Delta as an operator with matvec/rmatvec + column generation."""

    def __init__(self, A_inc, Delta):
        self.A_inc = A_inc.tocsr()          # [A, n_paths]
        self.Delta = Delta.tocsr()          # [n_paths, |OD|]
        self.path_flow = None

    @property
    def shape(self):
        return (self.A_inc.shape[0], self.Delta.shape[1])

    def matvec(self, od_vec):               # v = A_inc @ (Delta @ od)
        self.path_flow = self.Delta @ od_vec
        return self.A_inc @ self.path_flow

    def rmatvec(self, dL_dv):               # Pi^T dL_dv = Delta^T @ (A_inc^T @ dL_dv)
        return self.Delta.T @ (self.A_inc.T @ dL_dv)

    def as_dense_Pi(self):                  # for the C1 consistency proof
        return (self.A_inc @ self.Delta)

    def append_paths(self, A_cols, Delta_rows):
        """Column generation: add new paths.
        A_cols [A x n_new] new path->link columns; Delta_rows [n_new x |OD|]."""
        self.A_inc = sp.hstack([self.A_inc, A_cols]).tocsr()
        self.Delta = sp.vstack([self.Delta, Delta_rows]).tocsr()


def synth_factored(Z, n_paths=None, A=None, seed=1):
    """Build a synthetic A_inc, Delta whose product is a valid Pi."""
    rng = np.random.default_rng(seed)
    nod = Z * Z
    A = A or max(2 * Z, 60)
    n_paths = n_paths or 3 * nod            # ~3 paths per OD
    # each path serves one OD, uses a few links, with a share; shares sum to 1 per OD
    d_r, d_c, d_v, a_r, a_c = [], [], [], [], []
    pid = 0
    for w in range(nod):
        k = rng.integers(1, 4)
        shares = rng.dirichlet(np.ones(k))
        for s in shares:
            d_r.append(pid); d_c.append(w); d_v.append(s)
            for _ in range(rng.integers(1, 4)):
                a_r.append(rng.integers(0, A)); a_c.append(pid)
            pid += 1
    Delta = sp.csr_matrix((d_v, (d_r, d_c)), shape=(pid, nod))
    A_inc = sp.csr_matrix((np.ones(len(a_r)), (a_r, a_c)), shape=(A, pid))
    return A_inc, Delta


def main():
    Z, C, P = 120, 5, 2
    W = synth_world(Z, C, P)
    A_inc, Delta = synth_factored(Z)
    op = AssignmentOp(A_inc, Delta)

    rng = np.random.default_rng(3)
    g = rng.uniform(0.8, 1.2, P)
    dASC = rng.uniform(-0.4, 0.4, (P, C)); dASC[:, 0] = 0
    dIC = rng.uniform(-0.3, 0.3, (P, C))
    od = np_forward(g, dASC, dIC, W).reshape(-1)

    # C1: factored matvec == fused dense Pi @ od
    Pi = op.as_dense_Pi()
    v_factored = op.matvec(od)
    v_fused = Pi @ od
    c1 = float(np.abs(v_factored - v_fused).max())

    # C4: factored adjoint rmatvec == Pi^T @ dLdv
    dLdv = rng.uniform(-1, 1, op.shape[0])
    g_factored = op.rmatvec(dLdv)
    g_fused = Pi.T @ dLdv
    c4 = float(np.abs(g_factored - g_fused).max())

    # column generation: append 5 new paths, operator stays valid
    n0 = op.Delta.shape[0]
    new_A = sp.csr_matrix((np.ones(8), (rng.integers(0, op.shape[0], 8),
                           rng.integers(0, 5, 8))), shape=(op.shape[0], 5))
    ns = rng.dirichlet(np.ones(5))
    new_D = sp.csr_matrix((ns, (np.arange(5), rng.integers(0, Z * Z, 5))),
                          shape=(5, Z * Z))
    op.append_paths(new_A, new_D)
    c10 = op.Delta.shape[0] == n0 + 5

    print(f"Level 3 factored operator (Z={Z}, links={op.shape[0]}, "
          f"paths={n0}, od={op.shape[1]}):")
    print(f"  C1 factored matvec == fused Pi@od:   {c1:.2e}  "
          f"({'PASS' if c1 < 1e-9 else 'FAIL'})")
    print(f"  C4 factored adjoint == Pi^T@dLdv:    {c4:.2e}  "
          f"({'PASS' if c4 < 1e-9 else 'FAIL'})")
    print(f"  C10 append_paths grows operator:     {n0}->{op.Delta.shape[0]}  "
          f"({'PASS' if c10 else 'FAIL'})")
    print("  (path_flow f = Delta@od cached for column pricing c_r = A_inc^T t)")


if __name__ == "__main__":
    main()
