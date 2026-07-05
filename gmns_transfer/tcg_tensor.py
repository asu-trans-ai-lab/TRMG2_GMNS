"""tcg_tensor.py — PyTorch (tensor-platform) implementation of the layered
4-step computational graph, verifying AUTODIFF against the hand-derived
analytic adjoints of tcg_prototype.py on the same Sioux Falls world.

Math reference: TCG_MATH.tex (every line below cites its equation).
PASS criterion: torch autograd gradient == analytic gradient (rel err <= 1e-6)
and equal losses. This certifies that extending the graph with new layers
(person-level generation, MC logsums, feedback unrolling) can rely on
autodiff without re-deriving adjoints.
"""
import json
import sys
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tcg_prototype as tp  # noqa: E402

torch.set_default_dtype(torch.float64)


def torch_forward(x, W, counts, Pi_t):
    P, C, Z = len(tp.RATE), W["C"], W["Z"]
    g = x[:P]
    dASC = x[P:P + P * C].reshape(P, C)
    dIC = x[P + P * C:].reshape(P, C)

    mem = torch.as_tensor(W["member"])
    logM = torch.full((Z, C), -torch.inf)
    logM[torch.arange(Z), mem] = 0.0                    # mask ln M  [Z,C]
    onehot = torch.zeros(Z, C)
    onehot[torch.arange(Z), mem] = 1.0                  # origin-in-cluster
    t = torch.as_tensor(W["t"])
    lnE = torch.as_tensor(W["lnE"])
    pop = torch.as_tensor(W["pop"])
    th_c = torch.as_tensor(W["theta_c"])

    ODt = torch.zeros(Z, Z)
    for p in range(P):
        Prod = pop * tp.RATE[p] * g[p]                                  # (L1)
        U = tp.SIZECOEF[p] * lnE[None, :] + tp.TIMECOEF[p] * t          # (2)
        Uth = U / th_c[mem][None, :]                    # scale by dest cluster
        # masked LSE over destinations of each cluster  (4): W_[i,c]
        W_ic = th_c[None, :] * torch.logsumexp(
            Uth[:, :, None] + logM[None, :, :], dim=1)
        V = dASC[p][None, :] + dIC[p][None, :] * onehot + W_ic          # (6)
        Q = torch.softmax(V, dim=1)                                     # (6)
        # masked conditional within cluster              (5): R_[i,j]
        R = torch.softmax(Uth[:, :, None] + logM[None, :, :], dim=1)
        R = R[torch.arange(Z)[:, None], torch.arange(Z)[None, :], mem[None, :]
              .expand(Z, Z)]
        T = Prod[:, None] * Q[:, mem] * R                               # (7)
        X = tp.PAF[p] * T + (1 - tp.PAF[p]) * T.T                       # (8)
        ODt = ODt + tp.SHARE[p] * X                                     # (9)
    sr, sc, sd = W["sel"]
    odv = torch.zeros(Pi_t.shape[1])
    odv[torch.as_tensor(sd)] = ODt[torch.as_tensor(sr), torch.as_tensor(sc)]
    v = torch.sparse.mm(Pi_t, odv[:, None])[:, 0]                       # (10)
    c = torch.as_tensor(counts)
    wgt = 1.0 / torch.clamp(c, min=10.0)
    r = (v - c) * wgt
    return 0.5 * (r @ r)                                                # (11)


def main():
    Pi, od_col, n_links, pi_err = tp.run_kernel_and_extract()
    W = tp.build_world(Pi, od_col)
    P, C = len(tp.RATE), W["C"]

    coo = Pi.tocoo()
    Pi_t = torch.sparse_coo_tensor(
        np.vstack([coo.row, coo.col]), coo.data, coo.shape).coalesce()

    rng = np.random.default_rng(42)
    th_true = dict(g=np.array([1.25, 0.85]),
                   dASC=rng.uniform(-0.5, 0.5, (P, C)),
                   dIC=rng.uniform(-0.3, 0.3, (P, C)))
    th_true["dASC"][:, 0] = 0.0
    counts = tp.forward(th_true, W)["v"]

    x0 = tp.pack(dict(g=np.ones(P), dASC=np.zeros((P, C)), dIC=np.zeros((P, C))))

    # analytic (hand-derived adjoints, already gradcheck-verified)
    L_ana, g_ana = tp.loss_grad(x0, W, counts)

    # tensor platform (autodiff)
    x = torch.tensor(x0, requires_grad=True)
    L_t = torch_forward(x, W, counts, Pi_t)
    L_t.backward()
    g_t = x.grad.detach().numpy()

    dL = abs(float(L_t) - L_ana) / max(abs(L_ana), 1e-12)
    denom = np.maximum(np.abs(g_ana), np.abs(g_t))
    rel = np.abs(g_t - g_ana) / np.where(denom > 1e-12, denom, 1.0)
    worst = float(rel.max())
    ok = dL < 1e-10 and worst < 1e-6
    print(json.dumps(dict(
        loss_analytic=float(f"{L_ana:.12f}"), loss_torch=float(f"{float(L_t):.12f}"),
        loss_rel_diff=float(f"{dL:.2e}"),
        grad_worst_rel_diff=float(f"{worst:.2e}"),
        n_params=len(x0),
        VERDICT="PASS" if ok else "FAIL",
        meaning="autodiff on the tensor graph reproduces the hand-derived "
                "adjoints exactly -> new layers can rely on autodiff"),
        indent=1))


if __name__ == "__main__":
    main()
