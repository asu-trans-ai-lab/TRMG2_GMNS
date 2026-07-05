"""demand_graph.py — the single shared core for all levels and modules.

Provides:
  DemandLayers(Z,C,P)      torch nn.Module, L1-L4: (g,dASC,dIC) -> OD [Z,Z]
  np_forward(...)          numpy reference of the same (f64)
  synth_world(Z,C,P)       synthetic frozen tensors at a given scale
  make_pi(Z, density)      a synthetic sparse Pi [A x Z*Z] for level 2/3 tests
  load_pi_npz(path)        load a REAL Pi operator (../matrices/pi_*.npz)

Level 1 = DemandLayers (θ -> OD).  Level 2 = Pi @ vec(OD).  Level 3 = A @ (Δ @ vec(OD)).
All three import THIS module so the math is defined once.
"""
import numpy as np
import torch
import scipy.sparse as sp

torch.set_default_dtype(torch.float32)


def synth_world(Z, C, P, seed=0):
    rng = np.random.default_rng(seed)
    member = rng.integers(0, C, Z)
    t = rng.uniform(3, 60, (Z, Z)); np.fill_diagonal(t, 1e4)
    return dict(
        member=member, Z=Z, C=C, P=P,
        t=t, lnE=rng.uniform(4, 10, Z), pop=rng.uniform(100, 5000, Z),
        theta_c=np.linspace(0.7, 0.95, C),
        rate=rng.uniform(0.4, 0.9, P), sizec=rng.uniform(0.5, 0.9, P),
        timec=-rng.uniform(0.08, 0.2, P), share=rng.uniform(0.6, 0.8, P),
        paf=rng.uniform(0.5, 0.7, P))


def np_forward(g, dASC, dIC, W):
    Z, C, P, mem = W["Z"], W["C"], W["P"], W["member"]
    oh = np.eye(C)[mem]; ODt = np.zeros((Z, Z))
    for p in range(P):
        Prod = W["pop"] * W["rate"][p] * g[p]
        U = W["sizec"][p] * W["lnE"][None, :] + W["timec"][p] * W["t"]
        Uc = U / W["theta_c"][mem][None, :]
        m = np.full((Z, C), -1e30)
        for c in range(C):
            m[:, c] = Uc[:, mem == c].max(1)
        E = np.exp(Uc - m[:, mem]); S = np.zeros((Z, C))
        for c in range(C):
            S[:, c] = E[:, mem == c].sum(1)
        ls = W["theta_c"][None, :] * (m + np.log(np.maximum(S, 1e-300)))
        V = dASC[p][None, :] + dIC[p][None, :] * oh + ls
        EV = np.exp(V - V.max(1, keepdims=True)); Pc = EV / EV.sum(1, keepdims=True)
        T = Prod[:, None] * Pc[:, mem] * (E / S[:, mem])
        ODt += W["share"][p] * (W["paf"][p] * T + (1 - W["paf"][p]) * T.T)
    return ODt


class DemandLayers(torch.nn.Module):
    """Level-1 demand operator: (g, dASC, dIC) -> OD [Z,Z]."""

    def __init__(self, W):
        super().__init__()
        Z, C, P = W["Z"], W["C"], W["P"]
        self.Z, self.C, self.P = Z, C, P
        mem = torch.as_tensor(W["member"], dtype=torch.long)
        logM = torch.full((Z, C), float("-inf")); logM[torch.arange(Z), mem] = 0.0
        onehot = torch.zeros(Z, C); onehot[torch.arange(Z), mem] = 1.0
        buf = dict(mem=mem, logM=logM, onehot=onehot)
        for k in ("t", "lnE", "pop", "theta_c", "rate", "sizec", "timec", "share", "paf"):
            buf[k if k != "theta_c" else "th_c"] = torch.as_tensor(W[k], dtype=torch.float32)
        for n, v in buf.items():
            self.register_buffer(n, v)

    def forward(self, g, dASC, dIC):
        Z, mem = self.Z, self.mem; ODt = torch.zeros(Z, Z)
        for p in range(self.P):
            Prod = self.pop * self.rate[p] * g[p]
            U = self.sizec[p] * self.lnE[None, :] + self.timec[p] * self.t
            Uth = U / self.th_c[mem][None, :]
            masked = Uth[:, :, None] + self.logM[None, :, :]
            W_ic = self.th_c[None, :] * torch.logsumexp(masked, dim=1)
            V = dASC[p][None, :] + dIC[p][None, :] * self.onehot + W_ic
            Q = torch.softmax(V, dim=1); R3 = torch.softmax(masked, dim=1)
            idx = mem[None, :].expand(Z, Z)
            R = torch.gather(R3, 2, idx[:, :, None])[:, :, 0]
            T = Prod[:, None] * Q[:, mem] * R
            ODt = ODt + self.share[p] * (self.paf[p] * T + (1 - self.paf[p]) * T.T)
        return ODt


def make_pi(Z, A=None, seed=1):
    """Synthetic sparse Pi [A x Z*Z]: each OD routes over a few random links."""
    rng = np.random.default_rng(seed)
    A = A or max(2 * Z, 60)
    nod = Z * Z
    rows, cols, vals = [], [], []
    for w in range(nod):
        k = rng.integers(1, 4)
        for _ in range(k):
            rows.append(rng.integers(0, A)); cols.append(w); vals.append(1.0 / k)
    return sp.csr_matrix((vals, (rows, cols)), shape=(A, nod))


def load_pi_npz(path):
    """Load a REAL Pi operator produced by matrix_ops.py."""
    return sp.load_npz(path)
