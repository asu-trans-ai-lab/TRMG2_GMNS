"""consistency_check.py — SELF-CONTAINED representation-consistency gate.

No kernel, no real data, no geopandas — numpy + torch + onnx only. Proves the
demand computational graph is identical across representations on a synthetic
world, so a calibration step is the same in any of them:

    numpy (f64 reference)  ==  PyTorch (f32)  ==  ONNX (f32)
    then v = Pi @ vec(OD) agrees, and the gradient (numpy analytic adjoint ==
    torch autograd) agrees.

This is the portable version of the in-repo tcg_consistency.py; identical math,
synthetic frozen tensors + a synthetic sparse Pi. Run: python consistency_check.py
"""
import json
import os

import numpy as np
import torch
import scipy.sparse as sp

torch.set_default_dtype(torch.float64)   # match numpy for a clean gradient check
Z, C, P = 60, 4, 2
SIZEC = np.array([0.8, 0.65]); TIMEC = np.array([-0.09, -0.16])
RATE = np.array([0.55, 0.85]); SHARE = np.array([0.75, 0.62]); PAF = np.array([0.65, 0.5])


def synth(seed=1):
    rng = np.random.default_rng(seed)
    member = rng.integers(0, C, Z)
    t = rng.uniform(3, 60, (Z, Z)); np.fill_diagonal(t, 1e4)
    lnE = rng.uniform(4, 10, Z); pop = rng.uniform(100, 5000, Z)
    theta_c = np.linspace(0.7, 0.95, C)
    # synthetic Pi: each od routes over a few random links
    A = 200
    rows, cols, vals = [], [], []
    nod = Z * Z
    for w in range(nod):
        for _ in range(rng.integers(1, 4)):
            rows.append(rng.integers(0, A)); cols.append(w); vals.append(1.0)
    Pi = sp.csr_matrix((vals, (rows, cols)), shape=(A, nod))
    return dict(member=member, t=t, lnE=lnE, pop=pop, theta_c=theta_c, Pi=Pi)


def np_forward(g, dASC, dIC, W):
    mem = W["member"]; onehot = np.eye(C)[mem]
    ODt = np.zeros((Z, Z))
    for p in range(P):
        Prod = W["pop"] * RATE[p] * g[p]
        U = SIZEC[p] * W["lnE"][None, :] + TIMEC[p] * W["t"]
        Uc = U / W["theta_c"][mem][None, :]
        m = np.full((Z, C), -1e30)
        for c in range(C):
            m[:, c] = Uc[:, mem == c].max(1)
        E = np.exp(Uc - m[:, mem]); S = np.zeros((Z, C))
        for c in range(C):
            S[:, c] = E[:, mem == c].sum(1)
        logsum = W["theta_c"][None, :] * (m + np.log(np.maximum(S, 1e-300)))
        V = dASC[p][None, :] + dIC[p][None, :] * onehot + logsum
        EV = np.exp(V - V.max(1, keepdims=True)); Pc = EV / EV.sum(1, keepdims=True)
        Pjc = E / S[:, mem]; T = Prod[:, None] * Pc[:, mem] * Pjc
        ODt += SHARE[p] * (PAF[p] * T + (1 - PAF[p]) * T.T)
    return ODt


class DemandLayers(torch.nn.Module):
    def __init__(self, W):
        super().__init__()
        mem = torch.as_tensor(W["member"], dtype=torch.long)
        logM = torch.full((Z, C), float("-inf")); logM[torch.arange(Z), mem] = 0.0
        onehot = torch.zeros(Z, C); onehot[torch.arange(Z), mem] = 1.0
        for n, v in dict(mem=mem, logM=logM, onehot=onehot,
                         t=torch.as_tensor(W["t"]), lnE=torch.as_tensor(W["lnE"]),
                         pop=torch.as_tensor(W["pop"]), th_c=torch.as_tensor(W["theta_c"]),
                         rate=torch.as_tensor(RATE), sizec=torch.as_tensor(SIZEC),
                         timec=torch.as_tensor(TIMEC), share=torch.as_tensor(SHARE),
                         paf=torch.as_tensor(PAF)).items():
            self.register_buffer(n, v)

    def forward(self, g, dASC, dIC):
        mem = self.mem; ODt = torch.zeros(Z, Z, dtype=g.dtype)
        for p in range(P):
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


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    W = synth()
    rng = np.random.default_rng(3)
    g = rng.uniform(0.8, 1.2, P)
    dASC = rng.uniform(-0.4, 0.4, (P, C)); dASC[:, 0] = 0.0
    dIC = rng.uniform(-0.3, 0.3, (P, C))

    OD_np = np_forward(g, dASC, dIC, W)
    mod = DemandLayers(W).eval()
    gt, at, it = (torch.tensor(g), torch.tensor(dASC), torch.tensor(dIC))
    with torch.no_grad():
        OD_pt = mod(gt, at, it).numpy()

    # ONNX (export at f32)
    mod32 = DemandLayers(W).to(torch.float32).eval()
    a32 = (gt.float(), at.float(), it.float())
    onnx_fp = os.path.join(here, "_consistency.onnx")
    torch.onnx.export(mod32, a32, onnx_fp, input_names=["g", "dASC", "dIC"],
                      output_names=["OD"], dynamo=True)
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_fp)
    OD_ox = sess.run(None, {"g": g.astype(np.float32), "dASC": dASC.astype(np.float32),
                            "dIC": dIC.astype(np.float32)})[0]
    for junk in (onnx_fp, onnx_fp + '.data'):
        if os.path.exists(junk):
            os.remove(junk)

    sc = OD_np.max()
    d1 = float(np.abs(OD_np - OD_pt).max() / sc)
    d2 = float(np.abs(OD_pt - OD_ox).max() / sc)

    def v(OD):
        return W["Pi"] @ OD.ravel()
    vs = max(v(OD_np).max(), 1.0)
    dv1 = float(np.abs(v(OD_np) - v(OD_pt)).max() / vs)
    dv2 = float(np.abs(v(OD_pt) - v(OD_ox)).max() / vs)

    # gradient: numpy finite-diff vs torch autograd of L = 0.5||Pi vec(OD) - c||^2
    counts = v(OD_np)
    coo = W["Pi"].tocoo()
    Pit = torch.sparse_coo_tensor(np.vstack([coo.row, coo.col]), coo.data, coo.shape).coalesce()
    gt2 = torch.zeros(P, requires_grad=True); at2 = torch.zeros(P, C, requires_grad=True)
    it2 = torch.zeros(P, C, requires_grad=True)
    gt2.data = torch.ones(P)
    OD_t = mod(gt2, at2, it2)
    v_t = torch.sparse.mm(Pit, OD_t.reshape(-1, 1))[:, 0]
    L = 0.5 * ((v_t - torch.as_tensor(counts)) ** 2).sum(); L.backward()
    g_torch = np.concatenate([gt2.grad.numpy(), at2.grad.numpy().ravel(), it2.grad.numpy().ravel()])
    # numpy finite diff
    def loss(x):
        gg = x[:P]; da = x[P:P + P * C].reshape(P, C); di = x[P + P * C:].reshape(P, C)
        return 0.5 * ((v(np_forward(gg, da, di, W)) - counts) ** 2).sum()
    x0 = np.concatenate([np.ones(P), np.zeros(2 * P * C)])
    g_fd = np.zeros_like(x0); eps = 1e-6
    for k in range(len(x0)):
        xp = x0.copy(); xp[k] += eps; xm = x0.copy(); xm[k] -= eps
        g_fd[k] = (loss(xp) - loss(xm)) / (2 * eps)
    den = np.maximum(np.abs(g_fd), np.abs(g_torch))
    grad_rel = float((np.abs(g_torch - g_fd) / np.where(den > 1e-6, den, 1.0)).max())

    ok = d1 < 2e-9 and d2 < 1e-4 and dv1 < 2e-9 and dv2 < 1e-4 and grad_rel < 1e-4
    rep = dict(world=f"synthetic Z={Z} C={C} P={P}",
               OD_numpy_vs_torch=float(f"{d1:.2e}"), OD_torch_vs_onnx=float(f"{d2:.2e}"),
               v_numpy_vs_torch=float(f"{dv1:.2e}"), v_torch_vs_onnx=float(f"{dv2:.2e}"),
               grad_fd_vs_autograd=float(f"{grad_rel:.2e}"),
               VERDICT="PASS" if ok else "FAIL")
    print(json.dumps(rep, indent=1))
    outdir = os.path.join(here, "..", "results")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "consistency_result.json"), "w") as f:
        json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
