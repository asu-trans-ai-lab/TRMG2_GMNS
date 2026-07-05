"""export_onnx.py — SELF-CONTAINED export of the 4-step demand graph (L1-L4)
to ONNX + torch.export, with theta = (g, dASC, dIC) as real graph INPUTS.

No kernel, no data — numpy + torch + onnx only. Uses a synthetic world at a
chosen scale so it runs anywhere (Mac/Linux/Windows). Verifies the exported
ONNX reproduces PyTorch at random theta AND that the graph has real compute
nodes (the earlier bug was a folded 0-node constant when theta were baked
Parameters; here theta are inputs).

Usage:
  python export_onnx.py                 # Z=60 synthetic, quick
  python export_onnx.py --Z 3147 --C 12 --P 8   # TRMG2-scale graph
Outputs: artifacts/demand_layers.onnx + demand_layers.pt2
"""
import argparse
import os

import numpy as np
import torch

torch.set_default_dtype(torch.float32)


class DemandLayers(torch.nn.Module):
    """L1-L4 as (g, dASC, dIC) -> OD. Frozen world tensors are buffers."""

    def __init__(self, Z, C, P, seed=0):
        super().__init__()
        self.Z, self.C, self.P = Z, C, P
        rng = np.random.default_rng(seed)
        mem = torch.as_tensor(rng.integers(0, C, Z), dtype=torch.long)
        logM = torch.full((Z, C), float("-inf")); logM[torch.arange(Z), mem] = 0.0
        onehot = torch.zeros(Z, C); onehot[torch.arange(Z), mem] = 1.0
        buf = dict(
            mem=mem, logM=logM, onehot=onehot,
            t=torch.as_tensor(rng.uniform(3, 60, (Z, Z)), dtype=torch.float32),
            lnE=torch.as_tensor(rng.uniform(4, 10, Z), dtype=torch.float32),
            pop=torch.as_tensor(rng.uniform(100, 5000, Z), dtype=torch.float32),
            th_c=torch.as_tensor(np.linspace(0.7, 0.95, C), dtype=torch.float32),
            rate=torch.as_tensor(rng.uniform(0.4, 0.9, P), dtype=torch.float32),
            sizec=torch.as_tensor(rng.uniform(0.5, 0.9, P), dtype=torch.float32),
            timec=torch.as_tensor(-rng.uniform(0.08, 0.2, P), dtype=torch.float32),
            share=torch.as_tensor(rng.uniform(0.6, 0.8, P), dtype=torch.float32),
            paf=torch.as_tensor(rng.uniform(0.5, 0.7, P), dtype=torch.float32))
        for n, v in buf.items():
            self.register_buffer(n, v)

    def forward(self, g, dASC, dIC):
        Z, mem = self.Z, self.mem
        ODt = torch.zeros(Z, Z)
        for p in range(self.P):
            Prod = self.pop * self.rate[p] * g[p]                          # L1
            U = self.sizec[p] * self.lnE[None, :] + self.timec[p] * self.t
            Uth = U / self.th_c[mem][None, :]
            masked = Uth[:, :, None] + self.logM[None, :, :]
            W_ic = self.th_c[None, :] * torch.logsumexp(masked, dim=1)     # L2
            V = dASC[p][None, :] + dIC[p][None, :] * self.onehot + W_ic
            Q = torch.softmax(V, dim=1)
            R3 = torch.softmax(masked, dim=1)
            idx = mem[None, :].expand(Z, Z)
            R = torch.gather(R3, 2, idx[:, :, None])[:, :, 0]
            T = Prod[:, None] * Q[:, mem] * R
            X = self.paf[p] * T + (1 - self.paf[p]) * T.T                  # L4
            ODt = ODt + self.share[p] * X                                  # L3
        return ODt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Z", type=int, default=60)
    ap.add_argument("--C", type=int, default=4)
    ap.add_argument("--P", type=int, default=2)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, "..", "artifacts"); os.makedirs(outdir, exist_ok=True)

    mod = DemandLayers(a.Z, a.C, a.P).eval()
    rng = np.random.default_rng(1)
    g = torch.tensor(rng.uniform(0.7, 1.3, a.P), dtype=torch.float32)
    dASC = torch.tensor(rng.uniform(-0.4, 0.4, (a.P, a.C)), dtype=torch.float32)
    dIC = torch.tensor(rng.uniform(-0.3, 0.3, (a.P, a.C)), dtype=torch.float32)
    args = (g, dASC, dIC)
    with torch.no_grad():
        ref = mod(*args)

    pt2 = os.path.join(outdir, "demand_layers.pt2")
    torch.export.save(torch.export.export(mod, args), pt2)

    onnx_fp = os.path.join(outdir, "demand_layers.onnx")
    torch.onnx.export(mod, args, onnx_fp,
                      input_names=["g", "dASC", "dIC"], output_names=["OD"],
                      dynamo=True)

    import onnx
    gph = onnx.load(onnx_fp).graph
    n_nodes = len(gph.node)
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_fp)
    out = sess.run(None, {"g": g.numpy(), "dASC": dASC.numpy(), "dIC": dIC.numpy()})[0]
    diff = float(np.abs(out - ref.numpy()).max())

    print(f"[export] Z={a.Z} C={a.C} P={a.P}  OD total {ref.sum():,.0f}")
    print(f"  demand_layers.onnx: {os.path.getsize(onnx_fp)/1e3:.1f} KB, "
          f"{n_nodes} compute nodes, inputs {[i.name for i in gph.input]}")
    print(f"  onnxruntime(theta) vs torch(theta) max|diff| = {diff:.3e} "
          f"({'PASS' if diff < 1e-3 and n_nodes > 0 else 'FAIL'})")


if __name__ == "__main__":
    main()
