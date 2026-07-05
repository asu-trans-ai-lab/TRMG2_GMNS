"""export_all.py — pre-generate ONNX (+ torch.export) for ALL network scales,
ready to load on any machine. Self-contained (synthetic frozen tensors at each
network's real Z/C/P shape; no kernel/data). theta = (g, dASC, dIC) are graph
INPUTS. Each export is verified against PyTorch and written to artifacts/.

Run once here; the artifacts/ *.onnx files are then ready to test on the Mac
with onnxruntime — no regeneration needed.

  python export_all.py            # all networks
Outputs: artifacts/<net>_demand.onnx + .pt2, and artifacts/ONNX_MANIFEST.json
"""
import json
import os

import numpy as np
import torch

from export_onnx import DemandLayers   # reuse the self-contained module

torch.set_default_dtype(torch.float32)

NETWORKS = [
    dict(name="sioux_falls",     Z=24,   C=3,  P=2),
    dict(name="chicago_sketch",  Z=386,  C=6,  P=2),
    dict(name="chicago_regional", Z=1769, C=10, P=2),
    dict(name="trmg2",           Z=3147, C=12, P=8),
]


def export_one(net, outdir):
    Z, C, P = net["Z"], net["C"], net["P"]
    mod = DemandLayers(Z, C, P).eval()
    rng = np.random.default_rng(1)
    g = torch.tensor(rng.uniform(0.7, 1.3, P), dtype=torch.float32)
    dASC = torch.tensor(rng.uniform(-0.4, 0.4, (P, C)), dtype=torch.float32)
    dIC = torch.tensor(rng.uniform(-0.3, 0.3, (P, C)), dtype=torch.float32)
    args = (g, dASC, dIC)
    with torch.no_grad():
        ref = mod(*args)

    base = os.path.join(outdir, net["name"] + "_demand")
    torch.export.save(torch.export.export(mod, args), base + ".pt2")
    onnx_fp = base + ".onnx"
    torch.onnx.export(mod, args, onnx_fp,
                      input_names=["g", "dASC", "dIC"], output_names=["OD"],
                      dynamo=True)

    import onnx
    n_nodes = len(onnx.load(onnx_fp).graph.node)
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_fp)
    out = sess.run(None, {"g": g.numpy(), "dASC": dASC.numpy(), "dIC": dIC.numpy()})[0]
    diff = float(np.abs(out - ref.numpy()).max())
    sz = os.path.getsize(onnx_fp) + os.path.getsize(onnx_fp + ".data") \
        if os.path.exists(onnx_fp + ".data") else os.path.getsize(onnx_fp)
    rec = dict(network=net["name"], Z=Z, C=C, P=P,
               onnx_file=os.path.basename(onnx_fp),
               onnx_KB=round(sz / 1e3, 1), compute_nodes=n_nodes,
               inputs=["g", "dASC", "dIC"], output="OD [%d,%d]" % (Z, Z),
               ort_vs_torch_maxdiff=float(f"{diff:.2e}"),
               status="PASS" if diff < 2e-3 and n_nodes > 0 else "FAIL")
    print(f"[{net['name']:16s}] Z={Z:4d} C={C:2d} P={P} -> {rec['onnx_KB']:8.1f} KB, "
          f"{n_nodes} nodes, diff {diff:.1e} {rec['status']}")
    return rec


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, "..", "artifacts")
    os.makedirs(outdir, exist_ok=True)
    recs = []
    for net in NETWORKS:
        try:
            recs.append(export_one(net, outdir))
        except Exception as e:
            print(f"[{net['name']}] FAILED: {str(e)[:120]}")
            recs.append(dict(network=net["name"], status="ERROR", error=str(e)[:200]))
    with open(os.path.join(outdir, "ONNX_MANIFEST.json"), "w") as f:
        json.dump(recs, f, indent=1)
    ok = sum(1 for r in recs if r.get("status") == "PASS")
    print(f"\n{ok}/{len(NETWORKS)} ONNX models ready in artifacts/ "
          f"(manifest: ONNX_MANIFEST.json)")


if __name__ == "__main__":
    main()
