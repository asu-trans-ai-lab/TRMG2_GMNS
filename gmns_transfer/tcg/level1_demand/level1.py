"""Level 1 — demand operator: theta -> OD [Z,Z]. The ONNX sweet spot.

Exports live in ../common/export_all.py (all networks) and ../common/export_onnx.py
(one scale). Pre-built artifacts in ../artifacts/. Run those; this file is the
level marker + a quick torch check.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from demand_graph import synth_world, DemandLayers  # noqa
import torch, numpy as np  # noqa
if __name__ == "__main__":
    W = synth_world(60, 4, 2); mod = DemandLayers(W).eval()
    rng = np.random.default_rng(0)
    od = mod(torch.tensor(rng.uniform(.8,1.2,2),dtype=torch.float32),
             torch.zeros(2,4), torch.zeros(2,4))
    print(f"Level 1 demand: OD {tuple(od.shape)}, total {float(od.sum()):,.0f}")
    print("ONNX export: cd ../common && python export_all.py  (-> ../artifacts/)")
