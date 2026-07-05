"""tcg_consistency.py — end-to-end representation-consistency gate.

Verifies that the SAME theta produces the SAME demand and the SAME assigned
link volumes across every representation used in the pipeline:

    numpy analytic  (tcg_prototype.forward)        <- the reference math
        ==  PyTorch (export_onnx.DemandLayers)     <- the trainable NN
        ==  ONNX    (onnxruntime session)          <- the exported artifact
    then  v = Pi @ vec(OD)  agrees for all three   <- back into the big loop

and that the gradient path is consistent (torch autograd == numpy analytic
adjoint), so a calibration step taken in ANY representation is the same step.

Tolerances: numpy is f64, torch/onnx are f32 -> compare at f32 precision
(rel/abs ~1e-3). Run: python tcg_consistency.py
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..",
                                "tcg_research", "01_exports"))
import tcg_prototype as tp  # noqa: E402


def main():
    # ---- build one world + Pi (Sioux Falls) ----
    Pi, od_col, n_links, pi_err = tp.run_kernel_and_extract()
    W = tp.build_world(Pi, od_col)
    P, C = len(tp.RATE), W["C"]

    rng = np.random.default_rng(3)
    g = rng.uniform(0.8, 1.2, P)
    dASC = rng.uniform(-0.4, 0.4, (P, C)); dASC[:, 0] = 0.0
    dIC = rng.uniform(-0.3, 0.3, (P, C))
    th = dict(g=g, dASC=dASC, dIC=dIC)

    # ---- (1) numpy reference ----
    OD_np = tp.forward(th, W)["ODt"]

    # ---- (2) PyTorch ----
    from export_onnx import DemandLayers
    mod = DemandLayers(W).eval()
    gt = torch.tensor(g, dtype=torch.float32)
    at = torch.tensor(dASC, dtype=torch.float32)
    it = torch.tensor(dIC, dtype=torch.float32)
    with torch.no_grad():
        OD_pt = mod(gt, at, it).numpy()

    # ---- (3) ONNX (export fresh for this world, then run) ----
    onnx_fp = os.path.join(HERE, "_consistency_sf.onnx")
    torch.onnx.export(mod, (gt, at, it), onnx_fp,
                      input_names=["g", "dASC", "dIC"], output_names=["OD"],
                      dynamo=True)
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_fp)
    OD_ox = sess.run(None, {"g": g.astype(np.float32),
                            "dASC": dASC.astype(np.float32),
                            "dIC": dIC.astype(np.float32)})[0]

    # ---- OD agreement ----
    d_np_pt = float(np.abs(OD_np - OD_pt).max())
    d_pt_ox = float(np.abs(OD_pt - OD_ox).max())
    scale = OD_np.max()

    # ---- v = Pi @ vec(OD) for each -> back into the big loop ----
    sr, sc, sd = W["sel"]

    def to_v(OD):
        odv = np.zeros(W["n_od"])
        odv[sd] = OD[sr, sc]
        return Pi @ odv
    v_np, v_pt, v_ox = to_v(OD_np), to_v(OD_pt), to_v(OD_ox)
    dv_np_pt = float(np.abs(v_np - v_pt).max())
    dv_pt_ox = float(np.abs(v_pt - v_ox).max())
    vscale = max(v_np.max(), 1.0)

    # ---- gradient consistency: numpy analytic adjoint vs torch autograd ----
    counts = v_np.copy()
    x0 = tp.pack(dict(g=np.ones(P), dASC=np.zeros((P, C)), dIC=np.zeros((P, C))))
    _, g_np = tp.loss_grad(x0, W, counts)
    # torch autograd of the same loss
    gt2 = torch.ones(P, requires_grad=True)
    at2 = torch.zeros(P, C, requires_grad=True)
    it2 = torch.zeros(P, C, requires_grad=True)
    OD_t = mod(gt2, at2, it2)
    odv_t = torch.zeros(W["n_od"])
    odv_t[torch.as_tensor(sd)] = OD_t[torch.as_tensor(sr), torch.as_tensor(sc)]
    coo = Pi.tocoo()
    Pit = torch.sparse_coo_tensor(np.vstack([coo.row, coo.col]),
                                  coo.data.astype(np.float32), coo.shape).coalesce()
    v_t = torch.sparse.mm(Pit, odv_t[:, None].float())[:, 0]
    wgt = torch.as_tensor(1.0 / np.maximum(counts, 10.0)).float()
    L_t = 0.5 * (((v_t - torch.as_tensor(counts).float()) * wgt) ** 2).sum()
    L_t.backward()
    g_t = np.concatenate([gt2.grad.numpy(), at2.grad.numpy().ravel(),
                          it2.grad.numpy().ravel()])
    denom = np.maximum(np.abs(g_np), np.abs(g_t))
    grad_rel = float((np.abs(g_t - g_np) / np.where(denom > 1e-6, denom, 1.0)).max())

    os.remove(onnx_fp)
    ok = (d_np_pt / scale < 2e-3 and d_pt_ox / scale < 1e-4
          and dv_np_pt / vscale < 2e-3 and dv_pt_ox / vscale < 1e-4
          and grad_rel < 1e-2)
    report = dict(
        world="sioux_falls", Z=W["Z"], C=C, P=P, pi_valid_err=round(pi_err, 4),
        OD_numpy_vs_torch_relmax=float(f"{d_np_pt/scale:.2e}"),
        OD_torch_vs_onnx_relmax=float(f"{d_pt_ox/scale:.2e}"),
        v_numpy_vs_torch_relmax=float(f"{dv_np_pt/vscale:.2e}"),
        v_torch_vs_onnx_relmax=float(f"{dv_pt_ox/vscale:.2e}"),
        grad_numpy_vs_torch_relmax=float(f"{grad_rel:.2e}"),
        note="numpy f64 vs torch/onnx f32 -> ~1e-3 is exact agreement; "
             "onnx vs torch ~1e-4 (both f32).",
        VERDICT="PASS" if ok else "FAIL")
    os.makedirs(os.path.join(HERE, "review"), exist_ok=True)
    with open(os.path.join(HERE, "review", "tcg_consistency.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
