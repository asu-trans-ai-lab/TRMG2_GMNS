"""run_onnx.py — load the pre-built ONNX demand models and time inference.

No export, no torch needed — just onnxruntime + numpy. Loads each
artifacts/<net>_demand.onnx, runs it at random theta, and reports the
inference time. This is the zero-setup way to test the ONNX models on the Mac.

  python run_onnx.py            # all networks
  python run_onnx.py trmg2      # one
"""
import glob
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")


def run(fp):
    name = os.path.basename(fp).replace("_demand.onnx", "")
    sess = ort.InferenceSession(fp, providers=ort.get_available_providers())
    # infer theta shapes from the model inputs
    shapes = {i.name: i.shape for i in sess.get_inputs()}
    P = shapes["g"][0]
    C = shapes["dASC"][1]
    rng = np.random.default_rng(0)
    feed = {"g": rng.uniform(0.7, 1.3, P).astype(np.float32),
            "dASC": rng.uniform(-0.4, 0.4, (P, C)).astype(np.float32),
            "dIC": rng.uniform(-0.3, 0.3, (P, C)).astype(np.float32)}
    sess.run(None, feed)                       # warm
    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        od = sess.run(None, feed)[0]
    dt = (time.perf_counter() - t0) / n * 1e3
    return dict(network=name, P=int(P), C=int(C), OD_shape=list(od.shape),
                OD_total=round(float(od.sum())), infer_ms=round(dt, 2),
                providers=sess.get_providers())


def main():
    print("onnxruntime", ort.__version__, "| providers:", ort.get_available_providers())
    which = sys.argv[1] if len(sys.argv) > 1 else None
    files = sorted(glob.glob(os.path.join(ART, "*_demand.onnx")))
    if which:
        files = [f for f in files if which in os.path.basename(f)]
    recs = []
    for fp in files:
        try:
            r = run(fp)
            recs.append(r)
            print(f"  {r['network']:16s} OD {r['OD_shape']}  infer {r['infer_ms']:8.2f} ms")
        except Exception as e:
            print(f"  {os.path.basename(fp)}: ERROR {str(e)[:100]}")
    with open(os.path.join(ART, "onnx_inference_times.json"), "w") as f:
        json.dump(recs, f, indent=1)
    print("-> artifacts/onnx_inference_times.json")


if __name__ == "__main__":
    main()
