"""GPU-readiness benchmark of the demand layers (L1-L4).

This machine has torch CPU-only (no CUDA/MPS), so we cannot execute on a GPU.
What we CAN do, and what is decision-relevant:
  1. verify the module is DEVICE-PORTABLE (.to(dev) runs unchanged) — the
     only thing that has to be true for the same file to run on a GPU box;
  2. benchmark eager vs torch.compile (inductor) forward and forward+backward
     at 3 scales up to TRMG2 size (3147 zones, 12 clusters, 8 purposes);
     torch.compile fuses the softmax/LSE/einsum stack — that fusion IS the
     bulk of the GPU win, so the compile speedup here is a CPU-side proxy /
     lower bound for the GPU benefit;
  3. profile where time goes (the memory-bandwidth-bound Z^2 softmax stack),
     which is exactly what a GPU accelerates.

Only tensor SHAPES matter for timing, so we use synthetic frozen world tensors
at each scale (no kernel needed) — the arithmetic is identical to the real
demand layers.

Output: 05_gpu_training/gpu_bench.json
"""
import json
import os
import time

import numpy as np
import torch

torch.set_default_dtype(torch.float32)
DEV = "cuda" if torch.cuda.is_available() else (
    "mps" if getattr(torch.backends, "mps", None)
    and torch.backends.mps.is_available() else "cpu")

SCALES = [
    dict(name="sioux_falls", Z=24, C=3, P=2),
    dict(name="chicago_sketch", Z=386, C=6, P=2),
    dict(name="trmg2_scale", Z=3147, C=12, P=8),
]


def synth_world(Z, C, P, dev, seed=0):
    rng = np.random.default_rng(seed)
    member = torch.as_tensor(rng.integers(0, C, Z), dtype=torch.long, device=dev)
    logM = torch.full((Z, C), float("-inf"), device=dev)
    logM[torch.arange(Z, device=dev), member] = 0.0
    onehot = torch.zeros(Z, C, device=dev)
    onehot[torch.arange(Z, device=dev), member] = 1.0
    W = dict(
        member=member, logM=logM, onehot=onehot,
        t=torch.as_tensor(rng.uniform(3, 60, (Z, Z)), dtype=torch.float32, device=dev),
        lnE=torch.as_tensor(rng.uniform(4, 10, Z), dtype=torch.float32, device=dev),
        pop=torch.as_tensor(rng.uniform(100, 5000, Z), dtype=torch.float32, device=dev),
        th_c=torch.as_tensor(np.linspace(0.7, 0.95, C), dtype=torch.float32, device=dev),
        rate=torch.as_tensor(rng.uniform(0.4, 0.9, P), dtype=torch.float32, device=dev),
        sizec=torch.as_tensor(rng.uniform(0.5, 0.9, P), dtype=torch.float32, device=dev),
        timec=torch.as_tensor(-rng.uniform(0.08, 0.2, P), dtype=torch.float32, device=dev),
        share=torch.as_tensor(rng.uniform(0.6, 0.8, P), dtype=torch.float32, device=dev),
        paf=torch.as_tensor(rng.uniform(0.5, 0.7, P), dtype=torch.float32, device=dev),
        Z=Z, C=C, P=P)
    return W


def demand_forward(g, dASC, dIC, W):
    Z, C, P = W["Z"], W["C"], W["P"]
    mem = W["member"]
    ODt = torch.zeros(Z, Z, device=g.device)
    for p in range(P):
        Prod = W["pop"] * W["rate"][p] * g[p]
        U = W["sizec"][p] * W["lnE"][None, :] + W["timec"][p] * W["t"]
        Uth = U / W["th_c"][mem][None, :]
        masked = Uth[:, :, None] + W["logM"][None, :, :]
        W_ic = W["th_c"][None, :] * torch.logsumexp(masked, dim=1)
        V = dASC[p][None, :] + dIC[p][None, :] * W["onehot"] + W_ic
        Q = torch.softmax(V, dim=1)
        R3 = torch.softmax(masked, dim=1)
        idx = mem[None, :].expand(Z, Z)
        R = torch.gather(R3, 2, idx[:, :, None])[:, :, 0]
        T = Prod[:, None] * Q[:, mem] * R
        X = W["paf"][p] * T + (1 - W["paf"][p]) * T.T
        ODt = ODt + W["share"][p] * X
    return ODt


def timeit(fn, n, warm=2):
    for _ in range(warm):
        fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def bench_scale(cfg):
    Z, C, P = cfg["Z"], cfg["C"], cfg["P"]
    W = synth_world(Z, C, P, DEV)
    g = torch.ones(P, device=DEV, requires_grad=True)
    dASC = torch.zeros(P, C, device=DEV, requires_grad=True)
    dIC = torch.zeros(P, C, device=DEV, requires_grad=True)

    def fwd():
        return demand_forward(g, dASC, dIC, W)

    def fwd_bwd():
        for t_ in (g, dASC, dIC):
            t_.grad = None
        demand_forward(g, dASC, dIC, W).sum().backward()

    n = 50 if Z <= 400 else 10
    t_fwd = timeit(fwd, n)
    t_fb = timeit(fwd_bwd, n)

    # torch.compile (inductor fusion)
    compiled_ok, t_fwd_c, t_fb_c = True, None, None
    try:
        cfwd = torch.compile(demand_forward, fullgraph=False)

        def cf():
            return cfwd(g, dASC, dIC, W)

        def cfb():
            for t_ in (g, dASC, dIC):
                t_.grad = None
            cfwd(g, dASC, dIC, W).sum().backward()
        t_fwd_c = timeit(cf, n, warm=3)
        t_fb_c = timeit(cfb, n, warm=3)
    except Exception as e:
        compiled_ok = False
        t_fwd_c = str(e)[:80]

    out = dict(name=cfg["name"], Z=Z, C=C, P=P, device=DEV,
               od_cells=Z * Z,
               t_forward_ms=round(t_fwd * 1e3, 3),
               t_fwd_bwd_ms=round(t_fb * 1e3, 3))
    if compiled_ok:
        out.update(t_forward_compiled_ms=round(t_fwd_c * 1e3, 3),
                   t_fwd_bwd_compiled_ms=round(t_fb_c * 1e3, 3),
                   compile_speedup_fwd=round(t_fwd / t_fwd_c, 2),
                   compile_speedup_fwd_bwd=round(t_fb / t_fb_c, 2))
    else:
        out["compile_error"] = t_fwd_c
    return out


def main():
    print(f"device: {DEV}  (torch {torch.__version__}, "
          f"cuda_available={torch.cuda.is_available()})")
    print("NOTE: CPU-only build -> compile speedup is a CPU proxy/lower bound "
          "for the GPU fusion win.\n")
    results = []
    for cfg in SCALES:
        r = bench_scale(cfg)
        results.append(r)
        print(json.dumps(r, indent=1))
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, "..", "results")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "gpu_bench.json"), "w") as f:
        json.dump(dict(device=DEV, torch=torch.__version__,
                       cuda_available=torch.cuda.is_available(),
                       results=results), f, indent=1)
    print("-> gpu_bench.json")


if __name__ == "__main__":
    main()
