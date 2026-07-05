"""tcg_benchmark.py — computational-efficiency + guard-rail study of the
layered 4-step computational graph at increasing scale:

    Sioux Falls (24 z)  ->  Chicago Sketch (387 z)  ->  Chicago Regional (~1.8k z)
    [TRMG2 3,147 z row added from the regional pipeline once its 4-period
     operators are extracted]

Per dataset:
  1. TAPLite kernel run with route_output=1 (TAPLITE_ROUTE_VOL_MIN=0.01)
     -> pre-assignment results (OD -> path -> link).
  2. Extract Delta/A/Pi operators; validate Pi @ od == kernel loaded volumes.
  3. Attach the first-3-steps layers (generation, nested DC over C balanced
     geographic clusters, directionality/mode map) with theta =
     (g[2], dASC[2,C], dIC[2,C]); prior theta0 = neutral ("original coeffs").
  4. CALIBRATE to the BASE-YEAR LOADED volumes (the kernel's own link flows),
     twice:  (a) NO RAILS  (lambda = 0, wide bounds)
             (b) RAILS     (trust region lambda*||theta - theta0||^2 + bounds)
     Report %RMSE before/after and the parameter drift ||theta - theta0||_inf
     for both — the rails must bound drift with modest fit cost.
  5. Timings: kernel, extraction, forward, backward(+forward), optimizer.

Output: review/tcg_scaling.json + review/tcg_scaling.md
"""
import csv
import json
import os
import shutil
import time
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "C:/source_codes/0_source_code_new/dtalite_with_taplite_Cpp_kernel"
DATA = os.path.join(ROOT, "kernel", "data_sets")
KERNEL_EXE = os.environ.get("TAPLITE_EXE", ROOT + "/bin/DTALite.exe")
os.environ["TAPLITE_ROUTE_VOL_MIN"] = "0.01"

RATE = np.array([0.55, 0.85])
TIMECOEF = np.array([-0.09, -0.16])
SIZECOEF = np.array([0.8, 0.65])
SHARE = np.array([0.75, 0.62])
PAF = np.array([0.65, 0.5])

CASES = [
    dict(name="sioux_falls", src="02_Sioux_Falls", C=3, iters=25),
    dict(name="chicago_sketch", src="03_chicago_sketch", C=6, iters=25),
    dict(name="chicago_regional", src="04_chicago_regional", C=10, iters=20),
]

SETTINGS = ("number_of_iterations,number_of_processors,demand_period_starting_hours,"
            "demand_period_ending_hours,base_demand_mode,route_output,log_file,"
            "odme_mode,odme_vmt\n{iters},8,7,8,0,1,0,0,0\n")
MODETYPE = "mode_type_id,mode_type,name,vot,pce,occ,demand_file\n1,sov,SOV,10,1,1,demand.csv\n"


def prep_and_run(case):
    wd = os.path.join(HERE, f"bench_{case['name']}")
    os.makedirs(wd, exist_ok=True)
    src = os.path.join(DATA, case["src"])
    for fn in ("node.csv", "link.csv", "demand.csv"):
        shutil.copy(os.path.join(src, fn), wd)
    with open(os.path.join(wd, "settings.csv"), "w") as f:
        f.write(SETTINGS.format(iters=case["iters"]))
    mt = os.path.join(src, "mode_type.csv")
    if os.path.exists(mt):
        shutil.copy(mt, wd)
    else:
        with open(os.path.join(wd, "mode_type.csv"), "w") as f:
            f.write(MODETYPE)
    import pytaplite
    t0 = time.time()
    pytaplite.assign(wd, exe=KERNEL_EXE, prefer_inproc=False)
    return wd, time.time() - t0


def extract(wd):
    t0 = time.time()
    lidx = {}
    with open(os.path.join(wd, "link.csv"), encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        idc = "link_id" if "link_id" in rd.fieldnames else None
        for k, r in enumerate(rd):
            lidx[int(r[idc]) if idc else k + 1] = k
    od_vol = {}
    fp = os.path.join(wd, "route_assignment.csv")
    with open(fp) as f:
        for r in csv.DictReader(f):
            try:
                od = (int(r["o_zone_id"]), int(r["d_zone_id"]))
                od_vol[od] = od_vol.get(od, 0.0) + float(r["volume"] or 0)
            except (ValueError, KeyError):
                continue
    od_col = {od: k for k, od in enumerate(sorted(od_vol))}
    rr, cc, vv, ar, ac = [], [], [], [], []
    npth = 0
    with open(fp) as f:
        for r in csv.DictReader(f):
            try:
                od = (int(r["o_zone_id"]), int(r["d_zone_id"]))
                vol = float(r["volume"] or 0)
            except (ValueError, KeyError):
                continue
            if vol <= 0:
                continue
            p = npth; npth += 1
            rr.append(p); cc.append(od_col[od]); vv.append(vol / od_vol[od])
            for lid in r["link_ids"].split(";"):
                if lid and int(lid) in lidx:
                    ar.append(lidx[int(lid)]); ac.append(p)
    Delta = sp.csr_matrix((vv, (rr, cc)), shape=(npth, len(od_col)))
    A = sp.csr_matrix((np.ones(len(ar)), (ar, ac)), shape=(len(lidx), npth))
    Pi = (A @ Delta).tocsr()
    t_ex = time.time() - t0
    # loaded volumes (base year results) + validation
    key = {}
    with open(os.path.join(wd, "link.csv"), encoding="utf-8-sig") as f:
        for k, r in enumerate(csv.DictReader(f)):
            key[(int(r["from_node_id"]), int(r["to_node_id"]))] = k
    v_k = np.zeros(len(lidx))
    with open(os.path.join(wd, "link_performance.csv")) as f:
        for r in csv.DictReader(f):
            kk = key.get((int(r["from_node_id"]), int(r["to_node_id"])))
            if kk is not None:
                v_k[kk] = float(r["volume"] or 0)
    odv = np.zeros(len(od_col))
    for od, v in od_vol.items():
        odv[od_col[od]] = v
    v_re = Pi @ odv
    mask = v_k > 0
    r2 = 1 - ((v_re - v_k)[mask] ** 2).sum() / max(
        ((v_k[mask] - v_k[mask].mean()) ** 2).sum(), 1e-9)
    return dict(Pi=Pi, od_col=od_col, n_links=len(lidx), n_paths=npth,
                loaded=v_k, r2=float(r2), t_extract=t_ex)


def build_world(wd, X, C):
    od_col = X["od_col"]
    zones = sorted({o for o, _ in od_col} | {d for _, d in od_col})
    Z = len(zones)
    zpos = {z: i for i, z in enumerate(zones)}
    dem = np.zeros((Z, Z))
    with open(os.path.join(wd, "demand.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            o, d = int(r["o_zone_id"]), int(r["d_zone_id"])
            if o in zpos and d in zpos:
                dem[zpos[o], zpos[d]] += float(r["volume"] or 0)
    pop, emp = dem.sum(1), dem.sum(0)
    lnE = np.log(np.maximum(emp, 1.0))
    # balanced geographic clusters by x coordinate
    xs = {}
    with open(os.path.join(wd, "node.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("zone_id") and r["zone_id"] not in ("", "0"):
                xs[int(float(r["zone_id"]))] = float(r["x_coord"])
    order = np.argsort([xs.get(z, 0.0) for z in zones])
    member = np.zeros(Z, int)
    member[order] = np.minimum(np.arange(Z) * C // Z, C - 1)
    theta_c = np.linspace(0.7, 0.95, C)
    # fftt skim
    nid = {}
    with open(os.path.join(wd, "node.csv"), encoding="utf-8-sig") as f:
        for k, r in enumerate(csv.DictReader(f)):
            nid[int(r["node_id"])] = k
    rows, cols, w = [], [], []
    with open(os.path.join(wd, "link.csv"), encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        fftt_col = next((c for c in rd.fieldnames
                         if c.lower() in ("vdf_fftt", "fftt")), None)
        for r in rd:
            rows.append(nid[int(r["from_node_id"])])
            cols.append(nid[int(r["to_node_id"])])
            t = float(r.get(fftt_col) or 0) if fftt_col else 0
            if t <= 0:
                t = float(r.get("length") or 1) / max(
                    float(r.get("free_speed") or 30), 1) * 60
            w.append(max(t, 1e-3))
    G = sp.csr_matrix((w, (rows, cols)), shape=(len(nid),) * 2)
    src = [nid[z] for z in zones]
    t = dijkstra(G, directed=True, indices=src)[:, src]
    t = np.where(np.isfinite(t), t, 999.0)
    np.fill_diagonal(t, 1e4)
    sr, sc, sd = [], [], []
    for (o, d), c in od_col.items():
        sr.append(zpos[o]); sc.append(zpos[d]); sd.append(c)
    return dict(Z=Z, C=C, pop=pop, lnE=lnE, member=member, theta_c=theta_c,
                t=t, Pi=X["Pi"], n_od=X["Pi"].shape[1],
                sel=(np.array(sr), np.array(sc), np.array(sd)))


# ---- forward / loss+grad (identical math to tcg_prototype, C generalized) ----
def forward(th, W):
    P = len(RATE)
    Z, C, mem = W["Z"], W["C"], W["member"]
    onehot = np.eye(C)[mem]
    ODt = np.zeros((Z, Z))
    caches = []
    for p in range(P):
        Prod = W["pop"] * RATE[p] * th["g"][p]
        U = SIZECOEF[p] * W["lnE"][None, :] + TIMECOEF[p] * W["t"]
        Uc = U / W["theta_c"][mem][None, :]
        m = np.full((Z, C), -1e30)
        for c in range(C):
            m[:, c] = Uc[:, mem == c].max(1)
        E = np.exp(Uc - m[:, mem])
        S = np.zeros((Z, C))
        for c in range(C):
            S[:, c] = E[:, mem == c].sum(1)
        logsum = W["theta_c"][None, :] * (m + np.log(np.maximum(S, 1e-300)))
        V = th["dASC"][p][None, :] + th["dIC"][p][None, :] * onehot + logsum
        EV = np.exp(V - V.max(1, keepdims=True))
        Pc = EV / EV.sum(1, keepdims=True)
        Pjc = E / S[:, mem]
        T = Prod[:, None] * Pc[:, mem] * Pjc
        ODt += SHARE[p] * (PAF[p] * T + (1 - PAF[p]) * T.T)
        caches.append(dict(Prod=Prod, Pc=Pc, T=T))
    sr, sc, sd = W["sel"]
    odv = np.zeros(W["n_od"])
    odv[sd] = ODt[sr, sc]
    return dict(v=W["Pi"] @ odv, caches=caches, ODt=ODt)


def loss_grad(x, W, counts, prior, lam):
    P, C = len(RATE), W["C"]
    th = dict(g=x[:P], dASC=x[P:P + P * C].reshape(P, C),
              dIC=x[P + P * C:].reshape(P, C))
    fw = forward(th, W)
    wgt = 1.0 / np.maximum(counts, 10.0)
    r = (fw["v"] - counts) * wgt
    L = 0.5 * float(r @ r) + 0.5 * lam * float((x - prior) @ (x - prior))
    g_v = r * wgt
    g_odv = W["Pi"].T @ g_v
    sr, sc, sd = W["sel"]
    G_OD = np.zeros_like(fw["ODt"])
    G_OD[sr, sc] = g_odv[sd]
    mem = W["member"]
    Z = W["Z"]
    onehot = np.eye(C)[mem]
    g_out = dict(g=np.zeros(P), dASC=np.zeros((P, C)), dIC=np.zeros((P, C)))
    for p in range(P):
        ch = fw["caches"][p]
        G_T = SHARE[p] * (PAF[p] * G_OD + (1 - PAF[p]) * G_OD.T)
        GT_T = G_T * ch["T"]
        y = np.zeros((Z, C))
        for cc in range(C):
            y[:, cc] = GT_T[:, mem == cc].sum(1)
        Y = y.sum(1, keepdims=True)
        G_V = y - ch["Pc"] * Y
        g_out["dASC"][p] = G_V.sum(0)
        g_out["dIC"][p] = (G_V * onehot).sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            gP = np.where(ch["Prod"][:, None] > 0, GT_T / ch["Prod"][:, None], 0)
        g_out["g"][p] = float((gP.sum(1) * W["pop"] * RATE[p]).sum())
    grad = np.concatenate([g_out["g"], g_out["dASC"].ravel(), g_out["dIC"].ravel()])
    return L, grad + lam * (x - prior)


def prmse(v, c):
    m = c > 0
    n = int(m.sum())
    return float(np.sqrt(((v - c)[m] ** 2).sum() / max(n - 1, 1)) / (c[m].mean()) * 100)


def run_case(case):
    print(f"=== {case['name']} ===")
    wd, t_kernel = prep_and_run(case)
    X = extract(wd)
    W = build_world(wd, X, case["C"])
    P, C = len(RATE), case["C"]
    counts = X["loaded"]                      # BASE-YEAR LOADED RESULTS
    prior = np.concatenate([np.ones(P), np.zeros(2 * P * C)])   # original coeffs

    # timings
    th0 = dict(g=np.ones(P), dASC=np.zeros((P, C)), dIC=np.zeros((P, C)))
    t0 = time.time(); fw0 = forward(th0, W); t_fwd = time.time() - t0
    t0 = time.time(); loss_grad(prior, W, counts, prior, 0.0); t_lg = time.time() - t0
    rmse0 = prmse(fw0["v"], counts)

    def calibrate(lam, bound):
        bounds = [(0.3, 3.0)] * P + [(-bound, bound)] * (2 * P * C)
        t0 = time.time()
        res = minimize(loss_grad, prior, args=(W, counts, prior, lam), jac=True,
                       method="L-BFGS-B", bounds=bounds,
                       options=dict(maxiter=400, ftol=1e-12))
        th = dict(g=res.x[:P], dASC=res.x[P:P + P * C].reshape(P, C),
                  dIC=res.x[P + P * C:].reshape(P, C))
        v = forward(th, W)["v"]
        return dict(prmse=round(prmse(v, counts), 2), iters=int(res.nit),
                    secs=round(time.time() - t0, 1),
                    drift_max=round(float(np.abs(res.x - prior).max()), 3),
                    drift_mean=round(float(np.abs(res.x - prior).mean()), 3))

    free = calibrate(lam=0.0, bound=5.0)      # no rails
    rail = calibrate(lam=0.5, bound=1.5)      # guard rails

    out = dict(name=case["name"], zones=W["Z"], links=X["n_links"],
               od_pairs=W["n_od"], paths=X["n_paths"],
               pi_nnz=int(W["Pi"].nnz), pi_r2=round(X["r2"], 5),
               C=C, n_params=P + 2 * P * C,
               t_kernel=round(t_kernel, 1), t_extract=round(X["t_extract"], 1),
               t_forward=round(t_fwd, 3), t_loss_grad=round(t_lg, 3),
               prmse_before=round(rmse0, 2),
               no_rails=free, rails_lam0p5=rail)
    print(json.dumps(out, indent=1))
    return out


def main():
    results = [run_case(c) for c in CASES]
    os.makedirs(os.path.join(HERE, "review"), exist_ok=True)
    with open(os.path.join(HERE, "review", "tcg_scaling.json"), "w") as f:
        json.dump(results, f, indent=1)
    with open(os.path.join(HERE, "review", "tcg_scaling.md"), "w") as f:
        f.write("# Computational-graph scaling benchmark\n\n")
        f.write("Calibration target = base-year LOADED link volumes (kernel UE). "
                "Prior = neutral theta (the 'original coefficients'). "
                "Rails = trust region lam=0.5 + bounds +/-1.5.\n\n")
        f.write("| case | zones | links | od | paths | Pi R2 | t_kernel | t_extract "
                "| t_fwd | t_fwd+bwd | %RMSE start | no-rails %RMSE (drift_max) "
                "| rails %RMSE (drift_max) |\n")
        f.write("|" + "---|" * 13 + "\n")
        for r in results:
            f.write(f"| {r['name']} | {r['zones']} | {r['links']} | {r['od_pairs']:,} "
                    f"| {r['paths']:,} | {r['pi_r2']} | {r['t_kernel']}s "
                    f"| {r['t_extract']}s | {r['t_forward']}s | {r['t_loss_grad']}s "
                    f"| {r['prmse_before']} "
                    f"| {r['no_rails']['prmse']} ({r['no_rails']['drift_max']}) "
                    f"| {r['rails_lam0p5']['prmse']} ({r['rails_lam0p5']['drift_max']}) |\n")
        f.write("\nTRMG2 (3,147 zones) row: pending 4-period operator extraction "
                "(kernel patched; AM in flight). Improvement plans: see PERFORMANCE.md "
                "P0-P2 (f32, DemandLite fused C++ fwd+bwd, sparse pi export, "
                "concurrent periods).\n")
    print("-> review/tcg_scaling.{json,md}")


if __name__ == "__main__":
    main()
