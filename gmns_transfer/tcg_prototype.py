"""tcg_prototype.py — SMALL-SCALE VERIFICATION of the forward+backward
propagation pipeline for the 4-step process (Sioux Falls, 24 zones).

Purpose: prove the math and the matrix-form architecture end-to-end before
the regional (TRMG2) case:

  1. Run the TAPLite kernel ONCE on Sioux Falls -> route_assignment.csv
     = the pre-assignment results (OD -> path -> link).
  2. Extract the matrix operators Delta [paths x od], A [links x paths],
     Pi = A @ Delta [links x od]  — assignment enters the graph ONLY as Pi.
  3. Mini 4-step forward with the SAME layer structure as the regional file:
       L1 generation  P[p,i] = pop_i * rate_p * g_p
       L2 nested DC   T[p]   = P * Pr(j|i; U_p, dASC, dIC)  over district
                               clusters (theta_c fixed = frozen spine)
       L34 OD         OD     = sum_p share_p * (f*T + (1-f)*T^T)   (linear)
       L5 assignment  v      = Pi @ vec(OD)
       L6 loss        L      = 0.5 sum ((v - c)/max(c,10))^2
  4. ANALYTIC BACKWARD through every layer (cluster-nest softmax adjoint,
     transpose adjoint of the directionality map, Pi^T for assignment).
  5. GRADCHECK vs central finite differences — hard gate: worst rel err <= 1e-4.
  6. RECOVERY: counts generated at a hidden theta*; start from neutral theta0;
     L-BFGS-B with analytic gradients must recover theta*.

Pass criteria printed at the end. Run:  python tcg_prototype.py
"""
import csv
import json
import os
import shutil
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "C:/source_codes/0_source_code_new/dtalite_with_taplite_Cpp_kernel"
SRC = os.path.join(ROOT, "kernel", "data_sets", "02_Sioux_Falls")
PROTO = os.path.join(HERE, "proto_sf")
KERNEL_EXE = os.environ.get("TAPLITE_EXE", ROOT + "/bin/DTALite.exe")

RATE = np.array([0.55, 0.85])          # frozen person-trip rates, 2 purposes
TIMECOEF = np.array([-0.09, -0.16])    # frozen DC time coefficients
SIZECOEF = np.array([0.8, 0.65])       # frozen coef on ln(size)
SHARE = np.array([0.75, 0.62])         # frozen auto shares per purpose
PAF = np.array([0.65, 0.5])            # frozen PA->OD directionality
THETA_C = None                          # cluster thetas, set after clusters known


# ---------------------------------------------------------------- kernel + Pi
def run_kernel_and_extract():
    os.makedirs(PROTO, exist_ok=True)
    for fn in ("node.csv", "link.csv", "demand.csv", "mode_type.csv"):
        shutil.copy(os.path.join(SRC, fn), PROTO)
    with open(os.path.join(SRC, "settings.csv")) as f:
        rows = list(csv.reader(f))
    hdr = rows[0]
    if "route_output" in hdr:
        k = hdr.index("route_output")
        rows[1][k] = "1"
    with open(os.path.join(PROTO, "settings.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)
    import pytaplite
    pytaplite.assign(PROTO, exe=KERNEL_EXE, prefer_inproc=False)

    lidx = {}
    with open(os.path.join(PROTO, "link.csv"), encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        idcol = "link_id" if "link_id" in rd.fieldnames else None
        for k, r in enumerate(rd):
            lidx[int(r[idcol]) if idcol else k + 1] = k
    od_vol = {}
    with open(os.path.join(PROTO, "route_assignment.csv")) as f:
        for r in csv.DictReader(f):
            try:
                od = (int(r["o_zone_id"]), int(r["d_zone_id"]))
                od_vol[od] = od_vol.get(od, 0.0) + float(r["volume"] or 0)
            except (ValueError, KeyError):
                continue
    od_list = sorted(od_vol)
    od_col = {od: k for k, od in enumerate(od_list)}
    d_r, d_c, d_v, a_r, a_c = [], [], [], [], []
    npth = 0
    with open(os.path.join(PROTO, "route_assignment.csv")) as f:
        for r in csv.DictReader(f):
            try:
                od = (int(r["o_zone_id"]), int(r["d_zone_id"]))
                vol = float(r["volume"] or 0)
            except (ValueError, KeyError):
                continue
            if vol <= 0:
                continue
            p = npth
            npth += 1
            d_r.append(p); d_c.append(od_col[od]); d_v.append(vol / od_vol[od])
            for lid in r["link_ids"].split(";"):
                if lid and int(lid) in lidx:
                    a_r.append(lidx[int(lid)]); a_c.append(p)
    Delta = sp.csr_matrix((d_v, (d_r, d_c)), shape=(npth, len(od_list)))
    A = sp.csr_matrix((np.ones(len(a_r)), (a_r, a_c)), shape=(len(lidx), npth))
    Pi = (A @ Delta).tocsr()
    # validation: Pi @ od == kernel volumes
    odv = np.zeros(len(od_list))
    for od, v in od_vol.items():
        odv[od_col[od]] = v
    v_re = Pi @ odv
    v_k = np.zeros(len(lidx))
    with open(os.path.join(PROTO, "link_performance.csv")) as f:
        key = {}
        with open(os.path.join(PROTO, "link.csv")) as g:
            for k, r in enumerate(csv.DictReader(g)):
                key[(int(r["from_node_id"]), int(r["to_node_id"]))] = k
        for r in csv.DictReader(f):
            kk = key.get((int(r["from_node_id"]), int(r["to_node_id"])))
            if kk is not None:
                v_k[kk] = float(r["volume"] or 0)
    err = float(np.abs(v_re - v_k).max())
    print(f"[Pi] paths {npth}, od {len(od_list)}, links {len(lidx)}; "
          f"Pi@od vs kernel volumes max|err| = {err:.3f} veh")
    return Pi, od_col, len(lidx), err


# ---------------------------------------------------------------- mini world
def build_world(Pi, od_col):
    zones = sorted({o for o, _ in od_col} | {d for _, d in od_col})
    Z = len(zones)
    zpos = {z: i for i, z in enumerate(zones)}
    # SE proxies from the TNTP demand margins
    dem = np.zeros((Z, Z))
    with open(os.path.join(PROTO, "demand.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            o, d = int(r["o_zone_id"]), int(r["d_zone_id"])
            if o in zpos and d in zpos:
                dem[zpos[o], zpos[d]] = float(r["volume"])
    pop = dem.sum(1)
    emp = dem.sum(0)
    lnE = np.log(np.maximum(emp, 1.0))
    # clusters from node.csv district_id
    dist = {}
    with open(os.path.join(PROTO, "node.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("zone_id"):
                dist[int(r["zone_id"])] = int(float(r.get("district_id") or 1))
    # district_id is degenerate here (23/1 split -> single-member cluster
    # makes its dIC unidentifiable: a real lesson, see report). For a
    # well-conditioned verification use 3 balanced synthetic clusters.
    member = np.array([min(i * 3 // Z, 2) for i in range(Z)])
    C = 3
    theta_c = np.linspace(0.7, 0.95, C)               # frozen nest scales
    # FF time skim
    rows, cols, w = [], [], []
    nid = {}
    with open(os.path.join(PROTO, "node.csv"), encoding="utf-8-sig") as f:
        for k, r in enumerate(csv.DictReader(f)):
            nid[int(r["node_id"])] = k
    with open(os.path.join(PROTO, "link.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(nid[int(r["from_node_id"])])
            cols.append(nid[int(r["to_node_id"])])
            fftt = float(r.get("vdf_fftt") or 0)
            if fftt <= 0:
                fftt = float(r.get("length") or 1) / max(float(r.get("free_speed") or 30), 1) * 60
            w.append(max(fftt, 1e-3))
    G = sp.csr_matrix((w, (rows, cols)), shape=(len(nid),) * 2)
    src = [nid[z] for z in zones]
    t = dijkstra(G, directed=True, indices=src)[:, src]
    t = np.where(np.isfinite(t), t, 999.0)
    np.fill_diagonal(t, 1e4)   # intrazonal excluded (matches regional D1)
    # od selection map for Pi
    sel_r, sel_c, sel_dst = [], [], []
    for (o, d), c in od_col.items():
        sel_r.append(zpos[o]); sel_c.append(zpos[d]); sel_dst.append(c)
    return dict(Z=Z, C=C, pop=pop, lnE=lnE, member=member, theta_c=theta_c,
                t=t, Pi=Pi, n_od=Pi.shape[1],
                sel=(np.array(sel_r), np.array(sel_c), np.array(sel_dst)))


# ---------------------------------------------------------------- theta
def pack(th):
    return np.concatenate([th["g"], th["dASC"].ravel(), th["dIC"].ravel()])


def unpack(x, P, C):
    return dict(g=x[:P],
                dASC=x[P:P + P * C].reshape(P, C),
                dIC=x[P + P * C:].reshape(P, C))


# ---------------------------------------------------------------- forward
def forward(th, W, want_cache=False):
    P = len(RATE)
    Z, C = W["Z"], W["C"]
    mem = W["member"]
    onehot = np.eye(C)[mem]                        # [Z, C]
    caches = []
    ODt = np.zeros((Z, Z))
    for p in range(P):
        Prod = W["pop"] * RATE[p] * th["g"][p]                     # L1
        U = SIZECOEF[p] * W["lnE"][None, :] + TIMECOEF[p] * W["t"]  # frozen
        # --- cluster nest (textbook NL; regional variant differs only here) ---
        Uc = U / W["theta_c"][mem][None, :]        # scale by dest cluster theta
        m = np.full((Z, C), -1e30)
        for c in range(C):
            cols = np.where(mem == c)[0]
            m[:, c] = Uc[:, cols].max(1)
        E = np.exp(Uc - m[:, mem])                 # stabilized
        S = np.zeros((Z, C))
        for c in range(C):
            S[:, c] = E[:, mem == c].sum(1)
        logsum = W["theta_c"][None, :] * (m + np.log(np.maximum(S, 1e-300)))
        icdum = onehot                              # origin-in-cluster [Z,C]
        V = th["dASC"][p][None, :] + th["dIC"][p][None, :] * icdum + logsum
        mV = V.max(1, keepdims=True)
        EV = np.exp(V - mV)
        Pc = EV / EV.sum(1, keepdims=True)         # [Z, C]
        Pjc = E / S[:, mem]                        # [Z, Z] cond. within cluster
        T = Prod[:, None] * Pc[:, mem] * Pjc       # L2
        X = PAF[p] * T + (1 - PAF[p]) * T.T        # L34 directionality
        ODt += SHARE[p] * X
        caches.append(dict(Prod=Prod, Pc=Pc, Pjc=Pjc, T=T))
    sr, sc, sd = W["sel"]
    odv = np.zeros(W["n_od"])
    odv[sd] = ODt[sr, sc]
    v = W["Pi"] @ odv                              # L5
    out = dict(v=v, ODt=ODt, odv=odv)
    if want_cache:
        out["caches"] = caches
    return out


def loss_grad(x, W, counts):
    P, C = len(RATE), W["C"]
    th = unpack(x, P, C)
    fw = forward(th, W, want_cache=True)
    v = fw["v"]
    wgt = 1.0 / np.maximum(counts, 10.0)
    r = (v - counts) * wgt
    L = 0.5 * float(r @ r)
    # ---- backward ----
    g_v = r * wgt                                   # dL/dv
    g_odv = W["Pi"].T @ g_v                         # L5 adjoint
    sr, sc, sd = W["sel"]
    G_OD = np.zeros_like(fw["ODt"])
    G_OD[sr, sc] = g_odv[sd]
    mem = W["member"]
    Z = W["Z"]
    g_out = dict(g=np.zeros(P), dASC=np.zeros((P, C)), dIC=np.zeros((P, C)))
    onehot = np.eye(C)[mem]
    for p in range(P):
        ch = forward.__wrapped__ if False else None  # noqa
        c = fwc = fw["caches"][p]
        G_X = SHARE[p] * G_OD
        G_T = PAF[p] * G_X + (1 - PAF[p]) * G_X.T   # transpose adjoint
        T = fwc["T"]
        # cluster-level adjoint: y[i,c] = sum_{j in c} G_T*T ; dL/dV = y - Pc * Y
        GT_T = G_T * T
        y = np.zeros((Z, C))
        for cc in range(C):
            y[:, cc] = GT_T[:, mem == cc].sum(1)
        Y = y.sum(1, keepdims=True)
        G_V = y - fwc["Pc"] * Y                     # softmax adjoint [Z,C]
        g_out["dASC"][p] = G_V.sum(0)
        g_out["dIC"][p] = (G_V * onehot).sum(0)
        # generation adjoint: dL/dProd_i = sum_j G_T * T / Prod
        with np.errstate(invalid="ignore", divide="ignore"):
            gP = np.where(fwc["Prod"][:, None] > 0, GT_T / fwc["Prod"][:, None], 0)
        g_out["g"][p] = float((gP.sum(1) * W["pop"] * RATE[p]).sum())
    return L, pack(g_out)


# ---------------------------------------------------------------- main
def main():
    Pi, od_col, n_links, pi_err = run_kernel_and_extract()
    W = build_world(Pi, od_col)
    P, C = len(RATE), W["C"]
    print(f"[world] Z={W['Z']} zones, C={C} district clusters, "
          f"{W['n_od']} od pairs, {n_links} links")

    rng = np.random.default_rng(42)
    th_true = dict(g=np.array([1.25, 0.85]),
                   dASC=rng.uniform(-0.5, 0.5, (P, C)),
                   dIC=rng.uniform(-0.3, 0.3, (P, C)))
    # identifiability: softmax invariant to per-row constant -> pin dASC[:,0]=0
    th_true["dASC"][:, 0] = 0.0
    counts = forward(th_true, W)["v"]
    print(f"[truth] synthetic counts on {int((counts > 0).sum())} links, "
          f"total {counts.sum():,.0f}")

    th0 = dict(g=np.ones(P), dASC=np.zeros((P, C)), dIC=np.zeros((P, C)))
    x0 = pack(th0)

    # ---- gradcheck ----
    L0, g0 = loss_grad(x0, W, counts)
    eps = 1e-6
    worst = 0.0
    for k in range(len(x0)):
        xp = x0.copy(); xp[k] += eps
        xm = x0.copy(); xm[k] -= eps
        fd = (loss_grad(xp, W, counts)[0] - loss_grad(xm, W, counts)[0]) / (2 * eps)
        rel = abs(fd - g0[k]) / max(abs(fd), abs(g0[k]), 1e-12)
        worst = max(worst, rel)
    print(f"[gradcheck] {len(x0)} parameters, worst rel err = {worst:.2e} "
          f"({'PASS' if worst <= 1e-4 else 'FAIL'})")

    # ---- recovery ----
    # pin the same gauge on the optimizer side
    bounds = []
    P_, C_ = P, C
    for k in range(len(x0)):
        if P_ <= k < P_ + P_ * C_ and (k - P_) % C_ == 0:
            bounds.append((0.0, 0.0))       # dASC[:,0] pinned
        else:
            bounds.append((-3.0, 3.0)) if k >= P_ else bounds.append((0.2, 3.0))
    t0 = time.time()
    res = minimize(loss_grad, x0, args=(W, counts), jac=True,
                   method="L-BFGS-B", bounds=bounds,
                   options=dict(maxiter=500, ftol=1e-14, gtol=1e-12))
    th_hat = unpack(res.x, P, C)
    err_g = np.abs(th_hat["g"] - th_true["g"]).max()
    # IDENTIFIABLE quantities (softmax gauge): for origin-cluster a, the
    # cluster-utility row q[a, c] = dASC_c + dIC_c * 1[c == a], centered per
    # row. With C=2, raw dASC/dIC have a 1-dim null space per purpose - raw
    # parameter comparison is the WRONG test; row-centered q and the implied
    # trip tables are the right ones.
    def qmat(th):
        q = np.zeros((P, C, C))
        for p in range(P):
            for a in range(C):
                q[p, a] = th["dASC"][p] + th["dIC"][p] * (np.arange(C) == a)
            q[p] -= q[p].mean(axis=1, keepdims=True)
        return q
    err_q = np.abs(qmat(th_hat) - qmat(th_true)).max()
    OD_hat = forward(th_hat, W)["ODt"]
    OD_true = forward(th_true, W)["ODt"]
    err_od = float(np.abs(OD_hat - OD_true).max() / max(OD_true.max(), 1e-9))
    print(f"[recovery] L {L0:.4f} -> {res.fun:.3e} in {res.nit} iters "
          f"({time.time()-t0:.1f}s)")
    print(f"[recovery] g err {err_g:.2e}; identifiable-q err {err_q:.2e}; "
          f"OD-table rel err {err_od:.2e}")
    ok = (pi_err < 0.5 and worst <= 1e-4 and res.fun < 1e-6
          and err_g < 1e-3 and err_q < 1e-3 and err_od < 1e-6)
    print(json.dumps(dict(
        pi_max_err_veh=round(pi_err, 3), gradcheck_worst=float(f"{worst:.2e}"),
        final_loss=float(f"{res.fun:.3e}"), iters=res.nit,
        g_err=float(f"{err_g:.2e}"), identifiable_q_err=float(f"{err_q:.2e}"),
        od_table_rel_err=float(f"{err_od:.2e}"),
        note="params compared via identifiable functionals + OD equivalence "
             "(softmax gauge); degenerate clusters (SF district 23/1) leave dIC "
             "unidentified - the small-scale version of prior-pinned knobs",
        VERDICT="PASS" if ok else "FAIL"), indent=1))


if __name__ == "__main__":
    main()
