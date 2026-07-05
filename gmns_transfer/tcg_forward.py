"""tcg_forward.py — single-file reference: the ENTIRE travel-model forward
propagation, TRMG2-parameterized, from trip generation to assigned link
volumes and the count-comparison loss.

    L(theta) = loss( Assign( TOD( Mode( Distribute( Generate(theta) )))), counts )

Written as explicit LAYERS with tensor shapes documented at every interface,
so that four calibration paradigms can be built on the SAME forward:

  (1) CG          — computational graph: analytic backward per layer
                    (each layer below documents its Jacobian; assemble the
                    chain rule L6 -> L1 exactly as in calib_graph_demo).
  (2) NLP         — pure nonlinear optimization: treat forward() as a black
                    box, scipy.optimize.minimize / SPSA on theta directly
                    (works today; each evaluation = one forward).
  (3) COLUMNS     — super-column (path/route) based gradients: assignment is
                    linearized through the column-proportion operator PI
                    (routes from the kernel = columns); dL/dOD = PI^T r is
                    exact for fixed columns; refresh columns in an outer loop.
  (4) TENSOR      — flow-through-tensor: the whole chain is einsum algebra
                    (strings given per layer); swap numpy->torch, set
                    requires_grad on theta, and autodiff replaces (1).
                    Adding generation/distribution layers = adding tensors
                    to the einsum pipeline, nothing else changes.

THETA (calibratable parameter blocks — TRMG2's own knobs, nothing else):
  theta.g[p]        [P]         generation calibration factor per purpose
                                (TRMG2: generation/calibration_factors.csv)
  theta.dASC[p,c]   [P,C]       cluster delta-ASC (dc/*_cluster.csv column)
  theta.dIC[p,c]    [P,C]       cluster delta-IC  (dc/*_cluster.csv column)
  (frozen: all dc/*_zone.csv behavioral coefficients, TOD/dir/occ factors,
   VDF parameters — the reproduction spine.)

TENSOR SHAPES (Z zones, P purposes, C clusters, K periods, M classes, A links):
  HHPOP [Z]  rate [P]  ->  Prod [P,Z]
  U [P,K',Z,Z]  ->  Trips T [P,K',Z,Z]   (K'=2 skim variants AM/MD)
  share [P,M]  occ [P,K,M]  todf [P,K]  paf [P,K]  ->  OD [K,M,Z,Z]
  PI [K,A,Z*Z] (sparse columns)  ->  v [A]   ->  L scalar

Run:  python tcg_forward.py            (forward + loss, column backend)
      python tcg_forward.py --kernel   (forward + exact UE via TAPLite)
Requires: scenario/ built by build_gmns.py (network + counts in link.csv).
"""
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from se_loader import load_se, load_table  # noqa: E402
from make_od_4period import (parse_zone_table, parse_cluster_table,  # noqa: E402
                             read_kv, PURPOSES, PERIODS, SKIM_OF)

# =====================================================================
# L0 — DATA (frozen inputs; loaded once)
# =====================================================================
def load_data():
    D = {}
    # --- zones / crosswalk / SE ---
    se_rows = load_se(REPO, year=2020)
    se = {r["TAZ"]: r for r in se_rows if (r.get("Type") or "") == "Internal"}
    taz_of = {}
    with open(os.path.join(HERE, "node_crosswalk.csv")) as f:
        for r in csv.DictReader(f):
            if r["taz"]:
                taz_of[int(r["gmns_node_id"])] = int(r["taz"])
    import geopandas as gpd
    g = gpd.read_file(os.path.join(REPO, "docs", "data", "input", "tazs",
                                   "master_tazs.shp")).set_index("ID")
    taz_cluster = g["CLUSTER"].astype(int).to_dict()

    # --- network (from the validated GMNS build) ---
    links = []
    with open(os.path.join(HERE, "scenario", "link.csv")) as f:
        for r in csv.DictReader(f):
            links.append(r)
    nodes = sorted({int(r["from_node_id"]) for r in links}
                   | {int(r["to_node_id"]) for r in links})
    nix = {n: k for k, n in enumerate(nodes)}
    zones = sorted(z for z in taz_of if taz_of[z] in se)
    D.update(se=se, taz_of=taz_of, zones=zones, Z=len(zones), links=links,
             nodes=nodes, nix=nix,
             zone_cluster=[taz_cluster.get(taz_of[z], -1) for z in zones])

    # --- skims: 2 variants (AM, MD) on init congested times, fftt fallback ---
    init_rows = load_table(os.path.join(HERE, "params", "init_cong_time_2020.csv"),
                           os.path.join(REPO, "master", "networks", "init_cong_time_2020.bin"))
    init = {r["ID"]: r for r in init_rows}
    def time_graph(per):
        rows, cols, w = [], [], []
        for r in links:
            base, d = int(r["link_id"]) // 10, int(r["link_id"]) % 10
            key = ("AB" if d == 1 else "BA") + "InitCongTime" + per
            t = (init.get(base) or {}).get(key) or 0
            if not t or t <= 0:
                t = float(r["vdf_fftt"])
            rows.append(nix[int(r["from_node_id"])])
            cols.append(nix[int(r["to_node_id"])])
            w.append(max(t, 1e-3))
        return sp.csr_matrix((w, (rows, cols)), shape=(len(nodes),) * 2)
    src = [nix[z] for z in zones]
    D["skims"] = {}
    for sk in ("AM", "MD"):
        t = dijkstra(time_graph(sk), directed=True, indices=src)[:, src]
        t = np.where(np.isfinite(t), t, 1e4)
        np.fill_diagonal(t, 1e4)      # D1 deviation: no intrazonal (v1.6 fixes)
        D["skims"][sk] = t

    # --- TRMG2 parameter tables (frozen spine) ---
    D["zspec"] = {p: parse_zone_table(os.path.join(
        REPO, "master", "resident", "dc", p.lower() + "_zone.csv")) for p in PURPOSES}
    D["ctab"] = {p: parse_cluster_table(os.path.join(
        REPO, "master", "resident", "dc", p.lower() + "_cluster.csv")) for p in PURPOSES}
    D["todf"] = read_kv(os.path.join(REPO, "master", "resident", "tod",
                        "time_of_day_factors.csv"), ["trip_type", "tod"], "factor")
    D["paf"] = read_kv(os.path.join(REPO, "master", "resident", "tod",
                       "directionality_factors.csv"), ["trip_type", "tod"], "pa_fac")
    D["occ3"] = read_kv(os.path.join(REPO, "master", "resident", "tod",
                        "hov3_occ_factors_hb.csv"), ["trip_type", "tod"], "hov3")
    sp_rows = load_table(os.path.join(HERE, "params", "shadow_prices.csv"),
                         os.path.join(REPO, "master", "resident", "dc", "shadow_prices.bin"))
    sp_hbw = {r["TAZ"]: (r.get("hbw") or 0.0) for r in sp_rows}
    D["spvec"] = np.array([sp_hbw.get(taz_of[z], 0.0) for z in zones])

    # base person rates (stand-in generation; upgraded by task #4/#5)
    rates = defaultdict(lambda: [0.0, 0.0])
    with open(os.path.join(REPO, "master", "resident", "generation",
                           "production_rates.csv")) as f:
        for r in csv.DictReader(f):
            w = float(r["samples"] or 0)
            rates[r["trip_type"]][0] += w
            rates[r["trip_type"]][1] += w * float(r["rate"] or 0)
    D["rate"] = {p: ws / w for p, (w, ws) in rates.items() if w > 0}
    D["HHPOP"] = np.array([se[taz_of[z]]["HH_POP"] or 0 for z in zones], float)

    # size terms per purpose
    size_coef = defaultdict(dict)
    with open(os.path.join(REPO, "master", "resident", "dc", "dc_size_terms.csv")) as f:
        rd = csv.DictReader(f)
        for r in rd:
            for col in rd.fieldnames[1:]:
                if col in ("Field", "Description") or not r.get(col):
                    continue
                try:
                    v = float(r[col])
                except ValueError:
                    continue
                if v:
                    size_coef[col][r["Field"]] = v
    def se_field(z, field):
        r = se[z]
        if field.endswith(("_EH", "_EL")):
            base, pct = field[:-3], (r.get("PctHighPay") or 0) / 100.0
            return (r.get(base) or 0) * (pct if field.endswith("_EH") else 1 - pct)
        return r.get(field) or 0
    D["lnA"] = {}
    for p, (_, cols) in PURPOSES.items():
        A = np.zeros(D["Z"])
        for c in cols:
            for fld, cf in size_coef[c].items():
                A += np.array([cf * se_field(taz_of[z], fld) for z in zones])
        A /= len(cols)
        D["lnA"][p] = np.where(A > 0, np.log(np.maximum(A, 1e-9)), -np.inf)

    # mode shares stand-in (task #5 replaces with nested logit)
    D["share"] = {}
    with open(os.path.join(REPO, "docs", "data", "output", "survey_processing",
                           "eda_scheme6.csv")) as f:
        for r in csv.DictReader(f):
            if r["homebased"] != "HB":
                continue
            key = (r["tour_type"], r["purpose"], r["duration"])
            for p, (ek, _) in PURPOSES.items():
                if ek == key:
                    s = [float(r["pct_sov"]), float(r["pct_hov2"]), float(r["pct_hov3"])]
                    extra = float(r["pct_auto_pay"]) + float(r["pct_other_auto"])
                    tot = sum(s)
                    D["share"][p] = [x * (1 + extra / tot) / 100 for x in s]

    # observations: counts on links (two-way daily) + survey trips
    D["counts"] = {}   # base link id -> (count, [directed (a,b) keys])
    for r in links:
        if r["day_count"]:
            base = int(r["link_id"]) // 10
            key = (int(r["from_node_id"]), int(r["to_node_id"]))
            D["counts"].setdefault(base, [float(r["day_count"]), []])[1].append(key)
    return D


# =====================================================================
# L1 — GENERATION      Prod[p,i] = HHPOP[i] * rate[p] * g[p]
#   tensor:  einsum('i,p,p->pi', HHPOP, rate, g)
#   CG backward: dL/dg[p] = sum_i dL/dProd[p,i] * HHPOP[i] * rate[p]
# =====================================================================
def L1_generation(D, theta):
    return {p: D["HHPOP"] * D["rate"].get(p, 0) * theta["g"][k]
            for k, p in enumerate(PURPOSES)}


# =====================================================================
# L2 — DISTRIBUTION (TRMG2 nested DC; theta enters ONLY via dASC/dIC)
#   zone utility  U[i,j] = c_size*lnA[j] + c_t*t[i,j] + pieces + IZ + sp
#   nest:  logsum_c = theta_c * ln sum_{j in c} exp(U/theta_c)
#          V_c = (ASC+dASC) + (IC+dIC)*1[i in c] + logsum_c
#          T[i,j] = Prod[i] * softmax_c(V)[c(j)] * P(j | c(j), i)
#   CG backward (cluster-nest Jacobian):
#     dP(j|i)/ddASC_c = P(c|i)*(1{j in c} - P(c|i))*P(j|c,i)
#     dIC identical * 1[i in c]
#   tensor: two stacked softmaxes -> in torch: log_softmax over masked dims.
# =====================================================================
def L2_distribution(D, theta, Prod):
    from make_od_4period import nested_dc
    T = {}
    for k, p in enumerate(PURPOSES):
        zs, ct0 = D["zspec"][p], D["ctab"][p]
        # inject calibratable deltas on top of published values
        ct = {c: dict(theta=v["theta"],
                      asc=v["asc"] + theta["dASC"][k].get(c, 0.0),
                      ic=v["ic"] + theta["dIC"][k].get(c, 0.0))
              for c, v in ct0.items()}
        same = (np.array(D["zone_cluster"])[:, None]
                == np.array(D["zone_cluster"])[None, :])
        for sk, t in D["skims"].items():
            U = zs["size"] * D["lnA"][p][None, :] + zs["time"] * t
            for thr, cf in zs["piecewise"]:
                U = U + cf * np.maximum(0.0, t - thr)
            U = U + zs["iz"] * same          # D-veteran fix pending: diagonal only
            if zs["shadow"]:
                U = U + zs["shadow"] * D["spvec"][None, :]
            if zs["cutoff"] is not None:
                U = np.where(t > zs["cutoff"], -np.inf, U)
            U = np.where(np.isfinite(D["lnA"][p])[None, :], U, -np.inf)
            T[(p, sk)] = nested_dc(Prod[p], U, D["zone_cluster"], ct)
    return T


# =====================================================================
# L3+L4 — MODE / TOD / DIRECTIONALITY / OCCUPANCY  (all fixed linear maps)
#   OD[k,m] = sum_p  T[p, skim(k)] * todf[p,k] * dirmix * share[p,m] / occ[p,k,m]
#   tensor: einsum('pij,pk,pm->kmij', T, todf*dir, share/occ) + transpose term
#   CG backward: exact adjoints of the same linear maps (transpose included).
# =====================================================================
def L34_od(D, T):
    Z = D["Z"]
    OD = {(per, m): np.zeros((Z, Z)) for per in PERIODS for m in ("sov", "hov2", "hov3")}
    for p in PURPOSES:
        s = D["share"].get(p)
        if s is None:
            continue
        for per in PERIODS:
            Tm = T[(p, SKIM_OF[per])]
            PA = Tm * D["todf"].get((p, per), 0.0)
            f = D["paf"].get((p, per), 0.5)
            X = f * PA + (1 - f) * PA.T
            o3 = D["occ3"].get((p, per), 3.5)
            OD[(per, "sov")] += X * s[0]
            OD[(per, "hov2")] += X * s[1] / 2.0
            OD[(per, "hov3")] += X * s[2] / o3
    return OD


# =====================================================================
# L5 - ASSIGNMENT (MATRIX FORM - from the kernel's own equilibrium results)
#   Operators extracted by matrix_ops.py from scenario_{per}/route_assignment:
#     Delta[per]  [paths x od]    OD -> path shares     (pre-assignment result)
#     A[per]      [links x paths] path -> link incidence
#     Pi[per]     [links x od]    = A @ Delta
#   forward :  v = sum_per Pi[per] @ vec(OD[per])        -- NO recomputation
#   backward:  dL/dOD[per] = Pi[per]^T @ dL/dv           -- exact, fixed columns
#   refresh :  rerun the kernel (outer/bi-level loop), re-extract operators.
#   super-columns: work at (A, Delta) level - column generation appends path
#   rows; path cost c_p = A^T t prices new columns.
# =====================================================================
_PI_CACHE = {}


def load_pi(per):
    """Load Pi and the od-column index for a period (built by matrix_ops.py)."""
    if per in _PI_CACHE:
        return _PI_CACHE[per]
    fp = os.path.join(HERE, "matrices", f"pi_{per}.npz")
    if not os.path.exists(fp):
        raise FileNotFoundError(
            f"{fp} missing - run the kernel with route_output=1 for {per}, "
            f"then: python matrix_ops.py {per}")
    Pi = sp.load_npz(fp)
    odmap = {}
    with open(os.path.join(HERE, "matrices", f"od_index_{per}.csv")) as f:
        for r in csv.DictReader(f):
            odmap[(int(r["o_zone_id"]), int(r["d_zone_id"]))] = int(r["od_col"])
    _PI_CACHE[per] = (Pi, odmap)
    return Pi, odmap


def L5_assign_matrix(D, OD):
    """v = sum_per Pi[per] @ vec(OD[per]) using the extracted operators."""
    v = None
    for per in PERIODS:
        Pi, odmap = load_pi(per)
        odv = np.zeros(Pi.shape[1])
        od = sum(OD[(per, m)] for m in ("sov", "hov2", "hov3"))
        key = ("odsel", per)
        if key not in _PI_CACHE:
            zpos = {z: i for i, z in enumerate(D["zones"])}
            rows, cols, dest = [], [], []
            for (o, d), c in odmap.items():
                if o in zpos and d in zpos:
                    rows.append(zpos[o]); cols.append(zpos[d]); dest.append(c)
            _PI_CACHE[key] = (np.array(rows), np.array(cols), np.array(dest))
    # noqa
        rows, cols, dest = _PI_CACHE[key]
        odv[dest] = od[rows, cols]
        contrib = Pi @ odv
        v = contrib if v is None else v + contrib
    return v


# =====================================================================
# L6 — OBSERVATION / LOSS
#   r_a = (v_a - c_a) / max(c_a, cbar);  L = 0.5 * sum r^2  (+ survey terms)
#   CG backward: dL/dv = r / max(c, cbar) on observed links, 0 elsewhere.
# =====================================================================
def L6_loss(D, v):
    idx = {(int(r["from_node_id"]), int(r["to_node_id"])): k
           for k, r in enumerate(D["links"])}
    res, det = [], []
    for base, (cnt, keys) in D["counts"].items():
        est = sum(v[idx[k]] for k in keys if k in idx)
        res.append((est - cnt) / max(cnt, 5000.0))
        det.append((base, cnt, est))
    r = np.array(res)
    return 0.5 * float(r @ r) / len(r), det


# =====================================================================
# FORWARD
# =====================================================================
def theta_init():
    return dict(g=np.ones(len(PURPOSES)),
                dASC=[defaultdict(float) for _ in PURPOSES],
                dIC=[defaultdict(float) for _ in PURPOSES])


def forward(D, theta, backend="matrix"):
    t0 = time.time()
    Prod = L1_generation(D, theta)
    T = L2_distribution(D, theta, Prod)
    OD = L34_od(D, T)
    if backend == "matrix":
        v = L5_assign_matrix(D, OD)          # Pi from pre-assignment results
    else:
        raise NotImplementedError("kernel backend: write demand + pytaplite "
                                  "(see make_od_4period.py) then read volumes")
    L, det = L6_loss(D, v)
    return dict(loss=L, volumes=v, seconds=round(time.time() - t0, 1),
                total_veh=round(float(sum(o.sum() for o in OD.values()))),
                assigned_on_counted=round(sum(e for _, _, e in det)),
                counted_total=round(sum(c for _, c, _ in det)))


# =====================================================================
# CALIBRATION HOOKS (pseudocode for the four paradigms — same forward)
# =====================================================================
CALIBRATION_NOTES = """
(1) CG  — implement backward(): r -> dL/dv -> PI^T -> dL/dOD -> adjoint of
    L34 linear maps -> dL/dT -> cluster-nest Jacobian (formula in L2 header)
    -> dL/ddASC, dL/ddIC, and through L1 -> dL/dg.  Verify by finite
    differences (calib_graph_demo standard: rel err <= 1e-3), then L-BFGS-B.

(2) NLP — scipy.optimize.minimize(lambda th: forward(D, unpack(th))['loss'],
    x0, method='L-BFGS-B' (numeric grad) or 'Nelder-Mead'/SPSA;
    ~200 dof x forward cost -> expensive but zero extra math; the sanity
    baseline every gradient method must beat.

(3) COLUMNS — replace AON PI with the kernel's route set (route_output=1 or
    the sparse select-link export): v = PI @ vec(OD) with PI holding UE route
    proportions; gradients as in (1) but PI is the true equilibrium column
    operator; refresh PI (and skims) in an outer loop, MSA-damped.
    The kernel's DTAC column store (efficiency track) makes refresh cheap.

(4) TENSOR — port L1-L6 to torch:  Prod = HHPOP[None,:] * rate[:,None] * g[:,None]
    U as above (masked); within-nest and cluster softmaxes via
    torch.log_softmax over cluster-partitioned columns; OD via einsum
    ('pij,pk->kij' style contractions); v = torch.sparse.mm(PI, OD.flatten());
    loss.backward() gives all gradients — adding a NEW layer (e.g., person-
    level generation, feedback iteration) is just more tensor ops in the
    graph.  numpy forward above is the specification to port against.
"""


if __name__ == "__main__":
    D = load_data()
    th = theta_init()
    out = forward(D, th, backend="matrix")
    print(json.dumps(out, indent=1, default=str))
    print(CALIBRATION_NOTES)
