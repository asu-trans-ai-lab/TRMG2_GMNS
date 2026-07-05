"""Stage-level CPU benchmark of the demand pipeline (generation + distribution).

Times each stage of make_od_4period.py in isolation (no kernel runs):
  S1 load inputs (SE, rates, size terms, factors)
  S2 skims: time + distance dijkstra (3,247 sources x 33,963 nodes)
  S3 generation: P_i per purpose (vector ops)
  S4 size terms: A_j per purpose
  S5 one gravity evaluation (softmax 3247^2) — the inner-loop unit cost
  S6 full beta bisection (40 evals) x 8 purposes
  S7 OD assembly: 8 purposes x 4 periods x 3 classes accumulation
  S8 demand output: CSV write vs float32 .npy binary
Also reports float64 vs float32 gravity cost (vectorization headroom probe).

Writes review/bench_demand.json and prints a table.
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
from transcad_bin import read_bin  # noqa: E402
from make_od_4period import PURPOSES, PERIODS, read_kv  # noqa: E402

T = {}


def tic(name):
    T[name] = -time.perf_counter()


def toc(name):
    T[name] += time.perf_counter()


def main():
    # ---------------- S1 load ----------------
    tic("S1_load_inputs")
    eda = {}
    with open(os.path.join(REPO, "docs", "data", "output", "survey_processing",
                           "eda_scheme6.csv")) as f:
        for r in csv.DictReader(f):
            if r["homebased"] == "HB":
                eda[(r["tour_type"], r["purpose"])] = r
    tod_f = read_kv(os.path.join(REPO, "master", "resident", "tod",
                                 "time_of_day_factors.csv"), ["trip_type", "tod"], "factor")
    pa_f = read_kv(os.path.join(REPO, "master", "resident", "tod",
                                "directionality_factors.csv"), ["trip_type", "tod"], "pa_fac")
    rates = defaultdict(lambda: [0.0, 0.0])
    with open(os.path.join(REPO, "master", "resident", "generation",
                           "production_rates.csv")) as f:
        for r in csv.DictReader(f):
            w = float(r["samples"] or 0)
            rates[r["trip_type"]][0] += w
            rates[r["trip_type"]][1] += w * float(r["rate"] or 0)
    rate_p = {p: ws / w for p, (w, ws) in rates.items() if w > 0}
    _, se_rows = read_bin(os.path.join(REPO, "master", "sedata", "se_2020.bin"))
    se = {r["TAZ"]: r for r in se_rows if (r.get("Type") or "") == "Internal"}
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
                if v != 0:
                    size_coef[col][r["Field"]] = v
    toc("S1_load_inputs")

    # ---------------- S2 skims ----------------
    tic("S2_skims")
    gm0 = os.path.join(HERE, "scenario")
    taz_of = {}
    with open(os.path.join(HERE, "node_crosswalk.csv")) as f:
        for r in csv.DictReader(f):
            if r["taz"]:
                taz_of[int(r["gmns_node_id"])] = int(r["taz"])
    nodes = []
    with open(os.path.join(gm0, "node.csv")) as f:
        for r in csv.DictReader(f):
            nodes.append(int(r["node_id"]))
    nix = {n: k for k, n in enumerate(sorted(nodes))}
    rows, cols, wt, wd = [], [], [], []
    with open(os.path.join(gm0, "link.csv")) as f:
        for r in csv.DictReader(f):
            rows.append(nix[int(r["from_node_id"])])
            cols.append(nix[int(r["to_node_id"])])
            wt.append(max(float(r["vdf_fftt"]), 1e-3))
            wd.append(max(float(r["length"]), 1e-4))
    N = len(nodes)
    zones = sorted(z for z in taz_of if taz_of[z] in se)
    src = [nix[z] for z in zones]
    gt = sp.csr_matrix((wt, (rows, cols)), shape=(N, N))
    gd = sp.csr_matrix((wd, (rows, cols)), shape=(N, N))
    Tm = dijkstra(gt, directed=True, indices=src)[:, src]
    Dm = dijkstra(gd, directed=True, indices=src)[:, src]
    toc("S2_skims")
    Tm = np.where(np.isfinite(Tm), Tm, 1e4)
    Dm = np.where(np.isfinite(Dm), Dm, 1e4)
    np.fill_diagonal(Tm, 1e4)
    np.fill_diagonal(Dm, 1e4)
    Z = len(zones)

    # ---------------- S3 generation ----------------
    tic("S3_generation")
    HHPOP = np.array([se[taz_of[z]]["HH_POP"] or 0 for z in zones], dtype=float)
    P_by = {p: HHPOP * rate_p.get(p, 0.0) for p in PURPOSES}
    toc("S3_generation")

    # ---------------- S4 size terms ----------------
    tic("S4_size_terms")
    def se_field(z, field):
        r = se[z]
        if field.endswith("_EH") or field.endswith("_EL"):
            base = field[:-3]
            pct = (r.get("PctHighPay") or 0) / 100.0
            share = pct if field.endswith("_EH") else 1.0 - pct
            return (r.get(base) or 0) * share
        return r.get(field) or 0
    A_by = {}
    for ptype, (_, size_cols) in PURPOSES.items():
        A = np.zeros(Z)
        for col in size_cols:
            for field, coef in size_coef[col].items():
                A += np.array([coef * se_field(taz_of[z], field) for z in zones])
        A_by[ptype] = A / len(size_cols)
    toc("S4_size_terms")

    # ---------------- S5/S6 gravity ----------------
    lnA = np.where(A_by["W_HB_W_All"] > 0,
                   np.log(np.maximum(A_by["W_HB_W_All"], 1e-9)), -1e30)
    P = P_by["W_HB_W_All"]

    def gravity(beta, t, lna, p):
        u = lna[None, :] - beta * t
        u = u - u.max(axis=1, keepdims=True)
        ex = np.exp(np.clip(u, -700, 0))
        q = ex / np.maximum(ex.sum(axis=1, keepdims=True), 1e-12)
        return p[:, None] * q

    gravity(0.1, Tm, lnA, P)  # warm
    tic("S5_one_gravity_f64")
    for _ in range(3):
        M = gravity(0.1, Tm, lnA, P)
    toc("S5_one_gravity_f64")
    T["S5_one_gravity_f64"] /= 3

    T32, lnA32, P32 = Tm.astype(np.float32), lnA.astype(np.float32), P.astype(np.float32)
    gravity(0.1, T32, lnA32, P32)
    tic("S5_one_gravity_f32")
    for _ in range(3):
        gravity(0.1, T32, lnA32, P32)
    toc("S5_one_gravity_f32")
    T["S5_one_gravity_f32"] /= 3

    tic("S6_bisection_8x40")
    for ptype in PURPOSES:
        lna = np.where(A_by[ptype] > 0, np.log(np.maximum(A_by[ptype], 1e-9)), -1e30)
        p = P_by[ptype]
        lo, hi = 0.005, 0.6
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            M = gravity(mid, Tm, lna, p)
            al = (M * Dm).sum() / max(M.sum(), 1e-9)
            if al > 8.0:
                lo = mid
            else:
                hi = mid
    toc("S6_bisection_8x40")

    # ---------------- S7 OD assembly ----------------
    tic("S7_od_assembly")
    acc = {(per, m): np.zeros((Z, Z), dtype=np.float64)
           for per in PERIODS for m in ("sov", "hov2", "hov3")}
    for ptype in PURPOSES:
        M = gravity(0.1, Tm, np.where(A_by[ptype] > 0,
                    np.log(np.maximum(A_by[ptype], 1e-9)), -1e30), P_by[ptype])
        for per in PERIODS:
            PA = M * tod_f.get((ptype, per), 0.2)
            f = pa_f.get((ptype, per), 0.5)
            OD = f * PA + (1 - f) * PA.T
            acc[(per, "sov")] += OD * 0.6
            acc[(per, "hov2")] += OD * 0.1
            acc[(per, "hov3")] += OD * 0.05
    toc("S7_od_assembly")

    # ---------------- S8 output ----------------
    od = acc[("AM", "sov")]
    tic("S8_write_csv")
    oi, dj = np.nonzero(od >= 0.05)
    scratch = os.path.join(HERE, "review", "_bench_demand.csv")
    os.makedirs(os.path.dirname(scratch), exist_ok=True)
    with open(scratch, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["o", "d", "v"])
        for i, j in zip(oi, dj):
            w.writerow([zones[i], zones[j], round(float(od[i, j]), 3)])
    toc("S8_write_csv")
    tic("S8_write_npy")
    np.save(scratch.replace(".csv", ".npy"), od.astype(np.float32))
    toc("S8_write_npy")
    os.remove(scratch)
    os.remove(scratch.replace(".csv", ".npy"))

    out = {k: round(v, 3) for k, v in T.items()}
    out["_meta"] = dict(zones=Z, nodes=N, links=len(rows),
                        od_rows_written=int(len(oi)))
    with open(os.path.join(HERE, "review", "bench_demand.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
