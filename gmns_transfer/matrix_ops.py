"""matrix_ops.py — extract the ASSIGNMENT MATRIX OPERATORS from the kernel's
own equilibrium results (no assignment logic re-implemented in Python).

From scenario_{per}/route_assignment.csv (the pre-assignment results):

  Delta [n_paths x n_od]   OD -> path :  path share of its OD pair
                           (volume-weighted across classes: prob_r = vol_r / vol_od)
  A     [n_links x n_paths] path -> link incidence (1 if path uses link)
  Pi    [n_links x n_od]   = A @ Delta   (the composed link-use operator)

so that inside forward/backward propagation:

  forward :  v = Pi @ vec(OD)                     (matrix form, no recompute)
  backward:  dL/dOD = Pi^T @ dL/dv                (exact for fixed columns)
  path level (super-columns): keep A and Delta separate — column generation
  adds path rows to A/Delta; path costs c_p = A^T t give the column pricing.

Outputs per period in matrices/:
  pi_{per}.npz, A_{per}.npz, delta_{per}.npz     scipy sparse CSR
  od_index_{per}.csv                             od_col -> (o_zone, d_zone)
  path_meta_{per}.csv                            path_row -> od_col, share, n_links
  validation printed: v_rebuilt vs kernel link_performance volumes (R^2, max err)

Usage: python matrix_ops.py AM [MD PM NT]
"""
import csv
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "matrices")
os.makedirs(OUT, exist_ok=True)


def link_index():
    idx = {}
    with open(os.path.join(HERE, "scenario", "link.csv")) as f:
        for k, r in enumerate(csv.DictReader(f)):
            idx[int(r["link_id"])] = k
    return idx


def build_period(per, lidx):
    sdir = os.path.join(HERE, f"scenario_{per}")
    fp = os.path.join(sdir, "route_assignment.csv")
    t0 = time.time()

    # pass 1: OD volumes (all classes) for share normalization
    od_vol = {}
    with open(fp) as f:
        for r in csv.DictReader(f):
            try:
                od = (int(r["o_zone_id"]), int(r["d_zone_id"]))
                od_vol[od] = od_vol.get(od, 0.0) + float(r["volume"] or 0)
            except (ValueError, KeyError):
                continue
    od_list = sorted(od_vol)
    od_col = {od: k for k, od in enumerate(od_list)}

    # pass 2: paths -> Delta and A triplets
    d_rows, d_cols, d_vals = [], [], []      # Delta [paths x od]
    a_rows, a_cols = [], []                  # A [links x paths]
    meta = []
    n_paths = 0
    with open(fp) as f:
        for r in csv.DictReader(f):
            try:
                od = (int(r["o_zone_id"]), int(r["d_zone_id"]))
                vol = float(r["volume"] or 0)
            except (ValueError, KeyError):
                continue
            tot = od_vol.get(od, 0.0)
            if vol <= 0 or tot <= 0:
                continue
            share = vol / tot
            p = n_paths
            n_paths += 1
            c = od_col[od]
            d_rows.append(p)
            d_cols.append(c)
            d_vals.append(share)
            nl = 0
            for lid in r["link_ids"].split(";"):
                if lid:
                    k = lidx.get(int(lid))
                    if k is not None:
                        a_rows.append(k)
                        a_cols.append(p)
                        nl += 1
            meta.append((p, c, round(share, 6), nl))
    n_links = len(lidx)
    Delta = sp.csr_matrix((d_vals, (d_rows, d_cols)), shape=(n_paths, len(od_list)))
    A = sp.csr_matrix((np.ones(len(a_rows)), (a_rows, a_cols)),
                      shape=(n_links, n_paths))
    Pi = (A @ Delta).tocsr()
    print(f"[{per}] paths {n_paths:,}, od {len(od_list):,}, "
          f"Pi nnz {Pi.nnz:,} ({time.time()-t0:.0f}s)")

    sp.save_npz(os.path.join(OUT, f"pi_{per}.npz"), Pi)
    sp.save_npz(os.path.join(OUT, f"A_{per}.npz"), A)
    sp.save_npz(os.path.join(OUT, f"delta_{per}.npz"), Delta)
    with open(os.path.join(OUT, f"od_index_{per}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["od_col", "o_zone_id", "d_zone_id"])
        for od, k in od_col.items():
            w.writerow([k, od[0], od[1]])
    with open(os.path.join(OUT, f"path_meta_{per}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path_row", "od_col", "share", "n_links"])
        w.writerows(meta)

    # ---- validation: Pi @ od must reproduce the kernel's link volumes ----
    odv = np.zeros(len(od_list))
    for od, v in od_vol.items():
        odv[od_col[od]] = v
    # freeze f_OD WITH the operators so the snapshot is self-contained: FTT then
    # reproduces this run exactly (max|err|~0) regardless of later demand edits.
    np.save(os.path.join(OUT, f"f_od_{per}.npy"), odv)
    # fingerprint the snapshot (run identity) so a stale operator/demand mix is a
    # HARD error downstream, not a silent R^2 dip (reviewer R3-MINOR-5)
    import json
    meta = dict(period=per,
                route_mtime=os.path.getmtime(os.path.join(sdir, "route_assignment.csv")),
                f_od_total=float(odv.sum()), n_paths=int(n_paths),
                n_od=int(len(od_list)), pi_nnz=int(Pi.nnz))
    with open(os.path.join(OUT, f"snapshot_meta_{per}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    v_rebuilt = Pi @ odv
    v_kernel = np.zeros(n_links)
    with open(os.path.join(sdir, "link_performance.csv")) as f:
        key_of = {}
        with open(os.path.join(HERE, "scenario", "link.csv")) as g:
            for k, r in enumerate(csv.DictReader(g)):
                key_of[(int(r["from_node_id"]), int(r["to_node_id"]))] = k
        for r in csv.DictReader(f):
            k = key_of.get((int(r["from_node_id"]), int(r["to_node_id"])))
            if k is not None:
                v_kernel[k] = float(r["volume"] or 0)
    mask = v_kernel > 0
    ss_res = float(((v_rebuilt - v_kernel)[mask] ** 2).sum())
    ss_tot = float(((v_kernel[mask] - v_kernel[mask].mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot else 1.0
    print(f"[{per}] validation: R^2 {r2:.6f}, "
          f"max |err| {np.abs(v_rebuilt - v_kernel).max():.1f} veh, "
          f"totals {v_rebuilt.sum():,.0f} vs {v_kernel.sum():,.0f}")
    return r2


if __name__ == "__main__":
    periods = sys.argv[1:] or ["AM"]
    lidx = link_index()
    for per in periods:
        build_period(per, lidx)
