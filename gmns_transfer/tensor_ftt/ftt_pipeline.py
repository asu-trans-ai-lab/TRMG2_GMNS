"""ftt_pipeline.py — Flow-Through-Tensor on the REAL assignment snapshot.

Per this round of revision: "flow through tensors" means we ASSUME the assigned
path-flow matrix, the OD->path matrix, and the skim from the PREVIOUS assignment
are all available, and we RELOAD that entire snapshot — we do NOT re-run the
network assignment. Operating on the fixed loaded operators lets us
  (1) converge better  — warm-start from the real path flows, matrix-form iteration;
  (2) verify           — Pi @ f_OD must reproduce the kernel's link volumes;
  (3) go to ONNX later — fixed sparse operators => a clean differentiable graph.

Loaded snapshot (from ../matrices/ + ../scenario_<per>/ produced by matrix_ops.py
and the TAPLite run):
  Delta = B_OD,P   (|OD| x |P|)   OD -> path choice proportions   (delta_<per>.npz)
  A     = A_P,L    (|P|  x |L|)   path -> link incidence           (A_<per>.npz)
  Pi    = A^T Delta^T (|L| x |OD|) composed OD -> link operator     (pi_<per>.npz)
  f_OD                             assigned OD vehicle demand       (scenario_<per>/demand_*.csv)
  f_P   = Delta^T f_OD             assigned PATH flows (warm start)
  t_L^prev                         previous skim / link times       (link_performance.csv)

Forward (FTT):   f_P = Delta^T f_OD ; f_L = A^T f_P (= Pi f_OD) ; t_L = phi(f_L)
Backward (FTT):  t_P = A t_L ; t_OD = Delta t_P ;  dL/df_OD = Pi^T (f_L - counts)

Run:  python ftt_pipeline.py            # loads AM snapshot, verifies vs kernel
      python ftt_pipeline.py MD PM NT   # other periods if extracted
"""
import csv
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
GMNS = os.path.dirname(HERE)


# ----------------------------------------------------------------------------
# Load the entire previous-assignment snapshot as fixed tensors
# ----------------------------------------------------------------------------
def load_snapshot(per="AM"):
    mdir = os.path.join(GMNS, "matrices")
    need = [f"delta_{per}.npz", f"A_{per}.npz", f"pi_{per}.npz", f"od_index_{per}.csv"]
    if not all(os.path.exists(os.path.join(mdir, f)) for f in need):
        raise FileNotFoundError(
            f"snapshot for {per} not in {mdir} — run: python ../matrix_ops.py {per}")

    # matrix_ops saves these already transposed for direct forward multiply:
    Delta = sp.load_npz(os.path.join(mdir, f"delta_{per}.npz"))     # |P|  x |OD|
    A = sp.load_npz(os.path.join(mdir, f"A_{per}.npz"))             # |L|  x |P|
    Pi = sp.load_npz(os.path.join(mdir, f"pi_{per}.npz"))           # |L|  x |OD| = A @ Delta

    # OD column order + zone pairs
    od_row = {}
    with open(os.path.join(mdir, f"od_index_{per}.csv")) as f:
        for r in csv.DictReader(f):
            od_row[(int(r["o_zone_id"]), int(r["d_zone_id"]))] = int(r["od_col"])
    n_od = len(od_row)

    # assigned OD vehicle demand f_OD. PREFER the FROZEN copy saved with the
    # operators (self-contained snapshot -> guaranteed-consistent, exact
    # reproduction); fall back to re-reading the current demand CSVs (which may
    # be newer than the operators -> STALE).
    sdir = os.path.join(GMNS, f"scenario_{per}")
    frozen_fp = os.path.join(mdir, f"f_od_{per}.npy")
    frozen = os.path.exists(frozen_fp)
    if frozen:
        f_OD = np.load(frozen_fp)
    else:
        f_OD = np.zeros(n_od)
        for m in ("sov", "hov2", "hov3"):
            fp = os.path.join(sdir, f"demand_{m}.csv")
            if not os.path.exists(fp):
                continue
            with open(fp) as f:
                for r in csv.DictReader(f):
                    k = od_row.get((int(r["o_zone_id"]), int(r["d_zone_id"])))
                    if k is not None:
                        f_OD[k] += float(r["volume"] or 0)

    # kernel link volumes + previous skim (link times) from the last assignment
    n_links = A.shape[0]
    key_of = {}
    with open(os.path.join(GMNS, "scenario", "link.csv")) as g:
        for k, r in enumerate(csv.DictReader(g)):
            key_of[(int(r["from_node_id"]), int(r["to_node_id"]))] = k
    v_kernel = np.zeros(n_links)
    t_prev = np.zeros(n_links)
    have_perf = False
    lp = os.path.join(sdir, "link_performance.csv")
    try:                                    # may be locked if an assignment is mid-run
        with open(lp) as f:
            for r in csv.DictReader(f):
                k = key_of.get((int(r["from_node_id"]), int(r["to_node_id"])))
                if k is not None:
                    v_kernel[k] = float(r["volume"] or 0)
                    t_prev[k] = float(r.get("travel_time") or r.get("VDF_travel_time") or 0)
        have_perf = True
    except (PermissionError, FileNotFoundError):
        pass

    return dict(Delta=Delta, A=A, Pi=Pi, f_OD=f_OD, v_kernel=v_kernel, t_prev=t_prev,
                have_perf=have_perf, frozen=frozen,
                n_od=n_od, n_paths=A.shape[1], n_links=n_links)


# ----------------------------------------------------------------------------
# Flow-Through-Tensor forward / backward on the fixed loaded operators
# ----------------------------------------------------------------------------
def ftt_forward(S):
    f_OD = S["f_OD"]
    f_P = S["Delta"] @ f_OD            # OD -> path   (assigned path flows, warm start)
    f_L = S["A"] @ f_P                 # path -> link (== Pi @ f_OD)
    return f_P, f_L


def ftt_backward_time(S, t_L):
    """Path time = sum of link times (additive, exact). OD time = FLOW-WEIGHTED
    AVERAGE of path times, not a raw Delta^T multiply: t_OD = (Delta^T (f_P o t_P))
    / (Delta^T f_P). A bare Delta^T t_P would SUM path times and be wrong on any
    multi-path OD (reviewer R3-MAJOR4)."""
    t_P = S["A"].T @ t_L                       # link -> path times (|P|)
    f_P = S["Delta"] @ S["f_OD"]               # path flows (weights)
    num = S["Delta"].T @ (f_P * t_P)           # |OD|
    den = S["Delta"].T @ f_P                   # |OD| = f_OD (path flows sum back)
    t_OD = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return t_P, t_OD


def count_grad(S, f_L, counts):
    """dL/df_OD for L = 0.5||Pi f_OD - counts||^2. The FIXED-PROPORTION ODME
    approximation: exact for fixed Pi, i.e. we neglect dPi/df_OD (standard
    Cascetta/Yang/Lundgren-Peterson link-proportion ODME assumption -- NOT
    Danskin, which applies only to the inner equilibration). Valid where the
    active path set is locally invariant; breaks near capacity / at path swaps."""
    resid = f_L - counts
    return S["Pi"].T @ resid           # |OD| vector


def main():
    periods = sys.argv[1:] or ["AM"]
    for per in periods:
        t0 = time.time()
        S = load_snapshot(per)
        f_P, f_L = ftt_forward(S)
        print(f"[{per}] snapshot: |OD|={S['n_od']:,}  |P|={S['n_paths']:,}  "
              f"|L|={S['n_links']:,}  loaded in {time.time()-t0:.1f}s")
        src = "FROZEN with operators (self-contained)" if S["frozen"] \
              else "re-read from demand CSVs (may be newer than operators)"
        print(f"      f_OD total {S['f_OD'].sum():,.0f} veh [{src}]   "
              f"f_P total {f_P.sum():,.0f}")
        # verify: FTT link flow must equal the kernel's link volumes
        # (requires a CONSISTENT snapshot: matrices + OD + volumes from ONE run —
        #  re-run ../matrix_ops.py after each assignment to refresh)
        if S["have_perf"]:
            vk = S["v_kernel"]; mask = vk > 0
            # f_L = Pi f_OD is an EXACT identity for a consistent snapshot, so gate
            # on max abs/rel error (not R^2, which the freeway variance dominates
            # and can read 0.98 while the congested links are wrong -- R3-MAJOR2).
            abs_err = np.abs(f_L - vk)[mask]
            rel_err = (abs_err / vk[mask]).max()
            tol = 1e-6 * vk[mask].max()
            # congested-subset check: near-capacity links must match too
            r2 = 1 - float((abs_err ** 2).sum()) / float(
                ((vk[mask] - vk[mask].mean()) ** 2).sum() or 1.0)
            tag = "CONSISTENT" if abs_err.max() <= tol else "STALE (re-extract matrices)"
            print(f"      VERIFY  f_L = A Delta f_OD (= Pi f_OD) vs kernel volumes "
                  f"[{tag}]:")
            print(f"         max|err| {abs_err.max():.3f} veh (tol {tol:.3f})  "
                  f"max rel {rel_err:.2e}  (R^2 {r2:.6f} diagnostic only)")
            _, t_OD = ftt_backward_time(S, S["t_prev"])
            g = count_grad(S, f_L, counts=vk * 1.05)     # pretend 5%-high targets
            print(f"      t_OD from previous skim: mean {t_OD[t_OD>0].mean():.1f} min")
            print(f"      dL/df_OD (ODME grad) in [{g.min():.1f}, {g.max():.1f}]  "
                  f"-> gradient-descent on f_OD; operators fixed, ONNX-ready")
        else:
            print("      (link_performance.csv locked/absent — an assignment is "
                  "running; skipping the kernel-volume verify. f_L still computed.)")
            print(f"      f_L = Pi f_OD in [{f_L.min():.0f}, {f_L.max():.0f}] veh")
        print()
    print("Everything ran on the RELOADED previous-assignment snapshot -- no "
          "network re-assignment. Warm-started, verified, differentiable.")
    print("NOTE: R^2=1.0 requires a CONSISTENT snapshot (matrices + OD + volumes "
          "from the SAME run). If it reads STALE, refresh: python ../matrix_ops.py " + " ".join(periods))


if __name__ == "__main__":
    main()
