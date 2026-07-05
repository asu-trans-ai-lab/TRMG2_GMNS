"""column_tools.py — large-scale column management for the matrix-form
assignment layer (DynODME-style influence restriction + ADMM decomposition).

Problem: at regional scale Pi is [A x |Omega|] with |Omega| ~ 10^6 OD columns
and 10^6-10^7 paths. Two tools:

(1) INFLUENCE RESTRICTION (the DynODME idea, made exact by the operators):
    Let O = observed (counted) links. Columns of Pi_O := Pi[O, :] with zero
    norm CANNOT affect the count loss — their gradient through L_counts is
    identically zero. The active set
        Act = { w : ||Pi_O[:, w]||_1 > 0 }
    is the complete set of count-influencing OD columns; everything else is
    handled by the low-dimensional survey-total terms analytically. Typical
    reduction: |Act| << |Omega| when counts cover ~5-10% of links.

(2) ADMM SHARING DECOMPOSITION for the OD-adjustment (stage-2 / DynODME-like)
    layer. With od0 the seed OD, x per-OD multiplicative adjustments,
    B_o = Pi_O[:, cols(o)] diag(od0_o) the per-ORIGIN blocks:

        minimize  (1/2) || W (sum_o B_o x_o  -  c) ||^2            [counts]
                + (mu/2) sum_o || x_o - 1 ||^2                      [rails]
                  s.t. x in [lb, ub]

    This is the ADMM *sharing problem* (Boyd et al. 2011, sec 7.3):
      x_o-update: n_o-dim ridge least squares, INDEPENDENT per origin
                  (embarrassingly parallel; each B_o is a tiny sparse block)
      zbar-update: closed form (quadratic g), one vector op on |O| links
      dual u: scalar-form update on |O| links
    Convergence monitored by primal/dual residuals. mu is the drift rail:
    ||x - 1||_inf stays bounded, exactly the no-drift guarantee requested.

Outputs of the demo: review/column_scale_{case}.json with reduction stats,
ADMM residual trace, count fit before/after, drift statistics, timing.
"""
import csv
import json
import os
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# (1) influence restriction
# ---------------------------------------------------------------------------
def restrict_to_observed(Pi, obs_rows):
    """Pi [A x n_od] csr, obs_rows link indices -> (Pi_obs, active_cols, stats)."""
    Pi_obs = Pi[obs_rows, :].tocsc()
    colnorm = np.asarray(np.abs(Pi_obs).sum(axis=0)).ravel()
    active = np.where(colnorm > 0)[0]
    stats = dict(n_od_total=Pi.shape[1], n_od_active=int(active.size),
                 reduction=round(1 - active.size / Pi.shape[1], 4),
                 obs_links=len(obs_rows),
                 nnz_full=int(Pi.nnz), nnz_obs=int(Pi_obs.nnz))
    return Pi_obs[:, active].tocsc(), active, stats


# ---------------------------------------------------------------------------
# (2) ADMM sharing decomposition
# ---------------------------------------------------------------------------
def admm_od_adjust(Pi_obs, od0, counts, origin_of, mu=1.0, rho=1.0,
                   bounds=(0.5, 2.0), iters=150, w=None, verbose=True):
    """Sharing-problem ADMM (Boyd 7.3) for per-OD multiplicative adjustment.

    Pi_obs [nO x nW] csc (active columns only), od0 [nW] seed volumes,
    counts [nO], origin_of [nW] origin id per column (block key).
    Returns x [nW], trace list, timing dict.
    """
    nO, nW = Pi_obs.shape
    # normalize per-link residuals to O(1): scale rows by 1/max(c, 100)
    scale = 1.0 / np.maximum(counts, 100.0)
    counts = counts * scale
    w = np.ones(nO)
    B = Pi_obs.multiply(od0[None, :]).multiply(scale[:, None]).tocsc()
    # per-origin blocks
    origins = np.unique(origin_of)
    blocks = {o: np.where(origin_of == o)[0] for o in origins}
    N = len(origins)
    # prefactor each tiny ridge system  (mu I + rho B_o^T B_o)
    t0 = time.time()
    facs = {}
    for o, cols in blocks.items():
        Bo = B[:, cols]
        H = (mu * sp.identity(len(cols)) + rho * (Bo.T @ Bo)).tocsc()
        facs[o] = (Bo, H)
    t_factor = time.time() - t0

    def refactor(rho_):
        for o, cols in blocks.items():
            Bo = facs[o][0]
            facs[o] = (Bo, (mu * sp.identity(len(cols))
                            + rho_ * (Bo.T @ Bo)).tocsc())

    x = np.ones(nW)
    Bx = {o: facs[o][0] @ x[blocks[o]] for o in origins}
    zbar = sum(Bx.values()) / N
    u = np.zeros(nO)
    trace = []
    t0 = time.time()
    for it in range(iters):
        Bx_bar = sum(Bx.values()) / N
        # ---- x-updates: independent per-origin ridge LS (parallelizable) ----
        for o in origins:
            Bo, H = facs[o]
            target = Bx[o] - Bx_bar + zbar - u
            rhs = mu * np.ones(len(blocks[o])) + rho * (Bo.T @ target)
            xo = spsolve(H, rhs)
            x[blocks[o]] = np.clip(xo, bounds[0], bounds[1])
            Bx[o] = Bo @ x[blocks[o]]
        Bx_bar = sum(Bx.values()) / N
        # ---- zbar-update: closed form for g(v)=0.5||w(N zbar - c)||^2 ----
        zbar_old = zbar.copy()
        # argmin_z 0.5||w(Nz - c)||^2 + N rho/2 ||z - (Bx_bar + u)||^2
        #   => z = (w^2 c + rho (Bx_bar + u)) / (w^2 N + rho)
        zbar = (w * w * counts + rho * (Bx_bar + u)) / (w * w * N + rho)
        # ---- dual ----
        u = u + Bx_bar - zbar
        r_pri = float(np.linalg.norm(Bx_bar - zbar))
        r_dua = float(rho * np.linalg.norm(zbar - zbar_old))
        v = N * Bx_bar / scale
        c_raw = counts / scale
        m = c_raw > 0
        rmse = float(np.sqrt(((v - c_raw)[m] ** 2).sum() / max(m.sum() - 1, 1))
                     / c_raw[m].mean() * 100)
        trace.append(dict(it=it, r_pri=round(r_pri, 4), r_dual=round(r_dua, 4),
                          prmse=round(rmse, 2)))
        if verbose and (it % 10 == 0 or it == iters - 1):
            print(f"  admm it {it:3d}  r_pri {r_pri:10.3f}  r_dual {r_dua:10.3f} "
                  f" %RMSE {rmse:7.2f}")
        if r_pri < 1e-3 * np.linalg.norm(zbar) and r_dua < 1e-4:
            break
        # adaptive rho (Boyd 3.4.1): rebalance primal/dual residuals
        if it % 10 == 9:
            if r_pri > 10 * r_dua:
                rho *= 2.0; u /= 2.0; refactor(rho)
            elif r_dua > 10 * r_pri:
                rho /= 2.0; u *= 2.0; refactor(rho)
    return x, trace, dict(t_factor=round(t_factor, 1),
                          t_admm=round(time.time() - t0, 1),
                          n_blocks=N)


# ---------------------------------------------------------------------------
# demo driver on a benchmark case directory
# ---------------------------------------------------------------------------
def run_demo(case_dir, name, obs_share=0.10, perturb=0.35, mu=0.5, seed=7):
    """Perturb the seed OD by random per-origin factors, keep base-year loaded
    volumes as counts on a sample of links, and let influence-restricted ADMM
    pull the OD back — measuring reduction, convergence, fit, drift, time."""
    sys_path_fix = os.path.join(HERE)
    import sys
    sys.path.insert(0, sys_path_fix)
    from tcg_benchmark import extract
    X = extract(case_dir)
    Pi, od_col, loaded = X["Pi"], X["od_col"], X["loaded"]
    rng = np.random.default_rng(seed)

    # counted links: top obs_share by loaded volume (count-station realism)
    order = np.argsort(-loaded)
    obs_rows = order[: max(int(len(loaded) * obs_share), 20)]
    counts = loaded[obs_rows]

    # seed OD = kernel demand PERTURBED by per-origin factors (ground truth 1/f)
    od0_full = np.zeros(Pi.shape[1])
    origin_full = np.zeros(Pi.shape[1], dtype=int)
    with open(os.path.join(case_dir, "route_assignment.csv")) as f:
        pass  # od volumes come via od_col ordering below
    # reconstruct od volumes from route file aggregation done in extract():
    # extract() built od_col from route volumes; rebuild the vector directly
    odv = {}
    with open(os.path.join(case_dir, "route_assignment.csv")) as f:
        for r in csv.DictReader(f):
            try:
                key = (int(r["o_zone_id"]), int(r["d_zone_id"]))
                odv[key] = odv.get(key, 0.0) + float(r["volume"] or 0)
            except (ValueError, KeyError):
                continue
    for od, c in od_col.items():
        od0_full[c] = odv[od]
        origin_full[c] = od[0]
    fac = {}
    for o in np.unique(origin_full):
        fac[o] = rng.uniform(1 - perturb, 1 + perturb)
    od_pert = od0_full * np.array([fac[o] for o in origin_full])

    # influence restriction
    Pi_obs, active, stats = restrict_to_observed(Pi, obs_rows)
    od0 = od_pert[active]
    origin_of = origin_full[active]
    v_start = np.asarray(Pi_obs @ od0).ravel()
    m = counts > 0
    rmse0 = float(np.sqrt(((v_start - counts)[m] ** 2).sum()
                          / max(m.sum() - 1, 1)) / counts[m].mean() * 100)
    print(f"[{name}] reduction: {stats['n_od_active']:,} active of "
          f"{stats['n_od_total']:,} od columns ({stats['reduction']:.1%} dropped); "
          f"start %RMSE {rmse0:.1f}")

    x, trace, tinfo = admm_od_adjust(Pi_obs, od0, counts, origin_of, mu=mu)
    drift = np.abs(x - 1)
    out = dict(case=name, **stats, prmse_start=round(rmse0, 2),
               prmse_end=trace[-1]["prmse"], admm_iters=len(trace),
               drift_max=round(float(drift.max()), 3),
               drift_mean=round(float(drift.mean()), 3),
               true_perturb=perturb, mu=mu, **tinfo,
               trace_head=trace[:3], trace_tail=trace[-3:])
    os.makedirs(os.path.join(HERE, "review"), exist_ok=True)
    with open(os.path.join(HERE, "review", f"column_scale_{name}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("trace_head", "trace_tail")}, indent=1))
    return out


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "chicago_sketch"
    run_demo(os.path.join(HERE, f"bench_{name}"), name)
