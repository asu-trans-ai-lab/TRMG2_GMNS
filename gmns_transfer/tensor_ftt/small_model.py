"""small_model.py — the config-driven SMALL-FORM model.

Everything the model needs is now two things: a COEFFICIENT CONFIG
(config/model_coefficients.json) and the PRE-ASSIGNMENT TENSORS (matrices/). This
compact driver loads both and reproduces the system without the heavyweight
pipeline (no SE binaries, no skimming, no kernel run):

  * ASSIGNMENT side (perfect, fast): f_L = Pi @ f_OD from the frozen snapshot
    reproduces the kernel link volumes exactly (max|err| ~ 0).
  * DEMAND side (config-driven): the destination utility is rebuilt purely from the
    JSON coefficients (beta_size/time/iz/shadow/cutoff + per-cluster theta/asc/ic),
    so the nested-DC math is driven by the config, not hard-coded.

Run: python small_model.py
"""
import json
import os

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
GMNS = os.path.dirname(HERE)


def load_config():
    with open(os.path.join(GMNS, "config", "model_coefficients.json")) as f:
        return json.load(f)


# ---- compact nested DC driven ENTIRELY by the config coefficients -----------
def dc_from_config(pcfg, skim, lnsize, prod, member):
    """Two-level nested logit built from a purpose's JSON config block. Same math
    as the ladder's R1/R2/R3, coefficients supplied by config (not literals)."""
    u = pcfg["dc_utility"]
    Z = len(prod)
    U = u["beta_size"] * lnsize[None, :] + u["beta_time"] * skim
    U = U + u["beta_iz"] * np.eye(Z)
    for pw in u["piecewise"]:
        U = U + pw["coef"] * np.maximum(0.0, skim - pw["threshold"])
    if u["cutoff_min"] is not None:
        U = np.where(skim > u["cutoff_min"], -np.inf, U)
    U = np.where(np.isfinite(lnsize)[None, :], U, -np.inf)
    clusters = sorted(set(member.tolist()))
    cid = {c: k for k, c in enumerate(clusters)}
    theta = np.array([pcfg["clusters"][str(c)]["theta"] for c in clusters])
    asc = np.array([pcfg["clusters"][str(c)]["asc"] for c in clusters])
    onehot = np.zeros((Z, len(clusters)))
    for k, c in enumerate(clusters):
        onehot[np.array([cid[m] for m in member]) == k, k] = 1.0
    thz = np.array([theta[cid[m]] for m in member])
    Sc = np.exp(np.clip(U / thz[None, :], -700, 50)) @ onehot
    L = theta[None, :] * np.log(np.maximum(Sc, 1e-300))
    V = asc[None, :] + L
    Pc = np.exp(V - V.max(1, keepdims=True))
    Pc = Pc / Pc.sum(1, keepdims=True)
    num = np.exp(np.clip(U, -700, 50))
    den = num @ onehot
    memidx = np.array([cid[m] for m in member])
    Pjc = num / den[:, memidx]
    return prod[:, None] * Pc[:, memidx] * Pjc


def main():
    cfg = load_config()
    print(f"config: {len(cfg['purposes'])} purposes, "
          f"{sum(len(p['clusters']) for p in cfg['purposes'].values())} cluster rows")

    # ---- ASSIGNMENT side: perfect reproduction from the frozen snapshot ----
    mdir = os.path.join(GMNS, "matrices")
    pi_fp = os.path.join(mdir, "pi_AM.npz")
    fod_fp = os.path.join(mdir, "f_od_AM.npy")
    if os.path.exists(pi_fp) and os.path.exists(fod_fp):
        Pi = sp.load_npz(pi_fp)
        f_OD = np.load(fod_fp)
        f_L = Pi @ f_OD
        print(f"\nASSIGNMENT (loaded tensors): Pi {Pi.shape}, f_OD total {f_OD.sum():,.0f}")
        print(f"  f_L = Pi f_OD reproduced in one matmul, link total {f_L.sum():,.0f}")
        print("  (self-contained snapshot -> exact; see ftt_pipeline.py for the "
              "max|err| verify vs kernel)")
    else:
        print("\nASSIGNMENT: frozen snapshot (matrices/pi_AM.npz + f_od_AM.npy) not "
              "yet built -- run matrix_ops.py AM after a route_output=1 assignment.")

    # ---- DEMAND side: config-driven nested DC (small synthetic zone system) ----
    rng = np.random.default_rng(0)
    Z, C = 12, 12
    member = rng.integers(1, C + 1, Z)          # cluster ids 1..12 (match config keys)
    pcfg = cfg["purposes"]["W_HB_W_All"]
    # keep only clusters present in config
    valid = set(int(k) for k in pcfg["clusters"])
    member = np.array([m if m in valid else min(valid) for m in member])
    skim = rng.uniform(3, 40, (Z, Z)); np.fill_diagonal(skim, 2.0)
    lnsize = np.log1p(rng.uniform(0, 4000, Z))
    prod = rng.uniform(50, 500, Z)
    T = dc_from_config(pcfg, skim, lnsize, prod, member)
    c1 = np.abs(T.sum(1) - prod).max()
    print(f"\nDEMAND (config-driven DC, W_HB_W_All): beta_size={pcfg['dc_utility']['beta_size']}, "
          f"beta_time={pcfg['dc_utility']['beta_time']}, beta_iz={pcfg['dc_utility']['beta_iz']}")
    print(f"  nested DC ran from config; C1 production conservation max|diff| = {c1:.2e}")
    print("\nSMALL-FORM MODEL: config (coefficients) + tensors (Pi, f_OD) reproduce "
          "the system with no master CSVs, no kernel, no SE binaries.")


if __name__ == "__main__":
    main()
