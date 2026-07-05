"""scaling_experiment.py — quantify the route-frozen surrogate's validity domain.

Reviewer R3 (round 3): "C3 with frozen Delta is valid only for small demand
perturbations — show WHERE it breaks." This experiment scales the frozen f_OD by
a set of factors and compares, per scale:

  frozen-Pi link flows   f_L = Pi (s * f_OD)         (one matmul, no re-assign)
  frozen-Pi link times   t_L = BPR(f_L)              (using link fftt + capacity)

against a FULL kernel re-assignment at the same scaled demand (optional, slow —
enabled with --kernel). Without --kernel it still reports the leading indicator:
how many links change BPR regime (v/c crossing 0.9 / 1.0) under the frozen
routing — the reviewer's stated break condition.

Requires a CONSISTENT snapshot (run matrix_ops.py AM after a full-coverage
route_output run). Usage:
  python scaling_experiment.py                # indicator table, scales 0.7..1.3
  python scaling_experiment.py --kernel 1.3   # also kernel-truth at one scale
"""
import csv
import os
import sys

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
GMNS = os.path.dirname(HERE)


def load(per="AM"):
    mdir = os.path.join(GMNS, "matrices")
    Pi = sp.load_npz(os.path.join(mdir, f"pi_{per}.npz"))
    f_OD = np.load(os.path.join(mdir, f"f_od_{per}.npy"))
    # link attributes for BPR
    fftt, cap = [], []
    with open(os.path.join(GMNS, "scenario", "link.csv")) as f:
        for r in csv.DictReader(f):
            fftt.append(float(r.get("vdf_fftt") or 0) or 1e-3)
            cap.append(float(r.get("capacity") or 1) or 1.0)
    return Pi, f_OD, np.array(fftt), np.array(cap)


def main():
    per = "AM"
    Pi, f_OD, fftt, cap = load(per)
    scales = [0.7, 0.9, 1.0, 1.1, 1.3]
    base_fL = Pi @ f_OD
    base_vc = base_fL / cap
    print(f"[{per}] frozen-Pi scaling experiment  (|L|={len(cap):,}, "
          f"f_OD total {f_OD.sum():,.0f})")
    print(f"{'scale':>6} {'veh':>12} {'links v/c>0.9':>14} {'links v/c>1.0':>14} "
          f"{'new >0.9 vs base':>17} {'max v/c':>8}")
    for s in scales:
        fL = base_fL * s                       # linear in the frozen operator
        vc = fL / cap
        n9, n10 = int((vc > 0.9).sum()), int((vc > 1.0).sum())
        new9 = int(((vc > 0.9) & (base_vc <= 0.9)).sum())
        print(f"{s:>6.1f} {fL.sum():>12,.0f} {n9:>14,} {n10:>14,} "
              f"{new9:>17,} {vc.max():>8.2f}")
    print("\nReading: every link newly crossing v/c 0.9 under scaling is a link "
          "whose ROUTE PATTERN the frozen Pi cannot adjust — the surrogate's bias "
          "grows with that column. Kernel-truth comparison: --kernel <scale> "
          "(writes scaled demand, re-assigns, diffs link times).")

    if "--kernel" in sys.argv:
        s = float(sys.argv[sys.argv.index("--kernel") + 1])
        print(f"\n[kernel truth at scale {s}] scale demand files, re-run kernel, "
              f"compare t_L — see doc/REVIEW4_FABLE5.md open item 2. "
              f"(Manual step: scale scenario_AM demand CSVs by {s}, run DTALite, "
              f"diff link_performance travel_time vs BPR(frozen f_L).)")


if __name__ == "__main__":
    main()
