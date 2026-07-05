"""config_extract.py — consolidate ALL model coefficients into one config file.

Reads TRMG2's own coefficient tables (via make_od_4period's parsers) and writes a
single JSON, config/model_coefficients.json, so the model is config-driven: the
small-form model (tensor_ftt/small_model.py) reads this config + the pre-assignment
tensors and reproduces the system without touching the master CSVs.

Run: python config_extract.py   ->  config/model_coefficients.json
"""
import csv
import json
import os

import make_od_4period as M   # reuse the validated parsers (PURPOSES, parse_*)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DC = os.path.join(REPO, "master", "resident", "dc")
TOD = os.path.join(REPO, "master", "resident", "tod")


def _kv(path, key, *vals):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r[key]] = {v: r.get(v) for v in vals}
    return out


def main():
    cfg = {"_meta": {"source": "TRMG2 master/resident coefficient tables",
                     "purposes": list(M.PURPOSES), "note": "coefficients are "
                     "TRMG2's own; do not invent. See doc/tensor_4step_math.tex."},
           "purposes": {}}

    for ptype, (edakey, size_cols) in M.PURPOSES.items():
        z = M.parse_zone_table(os.path.join(DC, ptype.lower() + "_zone.csv"))
        c = M.parse_cluster_table(os.path.join(DC, ptype.lower() + "_cluster.csv"))
        cfg["purposes"][ptype] = {
            "eda_key": list(edakey),
            "size_cols": size_cols,
            "dc_utility": {   # the destination-utility coefficients (Step 2)
                "beta_size": z["size"], "beta_time": z["time"],
                "beta_iz": z["iz"], "beta_shadow": z["shadow"],
                "cutoff_min": z["cutoff"],
                "piecewise": [{"threshold": t, "coef": cc} for t, cc in z["piecewise"]],
                "deferred_terms": z["deferred"],   # mc_logsum / access (need transit skims)
            },
            "clusters": {str(k): {"theta": v["theta"], "asc": v["asc"], "ic": v["ic"]}
                         for k, v in c.items()},
        }

    # time-of-day, directionality, occupancy, other-shares, size-term field weights
    cfg["time_of_day"] = _kv(os.path.join(TOD, "time_of_day_factors.csv"),
                             "trip_type", "tod", "factor") or "see tod/time_of_day_factors.csv"
    cfg["size_term_fields"] = {}
    st = os.path.join(DC, "dc_size_terms.csv")
    if os.path.exists(st):
        with open(st) as f:
            rows = list(csv.DictReader(f))
        cols = [k for k in rows[0] if k not in ("Field", "Description")]
        for col in cols:
            cfg["size_term_fields"][col] = {r["Field"]: float(r[col])
                                            for r in rows if r.get(col) and float(r[col]) != 0}

    os.makedirs(os.path.join(HERE, "config"), exist_ok=True)
    out = os.path.join(HERE, "config", "model_coefficients.json")
    with open(out, "w") as f:
        json.dump(cfg, f, indent=2)
    npur = len(cfg["purposes"])
    ncl = sum(len(p["clusters"]) for p in cfg["purposes"].values())
    print(f"wrote {out}")
    print(f"  {npur} purposes, {ncl} cluster rows, "
          f"{len(cfg['size_term_fields'])} size-term columns")
    print(f"  size {os.path.getsize(out)//1024} KB")


if __name__ == "__main__":
    main()
