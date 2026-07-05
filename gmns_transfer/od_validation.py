"""od_validation.py — is the upstream 3-step OD reasonable? (MAJOR verification task)

We do NOT have TRMG2's official pre-assignment OD matrix. So we validate our
reproduced OD the only way that needs no extra input: assign it, sum all four
periods to a DAILY loaded network, and compare link volumes to the observed
2020 AWDT counts (day_count on ~4,200 two-way count stations) — overall and by
facility type, area type, screenline, and volume group.

VALIDATION ORDER (this is a learning discipline — follow it every time):
  STEP 1 · MAGNITUDE BIAS — always checked FIRST. Is the total loaded volume in
           the right ballpark? bias = (Sum model - Sum count) / Sum count. If the
           bias is large, the OD needs refinement (more trips / missing markets)
           and NOTHING ELSE MATTERS YET. Do not read correlation or RMSE while a
           big magnitude bias stands.
  STEP 2 · PATTERN — only AFTER magnitude bias is understood: scale-adjusted %RMSE,
           correlation R^2, GEH, and facility-type shares tell you whether the OD
           loads the network in the right *proportions*.
Our OD reproduces resident home-based AUTO travel only (~30% of trips), so its
magnitude bias is large and negative by construction — the first mark says
"refine: add NHB / commercial / external markets." Pattern is reported second, as
the diagnosis of *where* the shortfall concentrates.

Benchmarked against TRMG2's own published validation
(review/trmg2_benchmark_count_comparison_by_fac_type.csv, overall %RMSE 34.58).

When ITRE later shares the official OD matrix, call validate_vs_matrix() for a
direct OD-to-OD check on top — the count-based validation stays valid regardless.

Run: python od_validation.py   ->  review/od_validation.md + .json
"""
import csv
import json
import math
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PERIODS = ("AM", "MD", "PM", "NT")


def load_daily():
    """Return per two-way link: count (daily AWDT), modeled (daily, both dirs),
    facility type, area type, screenline."""
    # daily modeled volume per directed link = sum over periods
    vol = defaultdict(float)
    for per in PERIODS:
        fp = os.path.join(HERE, f"scenario_{per}", "link_performance.csv")
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            for r in csv.DictReader(f):
                vol[(r["from_node_id"], r["to_node_id"])] += float(r.get("volume") or 0)
    # attributes + counts from the base link file; aggregate to two-way stations
    twoway = {}
    with open(os.path.join(HERE, "scenario", "link.csv")) as f:
        for r in csv.DictReader(f):
            dc = float(r.get("day_count") or 0)
            if dc <= 0:
                continue
            a, b = r["from_node_id"], r["to_node_id"]
            key = tuple(sorted([a, b]))
            m = vol.get((a, b), 0.0)
            if key in twoway:
                twoway[key]["model"] += m           # add the reverse direction
            else:
                twoway[key] = dict(count=dc, model=m,
                                   fac=r.get("link_type_name", "?"),
                                   at=r.get("area_type", "?"),
                                   sl=r.get("screenline", "") or None)
    return list(twoway.values())


def metrics(pairs):
    """count/model metric bundle for a list of {count, model} dicts."""
    n = len(pairs)
    if n == 0:
        return None
    c = [p["count"] for p in pairs]
    m = [p["model"] for p in pairs]
    sc, sm = sum(c), sum(m)
    mean_c = sc / n
    # raw %RMSE (magnitude-sensitive)
    rmse = math.sqrt(sum((mi - ci) ** 2 for ci, mi in zip(c, m)) / n)
    prmse = 100 * rmse / mean_c if mean_c else 0
    # scale-adjusted: normalize model to same total, then %RMSE => PATTERN error
    k = sc / sm if sm else 0
    ms = [mi * k for mi in m]
    rmse_s = math.sqrt(sum((mi - ci) ** 2 for ci, mi in zip(c, ms)) / n)
    prmse_s = 100 * rmse_s / mean_c if mean_c else 0
    # correlation R^2
    mm = sm / n
    ssxy = sum((ci - mean_c) * (mi - mm) for ci, mi in zip(c, m))
    ssxx = sum((ci - mean_c) ** 2 for ci in c)
    ssyy = sum((mi - mm) ** 2 for mi in m)
    r2 = (ssxy ** 2 / (ssxx * ssyy)) if ssxx and ssyy else 0
    # GEH (on the scaled model, so it measures pattern not magnitude)
    geh = [math.sqrt(2 * (mi - ci) ** 2 / (mi + ci)) if (mi + ci) > 0 else 0
           for ci, mi in zip(c, ms)]
    geh5 = 100 * sum(1 for g in geh if g < 5) / n
    return dict(n=n, total_count=round(sc), total_model=round(sm),
                vol_ratio=round(sm / sc, 3) if sc else 0,
                pct_diff=round(100 * (sm - sc) / sc, 1) if sc else 0,
                prmse_raw=round(prmse, 1), prmse_scaled=round(prmse_s, 1),
                r2=round(r2, 3), geh5_pct=round(geh5, 1))


def by(pairs, keyfn):
    groups = defaultdict(list)
    for p in pairs:
        groups[keyfn(p)].append(p)
    return {k: metrics(v) for k, v in sorted(groups.items())}


def validate_vs_matrix(omx_or_csv_path):
    """FUTURE hook: when ITRE shares the official pre-assignment OD, load it and
    compare cell-by-cell to matrices/f_od_AM.npy (coincidence ratio, matrix %RMSE,
    per-district agreement). Count-based validation above is independent of this."""
    raise NotImplementedError(
        "Provide the official OD (OMX/CSV). Then compare vs matrices/f_od_*.npy: "
        "matrix %RMSE, coincidence ratio, 12-district cell agreement.")


def main():
    pairs = load_daily()
    overall = metrics(pairs)
    fac = by(pairs, lambda p: p["fac"])
    at = by(pairs, lambda p: p["at"])
    sl = by(pairs, lambda p: p["sl"] or "(none)")

    # TRMG2 benchmark by fac type (their published %RMSE)
    bench = {}
    bf = os.path.join(HERE, "review", "trmg2_benchmark_count_comparison_by_fac_type.csv")
    if os.path.exists(bf):
        for r in csv.DictReader(open(bf)):
            bench[r["HCMType"]] = dict(prmse=float(r["PRMSE"]), pctdiff=float(r["PctDiff"]))

    out = dict(overall=overall, by_facility=fac, by_area=at, by_screenline=sl,
               benchmark_fac=bench)
    os.makedirs(os.path.join(HERE, "review"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "review", "od_validation.json"), "w"), indent=1)

    bias = overall["pct_diff"]
    refine = abs(bias) > 10
    verdict = ("REFINE — magnitude bias is large; add the missing trip markets "
               "before trusting any pattern metric") if refine else \
              ("magnitude OK — proceed to read pattern")
    # markdown report — MAGNITUDE BIAS FIRST, ALWAYS
    L = ["# OD reasonableness — upstream 3-step OD vs observed counts\n",
         f"Daily loaded network (all 4 periods) vs 2020 AWDT on **{overall['n']:,} "
         "two-way count stations**. No official OD matrix required.\n",
         "> **How to verify an OD (the order matters).** Check **magnitude bias "
         "first**. Only if the total is in the right ballpark do you go on to read "
         "correlation or %RMSE. A big bias means the model needs more trips — no "
         "amount of good correlation fixes that.\n",
         "## ① Magnitude bias  ·  the first mark\n",
         f"**bias = (Σ model − Σ count) / Σ count = {bias:+.1f}%** "
         f"&nbsp;(modeled {overall['total_model']:,} vs counted {overall['total_count']:,} veh/day).\n",
         f"**Verdict: {verdict}.** Here the bias is {bias:+.1f}% — the OD reproduces "
         "resident home-based auto only (~30% of trips); NHB, commercial vehicles, "
         "university, airport and external markets are not yet added. That is the "
         "refinement the first mark calls for.\n",
         "## ② Pattern  ·  only after the magnitude bias is understood\n",
         f"With the magnitude gap set aside (model scaled to the count total), the "
         f"OD's *shape* gives scale-adjusted **%RMSE {overall['prmse_scaled']}**, "
         f"correlation **R² {overall['r2']}**, GEH<5 **{overall['geh5_pct']}%**. "
         "TRMG2's own published overall %RMSE is **34.58**. Read this only to see "
         "*where* the shortfall concentrates — not as a pass/fail.\n",
         "### By facility type\n",
         "| Facility | N | model/count | %RMSE (scaled) | R² | GEH<5% | TRMG2 %RMSE |",
         "|---|--:|--:|--:|--:|--:|--:|"]
    for k, v in sorted(fac.items(), key=lambda x: -(x[1]["total_count"] if x[1] else 0)):
        if not v:
            continue
        b = bench.get(k, {})
        L.append(f"| {k} | {v['n']} | {v['vol_ratio']} | {v['prmse_scaled']} | "
                 f"{v['r2']} | {v['geh5_pct']}% | {b.get('prmse','—')} |")
    L += ["\n### By area type\n",
          "| Area | N | model/count | %RMSE (scaled) | R² |", "|---|--:|--:|--:|--:|"]
    for k, v in at.items():
        if v:
            L.append(f"| {k} | {v['n']} | {v['vol_ratio']} | {v['prmse_scaled']} | {v['r2']} |")
    L += ["\n### By screenline\n", "| Screenline | N | model/count | R² |", "|---|--:|--:|--:|"]
    for k, v in sl.items():
        if v:
            L.append(f"| {k} | {v['n']} | {v['vol_ratio']} | {v['r2']} |")
    L += ["\n## Verdict (in order)\n",
          f"1. **Magnitude bias {bias:+.1f}% → {'REFINE' if refine else 'OK'}.** "
          "This is the first and decisive mark: add the missing trip markets "
          "(NHB, commercial, university, airport, external) to close it.",
          "2. **Only then, pattern.** The scale-adjusted %RMSE / R² above are the "
          "diagnosis of *where* the shortfall sits (largest on freeways — they carry "
          "the omitted through/commercial/long-distance travel), not a pass/fail.",
          "3. When ITRE shares the official pre-assignment OD, `validate_vs_matrix()` "
          "adds a direct OD-to-OD check. The magnitude-first count validation above "
          "stays valid regardless.\n"]
    open(os.path.join(HERE, "review", "od_validation.md"), "w", encoding="utf-8").write("\n".join(L))

    print("=== OD reasonableness · verification order ===")
    print(f"[1] MAGNITUDE BIAS (check first): {bias:+.1f}%  "
          f"(model {overall['total_model']:,} vs count {overall['total_count']:,} veh/day)")
    print(f"    -> {'REFINE: add missing trip markets' if refine else 'OK: magnitude in range'}")
    print(f"[2] pattern (only after #1): scaled %RMSE {overall['prmse_scaled']}, "
          f"R^2 {overall['r2']}, GEH<5 {overall['geh5_pct']}%  (TRMG2 published 34.58)")
    print("wrote review/od_validation.md + .json")


if __name__ == "__main__":
    main()
