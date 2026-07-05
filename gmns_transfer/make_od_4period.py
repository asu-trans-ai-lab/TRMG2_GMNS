"""v1.5 resident HB demand: od_veh_trips_{AM,MD,PM,NT} from the first 3 steps,
using TRMG2's OWN destination-choice tables — no invented parameters.

Chain (sources verified against src/*.rsc; see REPRODUCTION_MAP.md):
  1. Generation (05): P_i^p = HH_POP_i x sample-weighted person rate
     (master/resident/generation/production_rates.csv). [Stand-in for the
     person-level decision-tree application, pending population synthesis.]
  2. Destination choice (13 + utils/NestedDC.rsc): two-level nested logit,
     REPRODUCED FROM THEIR TABLES:
       - zone utility rows from master/resident/dc/<purpose>_zone.csv:
         coef x ln(size term), CongTime coef, piecewise max(0, t - X) terms,
         "t > X" -99 cutoffs, intra_cluster.IZ dummies, sp.hbw shadow prices
         (warm-started from shadow_prices.bin);
       - cluster nest from <purpose>_cluster.csv: Theta, ASC+Calibrated_DeltaASC,
         IC+Calibrated_DeltaIC (IC applied when origin is in the cluster);
       - clusters from master_tazs.shp CLUSTER field.
     Deferred rows (printed at run time): mc_logsums composites (needs step 12),
     se.access_* terms (needs step 03).
  3. Period skims (10/13): AM & PM use the AM congested times, MD & NT the MD
     times (13:216-225 tour-type convention), from init_cong_time_2020.bin.
  4. Mode (12 stand-in): survey shares per purpose (eda_scheme6.csv) with
     auto_pay/other_auto folded in; hov occupancies + TOD/directionality from
     master/resident/tod/*.csv (exact files, step 08/16).
Output per period: scenario_{per}/ + kernel run + daily count comparison.
v1.5 scope: resident HB only (no NHB/CV/univ/airport/externals yet).
"""
import csv
import json
import os
import shutil
import sys
import time
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

KERNEL_EXE = os.environ.get(
    "TAPLITE_EXE",
    "C:/source_codes/0_source_code_new/dtalite_with_taplite_Cpp_kernel/bin/DTALite.exe")

PERIODS = ["AM", "MD", "PM", "NT"]
PERIOD_HOURS = {"AM": 2.423, "MD": 5.5, "PM": 2.753, "NT": 4.0}
BUILD_HOURS = 2.423
# v1.6: each period uses its OWN congested skim (13:216-225 builds per-period
# sov_skim; the earlier AM->PM / MD->NT reuse was a v1.5 shortcut).
SKIM_OF = {"AM": "AM", "PM": "PM", "MD": "MD", "NT": "NT"}
PURPOSES = {   # trip_type -> (eda key (tour, purpose, duration), size-term columns)
    "W_HB_W_All":      (("W", "W", "All"),     ["w_hbw_size_H", "w_hbw_size_L"]),
    "W_HB_O_All":      (("W", "O", "All"),     ["w_hbo_size"]),
    "W_HB_EK12_All":   (("W", "EK12", "All"),  ["w_hbek12_size"]),
    "N_HB_OME_All":    (("N", "OME", "All"),   ["n_hbome_size"]),
    "N_HB_OMED_All":   (("N", "OMED", "All"),  ["n_hbomed_size"]),
    "N_HB_OD_Short":   (("N", "OD", "Short"),  ["n_hbods_size"]),
    "N_HB_OD_Long":    (("N", "OD", "Long"),   ["n_hbodl_size"]),
    "N_HB_K12_All":    (("N", "K12", "All"),   ["n_hbk12_size"]),
}
MIN_WRITE = 0.05


def read_kv(path, keycols, valcol):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[tuple(r[k] for k in keycols)] = float(r[valcol])
    return out


def parse_zone_table(path):
    """Parse a dc/<purpose>_zone.csv into structured coefficient spec."""
    spec = dict(size=0.0, time=0.0, piecewise=[], cutoff=None, iz=0.0,
                shadow=0.0, deferred=[])
    seen_size = []
    with open(path) as f:
        for r in csv.DictReader(f):
            expr = r["Expression"].strip().strip('"')
            try:
                coef = float(r["Coefficient"])
            except (ValueError, TypeError):
                continue
            if expr.startswith("se.") and "_size" in expr:
                seen_size.append(coef)
            elif expr == "sov_skim.CongTime":
                spec["time"] += coef
            elif expr.startswith("max(0, sov_skim.CongTime -"):
                thr = float(expr.split("-")[-1].rstrip(") "))
                spec["piecewise"].append((thr, coef))
            elif expr.startswith("sov_skim.CongTime >"):
                thr = float(expr.split(">")[-1])
                spec["cutoff"] = thr if spec["cutoff"] is None else min(spec["cutoff"], thr)
            elif expr == "intra_cluster.IZ":
                spec["iz"] += coef
            elif expr.startswith("sp."):
                spec["shadow"] += coef
            else:
                spec["deferred"].append(f"{expr} ({coef})")
    spec["size"] = float(np.mean(seen_size)) if seen_size else 1.0
    return spec


def parse_cluster_table(path):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[int(r["Cluster"])] = dict(
                theta=max(float(r["Theta"]), 0.05),
                asc=float(r["ASC"]) + float(r["Calibrated_DeltaASC"] or 0),
                ic=float(r["IC"]) + float(r["Calibrated_DeltaIC"] or 0))
    return out


def nested_dc(P, U, zone_cluster, ctab):
    """Two-level nested DC (NestedDC.rsc): returns T = P_i x P(c|i) x P(j|c,i).
    U: Z x Z zone utilities (-inf = unavailable)."""
    Z = U.shape[0]
    clusters = sorted(set(zone_cluster))
    Tm = np.zeros_like(U)
    Vc = np.full((Z, len(clusters)), -1e30)
    parts = []
    for k, c in enumerate(clusters):
        cj = np.where(np.array(zone_cluster) == c)[0]
        p = ctab.get(c, dict(theta=1.0, asc=0.0, ic=0.0))
        Uc = U[:, cj] / p["theta"]
        m = Uc.max(axis=1, keepdims=True)
        ok = m[:, 0] > -1e29
        E = np.where(np.isfinite(Uc), np.exp(np.clip(Uc - m, -700, 0)), 0.0)
        S = E.sum(axis=1)
        logsum = np.where(ok & (S > 0), m[:, 0] + np.log(np.maximum(S, 1e-300)), -1e30)
        ic_dummy = (np.array(zone_cluster) == c).astype(float)
        Vc[:, k] = np.where(logsum > -1e29,
                            p["asc"] + p["ic"] * ic_dummy + p["theta"] * logsum, -1e30)
        # v1.6: within-cluster conditional uses UNSCALED utilities (theta only
        # enters the logsum), per NestedDC.rsc CalcFinalProbs:395-405.
        Uu = U[:, cj]
        mu = Uu.max(axis=1, keepdims=True)
        Eu = np.where(np.isfinite(Uu), np.exp(np.clip(Uu - mu, -700, 0)), 0.0)
        with np.errstate(invalid="ignore"):
            Pjc = Eu / np.maximum(Eu.sum(axis=1)[:, None], 1e-300)
        parts.append((cj, Pjc))
    mV = Vc.max(axis=1, keepdims=True)
    EV = np.where(Vc > -1e29, np.exp(np.clip(Vc - mV, -700, 0)), 0.0)
    Pc = EV / np.maximum(EV.sum(axis=1, keepdims=True), 1e-300)
    for k, (cj, Pjc) in enumerate(parts):
        Tm[:, cj] = (P * Pc[:, k])[:, None] * Pjc
    return Tm


def main():
    t_start = time.time()
    # ---------------- shared inputs ----------------
    eda = {}
    with open(os.path.join(REPO, "docs", "data", "output", "survey_processing",
                           "eda_scheme6.csv")) as f:
        for r in csv.DictReader(f):
            if r["homebased"] == "HB":
                eda[(r["tour_type"], r["purpose"], r["duration"])] = r
    tod_f = read_kv(os.path.join(REPO, "master", "resident", "tod",
                                 "time_of_day_factors.csv"), ["trip_type", "tod"], "factor")
    pa_f = read_kv(os.path.join(REPO, "master", "resident", "tod",
                                "directionality_factors.csv"), ["trip_type", "tod"], "pa_fac")
    occ3 = read_kv(os.path.join(REPO, "master", "resident", "tod",
                                "hov3_occ_factors_hb.csv"), ["trip_type", "tod"], "hov3")
    # N2 fix: auto_pay/other_auto fold shares (NOT survey pct_sov/hov2/hov3)
    other_sh = {}
    with open(os.path.join(REPO, "master", "resident", "tod", "other_shares_hb.csv")) as f:
        for r in csv.DictReader(f):
            other_sh[r["trip_type"]] = (float(r["sov"]), float(r["hov2"]), float(r["hov3"]))
    rates = defaultdict(lambda: [0.0, 0.0])
    with open(os.path.join(REPO, "master", "resident", "generation",
                           "production_rates.csv")) as f:
        for r in csv.DictReader(f):
            w = float(r["samples"] or 0)
            rates[r["trip_type"]][0] += w
            rates[r["trip_type"]][1] += w * float(r["rate"] or 0)
    rate_p = {p: ws / w for p, (w, ws) in rates.items() if w > 0}

    from se_loader import load_se
    se_rows = load_se(REPO, year=2020)   # reads gmns/se_data/se_2020.csv (CSV-first)
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
    from se_loader import load_table
    sp_rows = load_table(os.path.join(HERE, "params", "shadow_prices.csv"),
                         os.path.join(REPO, "master", "resident", "dc", "shadow_prices.bin"))
    sp_hbw = {r["TAZ"]: (r.get("hbw") or 0.0) for r in sp_rows}

    # clusters from TAZ layer
    import geopandas as gpd
    g = gpd.read_file(os.path.join(REPO, "docs", "data", "input", "tazs",
                                   "master_tazs.shp")).set_index("ID")
    taz_cluster = g["CLUSTER"].astype(int).to_dict()
    taz_area = g["AREA"].astype(float).to_dict()   # sq mi, for intrazonal time

    # ---------------- network + zones ----------------
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
    N = len(nodes)
    zones = sorted(z for z in taz_of if taz_of[z] in se)
    Z = len(zones)
    src = [nix[z] for z in zones]
    zone_cluster = [taz_cluster.get(taz_of[z], -1) for z in zones]

    # per-period congested times from init_cong_time (fallback fftt)
    init_rows = load_table(os.path.join(HERE, "params", "init_cong_time_2020.csv"),
                           os.path.join(REPO, "master", "networks", "init_cong_time_2020.bin"))
    init = {r["ID"]: r for r in init_rows}
    link_rows = []
    with open(os.path.join(gm0, "link.csv")) as f:
        for r in csv.DictReader(f):
            link_rows.append(r)

    def build_time_graph(per):
        rows, cols, w = [], [], []
        for r in link_rows:
            base, d = int(r["link_id"]) // 10, int(r["link_id"]) % 10
            it = init.get(base)
            key = ("AB" if d == 1 else "BA") + "InitCongTime" + per
            t = (it or {}).get(key) or 0
            if not t or t <= 0:
                t = float(r["vdf_fftt"])
            rows.append(nix[int(r["from_node_id"])])
            cols.append(nix[int(r["to_node_id"])])
            w.append(max(t, 1e-3))
        return sp.csr_matrix((w, (rows, cols)), shape=(N, N))

    # v1.6: intrazonal time = (sqrt(area)*sqrt(2)/3) * (60/30) min at 30 mph
    # (10 - Skimming.rsc:78-80), applied to the skim diagonal so intrazonal
    # destinations are real alternatives (they don't load the network).
    iz_time = np.array([(taz_area.get(taz_of[z], 1.0) ** 0.5) * (2 ** 0.5) / 3
                        * (60.0 / 30.0) for z in zones])

    t0 = time.time()
    skims = {}
    for sk in PERIODS:   # v1.6: all four periods, own skim each
        d = dijkstra(build_time_graph(sk), directed=True, indices=src)[:, src]
        d = np.where(np.isfinite(d), d, 1e4)
        np.fill_diagonal(d, iz_time)
        skims[sk] = d
    print(f"period skims ({', '.join(PERIODS)}) in {time.time()-t0:.0f}s")

    HHPOP = np.array([se[taz_of[z]]["HH_POP"] or 0 for z in zones], dtype=float)
    # N4 fix: zone-level demographic generation base (uses real Pct_Worker/Child
    # per zone -> spatial variation), calibrated so regional total = survey wTrips.
    PCTW = np.array([(se[taz_of[z]].get("Pct_Worker") or 0) / 100.0 for z in zones])
    PCTC = np.array([(se[taz_of[z]].get("Pct_Child") or 0) / 100.0 for z in zones])
    BASE = {"W_HB_W_All": HHPOP * PCTW, "W_HB_O_All": HHPOP * PCTW,
            "W_HB_EK12_All": HHPOP * PCTC, "N_HB_K12_All": HHPOP * PCTC,
            "N_HB_OME_All": HHPOP, "N_HB_OMED_All": HHPOP,
            "N_HB_OD_Short": HHPOP, "N_HB_OD_Long": HHPOP}

    def se_field(z, field):
        r = se[z]
        if field.endswith("_EH") or field.endswith("_EL"):
            base = field[:-3]
            pct = (r.get("PctHighPay") or 0) / 100.0
            share = pct if field.endswith("_EH") else 1.0 - pct
            return (r.get(base) or 0) * share
        return r.get(field) or 0

    spvec = np.array([sp_hbw.get(taz_of[z], 0.0) for z in zones])
    # v1.6: intra_cluster.IZ is the INTRAZONAL dummy (i==j diagonal), per
    # dc/intrazonals.csv + 13 - Destination Choice.rsc — NOT same-cluster.
    intra_diag = np.eye(Z)

    # ---------------- per purpose: THEIR nested DC ----------------
    per_class_od = {(per, m): np.zeros((Z, Z))
                    for per in PERIODS for m in ("sov", "hov2", "hov3")}
    diag = {}
    t0 = time.time()
    for ptype, (edakey, size_cols) in PURPOSES.items():
        e = eda.get(edakey)
        if e is None:
            continue
        zspec = parse_zone_table(os.path.join(REPO, "master", "resident", "dc",
                                              ptype.lower() + "_zone.csv"))
        ctab = parse_cluster_table(os.path.join(REPO, "master", "resident", "dc",
                                                ptype.lower() + "_cluster.csv"))
        if zspec["deferred"]:
            print(f"  {ptype}: deferred utility rows -> {zspec['deferred']}")
        A = np.zeros(Z)
        for col in size_cols:
            for field, coef in size_coef[col].items():
                A += np.array([coef * se_field(taz_of[z], field) for z in zones])
        A /= len(size_cols)
        # v1.6: size term is Log(1+size) (13 - Destination Choice.rsc:149-153),
        # not Log(size); zones with 0 size stay finite (log1p(0)=0), so the
        # -inf availability mask must come from the size being truly empty.
        lnA = np.where(A > 0, np.log1p(A), -np.inf)
        # N4: production = demographic base x rate; rate set so regional total
        # matches the survey wTrips control total for this purpose.
        base_vec = BASE.get(ptype, HHPOP)
        wtrips = float(e["wTrips"])
        rate = wtrips / max(base_vec.sum(), 1e-9)
        P = base_vec * rate

        # mode shares (stand-in for step 12). N2 fix: fold auto_pay+other_auto
        # by other_shares_hb.csv factors (paid/other-auto skew to HOV), not by
        # the sov/hov survey shares. transit/walk-bike correctly excluded.
        extra = float(e["pct_auto_pay"]) + float(e["pct_other_auto"])
        osv, oh2, oh3 = other_sh.get(ptype, (0.5, 0.3, 0.2))
        s_sov = (float(e["pct_sov"]) + extra * osv) / 100.0
        s_h2 = (float(e["pct_hov2"]) + extra * oh2) / 100.0
        s_h3 = (float(e["pct_hov3"]) + extra * oh3) / 100.0

        def build_U(tmat, sp_local):
            U = zspec["size"] * lnA[None, :] + zspec["time"] * tmat
            for thr, coef in zspec["piecewise"]:
                U = U + coef * np.maximum(0.0, tmat - thr)
            U = U + zspec["iz"] * intra_diag
            if zspec["shadow"]:
                U = U + zspec["shadow"] * sp_local[None, :]
            if zspec["cutoff"] is not None:
                U = np.where(tmat > zspec["cutoff"], -np.inf, U)
            return np.where(np.isfinite(lnA)[None, :], U, -np.inf)

        # N9: HBW is doubly-constrained. Iterate the shadow price so modeled
        # attractions match the balanced attraction target (Sigma A = Sigma P),
        # per 13 - Destination Choice.rsc:275-330 (3 passes, step 0.85, stop
        # at <2% RMSE). Other 7 purposes have no shadow term -> singly-constrained.
        sp_local = spvec.copy()
        if zspec["shadow"]:
            avail = np.isfinite(lnA)
            target = np.where(avail, A, 0.0)
            target = target * (P.sum() / max(target.sum(), 1e-9))
            # v1.6c: convergence metric = RelRMSE between SUCCESSIVE shadow-price
            # vectors, matching 13 - Destination Choice.rsc:326-329 (the kernel
            # tests sp_new vs sp, NOT modeled-vs-target attractions). Attraction
            # RMSE still reported as a diagnostic.
            attr_rmse = 0.0
            sp_rel = 0.0
            for it in range(3):
                Tm = nested_dc(P, build_U(skims["AM"], sp_local), zone_cluster, ctab)
                modeled = Tm.sum(axis=0)
                mask = (target > 0) & (modeled > 0)
                attr_rmse = float(np.sqrt(np.mean(
                    ((modeled[mask] - target[mask]) / target[mask]) ** 2)))
                sp_new = sp_local + 0.85 * mask * (
                    np.log(np.maximum(target, 1e-9)) - np.log(np.maximum(modeled, 1e-9)))
                denom = np.abs(sp_local[mask])
                sp_rel = float(np.sqrt(np.mean(
                    ((sp_new - sp_local)[mask] / np.maximum(denom, 1e-6)) ** 2)))
                sp_local = sp_new
                if sp_rel < 0.02:      # kernel's own stopping rule
                    break
            print(f"    {ptype} shadow-price loop: {it + 1} iters, "
                  f"successive-sp RelRMSE {sp_rel:.3f} (kernel rule), "
                  f"attr RMSE {attr_rmse:.3f} (diagnostic)")

        Tm_by_skim = {sk: nested_dc(P, build_U(tmat, sp_local), zone_cluster, ctab)
                      for sk, tmat in skims.items()}
        # diagnostic average congested time (reported, NOT calibrated)
        TmAM = Tm_by_skim["AM"]
        att = float((TmAM * skims["AM"]).sum() / max(TmAM.sum(), 1e-9))
        diag[ptype] = dict(daily_person_trips=round(float(TmAM.sum())),
                           avg_am_time_min=round(att, 1))
        print(f"  {ptype}: {TmAM.sum():,.0f} person trips, avg AM time {att:.1f} min")

        for per in PERIODS:
            Tm = Tm_by_skim[SKIM_OF[per]]
            PA = Tm * tod_f.get((ptype, per), 0.0)
            f = pa_f.get((ptype, per), 0.5)
            OD = f * PA + (1 - f) * PA.T
            o3 = occ3.get((ptype, per), 3.5)
            per_class_od[(per, "sov")] += OD * s_sov
            per_class_od[(per, "hov2")] += OD * s_h2 / 2.0
            per_class_od[(per, "hov3")] += OD * s_h3 / o3
    print(f"nested DC (8 purposes x 2 skims) in {time.time()-t0:.0f}s")

    # ---------------- scenarios + kernel ----------------
    daily_flow = defaultdict(float)
    results = {}
    for per in PERIODS:
        sdir = os.path.join(HERE, f"scenario_{per}")
        os.makedirs(sdir, exist_ok=True)
        shutil.copy(os.path.join(gm0, "node.csv"), sdir)
        scale = PERIOD_HOURS[per] / BUILD_HOURS
        with open(os.path.join(sdir, "link.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(link_rows[0].keys()))
            w.writeheader()
            for r in link_rows:
                r2 = dict(r)
                r2["capacity"] = round(float(r["capacity"]) * scale, 1)
                w.writerow(r2)
        tot = {}
        for m in ("sov", "hov2", "hov3"):
            od = per_class_od[(per, m)]
            with open(os.path.join(sdir, f"demand_{m}.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["o_zone_id", "d_zone_id", "volume"])
                oi, dj = np.nonzero(od >= MIN_WRITE)
                for i, j in zip(oi, dj):
                    if i != j:
                        w.writerow([zones[i], zones[j], round(float(od[i, j]), 3)])
            tot[m] = float(od.sum())
        with open(os.path.join(sdir, "mode_type.csv"), "w", newline="") as f:
            f.write("mode_type_id,mode_type,name,vot,pce,occ,demand_file\n"
                    "1,sov,SOV,10,1,1,demand_sov.csv\n"
                    "2,hov2,HOV2,10,1,2,demand_hov2.csv\n"
                    "3,hov3,HOV3,10,1,3,demand_hov3.csv\n")
        with open(os.path.join(sdir, "settings.csv"), "w", newline="") as f:
            f.write("number_of_iterations,number_of_processors,demand_period_starting_hours,"
                    "demand_period_ending_hours,base_demand_mode,route_output,log_file,"
                    "odme_mode,odme_vmt\n30,8,7,8,0,0,0,0,0\n")
        import pytaplite
        t1 = time.time()
        pytaplite.assign(sdir, exe=KERNEL_EXE, prefer_inproc=False)
        vmt = 0.0
        with open(os.path.join(sdir, "link_performance.csv")) as f:
            for r in csv.DictReader(f):
                v = float(r["volume"] or 0)
                daily_flow[(int(r["from_node_id"]), int(r["to_node_id"]))] += v
                vmt += float(r.get("VMT") or 0)
        gap = None
        with open(os.path.join(sdir, "summary_log_file.txt")) as f:
            for line in f:
                if "iter No" in line:
                    gap = line.strip().split("gap =")[-1].strip()
        results[per] = dict(veh=round(sum(tot.values())), vmt=round(vmt),
                            secs=round(time.time() - t1), final_gap=gap)
        print(f"  [{per}] {results[per]}")

    # ---------------- daily count comparison ----------------
    per_link = {}
    for r in link_rows:
        if not r["day_count"]:
            continue
        base = int(r["link_id"]) // 10
        key = (int(r["from_node_id"]), int(r["to_node_id"]))
        d = per_link.setdefault(base, dict(count=float(r["day_count"]), vol=0.0,
                                           ft=r["link_type_name"], at=r["area_type"]))
        d["vol"] += daily_flow.get(key, 0.0)
    out = os.path.join(HERE, "review")
    os.makedirs(out, exist_ok=True)

    def table(groups, fname, label):
        rows_out = []
        for gname, items in sorted(groups.items()):
            n = len(items)
            tc = sum(i["count"] for i in items)
            tv = sum(i["vol"] for i in items)
            prmse = (np.sqrt(sum((i["vol"] - i["count"]) ** 2 for i in items)
                             / max(n - 1, 1)) / (tc / n) * 100) if n > 1 and tc > 0 else 0
            rows_out.append([gname, n, round(tc), round(tv),
                             round((tv - tc) / tc * 100, 2) if tc else "",
                             round(prmse, 2)])
        with open(os.path.join(out, fname), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([label, "N", "TotalCount", "TotalVolume", "PctDiff", "PRMSE"])
            w.writerows(rows_out)
        return rows_out

    by_ft = defaultdict(list)
    allx = list(per_link.values())
    for d in allx:
        by_ft[d["ft"]].append(d)
    by_ft["All"] = allx
    t1 = table(by_ft, "v15_count_comparison_by_fac_type.csv", "HCMType")

    summary = dict(dc="TRMG2 tables (v1.5, no invented parameters)",
                   purposes=diag, periods=results,
                   counted_links=len(per_link),
                   total_count=round(sum(d["count"] for d in allx)),
                   total_assigned=round(sum(d["vol"] for d in allx)),
                   captured_share=round(sum(d["vol"] for d in allx)
                                        / sum(d["count"] for d in allx), 3),
                   total_wall_secs=round(time.time() - t_start))
    with open(os.path.join(out, "v15_4period_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "purposes"}, indent=1))
    for row in t1:
        print("  ", row)


if __name__ == "__main__":
    main()
