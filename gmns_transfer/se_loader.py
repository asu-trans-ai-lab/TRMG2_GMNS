"""se_loader.py — read TRMG2 socioeconomic data from CSV (portable, no TransCAD
binary needed). Returns the same list-of-dicts (with numeric types) that
transcad_bin.read_bin produced, so it is a drop-in for the .bin path.

    from se_loader import load_se
    se_rows = load_se(REPO, year=2020)      # -> gmns/se_data/se_2020.csv

Falls back to the .bin via transcad_bin if the CSV is absent.
"""
import csv
import os


def _coerce(v):
    if v is None or v == "":
        return None
    try:
        i = int(v)
        return i
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def load_se(repo, year=2020):
    here = os.path.dirname(os.path.abspath(__file__))
    csv_fp = os.path.join(here, "se_data", f"se_{year}.csv")
    if os.path.exists(csv_fp):
        with open(csv_fp, newline="") as f:
            return [{k: _coerce(v) for k, v in r.items()} for r in csv.DictReader(f)]
    # fallback to the TransCAD binary
    import sys
    sys.path.insert(0, here)
    from transcad_bin import read_bin
    stem = {2020: "se_2020", 2035: "se_2035_Adopted",
            2045: "SE_2045_Adopted", 2055: "se_2055_Adopted"}.get(year, f"se_{year}")
    _, rows = read_bin(os.path.join(repo, "master", "sedata", stem + ".bin"))
    return rows


if __name__ == "__main__":
    rows = load_se(os.path.abspath(".."))
    print(f"loaded {len(rows)} TAZ x {len(rows[0])} fields from CSV")
    print("TAZ 1 HH:", rows[0]["HH"], "type:", type(rows[0]["HH"]).__name__)


def load_table(csv_path, bin_path=None):
    """Generic: read any TRMG2 table from CSV (preferred) or the .bin fallback,
    returning list-of-dicts with numeric types coerced (drop-in for read_bin)."""
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            return [{k: _coerce(v) for k, v in r.items()} for r in csv.DictReader(f)]
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from transcad_bin import read_bin
    _, rows = read_bin(bin_path)
    return rows
