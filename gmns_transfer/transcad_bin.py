"""TransCAD fixed-format binary (FFB) table reader: .BIN + text .DCB dictionary.

Public TransCAD model repos (TRMG2, OahuTDFM, Cleveland TN) ship their master data as
.dbd geographic files whose ATTRIBUTE tables are .BIN fixed-record binaries paired with
a plain-text .DCB data dictionary:

    line 1: table description
    line 2: record byte length
    lines : "Field Name",TYPE,start(1-based),width,decimals,...

Types: I = int32, S = int16, R = float64, F = float32, C = fixed char, DATE/TIME
variants read as raw ints. TransCAD null sentinels are mapped to None.

This recovers every link/node/SE attribute from those repos WITHOUT TransCAD. Topology
limitation (documented): line-layer endpoints and node coordinates live in the .dln/.pts
geometry sidecars (undocumented) -- so this reader supplies attributes keyed by ID;
pairing with an exported shapefile/geojson (or a future .pts reader) supplies geometry.

Provenance / legal posture: this is an INDEPENDENT implementation written solely from
the self-describing plain-text .DCB dictionary -- no Caliper software was used,
decompiled, or copied, and no access control was circumvented. Reading a data format
for interoperability is standard practice with long public precedent for this exact
format (e.g. the open-source wsp-sag/tcadr R package). "TransCAD" is a trademark of
Caliper Corporation, used here descriptively; no endorsement implied. The proprietary
.mtx matrix format is deliberately NOT implemented -- request OMX/CSV exports instead
(see docs/TRANSCAD_EXPORT_GUIDE.md).

Usage:
  python -m dtalite_qa.transcad_bin table.BIN [--dcb table.DCB] [--out table.csv] [--head 5]
"""
import csv
import os
import struct
import sys

# TransCAD missing-value sentinels
_NULL_I32 = -2147483647
_NULL_I16 = -32767
_NULL_F = -3.402823466e38          # float32 lowest (approx)
_NULL_R = -1.7976931348623157e308  # float64 lowest


def read_dcb(path):
    """-> (description, row_len, fields[(name, type, start0, width, dec)])."""
    with open(path, encoding="latin-1") as f:
        lines = [ln.rstrip("\n") for ln in f]
    desc = lines[0].strip()
    # TransCAD 9+ writes "609,codepage=1252"; older files have just "609"
    row_len = int(lines[1].strip().split(",")[0])
    fields = []
    for ln in lines[2:]:
        ln = ln.strip()
        if not ln.startswith('"'):
            continue
        name, rest = ln[1:].split('"', 1)
        parts = rest.lstrip(",").split(",")
        ftype = parts[0].strip().upper()
        start = int(parts[1])
        width = int(parts[2])
        dec = int(parts[3] or 0) if len(parts) > 3 and parts[3].strip() else 0
        fields.append((name, ftype, start - 1, width, dec))
    return desc, row_len, fields


def _decode(ftype, raw):
    if ftype == "I":
        v = struct.unpack("<i", raw)[0]
        return None if v == _NULL_I32 else v
    if ftype == "S":
        v = struct.unpack("<h", raw)[0]
        return None if v == _NULL_I16 else v
    if ftype == "R":
        v = struct.unpack("<d", raw)[0]
        return None if v <= _NULL_R * 0.99 else v
    if ftype == "F":
        v = struct.unpack("<f", raw)[0]
        return None if v <= _NULL_F * 0.99 else v
    if ftype == "C":
        return raw.decode("latin-1").rstrip("\x00 ").strip()
    # dates/times/unknown: return raw int if 2/4/8 bytes else hex
    n = len(raw)
    if n in (2, 4, 8):
        return struct.unpack("<" + {2: "h", 4: "i", 8: "q"}[n], raw)[0]
    return raw.hex()


def read_bin(bin_path, dcb_path=None):
    """-> (description, list-of-dict rows)."""
    dcb_path = dcb_path or os.path.splitext(bin_path)[0] + ".DCB"
    desc, row_len, fields = read_dcb(dcb_path)
    size = os.path.getsize(bin_path)
    if size % row_len:
        raise ValueError(f"{bin_path}: size {size} not a multiple of record length "
                         f"{row_len} (wrong DCB?)")
    rows = []
    with open(bin_path, "rb") as f:
        while True:
            rec = f.read(row_len)
            if len(rec) < row_len:
                break
            rows.append({name: _decode(t, rec[s:s + w]) for name, t, s, w, _ in fields})
    return desc, rows


def main():
    import argparse
    ap = argparse.ArgumentParser(description="TransCAD .BIN/.DCB -> CSV")
    ap.add_argument("bin")
    ap.add_argument("--dcb", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--head", type=int, default=0)
    a = ap.parse_args()
    desc, rows = read_bin(a.bin, a.dcb)
    print(f"{desc}: {len(rows):,} records, {len(rows[0]) if rows else 0} fields")
    if a.head and rows:
        for r in rows[:a.head]:
            print({k: v for k, v in list(r.items())[:8]})
    if a.out and rows:
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
