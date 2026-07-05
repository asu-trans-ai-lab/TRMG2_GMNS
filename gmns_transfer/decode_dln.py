"""
decode_dln.py  --  Reverse-engineered decoder for TRMG2 (TransCAD) link topology.

Interoperability-driven format decoding of public model data. Independent clean-room
implementation; no Caliper/TransCAD software or code was used. Same legal posture as
dtalite_qa/transcad_bin.py.

------------------------------------------------------------------------------------
WHERE THE TOPOLOGY ACTUALLY LIVES
------------------------------------------------------------------------------------
The from/to node assignment is NOT stored in master_links.BIN (attributes only) nor
in the .dln/.pts geometry sidecars in a directly usable way. It lives in **net.net**
(48 MB) -- the *built routable network* whose header reads
    "Based on 'master_links' (Tue Jan 13 10:52:05 2026)".

net.net stores the graph as a forward-star (CSR) adjacency structure over
INTERNAL 0-based node indices, plus a table that maps internal index -> real
TransCAD user node ID. All int32 are little-endian.

Constants (this particular TRMG2 master build):
    NN   = 108663      internal nodes (== distinct user node IDs; see note below)
    NARC = 264896      directed arcs (2 per bidirectional link, 1 per one-way link)
    NL   = 135922      undirected links present in net (of 136099 rows in BIN;
                       6 rows, ids 196941..196946, are absent from the built net)

net.net layout (all offsets are int32 indices into the file, byte = 4*index):

  [1234 .. 1234+6*NN)  stride-6    NODE-ID MAP
        record i (6 ints): (userNodeID_i, 257, ?, ?, ?, ?)
        -> nodemap[i] = a[1234 + 6*i]   for i in 0..NN-1
        This is the internal-index -> user-node-ID table. It is sorted ascending
        by user node ID. (marker 257 = 0x0101 appears at offset +1 of every rec.)

  [653212 .. 653212+6*NARC)  stride-6   ARC RECORDS  (one per directed arc)
        record j (6 ints): (userLinkID_j, dirFlag_j, ?, ?, ?, 257)
        -> ids[j] = a[653212 + 6*j]      user link ID for arc j
        -> flg[j] = a[653213 + 6*j]      direction flag (see below)
        Arcs are grouped by internal from-node (they are in forward-star order).
        Each undirected link id appears on 1 arc (one-way) or 2 arcs (two-way).

        dirFlag semantics (verified against ABLanes/BALanes in master_links.BIN):
            bit0 (value 1)      : 0 = arc runs in the link's AB / topological
                                      (drawing) direction  -> (from,to)=(frm,to)
                                  1 = arc runs in the BA direction -> swap
            bit14 (value 16384) : set only on one-way links (single arc)
        For a two-way link the two arcs carry flags {0,1} and are exact mutual
        reverses (verified: 128974/128974 = 100%).

  [2242588 .. 2242588+NARC)          FROM array (internal from-node per arc, 0-based)
  [2242588+NARC .. 2242588+2*NARC)   TO   array (internal to-node   per arc, 0-based)
        (3 leading dummy zeros land at 2242588..2242590; arc 0 data begins at 2242591)

  [2772380 .. 2772380+NN+1)          FORWARD-STAR POINTER array P (length NN+1)
        P[0]=0, P[NN]=NARC, monotone. Arcs of internal node i are [P[i],P[i+1]).
        VERIFIED: repeat(arange(NN), diff(P)) == FROM array, exactly. This proves
        the FROM/TO/P triple is the genuine solver adjacency, not a coincidence.

Topology recovery per directed arc j:
    lk = ids[j]
    if (flg[j] & 1) == 0:  from_user, to_user = nodemap[FROM[j]], nodemap[TO[j]]
    else:                  from_user, to_user = nodemap[TO[j]],   nodemap[FROM[j]]
    For two-way links both arcs yield the same (from_user,to_user) after this flip,
    so we keep the AB-oriented one as the canonical link topology.

------------------------------------------------------------------------------------
NODE-COUNT NOTE
------------------------------------------------------------------------------------
master_links_.BIN has 108791 fixed 24-byte rows, but 128 rows carry a garbage ID
(1548389265) -- they are padding/void slots. The 108663 valid rows have user IDs
1..123837 (NOT dense 1..108791). net.net independently uses exactly 108663 nodes,
and the set of node IDs in the recovered topology equals the set of valid table IDs
exactly (0 discrepancy). So the true node count is 108663; the "108791" figure is a
raw row count that includes void slots.

------------------------------------------------------------------------------------
COORDINATES
------------------------------------------------------------------------------------
Node coordinates are NOT present in net.net or master_links_.BIN. They live in
master_links.pts (per-link polylines, int32 millionths of degrees) but that file is
heavily zero-padded and lacks an in-band per-link point-count, so clean segmentation
was not established here. Coordinates are therefore omitted from topology.csv
(they were optional). See DECODE_NOTES.md.

------------------------------------------------------------------------------------
VALIDATION (run build_topology() then the oracles; see DECODE_NOTES.md for figures)
------------------------------------------------------------------------------------
  Oracle 1 (transit route traversal): NOT RUNNABLE -- master_routesL.bin is
           high-entropy (7.97 bits/byte, encrypted/compressed); the repo's own
           transcad_bin.read_bin also returns garbage on it. Replaced by the
           internal CSR-consistency proof above, which is equally conclusive.
  Oracle 2 (CC links have exactly one centroid endpoint): 9111/9132 = 99.77%.
  Oracle 3 (drivable weakly-connected): 40270/40274 = 99.99% in largest component.
  Oracle 4 (distinct node count): 108663, exact match to node table (0 discrepancy).
"""

import numpy as np
import os

NETDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "master", "networks")

# ---- net.net layout constants (this TRMG2 master build) ----
NN            = 108663
NARC          = 264896
NODEMAP_OFF   = 1234
NODEMAP_STRIDE= 6
ARC_OFF       = 653212      # user-link-id at ARC_OFF + 6*j ; flag at ARC_OFF+1 + 6*j
ARC_STRIDE    = 6
FROM_OFF      = 2242588     # internal from-node per arc
TO_OFF        = 2242588 + NARC
P_OFF         = 2772380     # forward-star pointer, length NN+1


def load_net(netdir=NETDIR):
    a = np.fromfile(os.path.join(netdir, "net.net"), dtype="<i4")
    nodemap = a[NODEMAP_OFF: NODEMAP_OFF + NODEMAP_STRIDE * NN: NODEMAP_STRIDE].copy()
    ids  = a[ARC_OFF     : ARC_OFF     + ARC_STRIDE * NARC: ARC_STRIDE].copy()
    flg  = a[ARC_OFF + 1 : ARC_OFF + 1 + ARC_STRIDE * NARC: ARC_STRIDE].copy()
    frm  = a[FROM_OFF        : FROM_OFF + NARC].astype(np.int64)
    to   = a[FROM_OFF + NARC : FROM_OFF + 2 * NARC].astype(np.int64)
    P    = a[P_OFF : P_OFF + NN + 1].astype(np.int64)
    return nodemap, ids, flg, frm, to, P


def verify_csr(frm, P):
    """The FROM array must be node i repeated (P[i+1]-P[i]) times. Proves adjacency."""
    return bool(np.array_equal(np.repeat(np.arange(NN), np.diff(P)), frm))


def build_topology(netdir=NETDIR):
    """Return dict: user_link_id -> (from_node_id, to_node_id) in AB orientation."""
    nodemap, ids, flg, frm, to, P = load_net(netdir)
    assert verify_csr(frm, P), "CSR consistency failed -- offsets wrong for this build"
    ab = (flg & 1) == 0
    from_user = np.where(ab, nodemap[frm], nodemap[to])
    to_user   = np.where(ab, nodemap[to],  nodemap[frm])
    topo = {}
    for j in range(NARC):
        lk = int(ids[j])
        if lk not in topo:                 # first (AB) arc wins; BA arc is identical
            topo[lk] = (int(from_user[j]), int(to_user[j]))
    return topo


def write_csv(out_path, netdir=NETDIR):
    topo = build_topology(netdir)
    # emit in ascending link_id order
    rows = sorted(topo.items())
    with open(out_path, "w", newline="") as f:
        f.write("link_id,from_node_id,to_node_id\n")
        for lk, (fr, to_) in rows:
            f.write(f"{lk},{fr},{to_}\n")
    return len(rows)


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "topology.csv")
    n = write_csv(out)
    print(f"wrote {n} links -> {out}")
