"""Connectivity-based identification of three-dimensional corner-sharing ABX3 perovskites.

All coordination thresholds are explicit parameters (``Params``). The defaults reproduce the classifier used in the
manuscript (pipeline/crystal.py::classify_perovskite). The thresholds are operational, not universal definitions.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, asdict, replace
import numpy as np

ANIONS = frozenset({"O", "F", "S", "Cl", "Se", "Br", "Te", "I", "N"})

PEROVSKITE = "3D corner-sharing perovskite"
NON_IDEAL = "3D octahedral network (non-ideal connectivity)"
LOW_A = "ABX3 stoichiometry, low-coordinated A-site"
NO_B = "ABX3 stoichiometry, no octahedral B-site"
NOT_ABX3 = "not ABX3"
FACE_EDGE = "3D octahedral network (face/edge-sharing present)"


@dataclass(frozen=True)
class Params:
    b_tol: float = 1.25        # B-site coordination shell: anions within b_tol x shortest cation-anion distance
    b_cn: int = 6              # required B-site coordination number
    a_tol: float = 1.3         # A-site shell: anions within a_tol x shortest distance
    a_min_cn: int = 8          # minimum A-site coordination
    bx_tol: float = 1.25       # B-X graph edges: d <= bx_tol x shortest B-X distance of that B atom
    strict_corner: bool = False  # also reject networks in which two B cations share more than one anion (face/edge sharing)
    cutoff: float = 4.5        # neighbour search radius (A)

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)


DEFAULT = Params()


def abx3_stoichiometry(s, anions=ANIONS):
    """Ordered, >=2 cation species, anion:cation = 3:2. Returns (bool, reason)."""
    if not s.is_ordered:
        return False, "disordered"
    comp = s.composition
    an = sum(a for e, a in comp.items() if e.symbol in anions)
    ca = sum(a for e, a in comp.items() if e.symbol not in anions)
    if an == 0 or ca == 0:
        return False, "no anion/cation split"
    if abs(an * 2 - ca * 3) > 1e-6:
        return False, "not 1:1:3"
    if len([e for e in comp if e.symbol not in anions]) < 2:
        return False, "single cation"
    return True, "ok"


def _bx_edges(nbrs, bmask, xmask, tol):
    edges = []
    for i in np.where(bmask)[0]:
        xs = [(n.index, n.image, n.nn_distance) for n in nbrs[i] if xmask[n.index]]
        if not xs:
            continue
        dmin = min(d for _, _, d in xs)
        for j, img, d in xs:
            if d <= tol * dmin:
                edges.append((int(i), int(j), tuple(int(v) for v in img)))
    return edges


def shares_face_or_edge(edges):
    """True if some pair of B cations is bridged by more than one anion (octahedra share a face or an edge)."""
    nb = {}
    for i, j, img in edges:                      # X j sits at +img from B i, i.e. B i sits at -img from X j
        nb.setdefault(j, []).append((i, tuple(-v for v in img)))
    pairs = Counter()
    for j, lst in nb.items():
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                (i1, o1), (i2, o2) = lst[a], lst[b]
                d = tuple(y - x for x, y in zip(o1, o2))
                k = (i1, i2, d); k2 = (i2, i1, tuple(-v for v in d))
                pairs[min(k, k2)] += 1
    return any(v > 1 for v in pairs.values())


def network_dimensionality(n, edges):
    """Dimensionality of a periodic graph: rank of the translation vectors that close cycles."""
    adj = {k: [] for k in range(n)}
    for i, j, img in edges:
        adj[i].append((j, np.array(img))); adj[j].append((i, -np.array(img)))
    start = next((k for k in adj if adj[k]), None)
    if start is None:
        return 0
    seen, cycles, stack = {start: np.zeros(3, int)}, [], [start]
    while stack:
        u = stack.pop()
        for v, img in adj[u]:
            off = seen[u] + img
            if v not in seen:
                seen[v] = off; stack.append(v)
            else:
                c = off - seen[v]
                if np.any(c):
                    cycles.append(c)
    return int(np.linalg.matrix_rank(np.array(cycles))) if cycles else 0


def classify(s, params: Params = DEFAULT, anions=ANIONS) -> dict:
    """Classify a pymatgen Structure. Returns a dict with family, B_sites, A_sites, x_bridging, dimensionality, reason."""
    ok, why = abx3_stoichiometry(s, anions)
    out = dict(abx3=ok, reason=why, family=NOT_ABX3, B_sites=None, A_sites=None, x_bridging=None, dimensionality=None)
    if not ok:
        return out
    syms = [sp.symbol for sp in s.species]
    xmask = np.array([x in anions for x in syms]); cmask = ~xmask
    nX = int(xmask.sum())
    if nX % 3:
        out.update(reason="anion count not multiple of 3"); return out
    nB = nX // 3
    nbrs = s.get_all_neighbors(params.cutoff)
    cn, mind = {}, {}
    for i in np.where(cmask)[0]:
        xs = sorted((n.nn_distance, n.index) for n in nbrs[i] if xmask[n.index])
        if not xs:
            cn[i] = 0; continue
        cn[i] = sum(1 for d, _ in xs if d <= params.b_tol * xs[0][0]); mind[i] = xs[0][0]
    cand = sorted([i for i in cn if cn[i] == params.b_cn], key=lambda i: mind[i])
    if len(cand) < nB:
        out.update(family=NO_B); return out
    bidx = cand[:nB]
    bmask = np.zeros(len(s), bool); bmask[bidx] = True
    aidx = np.where(cmask & ~bmask)[0]
    for i in aidx:
        xs = sorted(n.nn_distance for n in nbrs[i] if xmask[n.index])
        if len(xs) < params.a_min_cn or sum(d <= params.a_tol * xs[0] for d in xs) < params.a_min_cn:
            out.update(family=LOW_A, B_sites=sorted({syms[k] for k in bidx}), A_sites=sorted({syms[k] for k in aidx}))
            return out
    edges = _bx_edges(nbrs, bmask, xmask, params.bx_tol)
    xdeg = Counter(j for _, j, _ in edges)
    xi = np.where(xmask)[0]
    out["x_bridging"] = float(np.mean([xdeg.get(j, 0) for j in xi]))
    dim = network_dimensionality(len(s), edges)
    out.update(dimensionality=dim, B_sites=sorted({syms[i] for i in bidx}), A_sites=sorted({syms[i] for i in aidx}))
    corner = all(xdeg.get(j, 0) == 2 for j in xi)
    out["family"] = PEROVSKITE if (dim == 3 and corner) else NON_IDEAL if dim == 3 else f"{dim}D octahedral network"
    if params.strict_corner and out["family"] == PEROVSKITE and shares_face_or_edge(edges):
        out["family"] = FACE_EDGE
    return out


def is_perovskite(s, params: Params = DEFAULT) -> bool:
    return classify(s, params)["family"] == PEROVSKITE
