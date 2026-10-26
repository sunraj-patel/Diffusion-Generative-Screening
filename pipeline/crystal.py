"""Crystal-level checks shared by curation, screening and tests: ABX3 stoichiometry, perovskite topology,
charge balance, Goldschmidt tolerance factor, duplicate handling."""
from collections import Counter
from itertools import product
import numpy as np
from pymatgen.core import Structure, Composition, Element, Species
from pymatgen.analysis.structure_matcher import StructureMatcher

ANIONS = {"O", "F", "S", "Cl", "Se", "Br", "Te", "I", "N"}
MIN_DIST = 0.55                                                                             


def abx3_stoichiometry(s):
    """ABX3 in the general sense: only anions from ANIONS, anion:cation = 3:2, and (cation count)/2 B-type + A-type.
    Mixed sites are kept as is. Returns (bool, reason)."""
    comp = s.composition
    if not s.is_ordered:
        return False, "disordered"
    an = sum(a for e, a in comp.items() if e.symbol in ANIONS)
    ca = sum(a for e, a in comp.items() if e.symbol not in ANIONS)
    if an == 0 or ca == 0:
        return False, "no anion/cation split"
    if any(e.symbol in ANIONS for e in comp) and abs(an * 2 - ca * 3) > 1e-6:
        return False, "not 1:1:3"
    if len([e for e in comp if e.symbol not in ANIONS]) < 2:
        return False, "single cation"
    return True, "ok"


def _bx_graph(s, bmask, xmask, tol=1.25):
    """B-X edges (i, j, image offset) with d <= tol * shortest B-X distance of that B atom."""
    edges = []
    nbrs = s.get_all_neighbors(4.5)
    for i in np.where(bmask)[0]:
        xs = [(n.index, n.image, n.nn_distance) for n in nbrs[i] if xmask[n.index]]
        if not xs:
            continue
        dmin = min(d for _, _, d in xs)
        for j, img, d in xs:
            if d <= tol * dmin:
                edges.append((int(i), int(j), tuple(int(v) for v in img)))
    return edges


def _rank_of_network(n, edges):
    """Dimensionality of the periodic graph: rank of translation vectors closing cycles."""
    adj = {k: [] for k in range(n)}
    for i, j, img in edges:
        adj[i].append((j, np.array(img))); adj[j].append((i, -np.array(img)))
    seen, cycles = {}, []
    start = next((k for k in adj if adj[k]), None)
    if start is None:
        return 0, 0
    seen[start] = np.zeros(3, int); stack = [start]
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
    rank = np.linalg.matrix_rank(np.array(cycles)) if cycles else 0
    return rank, len(seen)


def classify_perovskite(s):
    """Return dict with family in {'3D corner-sharing perovskite','lower-dimensional ABX3', 'not ABX3', ...}.
    Criterion: a cation sub-lattice B (half of the cations) with octahedral X coordination, X bridging exactly two B,
    and a B-X network that is periodic in three dimensions. Tilting/distortion is allowed."""
    ok, why = abx3_stoichiometry(s)
    out = dict(abx3=ok, reason=why, family="not ABX3", B_sites=None, A_sites=None, x_bridging=None)
    if not ok:
        return out
    syms = [sp.symbol for sp in s.species]
    xmask = np.array([x in ANIONS for x in syms]); cmask = ~xmask
    nX = xmask.sum()
    if nX % 3:
        out.update(family="not ABX3", reason="anion count not multiple of 3"); return out
    nB = nX // 3
    best = None
    nbrs = s.get_all_neighbors(4.5)
                                                                                      
    cn, mind = {}, {}
    for i in np.where(cmask)[0]:
        xs = sorted((n.nn_distance, n.index) for n in nbrs[i] if xmask[n.index])
        if not xs:
            cn[i] = 0; continue
        cn[i] = sum(1 for d, _ in xs if d <= 1.25 * xs[0][0]); mind[i] = xs[0][0]
                                                                                               
    cand = sorted([i for i in cn if cn[i] == 6], key=lambda i: mind[i])
    if len(cand) < nB:
        out.update(family="ABX3 stoichiometry, no octahedral B-site"); return out
    bidx = cand[:nB]
    bmask = np.zeros(len(s), bool); bmask[bidx] = True
                                                                                                                  
                                                                                                                     
    for i in np.where(cmask & ~bmask)[0]:
        xs = sorted(n.nn_distance for n in nbrs[i] if xmask[n.index])
        if len(xs) < 8 or sum(d <= 1.3 * xs[0] for d in xs) < 8:
            out.update(family="ABX3 stoichiometry, low-coordinated A-site", B_sites=sorted({syms[k] for k in bidx}),
                       A_sites=sorted({syms[k] for k in np.where(cmask & ~bmask)[0]}))
            return out
    edges = _bx_graph(s, bmask, xmask)
    xdeg = Counter(j for _, j, _ in edges)
    out["x_bridging"] = float(np.mean([xdeg.get(j, 0) for j in np.where(xmask)[0]]))
    rank, _ = _rank_of_network(len(s), edges)
    out["B_sites"] = sorted({syms[i] for i in bidx}); out["A_sites"] = sorted({syms[i] for i in np.where(cmask & ~bmask)[0]})
    corner = all(xdeg.get(j, 0) == 2 for j in np.where(xmask)[0])
    if rank == 3 and corner:
        out["family"] = "3D corner-sharing perovskite"
    elif rank == 3:
        out["family"] = "3D octahedral network (non-ideal connectivity)"
    else:
        out["family"] = f"{rank}D octahedral network"
    return out


def charge_balance(comp):
    """Common-oxidation-state charge balance. Returns (True/False/None, guess); None when no data."""
    try:
        g = Composition(comp).oxi_state_guesses(max_sites=-1)
    except Exception:
        return None, None
    return bool(g), (g[0] if g else None)


def tolerance_factor(s, cls=None):
    """Goldschmidt t = (rA+rX)/(sqrt2 (rB+rX)) with Shannon radii (A: CN12, B: CN6, X: CN2 -> fall back on CN6).
    Site-averaged for mixed sites. None when radii/oxidation states unavailable."""
    cls = cls or classify_perovskite(s)
    if not cls.get("B_sites") or not cls.get("A_sites"):
        return None
    ok, guess = charge_balance(s.composition.reduced_composition)
    if not ok:
        return None
    def radius(sym, cn):
        ox = guess.get(sym)
        if ox is None:
            return None
        try:
            sp = Species(sym, round(ox))
            for c in (cn, {12: "XII", 6: "VI", 2: "II"}[cn]):
                try:
                    return float(sp.get_shannon_radius(c))
                except Exception:
                    pass
            for c in ("XII", "XI", "X", "IX", "VIII") if cn == 12 else ("VI",):
                try:
                    return float(sp.get_shannon_radius(c))
                except Exception:
                    pass
        except Exception:
            return None
        return None
    rA = [radius(e, 12) for e in cls["A_sites"]]; rB = [radius(e, 6) for e in cls["B_sites"]]
    xs = [e for e in {sp.symbol for sp in s.species} if e in ANIONS]
    rX = [radius(e, 2) or radius(e, 6) for e in xs]
    if None in rA or None in rB or None in rX or not rA or not rB or not rX:
        return None
    return (np.mean(rA) + np.mean(rX)) / (np.sqrt(2) * (np.mean(rB) + np.mean(rX)))


def geometry_ok(s, frac=MIN_DIST):
    """No pair closer than 0.55 x sum of atomic radii; positive volume; density in a sane window (g/cm3)."""
    if s.volume <= 0 or not (0.5 < s.density < 25):
        return False
    r = np.array([Element(x.symbol).atomic_radius or 1.2 for x in s.species], float)
    dm = s.distance_matrix + np.eye(len(s)) * 1e3
    thr = frac * (r[:, None] + r[None, :])
    return bool(np.all(dm > thr))


def make_matcher():
    return StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True, scale=True)


def dedup(structs, matcher=None):
    """Return (group_id per structure, first-index of each group). Groups only within equal reduced formula."""
    matcher = matcher or make_matcher()
    by_f = {}
    for k, s in enumerate(structs):
        by_f.setdefault(s.composition.reduced_formula, []).append(k)
    gid = [-1] * len(structs); reps = []
    for f, ks in by_f.items():
        local = []
        for k in ks:
            for g in local:
                if matcher.fit(structs[reps[g]], structs[k]):
                    gid[k] = g; break
            else:
                reps.append(k); local.append(len(reps) - 1); gid[k] = len(reps) - 1
    return gid, reps
