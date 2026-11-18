#!/usr/bin/env python3
"""riis.py - the 4 principal graph pairs of index in (4,5) of Section 1.

For the Haagerup, Asaeda-Haagerup, 3311 and 2221 pairs, this program
  * builds the graphs from their edge lists, rooted at vertex 0,
  * counts the closed walks c_n of length 2n at the root, in exact integers,
    on the principal and on the dual graph, and checks that they agree,
  * extracts the annular multiplicities a_n from c_n = C_n + sum binom(2n,n-k) a_k,
  * checks the closed formulas for the rooted laws stated in Theorem 1.8,
    by comparing their moments with the walk counts, in exact arithmetic in Q(sqrt d),
  * prints the atoms of the lift on the half-line, and the first branch data.

Standard library only. Output: terminal, and riis.log.
"""
from fractions import Fraction as Fr
from math import comb, isqrt
import sys

LOG = open(__file__.replace('.py', '.log'), 'w')
def out(*a):
    s = ' '.join(str(x) for x in a)
    print(s); LOG.write(s + '\n'); LOG.flush()

# ---------------------------------------------------------------- graphs
# vertices are numbered so that 0 is the root; edges as pairs
GRAPHS = {
 'Haagerup': {
   'index': (5, 13),                       # (5+sqrt 13)/2
   'principal': [(0,1),(1,2),(2,3),(3,4),(3,5),(4,6),(5,7),(6,8),(7,9)],
   'dual':      [(0,1),(1,2),(2,3),(3,4),(3,5),(4,6),(4,7)]},
 'Asaeda-Haagerup': {
   'index': (5, 17),
   'principal': [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(5,7),(6,8),(7,9),
                 (8,10),(9,11),(9,12),(10,13),(13,14)],
   'dual':      [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(5,7),(7,8),(7,9),(9,10),(10,11)]},
 '3311': {
   'index': None,                          # 3+sqrt 3
   'principal': [(0,1),(1,2),(2,3),(3,4),(3,5),(3,6),(4,7),(7,8)],
   'dual':      [(0,1),(1,2),(2,3),(3,4),(3,5),(3,6),(4,7),(7,8)]},
 '2221': {
   'index': (5, 21),
   'principal': [(0,1),(1,2),(2,3),(2,4),(2,5),(3,6),(4,7)],
   'dual':      [(0,1),(1,2),(2,3),(2,4),(2,5),(3,6),(4,7)]},
}

def adjacency(edges):
    n = 1 + max(max(e) for e in edges)
    A = [[0]*n for _ in range(n)]
    for a, b in edges:
        A[a][b] += 1; A[b][a] += 1
    return A

def depths(A):
    n = len(A); d = [None]*n; d[0] = 0; todo = [0]
    while todo:
        v = todo.pop(0)
        for w in range(n):
            if A[v][w] and d[w] is None:
                d[w] = d[v] + 1; todo.append(w)
    return d

def walk_counts(A, nmax):
    """c_n = sum_v p_n(v)^2, with p_n = A^n delta_0 (midpoint formula)."""
    n = len(A); p = [0]*n; p[0] = 1; c = [1]
    for _ in range(nmax):
        p = [sum(A[v][w]*p[w] for w in range(n)) for v in range(n)]
        c.append(sum(x*x for x in p))
    return c

def annular(c):
    """solve c_n = C_n + sum_{k=1}^n binom(2n,n-k) a_k"""
    a = [None]
    for n in range(1, len(c)):
        cat = comb(2*n, n)//(n+1)
        a.append(c[n] - cat - sum(comb(2*n, n-k)*a[k] for k in range(1, n)))
    return a

# ---------------------------------------------------------------- Q(sqrt d)
class Q:
    """numbers x + y sqrt(d), with x, y rational"""
    def __init__(s, x, y=0, d=1): s.x, s.y, s.d = Fr(x), Fr(y), d
    def __add__(s, o):
        o = o if isinstance(o, Q) else Q(o, 0, s.d); return Q(s.x+o.x, s.y+o.y, s.d)
    __radd__ = __add__
    def __mul__(s, o):
        o = o if isinstance(o, Q) else Q(o, 0, s.d)
        return Q(s.x*o.x + s.d*s.y*o.y, s.x*o.y + s.y*o.x, s.d)
    __rmul__ = __mul__
    def __pow__(s, n):
        r = Q(1, 0, s.d)
        for _ in range(n): r = r*s
        return r
    def rational(s):
        assert s.y == 0; return s.x

# the laws of Theorem 1.8, as lists of (mass, atom); conjugate pairs written with both signs
def laws():
    L = {}
    d = 13
    L['Haagerup'] = [(Q(Fr(1,3),0,d), Q(0,0,d)), (Q(Fr(1,3),0,d), Q(2,0,d))] + \
        [(Q(Fr(1,6), -s*Fr(3,6*13), d), Q(Fr(5,2), s*Fr(1,2), d)) for s in (1,-1)]
    d = 17
    L['Asaeda-Haagerup'] = [(Q(Fr(1,4),0,d), Q(0,0,d)), (Q(Fr(1,4),0,d), Q(1,0,d)),
        (Q(Fr(1,8),0,d), Q(2,0,d)), (Q(Fr(1,8),0,d), Q(3,0,d))] + \
        [(Q(Fr(1,8), -s*Fr(4,8*17), d), Q(Fr(5,2), s*Fr(1,2), d)) for s in (1,-1)]
    d = 3
    L['3311'] = [(Q(Fr(5,12),0,d), Q(0,0,d)), (Q(Fr(1,4),0,d), Q(2,0,d))] + \
        [(Q(Fr(2,12), -s*Fr(1,12), d), Q(3, s, d)) for s in (1,-1)]
    d = 21
    L['2221'] = [(Q(Fr(2,3),0,d), Q(1,0,d))] + \
        [(Q(Fr(1,6), -s*Fr(3,6*21), d), Q(Fr(5,2), s*Fr(1,2), d)) for s in (1,-1)]
    return L

def main():
    NMAX = 12
    L = laws()
    ok = True
    for name, g in GRAPHS.items():
        out('='*70); out(name)
        res = {}
        for side in ('principal', 'dual'):
            A = adjacency(g[side]); dep = depths(A)
            prof = [dep.count(r) for r in range(max(dep)+1)]
            c = walk_counts(A, NMAX); res[side] = c
            out('  %-9s vertices %2d, vertices by depth %s' % (side, len(A), prof))
        same = res['principal'] == res['dual']; ok &= same
        c = res['principal']; a = annular(c)
        out('  walk counts agree on the 2 graphs:', same)
        out('  c_1..c_%d =' % NMAX, c[1:])
        out('  a_1..a_%d =' % NMAX, a[1:])
        # first branch: supertransitivity
        A = adjacency(g['principal']); dep = depths(A)
        n = next(r for r in range(len(A)) if dep.count(r) > 1)
        out('  first depth with several vertices: n = %d, with m = %d vertices; a_n = %d' % (n, dep.count(n), a[n]))
        ok &= (a[n] == dep.count(n) - 1) and all(x == 0 for x in a[1:n])
        # closed formula
        mom = [sum((w*(x**k) for w, x in L[name]), Q(0,0,L[name][0][0].d)).rational() for k in range(NMAX+1)]
        good = [int(m) for m in mom] == c and all(m.denominator == 1 for m in mom)
        ok &= good
        out('  law of Theorem 1.8 has moments c_0..c_%d:' % NMAX, good)
        for w, x in L[name]:
            wf = float(w.x) + float(w.y)*w.d**0.5; xf = float(x.x) + float(x.y)*x.d**0.5
            out('     atom %.6f  mass %.6f' % (xf, wf))
        top = max(float(x.x) + float(x.y)*x.d**0.5 for w, x in L[name])
        z = ((top-2) + ((top-2)**2 - 4)**0.5)/2
        out('  index %.6f; atoms of the lift on the half-line: z = %.6f and 1/z = %.6f' % (top, z, 1/z))
    # closed formulas for the moments, n >= 1, with binary recurrences
    #   Haagerup         6 c_n = 2^(n+1) + r_n,          r_0 = 2, r_1 = 2, r_(n+2) = 5 r_(n+1) - 3 r_n
    #   Asaeda-Haagerup  8 c_n = 2 + 2^n + 3^n + r_n,    r_0 = 2, r_1 = 1, r_(n+2) = 5 r_(n+1) - 2 r_n
    #   3311            12 c_n = 3*2^n + r_n,            r_0 = 4, r_1 = 6, r_(n+2) = 6 r_(n+1) - 6 r_n
    #   2221             6 c_n = 4 + r_n,                r_0 = 2, r_1 = 2, r_(n+2) = 5 r_(n+1) - r_n
    out('='*70)
    rec = {'Haagerup': (6, lambda n: 2**(n+1), (2, 2), (5, 3)),
           'Asaeda-Haagerup': (8, lambda n: 2 + 2**n + 3**n, (2, 1), (5, 2)),
           '3311': (12, lambda n: 3*2**n, (4, 6), (6, 6)),
           '2221': (6, lambda n: 4, (2, 2), (5, 1))}
    for name, (den, f, r01, (p, q)) in rec.items():
        c = walk_counts(adjacency(GRAPHS[name]['dual']), 40)
        r = list(r01)
        while len(r) < 41: r.append(p*r[-1] - q*r[-2])
        good = all(den*c[n] == f(n) + r[n] for n in range(1, 41)); ok &= good
        out('%-16s closed formula for c_n, n <= 40: %s' % (name, good))
    out('ALL CHECKS PASSED' if ok else 'SOME CHECK FAILED')
    return 0 if ok else 1

if __name__ == '__main__':
    sys.exit(main())
