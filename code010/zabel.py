#!/usr/bin/env python3
"""zabel.py - the exceptional series A1, A2, G2, D4, F4, E6, E7, E8 of Section 2.

For each of the 8 simple Lie algebras of Deligne's series, this program
  * generates the positive roots from the Cartan matrix,
  * computes h^vee, the parameter mu = 6/h^vee, and checks the dimension formula
        dim g = 2(5-mu)(6+mu) / (mu(1+mu)),
  * checks Lemma 2.4: <alpha, theta^vee> is 0, 1 or 2, with 2h^vee - 4 roots at level 1,
  * checks the identity of Proposition 2.5 for the numbers x = 12(rho, alpha) at level 1,
        sum (delta_{x+mu} - delta_x) = delta_4 + delta_{5-mu} + delta_{6-mu}
                                        - delta_mu - delta_{1+mu} - delta_2,
  * checks the endpoint formula for the signed measures L(k theta), k <= 6, and the
    resulting product formula for dim V_{k theta} against the Weyl dimension formula.
It also checks the circular measures of Section 1.2 (A, D and affine diagrams), those of the
Dynkin diagrams E6, E7, E8 of Theorem 2.2, and the law of E6 at its short tip (Proposition 2.1),
against exact walk counts.

Exact rational arithmetic for the Lie part. Standard library only.
Output: terminal, and zabel.log.
"""
from fractions import Fraction as Fr
from collections import Counter
from math import comb, cos, pi
import cmath, sys

LOG = open(__file__.replace('.py', '.log'), 'w')
def out(*a):
    s = ' '.join(str(x) for x in a)
    print(s); LOG.write(s + '\n'); LOG.flush()

# ------------------------------------------------------------------ root systems
# Dynkin data: edges (i, j, m) meaning m bonds with arrow from long i to short j (m = 1: simply laced)
TYPES = {
 'A1': (1, []),
 'A2': (2, [(0,1,1)]),
 'G2': (2, [(1,0,3)]),                       # alpha_0 short, alpha_1 long
 'D4': (4, [(0,1,1),(1,2,1),(1,3,1)]),
 'F4': (4, [(0,1,1),(1,2,2),(2,3,1)]),       # alpha_0, alpha_1 long; alpha_2, alpha_3 short
 'E6': (6, [(0,2,1),(1,3,1),(2,3,1),(3,4,1),(4,5,1)]),
 'E7': (7, [(0,2,1),(1,3,1),(2,3,1),(3,4,1),(4,5,1),(5,6,1)]),
 'E8': (8, [(0,2,1),(1,3,1),(2,3,1),(3,4,1),(4,5,1),(5,6,1),(6,7,1)]),
}

def gram(name):
    """Gram matrix of the simple roots, long roots of squared length 2."""
    r, edges = TYPES[name]
    # squared lengths: propagate along edges
    sq = [None]*r; sq[0] = Fr(1)
    changed = True
    while changed:
        changed = False
        for i, j, m in edges:          # i long, j short when m > 1
            if sq[i] is not None and sq[j] is None: sq[j] = sq[i]/m; changed = True
            if sq[j] is not None and sq[i] is None: sq[i] = sq[j]*m; changed = True
    top = max(sq); sq = [2*s/top for s in sq]
    B = [[Fr(0)]*r for _ in range(r)]
    for i in range(r): B[i][i] = sq[i]
    for i, j, m in edges:
        B[i][j] = B[j][i] = -max(sq[i], sq[j])/2      # (a_i, a_j) = -(long length)/2
    return B

def positive_roots(B):
    r = len(B)
    def refl(c, i):                      # reflection in simple root i
        k = 2*sum(c[j]*B[j][i] for j in range(r))/B[i][i]
        d = list(c); d[i] -= k; return tuple(d)
    simple = [tuple(Fr(int(i == j)) for j in range(r)) for i in range(r)]
    seen = set(simple); todo = list(simple)
    while todo:
        c = todo.pop()
        for i in range(r):
            d = refl(c, i)
            if d not in seen: seen.add(d); todo.append(d)
    return sorted(c for c in seen if all(x >= 0 for x in c))

def pair(B, a, b):
    r = len(B); return sum(a[i]*B[i][j]*b[j] for i in range(r) for j in range(r))

def measure(points_plus, points_minus):
    m = Counter()
    for x in points_plus: m[x] += 1
    for x in points_minus: m[x] -= 1
    return {x: v for x, v in m.items() if v != 0}

def lie_checks():
    ok = True
    out('%-3s %5s %5s %6s %5s %6s   key identity  L(k theta), k<=6  dim V_(k theta), k = 1, 2, 3' %
        ('g', 'rank', 'h^v', 'mu', 'dim', '#g_1'))
    for name in TYPES:
        B = gram(name); r = len(B); P = positive_roots(B)
        # rho in simple root coordinates: (rho, a_i) = (a_i, a_i)/2
        rho = tuple(sum(c[i] for c in P)/2 for i in range(r))
        assert all(pair(B, rho, tuple(Fr(int(i == j)) for j in range(r))) == B[i][i]/2 for i in range(r))
        theta = max(P, key=lambda c: sum(c))
        assert pair(B, theta, theta) == 2
        hv = pair(B, rho, theta) + 1                 # dual Coxeter number
        mu = Fr(6)/hv
        dim = r + 2*len(P)
        ok &= (dim == 2*(5-mu)*(6+mu)/(mu*(1+mu)))
        x = {c: mu*pair(B, rho, c) for c in P}       # x = 12 (rho, alpha), Killing normalization
        lev = {c: pair(B, c, theta) for c in P}      # <alpha, theta^vee>, as (theta, theta) = 2
        ok &= set(lev.values()) <= {0, 1, 2} and [c for c in P if lev[c] == 2] == [theta]
        g1 = [c for c in P if lev[c] == 1]
        ok &= (len(g1) == 2*hv - 4) and (x[theta] == 6 - mu)
        # key identity
        lhs = measure([x[c] + mu for c in g1], [x[c] for c in g1])
        rhs = measure([Fr(4), 5-mu, 6-mu], [mu, 1+mu, Fr(2)])
        key = (lhs == rhs); ok &= key
        # L(k theta) and dimensions
        good = True; dims = []
        for k in range(1, 7):
            L = measure([x[c] + k*mu*lev[c] for c in P], [x[c] for c in P])
            pos = [4+(j-1)*mu for j in range(1, k+1)] + [5+(j-2)*mu for j in range(1, k+1)] \
                + [6+(j-1)*mu for j in range(1, k)] + [6+(2*k-1)*mu]
            neg = [j*mu for j in range(1, k+1)] + [1+j*mu for j in range(1, k+1)] \
                + [2+(j-1)*mu for j in range(1, k+1)]
            good &= (L == measure(pos, neg))
            weyl = Fr(1)
            for c in P: weyl *= (x[c] + k*mu*lev[c])/x[c]
            prod = Fr(1)
            for y in pos: prod *= y
            for y in neg: prod /= y
            good &= (weyl == prod) and weyl.denominator == 1
            dims.append(int(weyl))
            # first moment 24 (lambda, rho) = 2k(6 - mu); second moment 72 c_lambda = 12k(6 + (k-1)mu)
            good &= sum(y*v for y, v in L.items()) == 2*k*(6 - mu)
            good &= sum(y*y*v for y, v in L.items()) == 12*k*(6 + (k-1)*mu)
        ok &= good
        out('%-3s %5d %5s %6s %5d %6d   %-12s  %-16s  %s' % (name, r, hv, mu, dim, len(g1), key, good, dims[:3]))
        out('     level 1 values x/mu = (rho, alpha) :', ' '.join(str(v) for v in sorted(pair(B, rho, c) for c in g1)))
    return ok

# ------------------------------------------------------------------ Dynkin diagrams E6, E7, E8
def walk(edges, root, nmax):
    n = 1 + max(max(e) for e in edges)
    A = [[0]*n for _ in range(n)]
    for a, b in edges: A[a][b] = A[b][a] = 1
    p = [0]*n; p[root] = 1; c = [1]
    for _ in range(nmax):
        p = [sum(A[v][w]*p[w] for w in range(n)) for v in range(n)]
        c.append(sum(t*t for t in p))
    return c

def d_unif(n):   return [(cmath.exp(2j*pi*k/(2*n)), 1/(2*n)) for k in range(2*n)]
def d_prime(n):  return [(cmath.exp(2j*pi*(2*k+1)/(4*n)), 1/(2*n)) for k in range(2*n)]
def d_second(l):
    pts = [j for j in range(12*l) if j % 6 in (1, 5)]
    return [(cmath.exp(2j*pi*j/(12*l)), 1/len(pts)) for j in pts]
def d_third():   return [(cmath.exp(2j*pi*k/6), 0.25) for k in (1, 2, 4, 5)]
def h(j):        return lambda u: 1 - (u**(2*j)).real
def moments(terms, nmax):
    """terms: list of (coefficient, density or None, measure)"""
    m = [0.0]*(nmax+1)
    for coef, dens, meas in terms:
        for u, w in meas:
            f = coef*w*(dens(u) if dens else 1.0); s = abs(u + 1/u)**2
            for k in range(nmax+1): m[k] += f*s**k
    return m

def dynkin_checks():
    ok = True; NM = 12
    E = {6: [(0,1),(1,2),(2,3),(3,4),(2,5)],            # arms 2, 2, 1 at vertex 2; long tips 0, 4; short tip 5
         7: [(0,1),(1,2),(2,3),(3,4),(4,5),(3,6)],      # arms 3, 2, 1 at vertex 3; root 0
         8: [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(4,7)]}# arms 4, 2, 1 at vertex 4; root 0
    tern = {6: [(1/6, None, d_second(2)), (2/6, h(1), d_second(2)), (3/6, None, d_third())],
            7: [(2/3, h(2), d_second(3)), (1/3, None, d_prime(1))],
            8: [(2/3, lambda u: h(1)(u)+h(3)(u), d_second(5)), (-1/3, None, d_second(1))]}
    cyc = {6: [(1, h(1), d_unif(12)), (.5, None, d_unif(12)), (-.5, None, d_unif(6)), (-.5, None, d_unif(4)), (.5, None, d_unif(3))],
           7: [(1, h(2), d_prime(9)), (.5, None, d_prime(1)), (-.5, None, d_prime(3))],
           8: [(1, lambda u: h(1)(u)+h(3)(u), d_prime(15)), (-.5, None, d_prime(5)), (-.5, None, d_prime(3))]}
    for n in (6, 7, 8):
        c = walk(E[n], 0, NM)
        for label, terms in (('ternary', tern[n]), ('cyclotomic', cyc[n])):
            m = moments(terms, NM)
            good = all(abs(m[k] - c[k]) < 1e-7*max(1, c[k]) for k in range(NM+1)); ok &= good
            out('E%d, root at the tip of the long arm: %-10s expansion matches c_0..c_%d: %s' % (n, label, NM, good))
        out('    c_1..c_8 =', c[1:9])
    # E6 at the short tip
    c = walk(E[6], 5, NM)
    m = moments([(1/3, None, d_second(2)), (2/3, h(1), d_second(2))], NM)
    good = all(abs(m[k] - c[k]) < 1e-7*max(1, c[k]) for k in range(NM+1)); ok &= good
    s3 = 3**0.5
    m2 = [(1+1/s3)/2*(2-s3)**k + (1-1/s3)/2*(2+s3)**k for k in range(NM+1)]
    good2 = all(abs(m2[k] - c[k]) < 1e-7*max(1, c[k]) for k in range(NM+1)); ok &= good2
    out('E6, root at the short tip: circular measure (1+2h_1)/3 d\'\'_2:', good, '; 2 atoms at 2 -+ sqrt 3:', good2)
    cl = walk(E[6], 0, NM)
    good3 = all(abs(cl[k] - (c[k]/2 + 0.5)) < 1e-9 for k in range(1, NM+1)); ok &= good3
    out('E6: law at the long tip = (law at the short tip + delta_1)/2:', good3)
    return ok

def classical_checks():
    """the table of Section 1.2: A_(n-1), cycle with 2n vertices, D_(n+1), affine D_(n+2)"""
    ok = True; NM = 10
    for n in range(3, 9):
        tests = []
        path = [(i, i+1) for i in range(n-2)]                 # A_(n-1), vertices 0..n-2
        for j in range(1, n):                                 # rooted at its j-th vertex
            tests.append((walk(path, j-1, NM), [(1, h(j), d_unif(n))]))
        # cycle with 2n vertices
        tests.append((walk([(i, (i+1) % (2*n)) for i in range(2*n)], 0, NM), [(1, None, d_unif(n))]))
        # D_(n+1): path 0..n-2, and 2 tips n-1, n attached to n-2; root 0
        tests.append((walk(path + [(n-2, n-1), (n-2, n)], 0, NM), [(1, h(1), d_prime(n))]))
        # affine D_(n+2): path 1..n-1, tips 0 and n+1 attached to 1, tips n and n+2 attached to n-1; root 0
        tests.append((walk([(i, i+1) for i in range(1, n-1)] + [(0, 1), (1, n+1), (n-1, n), (n-1, n+2)], 0, NM),
                      [(.5, None, d_unif(n)), (.5, None, d_prime(1))]))
        for c, terms in tests:
            m = moments(terms, NM)
            ok &= all(abs(m[k] - c[k]) < 1e-7*max(1, c[k]) for k in range(NM+1))
    out('Section 1.2 (A, cyclic, D and affine D diagrams, n = 3..8, moments up to order %d):' % NM, ok)
    return ok

def main():
    ok = lie_checks()
    out('-'*100)
    ok &= classical_checks()
    out('-'*100)
    ok &= dynkin_checks()
    out('ALL CHECKS PASSED' if ok else 'SOME CHECK FAILED')
    return 0 if ok else 1

if __name__ == '__main__':
    sys.exit(main())
