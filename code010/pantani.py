#!/usr/bin/env python3
"""pantani.py - the generators of the phase group and the transposition (Section 4).

The entries of a dephased parameter matrix Q, indexed by X x Y, are taken as formal independent
variables, so that every Laurent monomial in them is an integer vector, and all the checks are
exact identities in the free abelian group on the (M-1)(N-1) free entries of Q.

  1. Proposition 4.2: R_k(i,e) = Q_{i,e} Q_{-k,e} / (Q_{i-k,e} Q_{0,e}) and
     g_{k,e}(i,j) = R_k(i,e+j) R_k(i,0) / (R_k(i,e) R_k(i,j)).
  2. Proof of Theorem 4.6: with the multiplicative double difference
        dQ(a,b;c,d) = Q_{a+b,c+d} Q_{a,c} Q_{b,c} Q_{a,d} Q_{b,d}
                      / (Q_{a,c+d} Q_{b,c+d} Q_{a+b,c} Q_{a+b,d})
     we have g_{k,e}(i,j) = dQ(i,-k;e,j)^(-1), dQ is symmetric in (a,b) and in (c,d), and
     dQ^t(c,d;a,b) = dQ(a,b;c,d).
  3. Theorem 4.6 (1): the generators for Q^t are the transposes of those for Q, in the form
     g'_{e,k}(j,i) = g_{-k,-e}(i,j).
  4. The lattice spanned by the generators has full rank (M-1)(N-1), in agreement with
     K_Q being the whole torus for generic Q.
  5. Theorem A: for a matrix Q which is not dephased, with all its MN entries as independent
     variables, the dephased form (W_ij W_00 / W_i0 W_0j) of the second difference
        W_ij = Q_{i+k,j+e} Q_{i,j} / (Q_{i+k,j} Q_{i,j+e})
     is the inverse of the generator g_{-k,e} of Proposition 4.2.

The groups X, Y run over all the abelian groups of order at most 6, given by their cyclic factors.

Output: terminal, and pantani.log.
"""
from fractions import Fraction as Fr
from itertools import product

LOG = open(__file__.replace('.py', '.log'), 'w')
def out(*a):
    s = ' '.join(str(x) for x in a)
    print(s); LOG.write(s + '\n'); LOG.flush()

GROUPS = [(2,), (3,), (4,), (2, 2), (5,), (6,), (2, 3)]

def elements(G): return list(product(*[range(n) for n in G]))
def add(G, a, b): return tuple((x+y) % n for x, y, n in zip(a, b, G))
def neg(G, a): return tuple((-x) % n for x, n in zip(a, G))
def zero(G): return tuple(0 for _ in G)

class Mono:
    """Laurent monomials in the free entries of Q, as integer vectors."""
    def __init__(self, X, Y, dephased=True):
        self.X, self.Y = X, Y
        self.free = [(i, b) for i in elements(X) for b in elements(Y)
                     if not dephased or (i != zero(X) and b != zero(Y))]
        self.pos = {ib: n for n, ib in enumerate(self.free)}
        self.dim = len(self.free)
    def one(self): return (0,)*self.dim
    def Q(self, i, b):
        v = [0]*self.dim
        if (i, b) in self.pos: v[self.pos[(i, b)]] = 1
        return tuple(v)
    def mul(self, *vs): return tuple(sum(c) for c in zip(*vs))
    def inv(self, v): return tuple(-c for c in v)

def generators(m, Qf):
    """The matrices g_{k,e} of Proposition 4.2, for the parameter function Qf(i,b)."""
    X, Y = m.X, m.Y
    def R(k, i, e):
        return m.mul(Qf(i, e), Qf(neg(X, k), e), m.inv(Qf(add(X, i, neg(X, k)), e)), m.inv(Qf(zero(X), e)))
    g = {}
    for k in elements(X):
        for e in elements(Y):
            for i in elements(X):
                for j in elements(Y):
                    g[(k, e, i, j)] = m.mul(R(k, i, add(Y, e, j)), R(k, i, zero(Y)), m.inv(R(k, i, e)), m.inv(R(k, i, j)))
    return g

def dQ(m, Qf, GA, GC, a, b, c, d):
    ab, cd = add(GA, a, b), add(GC, c, d)
    return m.mul(Qf(ab, cd), Qf(a, c), Qf(b, c), Qf(a, d), Qf(b, d),
                 m.inv(Qf(a, cd)), m.inv(Qf(b, cd)), m.inv(Qf(ab, c)), m.inv(Qf(ab, d)))

def rank(rows):
    rows = [[Fr(x) for x in r] for r in rows]; r = 0
    ncol = len(rows[0]) if rows else 0
    for c in range(ncol):
        p = next((i for i in range(r, len(rows)) if rows[i][c] != 0), None)
        if p is None: continue
        rows[r], rows[p] = rows[p], rows[r]
        for i in range(len(rows)):
            if i != r and rows[i][c] != 0:
                f = rows[i][c]/rows[r][c]
                rows[i] = [x-f*y for x, y in zip(rows[i], rows[r])]
        r += 1
    return r

def main():
    ok = True
    out('X        Y        free  generators  double difference  symmetries  transposition  rank  second differences')
    for X in GROUPS:
        for Y in GROUPS:
            m = Mono(X, Y)
            Qf = lambda i, b: m.Q(i, b)
            Qt = lambda b, i: m.Q(i, b)                      # the transposed matrix, indexed by Y x X
            g = generators(m, Qf)
            EX, EY = elements(X), elements(Y)
            c2 = all(g[(k, e, i, j)] == m.inv(dQ(m, Qf, X, Y, i, neg(X, k), e, j))
                     for k in EX for e in EY for i in EX for j in EY)
            c3 = all(dQ(m, Qf, X, Y, a, b, c, d) == dQ(m, Qf, X, Y, b, a, c, d) == dQ(m, Qf, X, Y, a, b, d, c)
                     == dQ(m, Qt, Y, X, c, d, a, b)
                     for a in EX for b in EX for c in EY for d in EY)
            mt = Mono(Y, X)                                  # same variables, read through Q^t
            mt.Q = lambda b, i: m.Q(i, b); mt.dim = m.dim
            gt = generators(mt, mt.Q)
            c4 = all(gt[(e, k, j, i)] == g[(neg(X, k), neg(Y, e), i, j)]
                     for k in EX for e in EY for i in EX for j in EY)
            # exponent matrix of the generators: row (k,e), column (i,j), entry = vector; the phase
            # group is the whole torus exactly when the characters trivial on all g_{k,e} are trivial
            rows = []
            for v in range(m.dim):
                for k in EX:
                    for e in EY:
                        rows.append([g[(k, e, i, j)][v] for (i, j) in m.free])
            rk = rank(rows)
            c5 = (rk == m.dim)
            # second differences, for Q not dephased
            mf = Mono(X, Y, dephased=False)
            Qn = lambda i, b: mf.Q(i, b)
            gn = generators(mf, Qn)
            def Wd(k, e, i, j):
                W = lambda a, b: mf.mul(Qn(add(X, a, k), add(Y, b, e)), Qn(a, b),
                                        mf.inv(Qn(add(X, a, k), b)), mf.inv(Qn(a, add(Y, b, e))))
                return mf.mul(W(i, j), W(zero(X), zero(Y)), mf.inv(W(i, zero(Y))), mf.inv(W(zero(X), j)))
            c6 = all(Wd(k, e, i, j) == mf.inv(gn[(neg(X, k), e, i, j)])
                     for k in EX for e in EY for i in EX for j in EY)
            good = c2 and c3 and c4 and c5 and c6; ok = ok and good
            out('%-8s %-8s %4d  %10d  %17s  %10s  %13s  %4d  %18s' % (X, Y, m.dim, len(EX)*len(EY), c2, c3, c4, rk, c6))
    out('ALL CHECKS PASSED' if ok else 'SOME CHECK FAILED')

if __name__ == '__main__':
    main()
