#!/usr/bin/env python3
"""zulle.py - growth of the moments, c_n ~ C (MN)^n n^(-d/2) (Theorem 6.1, Corollary 7.3, Theorem 7.6).

A parameter matrix is given by its exponents: Q_ij = e(a_ij/L) * prod_s gamma_s^(b_ij[s]), where
the gamma_s are multiplicatively independent numbers of infinite order. From this data the program
computes, in exact integer arithmetic,
    the generators g_(k,e) of the phase group K_Q (Proposition 4.2),
    d      = dim K_Q,
    kappa  = number of connected components of K_Q,
    Delta  = determinant of the quadratic form q on the lattice Lambda* (Theorem 6.1),
    C    = (MN)^(-1) kappa^(-1) (MN/4 pi)^(d/2) Delta^(-1/2),
and compares C (MN)^n n^(-d/2) with the moments c_n, which are computed exactly when a closed
formula is available, and otherwise by integration of Tr((ZZ*)^n) over K_Q on a grid.

Needs numpy for the grid integrations only. Output: terminal, and zulle.log.
"""
from fractions import Fraction as Fr
from math import comb, gcd, pi, sqrt
from itertools import product
import sys

LOG = open(__file__.replace('.py', '.log'), 'w')
def out(*a):
    s = ' '.join(str(x) for x in a)
    print(s); LOG.write(s + '\n'); LOG.flush()

# ------------------------------------------------------------------ integer linear algebra
def kernel_lattice(rows, D):
    """basis of {m in Z^D : <r, m> = 0 for all rows r}, a saturated lattice (column reduction)."""
    A = [list(r) for r in rows]; U = [[int(i == j) for j in range(D)] for i in range(D)]
    piv = 0
    for r in range(len(A)):
        if piv == D: break
        while True:
            nz = [j for j in range(piv, D) if A[r][j] != 0]
            if len(nz) <= 1: break
            j0 = min(nz, key=lambda j: abs(A[r][j]))
            for j in nz:
                if j == j0: continue
                q = A[r][j]//A[r][j0]
                for M_ in (A, U):
                    for row in M_: row[j] -= q*row[j0]
        nz = [j for j in range(piv, D) if A[r][j] != 0]
        if nz:
            j0 = nz[0]
            for M_ in (A, U):
                for row in M_: row[piv], row[j0] = row[j0], row[piv]
            piv += 1
    return [[U[i][j] for i in range(D)] for j in range(piv, D)]

def det_fr(M_):
    M_ = [list(map(Fr, r)) for r in M_]; n = len(M_); d = Fr(1)
    for i in range(n):
        p = next((r for r in range(i, n) if M_[r][i] != 0), None)
        if p is None: return Fr(0)
        if p != i: M_[i], M_[p] = M_[p], M_[i]; d = -d
        d *= M_[i][i]
        for r in range(i+1, n):
            f = M_[r][i]/M_[i][i]
            for c in range(i, n): M_[r][c] -= f*M_[i][c]
    return d

# ------------------------------------------------------------------ groups, generators
def elements(orders): return list(product(*[range(n) for n in orders]))
def add(a, b, o): return tuple((x + y) % n for x, y, n in zip(a, b, o))
def neg(a, o): return tuple((-x) % n for x, n in zip(a, o))

class Data:
    """ox, oy: orders of the cyclic factors of X and Y; L; a[(i,j)] in Z/L; b[(i,j)] tuple in Z^m (free entries only)."""
    def __init__(s, ox, oy, L, a, b, m):
        s.ox, s.oy, s.L, s.m = ox, oy, L, m
        s.X, s.Y = elements(ox), elements(oy); s.M, s.N = len(s.X), len(s.Y)
        s.free = [(i, j) for i in s.X[1:] for j in s.Y[1:]]; s.D = len(s.free)
        z = tuple([0]*m)
        s.a = {(i, j): 0 for i in s.X for j in s.Y}; s.b = {(i, j): z for i in s.X for j in s.Y}
        for key, v in a.items(): s.a[key] = v % L
        for key, v in b.items(): s.b[key] = tuple(v)
    def expo(s, i, j):                       # exponent vector (torsion part, free part) of Q_ij
        return (s.a[(i, j)],) + s.b[(i, j)]
    def R(s, k, i, e):
        v = [0]*(1 + s.m)
        for sign, (p, q) in ((1, (i, e)), (1, (neg(k, s.ox), e)), (-1, (add(i, neg(k, s.ox), s.ox), e)), (-1, (s.X[0], e))):
            w = s.expo(p, q)
            for t in range(1 + s.m): v[t] += sign*w[t]
        return v
    def generators(s):
        gens = []
        for k in s.X:
            for e in s.Y:
                g = []
                for (i, j) in s.free:
                    v = [0]*(1 + s.m)
                    for sign, w in ((1, s.R(k, i, add(e, j, s.oy))), (1, s.R(k, i, s.Y[0])), (-1, s.R(k, i, e)), (-1, s.R(k, i, j))):
                        for t in range(1 + s.m): v[t] += sign*w[t]
                    g.append(v)
                gens.append(g)                # g[coordinate][0] = torsion exponent mod L, g[coordinate][1:] = free exponents
        return gens
    def invariants(s):
        gens = s.generators(); D = s.D
        W = [[g[c][1 + t] for c in range(D)] for g in gens for t in range(s.m)]     # free parts
        N1 = kernel_lattice(W, D)             # saturation of N_Q
        d = D - len(N1)
        # kappa = [N1 : N_Q] = order of the image of N1 in (Z/L)^(number of generators)
        img = {tuple([0]*len(gens))}; todo = list(img)
        vecs = [tuple(sum(m_[c]*g[c][0] for c in range(D)) % s.L for g in gens) for m_ in N1]
        while todo:
            x = todo.pop()
            for v in vecs:
                y = tuple((p + q) % s.L for p, q in zip(x, v))
                if y not in img: img.add(y); todo.append(y)
        kappa = len(img)
        Lam = kernel_lattice(N1, D) if N1 else [[int(i == j) for j in range(D)] for i in range(D)]
        # Gram matrix of q = (1 - P_M) (x) (1 - P_N) on the free coordinates
        def G(c1, c2):
            (i, j), (i2, j2) = s.free[c1], s.free[c2]
            return (Fr(int(i == i2)) - Fr(1, s.M))*(Fr(int(j == j2)) - Fr(1, s.N))
        gram = [[sum(u[c1]*G(c1, c2)*v[c2] for c1 in range(D) for c2 in range(D)) for v in Lam] for u in Lam]
        Delta = det_fr(gram) if Lam else Fr(1)
        C = 1/(s.M*s.N*kappa)*(s.M*s.N/(4*pi))**(d/2)/sqrt(float(Delta))
        return d, kappa, Delta, C, Lam, gens

# ------------------------------------------------------------------ moments
def W(N, j, memo={}):
    if (N, j) in memo: return memo[(N, j)]
    r = 1 if N == 1 else sum(comb(j, a)**2*W(N-1, j-a) for a in range(j+1))
    memo[(N, j)] = r; return r

def two_rows(N, even_moment):
    """c_n = (1/N) sum binom(n,2j) N^(n-2j) E|S|^(2j), exact"""
    return lambda n: Fr(sum(comb(n, 2*j)*N**(n-2*j)*even_moment(j) for j in range(n//2+1)), N)

def grid_moment(data, n, pts):
    """(1/MN) * integral over K_Q of Tr((ZZ*)^n), divided by (MN)^n; grid on the identity component,
    finite sum over the components (found by enumeration of the torsion directions)."""
    import numpy as np
    d, kappa, Delta, C, Lam, gens = data.invariants()
    D, M, N, L = data.D, data.M, data.N, data.L
    # torsion representatives: subgroup generated by the generators, modulo the identity component
    # we enumerate the group generated by the torsion parts on the saturated complement; simple approach:
    # K_Q = closure of <g>; a point of K_Q is exp(2 pi i (sum_s t_s B_s + T/L)) with B_s the free exponent
    # vectors, t real, T in the finite group generated by the torsion exponent vectors.
    tors = {tuple([0]*D)}; todo = list(tors)
    tv = [tuple(g[c][0] % L for c in range(D)) for g in gens]
    while todo:
        x = todo.pop()
        for v in tv:
            y = tuple((p + q) % L for p, q in zip(x, v))
            if y not in tors: tors.add(y); todo.append(y)
    # identity component: theta = sum_r u_r Lam_r, u in [0,1)^d
    import itertools
    us = np.linspace(0, 1, pts, endpoint=False)
    grids = np.meshgrid(*([us]*d), indexing='ij') if d else []
    theta0 = np.zeros((D,) + ((pts,)*d if d else ()))
    for r in range(d):
        for c in range(D): theta0[c] += Lam[r][c]*grids[r]
    # distinct components: torsion points modulo the identity component give repeated values; average over all
    # torsion points is correct anyway (each component is hit the same number of times)
    total = 0.0
    for T in tors:
        Zfree = np.exp(2j*np.pi*(theta0 + np.array(T, dtype=float).reshape((D,) + (1,)*d)/L))
        shape = Zfree.shape[1:]
        Z = np.ones((M, N) + shape, dtype=complex)
        for c, (i, j) in enumerate(data.free):
            Z[data.X.index(i), data.Y.index(j)] = Zfree[c]
        Zm = np.moveaxis(Z, (0, 1), (-2, -1))
        B = Zm @ np.conj(np.swapaxes(Zm, -1, -2))
        ev = np.linalg.eigvalsh(B)/(M*N)
        total += float(np.mean(np.sum(ev**n, axis=-1)))
    return total/len(tors)/(M*N)

def main():
    ok = True
    one = (1,)
    cases = []
    # generic 2 x N
    for N in (2, 3, 4, 5):
        b = {((1,), (j,)): tuple(int(t == j-1) for t in range(N-1)) for j in range(1, N)}
        cases.append(('generic F_2 x F_%d' % N, Data((2,), (N,), 1, {}, b, N-1), two_rows(N, lambda j, N=N: W(N, j)), None))
    # index 8, Y = Z_4, case 2 with m = 1: row (1, a, a, i)
    cases.append(('F_2 x F_4, row (1,a,a,i)', Data((2,), (4,), 4, {(one, (3,)): 1}, {(one, (1,)): (1,), (one, (2,)): (1,)}, 1),
                  two_rows(4, lambda j: comb(2*j, j)**2), None))
    # case 2 with m = 2 and m = 3: row (1, a, a, c) with c a primitive 4m-th root of unity
    for m in (2, 3):
        cases.append(('F_2 x F_4, row (1,a,a,e(1/%d))' % (4*m), Data((2,), (4,), 4*m, {(one, (3,)): 1}, {(one, (1,)): (1,), (one, (2,)): (1,)}, 1), None, 96))
    # case 3: rows (1, q, 1, q) and (1, q, i, q)
    cases.append(('F_2 x F_4, row (1,q,1,q)', Data((2,), (4,), 1, {}, {(one, (1,)): (1,), (one, (3,)): (1,)}, 1),
                  lambda n: Fr(2**n*comb(2*n, n), 4), None))
    cases.append(('F_2 x F_4, row (1,q,i,q)', Data((2,), (4,), 4, {(one, (2,)): 1}, {(one, (1,)): (1,), (one, (3,)): (1,)}, 1),
                  lambda n: Fr(4**n, 8) + Fr(2**n*comb(2*n, n), 8), None))
    # Klein: row (1, p, q, pq), order 00, 10, 01, 11
    cases.append(('F_2 x F_(2,2), row (1,p,q,pq)', Data((2,), (2, 2), 1, {}, {(one, (1, 0)): (1, 0), (one, (0, 1)): (0, 1), (one, (1, 1)): (1, 1)}, 2),
                  two_rows(4, lambda j: comb(2*j, j)**2), None))
    # index 9 slice: Q_11 = q, Q_22 = 1/q
    cases.append(('F_3 x F_3, Q_11 = q, Q_22 = 1/q', Data((3,), (3,), 1, {}, {((1,), (1,)): (1,), ((2,), (2,)): (-1,)}, 1),
                  lambda n: Fr(W(3, n), 3), None))
    # generic 3 x 3, by grid integration
    b = {((i,), (j,)): tuple(int(t == 2*(i-1) + (j-1)) for t in range(4)) for i in (1, 2) for j in (1, 2)}
    cases.append(('generic F_3 x F_3', Data((3,), (3,), 1, {}, b, 4), None, 36))
    # circulant family of Theorem 7.1: Q_ij = q^(1 + delta(i+j,0)) for i,j nonzero, on cyclic and noncyclic groups
    for og in ((3,), (4,), (2, 2), (5,), (6,)):
        G = elements(og); N = len(G); zero = G[0]
        b = {(i, j): (1 + int(add(i, j, og) == zero),) for i in G[1:] for j in G[1:]}
        cases.append(('circulant, G = %s' % (og,), Data(og, og, 1, {}, b, 1), lambda n, N=N: Fr(W(N, n), N), None))
    out('%-34s %2s %5s %8s %12s   %s' % ('matrix', 'd', 'kappa', 'Delta', 'C', 'c_n / (C (MN)^n n^(-d/2))  at n = 100, 400, 1600'))
    for name, data, exact, pts in cases:
        d, kappa, Delta, C, Lam, gens = data.invariants()
        MN = data.M*data.N; ratios = []
        if exact is not None:
            for n in (100, 400, 1600):
                c = exact(n)
                ratios.append(float(Fr(c)/Fr(MN)**n)/(C*n**(-d/2)))
            last = ratios[-1]
        else:
            ns = (100, 400) if d < 4 else (60, 120)
            for n in ns:
                ratios.append(grid_moment(data, n, pts)/(C*n**(-d/2)))
            last = ratios[-1]
        tol = 0.002 if exact is not None else (0.02 if d < 4 else 0.08)
        good = abs(last - 1) < tol and abs(ratios[-1] - 1) <= abs(ratios[0] - 1) + 1e-9; ok &= good
        out('%-34s %2d %5d %8s %12.8f   %s  %s%s' % (name, d, kappa, Delta, C, ' '.join('%.6f' % r for r in ratios),
            '' if exact is not None else ('(grid, n = %s) ' % (', '.join(map(str, ns)))), 'ok' if good else 'FAIL'))
        if name.startswith('circulant'):
            # Corollary 7.3 against Theorem 6.1: d = N-1, kappa = 1, Delta = N^(N-4), C = N^((N-2)/2) / (2^(N-1) pi^((N-1)/2))
            N = data.N
            good = d == N-1 and kappa == 1 and Delta == Fr(N)**(N-4) and abs(C - N**((N-2)/2)/(2**(N-1)*pi**((N-1)/2))) < 1e-12; ok &= good
            out('    Corollary 7.3: d = N-1, kappa = 1, Delta = N^(N-4) = %s, constant of the corollary: %s' % (Fr(N)**(N-4), 'ok' if good else 'FAIL'))
    # finite grids of Theorem 7.6: all free entries equal to q, of order l coprime to MN; then K_Q has l^d elements
    out('')
    out('%-34s %2s %8s %8s %10s' % ('constant free entries', 'd', 'kappa', 'l^D', '|G_H|'))
    for ox, oy, l in (((2,), (3,), 5), ((2,), (4,), 3), ((3,), (3,), 2), ((3,), (3,), 4), ((2, 2), (3,), 5), ((2,), (5,), 3), ((3,), (4,), 5)):
        X, Y = elements(ox), elements(oy)
        data = Data(ox, oy, l, {(i, j): 1 for i in X[1:] for j in Y[1:]}, {}, 0)
        d, kappa, Delta, C, Lam, gens = data.invariants()
        good = d == 0 and kappa == l**data.D and Delta == 1; ok &= good
        out('%-34s %2d %8d %8d %10d  %s' % ('X = %s, Y = %s, l = %d' % (ox, oy, l), d, kappa, l**data.D, data.M*data.N*kappa, 'ok' if good else 'FAIL'))
    # the assumption on l is needed: at M = N = 2 the group is generated by q^4
    for l in (2, 4, 8, 12):
        data = Data((2,), (2,), l, {((1,), (1,)): 1}, {}, 0)
        d, kappa, Delta, C, Lam, gens = data.invariants()
        good = kappa == l//gcd(l, 4); ok &= good
        out('%-34s %2d %8d %8s %10d  %s' % ('X = Y = (2,), l = %d' % l, d, kappa, '-', 4*kappa, 'ok' if good else 'FAIL'))
    out('ALL CHECKS PASSED' if ok else 'SOME CHECK FAILED')
    return 0 if ok else 1

if __name__ == '__main__':
    sys.exit(main())
