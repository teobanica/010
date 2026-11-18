#!/usr/bin/env python3
"""Exact checks for Section 7 of the paper (010.tex).

Run: python3 coppi.py
Python standard library only. No numerical phase integrations are used.

1. For Q[i,j] = q**[i+j=0], verify the phase generators are dephasings
   of group-circulant vectors, their summed exponents, and rank |G|-1.
2. Independently count return words using the block action of the quantum
   group. Compare with (1/|G|) times the squared multinomial sum.
3. Check the generic first three moments by balanced Gram monomials.
4. Check the delayed index-eight Hankel and Jacobi defects for m=1,...,5,
   using direct character counts on T^2 x the group of m-th roots.
5. Check the odd-degree Jacobi defect for the new square-index family
   and the determinant of the quadratic form used in its growth estimate.

6. For the constant free-entry matrix Q, check the exponent determinant,
   finite-grid moment agreement, and the first Hankel/Jacobi discrepancy.
   Compare finite-grid laws with independent block-action word counts.

These finite checks accompany Theorem 7.1, Proposition 7.2, Corollary 7.3,
Propositions 7.4-7.5, Theorem 7.6 and Corollary 7.7 of the paper.
The uniform proofs are in the text.
"""

from collections import Counter
from fractions import Fraction
from functools import lru_cache
from itertools import product
from math import comb, factorial, gcd, prod


def group_tables(shape):
    elements = list(product(*(range(n) for n in shape)))
    where = {x: i for i, x in enumerate(elements)}
    add = [[where[tuple((x+y) % n for x, y, n in zip(a, b, shape))]
            for b in elements] for a in elements]
    sub = [[where[tuple((x-y) % n for x, y, n in zip(a, b, shape))]
            for b in elements] for a in elements]
    return add, sub


def rank(matrix):
    a = [list(map(Fraction, row)) for row in matrix]
    r = 0
    for j in range(len(a[0])):
        pivot = next((i for i in range(r, len(a)) if a[i][j]), None)
        if pivot is None:
            continue
        a[r], a[pivot] = a[pivot], a[r]
        scale = a[r][j]
        a[r] = [x/scale for x in a[r]]
        for i in range(r+1, len(a)):
            scale = a[i][j]
            if scale:
                a[i] = [x-scale*y for x, y in zip(a[i], a[r])]
        r += 1
    return r


def determinant(matrix):
    """Fraction-free Bareiss determinant, with row pivoting."""
    a = [row[:] for row in matrix]
    n = len(a)
    if n == 0:
        return 1
    previous, sign = 1, 1
    for k in range(n-1):
        pivot = next((i for i in range(k, n) if a[i][k]), None)
        if pivot is None:
            return 0
        if pivot != k:
            a[k], a[pivot] = a[pivot], a[k]
            sign *= -1
        value = a[k][k]
        for i in range(k+1, n):
            for j in range(k+1, n):
                numerator = a[i][j]*value - a[i][k]*a[k][j]
                assert numerator % previous == 0
                a[i][j] = numerator // previous
            a[i][k] = 0
        previous = value
    return sign*a[-1][-1]


def phase_check(shape):
    add, sub = group_tables(shape)
    n = len(add)
    q = [[int(add[i][j] == 0) for j in range(n)] for i in range(n)]

    def r(k, i, e):
        return q[i][e]+q[sub[0][k]][e]-q[sub[i][k]][e]-q[0][e]

    rows = []
    for e in range(n):
        vectors = []
        for k in range(n):
            v = [int(add[t][e] == 0)+int(sub[t][k] == 0)
                 - int(sub[add[t][e]][k] == 0)-int(t == 0)
                 for t in range(n)]
            vectors.append(v)
            g = [r(k, i, add[e][j])+r(k, i, 0)-r(k, i, e)-r(k, i, j)
                 for i in range(1, n) for j in range(1, n)]
            dephased = [v[add[i][j]]+v[0]-v[i]-v[j]
                        for i in range(1, n) for j in range(1, n)]
            assert g == dephased
            rows.append(g)
        assert [sum(v[t] for v in vectors) for t in range(n)] == [
            n*(int(add[t][e] == 0)-int(t == 0)) for t in range(n)]
    assert rank(rows) == n-1
    return q


def return_word_moments(shape, q, depth):
    """The block-action return-word computation, for an infinite-order q.

    A state has its group degree and the integer phase exponents, reduced
    by a scalar on each block. Exponents are never reduced modulo an order.
    """
    add, sub = group_tables(shape)
    n = len(add)
    positions = list(product(range(n), repeat=2))
    zero = (0,)*(n*n)
    origin = (0, zero)

    # A word is read using the same block action as wordgram.py, but in Z.
    increments = {}
    for degree in range(n):
        targets = []
        for i, b in positions:
            increment = []
            for k, e in positions:
                shifted = sub[e][degree]
                shifted_b = sub[shifted][b]
                increment.append(q[i][shifted_b]+q[sub[i][k]][shifted]
                                 -q[i][shifted]-q[sub[i][k]][shifted_b])
            increment = tuple(increment[k*n+e]-increment[k*n]
                              for k, e in positions)
            targets.append((add[degree][b], increment))
        increments[degree] = Counter(targets)

    paths = Counter({origin: 1})
    result = [1]
    for step in range(1, depth+1):
        following = Counter()
        for (degree, phase), count in paths.items():
            for (new_degree, increment), multiplicity in increments[degree].items():
                new_phase = tuple(a+b for a, b in zip(phase, increment))
                following[new_degree, new_phase] += multiplicity*count
        paths = following
        assert paths[origin] % n == 0
        result.append(paths[origin]//n)
    return result


def contents(length, size):
    if size == 1:
        yield (length,)
        return
    for first in range(length+1):
        for rest in contents(length-first, size-1):
            yield (first,)+rest


def multinomial(content):
    return factorial(sum(content))//prod(factorial(x) for x in content)


@lru_cache(None)
def walk_moment(n, size):
    return sum(multinomial(a)**2 for a in contents(n, size))


def square_moment(n, size):
    if n == 0:
        return 1
    value = walk_moment(n, size)
    assert value % size == 0
    return value//size


def generic_gram_moment(n, size):
    """Integrate the Gram trace using balance of independent entry phases."""
    count = 0
    for rows in product(range(size), repeat=n):
        shifted = rows[1:]+rows[:1]
        for cols in product(range(size), repeat=n):
            count += sorted(zip(rows, cols)) == sorted(zip(shifted, cols))
    assert count % (size*size) == 0
    return count//(size*size)


def eight_moments(order, depth):
    """The row (1,U,EUV,V), with E of order 'order'; None is generic."""
    even_abs = []
    for j in range(depth//2+1):
        grouped = Counter()
        for a in contents(j, 4):
            key = a if order is None else (a[1]+a[2], a[2]+a[3], a[2] % order)
            grouped[key] += multinomial(a)
        even_abs.append(sum(value*value for value in grouped.values()))
    result = [1]
    for n in range(1, depth+1):
        value = sum(comb(n, 2*j)*4**(n-2*j)*even_abs[j] for j in range(n//2+1))
        assert value % 4 == 0
        result.append(value//4)
    return result


def hankel(moments, size):
    return determinant([[moments[i+j] for j in range(size)] for i in range(size)])


def constant_generator_check(xshape, yshape):
    """Proposition 4.2 evaluated on Q[i,j] = q for i,j nonzero."""
    _, sx = group_tables(xshape)
    ay, sy = group_tables(yshape)
    m, n = len(sx), len(sy)
    q = [[int(i != 0 and j != 0) for j in range(n)] for i in range(m)]

    def r(k, i, e):
        return q[i][e]+q[sx[0][k]][e]-q[sx[i][k]][e]-q[0][e]

    matrix = []
    for k in range(1, m):
        for e in range(1, n):
            row = [r(k, i, ay[e][j])+r(k, i, 0)-r(k, i, e)-r(k, i, j)
                   for i in range(1, m) for j in range(1, n)]
            expected = [-(1+int(i == k))*(1+int(j == sy[0][e]))
                        for i in range(1, m) for j in range(1, n)]
            assert row == expected
            matrix.append(row)
    det = abs(determinant(matrix))
    assert det == m**(n-1)*n**(m-1)
    return det


@lru_cache(None)
def two_row_grid_moments(n, order):
    """Generic and finite-grid moments through the first extra degree 2*order."""
    finite_abs, generic_abs = [], []
    for j in range(order+1):
        grouped = Counter()
        for a in contents(j, n):
            grouped[tuple(v % order for v in a)] += multinomial(a)
        finite_abs.append(sum(v*v for v in grouped.values()))
        generic_abs.append(walk_moment(j, n))
    laws = []
    for moments in (generic_abs, finite_abs):
        result = [1]
        for degree in range(1, 2*order+1):
            total = sum(comb(degree, 2*j)*n**(degree-2*j)*moments[j]
                        for j in range(degree//2+1))
            assert total % n == 0
            result.append(total//n)
        laws.append(result)
    return laws


@lru_cache(None)
def rectangular_grid_moments(m, n, order, depth):
    """Direct expansion of Gram traces, with zero or modular phase balance."""
    generic, finite = [1], [1]
    for degree in range(1, depth+1):
        a = b = 0
        for rows in product(range(m), repeat=degree):
            shifted = rows[1:]+rows[:1]
            for cols in product(range(n), repeat=degree):
                exponents = Counter(zip(rows, cols))
                exponents.subtract(Counter(zip(shifted, cols)))
                a += all(v == 0 for v in exponents.values())
                b += all(v % order == 0 for v in exponents.values())
        assert a % (m*n) == b % (m*n) == 0
        generic.append(a//(m*n))
        finite.append(b//(m*n))
    return generic, finite


def finite_block_moments(xshape, yshape, order, depth):
    """Independent return words from the block action of Proposition 3.6."""
    _, sx = group_tables(xshape)
    ay, sy = group_tables(yshape)
    m, n = len(sx), len(sy)
    q = [[int(i != 0 and e != 0) for e in range(n)] for i in range(m)]
    positions = list(product(range(m), range(n)))
    origin = (0, (0,)*(m*n))
    increments = {}
    for degree in range(n):
        targets = []
        for i, b in positions:
            raw = []
            for k, e in positions:
                shifted = sy[e][degree]
                shifted_b = sy[shifted][b]
                raw.append(q[i][shifted_b]+q[sx[i][k]][shifted]
                           -q[i][shifted]-q[sx[i][k]][shifted_b])
            increment = tuple((raw[k*n+e]-raw[k*n]) % order
                              for k, e in positions)
            targets.append((ay[degree][b], increment))
        increments[degree] = Counter(targets)
    paths = Counter({origin: 1})
    result = [1]
    for _ in range(depth):
        following = Counter()
        for (degree, phase), count in paths.items():
            for (new_degree, increment), multiplicity in increments[degree].items():
                new_phase = tuple((a+b) % order for a, b in zip(phase, increment))
                following[new_degree, new_phase] += count*multiplicity
        paths = following
        assert paths[origin] % m == 0
        result.append(paths[origin]//m)
    return result


def finite_prefix_checks():
    print('\nFINITE GRIDS: GENERATOR DETERMINANTS')
    pairs = [((2,), (3,)), ((2,), (4,)), ((3,), (3,)),
             ((2, 2), (3,)), ((3,), (2, 2)), ((2, 2), (2, 2))]
    for xshape, yshape in pairs:
        det = constant_generator_check(xshape, yshape)
        print('  X =', xshape, 'Y =', yshape, ': absolute determinant =', det)

    print('\nFINITE GRIDS: FIRST MOMENT, HANKEL AND JACOBI DEFECTS')
    for n in range(2, 6):
        for order in range(2, 8):
            if gcd(2*n, order) != 1:
                continue
            generic, finite = two_row_grid_moments(n, order)
            defect = n-1
            assert generic[:-1] == finite[:-1]
            assert finite[-1]-generic[-1] == defect
            r = order
            for k in range(1, r+1):
                assert hankel(generic, k) == hankel(finite, k) > 0
            d = hankel(generic, r)
            prev = hankel(generic, r-1)
            a, b = hankel(generic, r+1), hankel(finite, r+1)
            assert b-a == defect*d
            jump = Fraction(b*prev, d*d)-Fraction(a*prev, d*d)
            assert jump == Fraction(defect*prev, d)
            print('  M=2 N=%d l=%d: first degree %d; defect %d; Jacobi check passed'
                  % (n, order, 2*order, defect))

    for m, n, order in [(3, 3, 2), (3, 5, 2)]:
        generic, finite = rectangular_grid_moments(m, n, order, 2*order)
        assert generic[:-1] == finite[:-1]
        assert finite[-1]-generic[-1] == (m-1)*(n-1)
        print('  Direct Gram count M=%d N=%d l=%d:' % (m, n, order), finite)
        if (m, n) == (3, 3):
            assert generic == [1, 1, 5, 29, 181]
            assert finite == [1, 1, 5, 29, 185]
            assert (hankel(generic, 3), hankel(finite, 3)) == (48, 64)
            assert (Fraction(hankel(generic, 3), 16),
                    Fraction(hankel(finite, 3), 16)) == (3, 4)

    print('\nFINITE GRIDS: INDEPENDENT BLOCK-ACTION COUNTS')
    cases = [((2,), (3,), 5, 10), ((2,), (4,), 3, 6),
             ((3,), (3,), 2, 4), ((3,), (5,), 2, 4),
             ((2, 2), (3,), 5, 3)]
    for xshape, yshape, order, depth in cases:
        m, n = prod(xshape), prod(yshape)
        expected = (two_row_grid_moments(n, order)[1][:depth+1] if m == 2 else
                    rectangular_grid_moments(m, n, order, depth)[1])
        measured = finite_block_moments(xshape, yshape, order, depth)
        assert measured == expected
        print('  X =', xshape, 'Y =', yshape, 'l =', order, ':', measured)


def main():
    print('PHASE GENERATORS: exact identities and ranks')
    shapes = [(2,), (3,), (4,), (2, 2), (5,), (6,), (7,), (8,), (4, 2), (2, 2, 2), (3, 3)]
    for shape in shapes:
        phase_check(shape)
        print('  G =', shape, ': rank =', prod(shape)-1, '; identities passed')

    print('\nINDEPENDENT RETURN-WORD CHECKS')
    for shape, depth in [((2,), 6), ((3,), 6), ((4,), 5), ((2, 2), 5), ((5,), 3)]:
        n = prod(shape)
        measured = return_word_moments(shape, phase_check(shape), depth)
        expected = [square_moment(j, n) for j in range(depth+1)]
        assert measured == expected
        print('  G =', shape, ': c_0,...,c_%d =' % depth, measured)

    print('\nGENERIC MOMENTS AND THE ODD JACOBI DEFECT')
    for n in range(2, 6):
        generic = [generic_gram_moment(j, n) for j in range(1, 4)]
        assert generic == [1, 2*n-1, 5*n*n-6*n+2]
        exceptional = [square_moment(j, n) for j in range(1, 4)]
        assert exceptional == [1, 2*n-1, 6*n*n-9*n+4]
        assert exceptional[2]-generic[2] == (n-1)*(n-2)
        beta_generic = Fraction(generic[2]-2*generic[1]+1, generic[1]-1)
        beta_exceptional = Fraction(exceptional[2]-2*exceptional[1]+1, exceptional[1]-1)
        assert beta_exceptional-beta_generic == Fraction(n-2, 2)
        print('  N =', n, ': generic', generic, '; circulant', exceptional,
              '; beta_1 jump =', beta_exceptional-beta_generic)

    print('\nDELAYED HANKEL AND EVEN JACOBI DEFECTS')
    for m in range(1, 6):
        generic = eight_moments(None, 4*m)
        exceptional = eight_moments(m, 4*m)
        delta = comb(2*m, m)**2//2
        assert exceptional[:4*m] == generic[:4*m]
        assert exceptional[4*m]-generic[4*m] == delta
        r = 2*m
        for k in range(1, r+1):
            assert hankel(generic, k) == hankel(exceptional, k) > 0
        d = hankel(generic, r)
        prev = hankel(generic, r-1)
        a, b = hankel(generic, r+1), hankel(exceptional, r+1)
        assert b-a == delta*d
        assert Fraction((b-a)*prev, d*d) == Fraction(delta*prev, d)
        print('  m =', m, ': first Hankel size', r+1, '; defect/cofactor =', delta,
              '; lambda_%d jump =' % r, Fraction(delta*prev, d))
        if m == 1:
            assert (a, b) == (39, 47)
            print('    D_3: 39 -> 47; lambda_2: 39/16 -> 47/16')

    print('\nLAPLACE QUADRATIC FORM')
    for n in range(2, 11):
        a = [[n*int(i == j)-1 for j in range(n-1)] for i in range(n-1)]
        assert determinant(a) == n**(n-2)
    print('  det(N I_(N-1) - J_(N-1)) = N^(N-2), N=2,...,10')
    finite_prefix_checks()
    print('\nALL CHECKS PASSED')


if __name__ == '__main__':
    main()
