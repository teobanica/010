"""Exact moments for the Klein four deformation in the paper.

For S = (1+U)(1+V), U and V are independent Haar phases on finite cyclic
subgroups or on the circle. The integer order 1 denotes the trivial group;
None denotes the circle. Run: python3 klein_moments.py
Only the Python standard library is required.
"""

from fractions import Fraction
from math import comb


def phase_moment(order, half_power):
    """E|1+U|^(2*half_power), using the Laurent coefficients of (1+U)^(2m)/U^m."""
    if order is None:
        return comb(2 * half_power, half_power)
    if not isinstance(order, int) or order < 1:
        raise ValueError('A finite cyclic order must be a positive integer.')
    return sum(
        comb(2 * half_power, power)
        for power in range(2 * half_power + 1)
        if (power - half_power) % order == 0
    )


def rooted_moment(first_order, second_order, degree):
    """Moment of 3/4 delta_0 + 1/4 law(4 + epsilon*|S|)."""
    if degree == 0:
        return Fraction(1)
    return Fraction(sum(
        comb(degree, 2*j) * 4 ** (degree-2*j)
        * phase_moment(first_order, j) * phase_moment(second_order, j)
        for j in range(degree // 2 + 1)
    ), 4)


def main():
    cases = [(1, 1), (1, 5), (5, 1), (2, 2), (3, 7), (None, 4), (None, None)]
    for first_order, second_order in cases:
        moments = [rooted_moment(first_order, second_order, n) for n in range(1, 5)]
        assert all(value.denominator == 1 for value in moments)
        print('Orders', first_order, second_order, 'moments', list(map(int, moments)))
    assert [rooted_moment(1, 1, n) for n in range(1, 5)] == [1, 8, 64, 512]
    assert [rooted_moment(1, 5, n) for n in range(1, 5)] == [1, 6, 40, 280]
    assert [rooted_moment(5, 1, n) for n in range(1, 5)] == [1, 6, 40, 280]
    for orders in [(2, 2), (3, 7), (None, 4), (None, None)]:
        moments = [rooted_moment(*orders, n) for n in range(1, 5)]
        assert moments[:3] == [1, 5, 28] and moments[3] >= 169
    assert [rooted_moment(None, None, n) for n in range(1, 5)] == [1, 5, 28, 169]
    print('All Klein four checks passed. Generic moments are 1, 5, 28, 167.')


if __name__ == '__main__':
    main()
