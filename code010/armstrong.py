"""Hadamard projection checks for the spectral blowup paper.

Construct Dita-deformed Fourier matrices and the finite tensor moment
matrices. Count eigenvalues within 1e-6 of 1 and compare with the formulas
in the paper. These comparisons use floating-point arithmetic.

Run: python3 armstrong.py
Requires NumPy. The former --quick and --rapide flags are accepted.
"""

from itertools import product
from math import comb, factorial, prod
from fractions import Fraction

import numpy as np


def planar_walk_moment(columns, half_power):
    """Exact E|z_1+...+z_N|^(2j) for independent circle phases."""
    return sum(
        (factorial(half_power) // prod(factorial(count) for count in content)) ** 2
        for content in product(range(half_power + 1), repeat=columns)
        if sum(content) == half_power
    )


def deformed_fourier(rows, columns, parameters):
    matrix = np.empty((rows * columns, rows * columns), dtype=complex)
    for i, a, j, b in product(range(rows), range(columns), range(rows), range(columns)):
        matrix[i * columns + a, j * columns + b] = (
            parameters[i, b] * np.exp(2j * np.pi * (i * j / rows + a * b / columns))
        )
    return matrix


def fixed_dimension(matrix, degree):
    """Numerical multiplicity of 1 in the tensor moment matrix.

    Its (i,j) entry is the normalized trace of the cyclic product of the
    rank-one projections p[i_1,j_1],...,p[i_degree,j_degree].
    """
    size = len(matrix)
    vectors = np.array([
        [matrix[row] / matrix[column] / np.sqrt(size) for column in range(size)]
        for row in range(size)
    ])
    gram = np.einsum('rsx,tux->rstu', vectors.conj(), vectors).reshape(size ** 2, size ** 2)
    if degree == 1:
        moment_matrix = np.full((size, size), 1 / size, dtype=complex)
    else:
        cyclic_product = np.ones((size ** 2,) * degree, dtype=complex)
        for position in range(degree):
            following = (position + 1) % degree
            shape = [1] * degree
            shape[position] = size ** 2
            shape[following] = size ** 2
            factor = gram if position < following else gram.T
            cyclic_product *= factor.reshape(shape)
        row_axes = list(range(0, 2 * degree, 2))
        column_axes = list(range(1, 2 * degree, 2))
        moment_matrix = (cyclic_product / size).reshape((size, size) * degree)
        moment_matrix = moment_matrix.transpose(row_axes + column_axes).reshape(size ** degree, size ** degree)
    eigenvalues = np.linalg.eigvals(moment_matrix)
    return int(np.count_nonzero(np.abs(eigenvalues - 1) < 1e-6))


def main():
    generic = [
        Fraction(sum(
            comb(degree, 2*j) * 3 ** (degree-2*j) * planar_walk_moment(3, j)
            for j in range(degree // 2 + 1)
        ), 3)
        for degree in range(1, 5)
    ]
    assert generic == [1, 4, 18, 86]
    print('Exact generic index-six moments', list(map(int, generic)), flush=True)

    random = np.random.default_rng(3)
    for rows, columns in [(2, 3), (3, 2)]:
        parameters = np.exp(2j * np.pi * random.random((rows, columns)))
        parameters[0, :] = 1
        parameters[:, 0] = 1
        matrix = deformed_fourier(rows, columns, parameters)
        assert np.allclose(matrix @ matrix.conj().T, (rows * columns) * np.eye(rows * columns))
        dimensions = [fixed_dimension(matrix, degree) for degree in (1, 2, 3)]
        assert dimensions == [1, 4, 18]
        print('Sample deformation', rows, columns, 'fixed dimensions', dimensions, flush=True)

    cases = [
        (4, [1, 4, 16, 64, 256]),
        (8, [1, 3, 10, 36, 136]),
        (3, [1, 3, 10, 35, 126]),
    ]
    for order, expected in cases:
        phase = np.exp(2j * np.pi / order)
        matrix = deformed_fourier(2, 2, np.array([[1, 1], [1, phase]]))
        dimensions = [fixed_dimension(matrix, degree) for degree in range(1, 6)]
        assert dimensions == expected
        print('Index four, phase order', order, 'fixed dimensions', dimensions, flush=True)
    print('All Hadamard projection checks passed.')


if __name__ == '__main__':
    main()
