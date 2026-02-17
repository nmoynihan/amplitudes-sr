from __future__ import annotations

import random
from fractions import Fraction


def angle(spinor_i, spinor_j):
    """Angle bracket <ij> = lambda_i^1 lambda_j^2 - lambda_i^2 lambda_j^1."""
    li, lj = spinor_i["lam"], spinor_j["lam"]
    return li[0] * lj[1] - li[1] * lj[0]


def square(spinor_i, spinor_j):
    """Square bracket [ij] = lambdat_i^1 lambdat_j^2 - lambdat_i^2 lambdat_j^1."""
    li, lj = spinor_i["lamt"], spinor_j["lamt"]
    return li[0] * lj[1] - li[1] * lj[0]


def mandelstam_2pt(spinor_i, spinor_j):
    """Two-particle Mandelstam invariant s_ij = <ij>[ji]."""
    return angle(spinor_i, spinor_j) * square(spinor_j, spinor_i)


def mandelstam(spinor_i, spinor_j):
    """Alias for two-particle Mandelstam used by 5pt scripts."""
    return mandelstam_2pt(spinor_i, spinor_j)


def _outer(lam, lamt):
    return [
        [lam[0] * lamt[0], lam[0] * lamt[1]],
        [lam[1] * lamt[0], lam[1] * lamt[1]],
    ]


def _add2(A, B):
    return [
        [A[0][0] + B[0][0], A[0][1] + B[0][1]],
        [A[1][0] + B[1][0], A[1][1] + B[1][1]],
    ]


def _matmul2(A, B):
    return [
        [A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
        [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]],
    ]


def _inv2(M):
    a, b = M[0]
    c, d = M[1]
    det = a * d - b * c
    if det == 0:
        raise ValueError("singular 2x2")
    return [[d / det, -b / det], [-c / det, a / det]]


def generate_exact_spinors(n, int_range=4, max_tries=100_000):
    """
    Generate exact rational (2,2)-signature on-shell spinors for n particles
    with exact momentum conservation.
    """
    for _ in range(max_tries):
        lam = [
            (Fraction(random.randint(-int_range, int_range)), Fraction(random.randint(-int_range, int_range)))
            for _ in range(n)
        ]

        L = [[lam[n - 2][0], lam[n - 1][0]], [lam[n - 2][1], lam[n - 1][1]]]
        if L[0][0] * L[1][1] - L[0][1] * L[1][0] == 0:
            continue

        lamt = [
            (Fraction(random.randint(-int_range, int_range)), Fraction(random.randint(-int_range, int_range)))
            for _ in range(n - 2)
        ]

        S = [[Fraction(0), Fraction(0)], [Fraction(0), Fraction(0)]]
        for i in range(n - 2):
            S = _add2(S, _outer(lam[i], lamt[i]))

        Linv = _inv2(L)
        minusS = [[-S[0][0], -S[0][1]], [-S[1][0], -S[1][1]]]
        T = _matmul2(Linv, minusS)
        lamt.append((T[0][0], T[0][1]))
        lamt.append((T[1][0], T[1][1]))

        spinors = [{"lam": lam[i], "lamt": lamt[i]} for i in range(n)]

        Tot = [[Fraction(0), Fraction(0)], [Fraction(0), Fraction(0)]]
        for sp in spinors:
            Tot = _add2(Tot, _outer(sp["lam"], sp["lamt"]))
        if Tot != [[0, 0], [0, 0]]:
            continue

        good = True
        for i in range(n):
            for j in range(i + 1, n):
                if angle(spinors[i], spinors[j]) == 0:
                    good = False
                    break
            if not good:
                break
        if not good:
            continue

        return spinors

    raise RuntimeError(
        f"Failed to generate non-degenerate spinors for n={n} "
        f"after {max_tries} attempts. Try increasing int_range."
    )
