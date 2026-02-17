from __future__ import annotations

from fractions import Fraction

from .kinematics import angle, mandelstam, square


def parke_taylor_mhv(spinors, ordering, neg=(1, 2)):
    """
    Parke-Taylor MHV gluon amplitude:
      A_n(sigma; i-, j-) = <ij>^4 / prod_k <sigma_k sigma_{k+1}>.
    """
    n = len(ordering)
    i, j = neg[0] - 1, neg[1] - 1
    num = angle(spinors[i], spinors[j]) ** 4

    den = Fraction(1)
    for k in range(n):
        a = ordering[k] - 1
        b = ordering[(k + 1) % n] - 1
        den *= angle(spinors[a], spinors[b])

    if den == 0:
        raise ZeroDivisionError("Degenerate kinematics: vanishing Parke-Taylor denominator.")
    return num / den


def hodges_gravity_mhv_5pt(spinors, neg=(1, 2)):
    """
    Hodges determinant formula for 5-point MHV gravity.
    Supports any neg pair by relabeling.
    """
    if len(spinors) != 5:
        raise ValueError("Hodges graviton mode is implemented for n=5 only.")

    a, b = neg
    if a == b:
        raise ValueError("Negative-helicity legs must be distinct.")
    if a < 1 or a > 5 or b < 1 or b > 5:
        raise ValueError("Negative-helicity legs must lie in 1..5.")

    perm = [a, b] + [i for i in range(1, 6) if i not in (a, b)]
    sp = [spinors[i - 1] for i in perm]

    n = 5
    Phi = [[Fraction(0) for _ in range(n)] for __ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            Phi[i][j] = square(sp[i], sp[j]) / angle(sp[i], sp[j])

    x, y = 0, 1
    for i in range(n):
        if i in (x, y):
            continue
        s = Fraction(0)
        for j in range(n):
            if j == i:
                continue
            s += (
                square(sp[i], sp[j])
                * angle(sp[j], sp[x])
                * angle(sp[j], sp[y])
                / (angle(sp[i], sp[j]) * angle(sp[i], sp[x]) * angle(sp[i], sp[y]))
            )
        Phi[i][i] = -s

    i4, i5 = 3, 4
    det_minor = Phi[i4][i4] * Phi[i5][i5] - Phi[i4][i5] * Phi[i5][i4]

    c123 = 1 / (angle(sp[0], sp[1]) * angle(sp[1], sp[2]) * angle(sp[2], sp[0]))
    barM5 = (c123 * c123) * det_minor

    return (angle(sp[0], sp[1]) ** 8) * barM5


def klt_gravity_5pt_for_validation(spinors, neg=(1, 2)):
    """
    Standard 5-point field-theory KLT relation (for validation only):
      M_5 = s_12 s_34 A(1,2,3,4,5)A(2,1,4,3,5) + s_13 s_24 A(1,3,2,4,5)A(3,1,4,2,5).
    """
    s12 = mandelstam(spinors[0], spinors[1])
    s34 = mandelstam(spinors[2], spinors[3])
    s13 = mandelstam(spinors[0], spinors[2])
    s24 = mandelstam(spinors[1], spinors[3])

    A12345 = parke_taylor_mhv(spinors, (1, 2, 3, 4, 5), neg)
    A21435 = parke_taylor_mhv(spinors, (2, 1, 4, 3, 5), neg)
    A13245 = parke_taylor_mhv(spinors, (1, 3, 2, 4, 5), neg)
    A31425 = parke_taylor_mhv(spinors, (3, 1, 4, 2, 5), neg)

    return s12 * s34 * A12345 * A21435 + s13 * s24 * A13245 * A31425
