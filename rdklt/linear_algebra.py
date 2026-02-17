from __future__ import annotations

import numpy as np
from scipy.linalg import qr


def cpqr_analysis(M, label_names, tol=None):
    """
    Column-pivoted QR decomposition and rank extraction.
    """
    _, R, P = qr(M, pivoting=True)

    diag_R = np.abs(np.diag(R))
    if tol is None:
        lead = diag_R[0] if diag_R.size > 0 and diag_R[0] > 0 else 1.0
        tol = max(M.shape) * np.finfo(float).eps * lead
    rank = int(np.sum(diag_R > tol))

    selected = [label_names[P[i]] for i in range(rank)]
    dependent = [label_names[P[i]] for i in range(rank, len(label_names))]

    return {
        "rank": rank,
        "selected": selected,
        "dependent": dependent,
        "diag_R": diag_R,
        "perm": P,
        "R": R,
        "tol": tol,
    }


def matrix_rank_and_cond(X):
    """
    Return (rank, effective_condition_number) from SVD.
    """
    if X.size == 0:
        return 0, float("inf")
    try:
        s = np.linalg.svd(X, compute_uv=False)
    except np.linalg.LinAlgError:
        return 0, float("inf")
    if s.size == 0 or not np.isfinite(s[0]):
        return 0, float("inf")
    tol = float(s[0]) * max(X.shape) * np.finfo(float).eps
    rank = int(np.sum(s > tol))
    if rank <= 0:
        return 0, float("inf")
    if rank == 1:
        return 1, 1.0
    return rank, float(s[0] / s[rank - 1])


def nice_coeff(c, tol=1e-8):
    """Try to display a float as a small integer or simple fraction."""
    if abs(c - round(c)) < tol:
        v = int(round(c))
        if v == 1:
            return ""
        if v == -1:
            return "-"
        return f"{v}*"
    for d in range(2, 13):
        num = c * d
        if abs(num - round(num)) < tol:
            n = int(round(num))
            return f"({n}/{d})*"
    return f"{c:.6g}*"
