#!/usr/bin/env python3
"""
Mandelstam rank finder
======================

Discovers the rank (number of independent Mandelstam variables) for
n-point massless kinematics in (2,2)-signature spacetime, using CPQR.

For each of N random phase-space points (exact rational spinors),
the script evaluates every distinct Mandelstam invariant s_I, assembles
the N x (# Mandelstams) matrix, and runs Column-Pivoted QR to find:
  (a) the numerical rank, and
  (b) a minimal independent set of Mandelstams.

The (2,2)-signature spinor construction follows the same approach as
klt_pysr_simple5pt.py: rational Fraction arithmetic guarantees exact
momentum conservation and on-shell conditions.

Usage examples:
  python mandelstam_rank.py -n 5 -N 200
  python mandelstam_rank.py -n 6 -N 500 --include-multiparticle
  python mandelstam_rank.py -n 8 -N 1000 --print-relations
"""

from __future__ import annotations

import argparse
import random
from fractions import Fraction

import numpy as np
from scipy.linalg import solve_triangular

from rdklt.kinematics import generate_exact_spinors, mandelstam_2pt
from rdklt.linear_algebra import cpqr_analysis, nice_coeff
from rdklt.mandelstams import compute_mandelstam, generate_mandelstam_labels, mandelstam_label


def print_relations(cpqr_result, label_names):
    """
    Print how each dependent Mandelstam is expressed as a linear
    combination of the independent ones (from the CPQR factorisation).
    """
    rank = cpqr_result["rank"]
    R = cpqr_result["R"]
    P = cpqr_result["perm"]
    n_vars = len(label_names)

    if rank >= n_vars:
        print("  All variables are independent — no relations.")
        return

    R11 = R[:rank, :rank]
    R12 = R[:rank, rank:]

    # Solve R11 @ C = R12  =>  M_dep ≈ M_indep @ C
    coeffs = solve_triangular(R11, R12)

    indep_names = [label_names[P[i]] for i in range(rank)]
    dep_names = [label_names[P[i]] for i in range(rank, n_vars)]

    for j, dep_name in enumerate(dep_names):
        terms = []
        for i, c in enumerate(coeffs[:, j]):
            if abs(c) < 1e-10:
                continue
            # Try to detect simple rational coefficients
            label = indep_names[i]
            c_nice = nice_coeff(c)
            terms.append(f"{c_nice} {label}")
        rhs = " + ".join(terms).replace("+ -", "- ") if terms else "0"
        print(f"  {dep_name}  =  {rhs}")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Discover the rank of the Mandelstam matrix via CPQR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s -n 5 -N 200
  %(prog)s -n 6 -N 500 --include-multiparticle
  %(prog)s -n 7 -N 1000 --print-relations
""",
    )
    ap.add_argument("-n", "--nparticles", type=int, default=5,
                    help="Number of massless particles (default: 5)")
    ap.add_argument("-N", "--nsamples", type=int, default=200,
                    help="Number of phase-space points / rows (default: 200)")
    ap.add_argument("--int-range", type=int, default=4,
                    help="Integer range for random spinor components (default: 4)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")
    ap.add_argument("--include-multiparticle", action="store_true",
                    help="Include multi-particle Mandelstams s_{ijk}, s_{ijkl}, ...")
    ap.add_argument("--max-subset-size", type=int, default=None,
                    help="Max subset size for Mandelstams (default: 2, or n-2 with --include-multiparticle)")
    ap.add_argument("--tol", type=float, default=None,
                    help="Tolerance for CPQR rank determination (default: auto)")
    ap.add_argument("--print-relations", action="store_true",
                    help="Print null-space relations (dependent = linear combo of independent)")
    ap.add_argument("--verify", action="store_true",
                    help="Verify momentum conservation on first few samples")
    ap.add_argument("--max-abs-value", type=float, default=1e10,
                    help="Reject samples where any |s_I| exceeds this (default: 1e10)")
    args = ap.parse_args()

    n = args.nparticles
    N = args.nsamples

    if n < 3:
        print("Need at least 3 particles.")
        return

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ---- Determine which Mandelstams to compute ----
    if args.include_multiparticle:
        max_k = args.max_subset_size if args.max_subset_size else n - 2
    else:
        max_k = args.max_subset_size if args.max_subset_size else 2
    max_k = min(max_k, n - 2)

    labels = generate_mandelstam_labels(n, max_subset_size=max_k)
    label_names = [mandelstam_label(s) for s in labels]
    n_mandel = len(labels)

    # For 2-particle Mandelstams, the number of linear constraints from
    # momentum conservation + masslessness is exactly n (for n >= 3),
    # giving rank = C(n,2) - n = n(n-3)/2.
    # NOTE: Gram constraints (det G = 0 etc.) are nonlinear in s_ij and
    #       do NOT reduce the LINEAR rank of the data matrix.
    expected_rank = n * (n - 3) // 2

    print("=" * 65)
    print("  Mandelstam Rank Finder  (CPQR on exact (2,2)-signature data)")
    print("=" * 65)
    print(f"  Particles:            n = {n}")
    print(f"  Phase-space points:   N = {N}")
    print(f"  Max subset size:      {max_k}")
    print(f"  Mandelstam variables: {n_mandel}")
    if max_k == 2:
        print(f"  Expected rank:        {expected_rank}  [= n(n-3)/2]")
    print()

    # List variables
    if n_mandel <= 60:
        # Print in rows of ~8
        chunk = 8
        for i in range(0, n_mandel, chunk):
            batch = label_names[i:i + chunk]
            print(f"  {', '.join(batch)}")
        print()

    # ---- Generate data matrix ----
    print(f"Generating {N} phase-space points...")
    M = np.empty((N, n_mandel), dtype=float)
    n_generated = 0
    n_attempts = 0

    while n_generated < N:
        n_attempts += 1
        try:
            spinors = generate_exact_spinors(n=n, int_range=args.int_range)
        except RuntimeError:
            print(f"  WARNING: spinor generation failed after many attempts at sample {n_generated}.")
            print(f"  Try increasing --int-range (currently {args.int_range}).")
            return

        # Verify momentum conservation on first few samples
        if args.verify and n_generated < 3:
            for i in range(1, n + 1):
                total = Fraction(0)
                for j in range(1, n + 1):
                    if j == i:
                        continue
                    total += mandelstam_2pt(spinors[i - 1], spinors[j - 1])
                assert total == 0, f"Momentum conservation violated for particle {i}"
            if n_generated == 0:
                print("  Momentum conservation verified (exact) on first sample.")

        # Evaluate all Mandelstams
        row = []
        bad = False
        for subset in labels:
            val = float(compute_mandelstam(spinors, subset))
            if not np.isfinite(val):
                bad = True
                break
            row.append(val)
        if bad:
            continue

        if max(abs(v) for v in row) > args.max_abs_value:
            continue

        M[n_generated] = row
        n_generated += 1

        if n_generated % max(1, N // 5) == 0:
            print(f"  {n_generated:6d} / {N} points generated  ({n_attempts} attempts)")

    print(f"  Done: {N} points from {n_attempts} attempts "
          f"(acceptance rate {N / n_attempts:.1%})\n")

    # ---- CPQR analysis ----
    print("Running CPQR...")
    result = cpqr_analysis(M, label_names, tol=args.tol)
    rank = result["rank"]
    diag_R = result["diag_R"]
    P = result["perm"]

    print()
    print("=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(f"  Matrix shape:       {M.shape[0]} × {M.shape[1]}")
    print(f"  Numerical rank:     {rank}  (of {n_mandel} columns)")
    print(f"  Null-space dim:     {n_mandel - rank}")
    if max_k == 2:
        print(f"  Expected rank:      {expected_rank}  [= n(n-3)/2]")
    print()

    # CPQR pivots
    n_show = min(n_mandel, rank + 5)
    print("  CPQR pivot diagonal |R_{ii}|:")
    for i in range(n_show):
        name = label_names[P[i]]
        marker = ""
        if i == rank:
            marker = "  <-- rank cutoff"
        elif i == rank - 1 and rank < n_mandel:
            gap = diag_R[rank - 1] / diag_R[rank] if diag_R[rank] > 0 else float("inf")
            marker = f"  (gap to next: {gap:.2e})"
        print(f"    {i + 1:4d}. {name:16s}  |R| = {diag_R[i]:.6e}{marker}")
    if n_show < n_mandel:
        print(f"    ... ({n_mandel - n_show} more below cutoff)")
    print()

    # Independent set
    print(f"  Independent set ({rank} variables):")
    for i, name in enumerate(result["selected"]):
        print(f"    {i + 1:3d}. {name}")
    print()

    # Dependent set
    if result["dependent"]:
        print(f"  Dependent variables ({len(result['dependent'])}):")
        for i, name in enumerate(result["dependent"]):
            print(f"    {i + 1:3d}. {name}")
        print()

    # Relations
    if args.print_relations and rank < n_mandel:
        print("  Null-space relations:")
        print("  (each dependent variable as a linear combination of independent ones)")
        print()
        print_relations(result, label_names)
        print()

    # ---- SVD cross-check ----
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    svd_tol = max(M.shape) * np.finfo(float).eps * (S[0] if S[0] > 0 else 1.0)
    svd_rank = int(np.sum(S > svd_tol))

    print("  SVD cross-check:")
    print(f"    SVD rank:  {svd_rank}")
    n_show_sv = min(len(S), svd_rank + 3)
    for i in range(n_show_sv):
        marker = "  <-- cutoff" if i == svd_rank else ""
        print(f"      σ_{i + 1:d} = {S[i]:.6e}{marker}")
    if n_show_sv < len(S):
        print(f"      ... ({len(S) - n_show_sv} more below cutoff)")

    if svd_rank > 0:
        cond = S[0] / S[svd_rank - 1]
        print(f"    Condition number (independent block): {cond:.6e}")

    if svd_rank != rank:
        print(f"\n    NOTE: SVD rank ({svd_rank}) != CPQR rank ({rank}). "
              f"Consider adjusting --tol.")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
