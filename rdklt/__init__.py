"""Shared utilities for RDKLT scripts."""

from .amplitudes import hodges_gravity_mhv_5pt, klt_gravity_5pt_for_validation, parke_taylor_mhv
from .kinematics import angle, generate_exact_spinors, mandelstam, mandelstam_2pt, square
from .linear_algebra import cpqr_analysis, matrix_rank_and_cond, nice_coeff
from .mandelstams import (
    canonical_subset,
    compute_mandelstam,
    generate_mandelstam_labels,
    mandelstam_label,
    pair_mandelstam_labels,
    pair_mandelstam_name,
    standardize_mandelstam_name,
)
from .orderings import (
    canonical_dihedral,
    orderings_fix_first,
    orderings_fix_first_and_last,
    orderings_fix_first_and_second,
    orderings_fix_first_reflection_quotient,
)

__all__ = [
    "angle",
    "square",
    "mandelstam_2pt",
    "mandelstam",
    "generate_exact_spinors",
    "parke_taylor_mhv",
    "hodges_gravity_mhv_5pt",
    "klt_gravity_5pt_for_validation",
    "standardize_mandelstam_name",
    "pair_mandelstam_labels",
    "pair_mandelstam_name",
    "canonical_subset",
    "generate_mandelstam_labels",
    "mandelstam_label",
    "compute_mandelstam",
    "canonical_dihedral",
    "orderings_fix_first",
    "orderings_fix_first_reflection_quotient",
    "orderings_fix_first_and_last",
    "orderings_fix_first_and_second",
    "cpqr_analysis",
    "matrix_rank_and_cond",
    "nice_coeff",
]
