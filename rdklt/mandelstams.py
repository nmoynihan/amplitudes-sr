from __future__ import annotations

import itertools
from fractions import Fraction

from .kinematics import mandelstam_2pt


def standardize_mandelstam_name(name):
    """
    Return canonical column name s_ij (i<j) for labels like s_12, s12, 12, s_21.
    """
    digits = [c for c in str(name) if c.isdigit()]
    if len(digits) != 2:
        raise ValueError(f"Bad Mandelstam label {name!r}; expected something like s_12.")
    i = int(digits[0])
    j = int(digits[1])
    if i == j:
        raise ValueError(f"Bad Mandelstam label {name!r}; i==j.")
    if i > j:
        i, j = j, i
    return f"s_{i}{j}"


def pair_mandelstam_labels(n):
    """All pairwise Mandelstam labels (i<j)."""
    return [(i, j) for i in range(1, n + 1) for j in range(i + 1, n + 1)]


def pair_mandelstam_name(label):
    i, j = label
    return f"s_{i}{j}"


def canonical_subset(subset, n):
    """
    Canonical representative for s_I = s_complement(I).
    Returns the lexicographically smaller of subset and complement.
    """
    comp = tuple(i for i in range(1, n + 1) if i not in subset)
    if len(subset) < len(comp):
        return subset
    if len(subset) > len(comp):
        return comp
    return min(subset, comp)


def generate_mandelstam_labels(n, max_subset_size=2):
    """
    Generate all distinct Mandelstam labels s_I for subsets 2 <= |I| <= max_subset_size,
    identifying s_I with s_complement(I).
    """
    labels = []
    seen = set()
    upper = min(max_subset_size, n - 2)

    for k in range(2, upper + 1):
        for subset in itertools.combinations(range(1, n + 1), k):
            canon = canonical_subset(subset, n)
            if canon not in seen:
                seen.add(canon)
                labels.append(canon)
    return labels


def mandelstam_label(subset):
    """Human-readable label, e.g. s_{1,2} or s_{1,2,3}."""
    return "s_{" + ",".join(str(i) for i in subset) + "}"


def compute_mandelstam(spinors, subset):
    """
    s_I = (sum_{i in I} p_i)^2 = sum_{i<j in I} s_{ij} for massless external legs.
    """
    val = Fraction(0)
    members = list(subset)
    for a in range(len(members)):
        for b in range(a + 1, len(members)):
            val += mandelstam_2pt(spinors[members[a] - 1], spinors[members[b] - 1])
    return val
