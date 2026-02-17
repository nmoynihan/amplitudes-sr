from __future__ import annotations

import itertools


def canonical_dihedral(ordering):
    """
    Canonical representative under cyclic rotations plus reversal.
    For color-ordered amplitudes, cyclic rotations are equivalent.
    """
    n = len(ordering)
    rots = [ordering[i:] + ordering[:i] for i in range(n)]
    rev = tuple(reversed(ordering))
    rots_rev = [rev[i:] + rev[:i] for i in range(n)]
    return min(rots + rots_rev)


def orderings_fix_first(n, first=1):
    """All orderings with first leg fixed (cyclic gauge-fixing)."""
    legs = [x for x in range(1, n + 1) if x != first]
    return [(first,) + p for p in itertools.permutations(legs)]


def orderings_fix_first_reflection_quotient(n, first=1):
    """
    One representative per reflection pair with first leg fixed:
      (first, a2, ..., an) ~ (first, an, ..., a2).
    """
    legs = [x for x in range(1, n + 1) if x != first]
    out = []
    for p in itertools.permutations(legs):
        q = (first,) + p
        r = (first,) + tuple(reversed(p))
        if q <= r:
            out.append(q)
    return out


def orderings_fix_first_and_last(n=5, first=1, last=5):
    """Generate orderings with first and last legs fixed."""
    legs = [x for x in range(1, n + 1) if x not in (first, last)]
    return [(first,) + p + (last,) for p in itertools.permutations(legs)]


def orderings_fix_first_and_second(n=5, first=1, second=2):
    """Generate orderings with first two legs fixed."""
    legs = [x for x in range(1, n + 1) if x not in (first, second)]
    return [(first, second) + p for p in itertools.permutations(legs)]
