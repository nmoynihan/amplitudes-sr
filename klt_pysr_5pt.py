#!/usr/bin/env python3
"""
5-point KLT from symbolic regression

What this script does:
1) Select LEFT/RIGHT BCJ bases from data, using XGBoost.
2) Select a small Mandelstam subset from the same data.
3) Build a dataset from the discovered bases + Mandelstams.
4) Run PySR and pick the best equation by score.
5) Evaluate that best equation on a fresh test set.
"""

from __future__ import annotations

import csv
import itertools
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from rdklt.amplitudes import hodges_gravity_mhv_5pt, parke_taylor_mhv
from rdklt.kinematics import generate_exact_spinors, mandelstam
from rdklt.linear_algebra import matrix_rank_and_cond
from rdklt.orderings import canonical_dihedral, orderings_fix_first_and_last, orderings_fix_first_and_second


# ============================================================
# Options (edit here)
# ============================================================

# Data generation / discovery
SEED = 1096  # Seeds that converge quickly with the default options: 1434, 1885, 2166, 2329, 1096
INT_RANGE = 4
NSAMPLES = 250 
BASIS_SELECT_NSAMPLES = 80
BASIS_SELECT_N_ESTIMATORS = 120
BASIS_CANDIDATES_PER_GROUP = 3
BASIS_SELECT_TOPK_PAIRS = 10
BASIS_PRINT_REPORT = True
BASIS_REPORT_MAX_ROWS = 35
CANDIDATE_POOL_MODE = "legacy"  # "legacy", "all_fixings", "legacy_plus_missing"
PRIORITIZE_LEGACY_CANDIDATES = True  # Keep old seed behavior stable when extra candidates are added.
STRICT_LEGACY_SELECTION_WHEN_AVAILABLE = True  # In mixed mode, evaluate only legacy candidates so old seeds reproduce exactly.
MAX_MANDELSTAMS = 5
SUBSET_VAL_FRACTION = 0.25 
SUBSET_RIDGE_ALPHA = 1e-8 
S_MEDIAN_MAX = 1e6
NU2 = 1e8 # Energy scale used to rescale features. Adjust as needed, but should be large enough to avoid numerical issues with small Mandelstams.
MAX_ABS_VALUE = 1e12
ENFORCE_GLUON_AMP_MAGNITUDE_WINDOW = True
GLUON_AMP_ABS_MIN = 1.0
GLUON_AMP_ABS_MAX = 1e3
SAVE_CSV_PATH = "klt_discovery_dataset.csv"

# PySR
RUN_PYSR = True # Set to True to run PySR search after basis selection and dataset generation.
PYSR_NITERATIONS = 5000 # Adjust as needed; more iterations may find better equations but take longer to run.
PYSR_POPULATIONS = 16
PYSR_POPULATION_SIZE = 48
PYSR_MAXSIZE = 35
PYSR_BINARY_OPERATORS = ["+", "-", "*"]
PYSR_UNARY_OPERATORS = ["inv"]
PYSR_PARSIMONY = 0.0
PYSR_NESTED_CONSTRAINTS = {"inv": {"inv": 0}}  # Disallow inv(inv(x)).
PYSR_ENFORCE_DIMENSIONAL_CONSTRAINTS = True  # Enforce physical units in symbolic search. Not strictly necessary, but helps guide search away from unit-violating expressions that can still (badly) fit the data.
PYSR_DIMENSIONAL_CONSTRAINT_PENALTY = 1e6  # Penalty for unit-violating expressions.
PYSR_DIMENSIONLESS_CONSTANTS_ONLY = True  # Learned constants are unitless.
PYSR_TARGET_UNIT = "m^2"  # Gravity target has mass-dimension +2.
PYSR_MANDELSTAM_UNIT = "m^2"  # Each Mandelstam feature has mass-dimension +2.
PYSR_BILINEAR_UNIT = "m^-2"  # Each gluon-bilinear feature has mass-dimension -2.

# Fresh test-set evaluation for the selected best equation
TEST_NSAMPLES = 250
TEST_SEED_OFFSET = 1e3 # Offset added to SEED to generate a different test set. Adjust as needed, but make sure it's large enough to avoid overlap with the training/validation set.
TEST_INT_RANGE = 500 # Use a larger integer range for test set to check extrapolation
TEST_NU2 = NU2 # 
TEST_MAX_ABS_VALUE = 1e20 # Use a larger window for test-set evaluation to avoid false negatives due to outliers. Old window was 1e3
TEST_S_MEDIAN_MAX = None # Set to None for no maximum on median Mandelstam in test set, or set to a finite value to exclude test samples with large Mandelstams
TEST_MAX_ATTEMPTS = 1_000_000


# Shared config map used by other local modules.
DEFAULT_OPTIONS = {
    "seed": SEED,
    "int_range": INT_RANGE,
    "nsamples": NSAMPLES,
    "basis_select_nsamples": BASIS_SELECT_NSAMPLES,
    "basis_select_n_estimators": BASIS_SELECT_N_ESTIMATORS,
    "basis_candidates_per_group": BASIS_CANDIDATES_PER_GROUP,
    "basis_select_topk_pairs": BASIS_SELECT_TOPK_PAIRS,
    "max_mandelstams": MAX_MANDELSTAMS,
    "fixed_energy_scale_nu2": NU2,
    "skip_pysr": not RUN_PYSR,
    "pysr_niterations": PYSR_NITERATIONS,
}


def ordering_text(ordering):
    return "".join(str(x) for x in ordering)


def all_pairwise_mandelstams(spinors):
    out = {}
    for i in range(5):
        for j in range(i + 1, 5):
            out[f"s_{i + 1}{j + 1}"] = float(mandelstam(spinors[i], spinors[j]))
    return out


def is_finite_and_bounded(values, max_abs_value):
    if not all(np.isfinite(v) for v in values):
        return False
    return max(abs(v) for v in values) <= max_abs_value


def scale_features(raw_feats, raw_y, nu2):
    out = {}
    for name, value in raw_feats.items():
        if name.startswith("s_"):
            out[name] = value / nu2
        elif name.startswith("B_"):
            out[name] = value * nu2
        else:
            out[name] = value
    return out, raw_y / nu2


def rows_to_matrix(rows, columns):
    return np.asarray([[row[c] for c in columns] for row in rows], dtype=float)


def feature_unit(name):
    if name.startswith("s_"):
        return PYSR_MANDELSTAM_UNIT
    if name.startswith("B_"):
        return PYSR_BILINEAR_UNIT
    return "1"


def format_table_value(value):
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "nan"
        return f"{float(value):.3e}"
    return str(value)


def print_table(title, columns, rows):
    print(f"\n{title}")
    if not rows:
        print("  [empty]")
        return

    str_rows = [[format_table_value(row.get(col, "")) for col in columns] for row in rows]
    widths = [len(col) for col in columns]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_line(cells):
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    print(fmt_line(columns))
    print("-+-".join("-" * width for width in widths))
    for row in str_rows:
        print(fmt_line(row))


def format_orderings(orderings):
    return "[" + ", ".join(ordering_text(ordering) for ordering in orderings) + "]"


def normalize_seed(seed):
    if isinstance(seed, (float, np.floating)):
        if not np.isfinite(seed):
            raise ValueError(f"Seed must be finite; got {seed!r}.")
    try:
        return int(seed)
    except Exception as exc:
        raise ValueError(f"Seed must be int-like; got {seed!r}.") from exc


def make_train_val_split(nrows, val_fraction, seed):
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be in (0, 1).")
    if nrows < 4:
        raise ValueError("Need at least 4 rows for a train/val split.")

    rng = np.random.default_rng(normalize_seed(seed))
    perm = rng.permutation(nrows)
    n_train = int(round((1.0 - float(val_fraction)) * nrows))
    n_train = max(1, min(nrows - 1, n_train))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    return train_idx, val_idx


def linear_val_mse_no_intercept(X, y, train_idx, val_idx):
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
    beta, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
    pred = X_val @ beta
    return float(np.mean((pred - y_val) ** 2))


def linear_val_mse_with_standardize(X, y, train_idx, val_idx):
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_val_std = scaler.transform(X_val)

    A_train = np.column_stack([np.ones(X_train_std.shape[0]), X_train_std])
    A_val = np.column_stack([np.ones(X_val_std.shape[0]), X_val_std])
    beta, *_ = np.linalg.lstsq(A_train, y_train, rcond=None)
    pred = A_val @ beta
    return float(np.mean((pred - y_val) ** 2))


def canonical_pair_key(ordering_pair):
    c1 = canonical_dihedral(ordering_pair[0])
    c2 = canonical_dihedral(ordering_pair[1])
    return tuple(sorted((c1, c2)))


def generate_legacy_bcj_basis_candidates_5pt(first=1):
    candidates = []

    for last in [2, 3, 4, 5]:
        orderings = orderings_fix_first_and_last(n=5, first=first, last=last)
        for idx, (o1, o2) in enumerate(itertools.combinations(orderings, 2)):
            candidates.append(
                {
                    "scheme": "first_last",
                    "fixed_leg": last,
                    "tag": f"last={last}:pair={idx}",
                    "orderings": (o1, o2),
                    "tier": 0,
                }
            )

    for second in [2, 3, 4, 5]:
        orderings = orderings_fix_first_and_second(n=5, first=first, second=second)
        for idx, (o1, o2) in enumerate(itertools.combinations(orderings, 2)):
            candidates.append(
                {
                    "scheme": "first_second",
                    "fixed_leg": second,
                    "tag": f"second={second}:pair={idx}",
                    "orderings": (o1, o2),
                    "tier": 0,
                }
            )

    seen = set()
    out = []
    for c in candidates:
        if canonical_dihedral(c["orderings"][0]) == canonical_dihedral(c["orderings"][1]):
            continue
        key = frozenset(c["orderings"])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def generate_all_fixings_canonical_pairs_5pt():
    n = 5
    legs = list(range(1, n + 1))
    pair_keys = set()

    for first_leg in legs:
        for last_leg in legs:
            if last_leg == first_leg:
                continue
            orderings = orderings_fix_first_and_last(n=n, first=first_leg, last=last_leg)
            for o1, o2 in itertools.combinations(orderings, 2):
                c1 = canonical_dihedral(o1)
                c2 = canonical_dihedral(o2)
                if c1 == c2:
                    continue
                pair_keys.add(tuple(sorted((c1, c2))))

        for second_leg in legs:
            if second_leg == first_leg:
                continue
            orderings = orderings_fix_first_and_second(n=n, first=first_leg, second=second_leg)
            for o1, o2 in itertools.combinations(orderings, 2):
                c1 = canonical_dihedral(o1)
                c2 = canonical_dihedral(o2)
                if c1 == c2:
                    continue
                pair_keys.add(tuple(sorted((c1, c2))))

    return sorted(pair_keys)


def generate_bcj_basis_candidates_5pt(first=1, mode=None):
    mode = str(mode or CANDIDATE_POOL_MODE).strip().lower()

    if mode == "legacy":
        return generate_legacy_bcj_basis_candidates_5pt(first=first)

    all_pairs = generate_all_fixings_canonical_pairs_5pt()
    if mode == "all_fixings":
        return [
            {
                "scheme": "all_fixings",
                "fixed_leg": idx + 1,
                "tag": f"allfix:pair={idx}",
                "orderings": pair,
                "tier": 1,
            }
            for idx, pair in enumerate(all_pairs)
        ]

    if mode != "legacy_plus_missing":
        raise ValueError(f"Unsupported CANDIDATE_POOL_MODE={mode!r}.")

    legacy = generate_legacy_bcj_basis_candidates_5pt(first=first)
    legacy_pair_keys = {canonical_pair_key(c["orderings"]) for c in legacy}

    missing = [pair for pair in all_pairs if pair not in legacy_pair_keys]
    extra = [
        {
            "scheme": "missing_fixings",
            "fixed_leg": idx + 1,
            "tag": f"missing:pair={idx}",
            "orderings": pair,
            "tier": 1,
        }
        for idx, pair in enumerate(missing)
    ]
    return legacy + extra


def auto_select_bcj_bases_xgb(
    candidates_left,
    candidates_right,
    spinors_pool,
    neg=(1, 2),
    seed=0,
    n_estimators=120,
    candidates_per_group=3,
    topk_pairs=10,
    print_report=True,
    report_max_rows=35,
):
    """Same basis-selection strategy as before: linear pre-screen, then XGBoost final ranking."""
    if not spinors_pool:
        raise ValueError("spinors_pool is empty.")

    mode = str(CANDIDATE_POOL_MODE).strip().lower()
    if STRICT_LEGACY_SELECTION_WHEN_AVAILABLE and mode == "legacy_plus_missing":
        legacy_left = [c for c in candidates_left if int(c.get("tier", 0)) == 0]
        legacy_right = [c for c in candidates_right if int(c.get("tier", 0)) == 0]
        if legacy_left and legacy_right:
            candidates_left_eval = legacy_left
            candidates_right_eval = legacy_right
        else:
            candidates_left_eval = list(candidates_left)
            candidates_right_eval = list(candidates_right)
    else:
        candidates_left_eval = list(candidates_left)
        candidates_right_eval = list(candidates_right)

    unique_orderings = sorted({o for c in (candidates_left_eval + candidates_right_eval) for o in c["orderings"]})
    canonical_map = {o: canonical_dihedral(o) for o in unique_orderings}

    amps_per_sample = []
    mu2_per_sample = []
    y_scaled = []

    for sp in spinors_pool:
        try:
            s_vals = all_pairwise_mandelstams(sp)
            mu2 = float(np.median([abs(v) for v in s_vals.values()]))
            if (not np.isfinite(mu2)) or mu2 <= 0 or mu2 > 1e6:
                continue

            y = float(hodges_gravity_mhv_5pt(sp, neg=neg)) / mu2
            amp_map = {o: float(parke_taylor_mhv(sp, o, neg=neg)) for o in unique_orderings}
        except ZeroDivisionError:
            continue

        if not is_finite_and_bounded(list(amp_map.values()) + [y], MAX_ABS_VALUE):
            continue

        amps_per_sample.append(amp_map)
        mu2_per_sample.append(mu2)
        y_scaled.append(y)

    y_arr = np.asarray(y_scaled, dtype=float)
    if y_arr.size < 50:
        raise RuntimeError("Too few valid samples for basis selection.")

    global_pool_schemes = {"all_orderings", "all_fixings"}
    is_all_orderings_pool = (
        all(c.get("scheme") in global_pool_schemes for c in candidates_left_eval)
        and all(c.get("scheme") in global_pool_schemes for c in candidates_right_eval)
    )

    if (candidates_per_group is not None and candidates_per_group > 0) and (not is_all_orderings_pool):
        order_to_col = {o: i for i, o in enumerate(unique_orderings)}
        A = np.empty((y_arr.size, len(unique_orderings)), dtype=float)
        for row_idx, amp_map in enumerate(amps_per_sample):
            A[row_idx] = [amp_map[o] for o in unique_orderings]

        def prune(cands):
            grouped = defaultdict(list)
            for c in cands:
                cols = [order_to_col[c["orderings"][0]], order_to_col[c["orderings"][1]]]
                X = A[:, cols]
                Xs = StandardScaler().fit_transform(X)
                rank, cond = matrix_rank_and_cond(Xs)
                if rank < 2:
                    continue
                grouped[(c["scheme"], c["fixed_leg"])].append((cond, c))

            out = []
            for _, items in grouped.items():
                items.sort(key=lambda t: t[0])
                out.extend([c for _, c in items[:candidates_per_group]])
            return out

        candidates_left_eval = prune(candidates_left_eval)
        candidates_right_eval = prune(candidates_right_eval)
        if not candidates_left_eval or not candidates_right_eval:
            raise RuntimeError("Basis candidate pruning removed all candidates.")

    train_idx, val_idx = make_train_val_split(y_arr.size, SUBSET_VAL_FRACTION, seed)

    def fit_and_score(X):
        X_train = X[train_idx]
        X_val = X[val_idx]
        y_train = y_arr[train_idx]
        y_val = y_arr[val_idx]

        model = xgb.XGBRegressor(
            n_estimators=int(n_estimators),
            learning_rate=0.03,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=int(seed),
            tree_method="hist",
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)
        train_mse = float(np.mean((pred_train - y_train) ** 2))
        val_mse = float(np.mean((pred_val - y_val) ** 2))
        return {"train_mse": train_mse, "val_mse": val_mse}

    screened = []
    for left in candidates_left_eval:
        left_canon = {canonical_map[left["orderings"][0]], canonical_map[left["orderings"][1]]}
        for right in candidates_right_eval:
            right_canon = {canonical_map[right["orderings"][0]], canonical_map[right["orderings"][1]]}
            if left_canon & right_canon:
                continue

            X = np.empty((y_arr.size, 4), dtype=float)
            for row_idx, (amp_map, mu2) in enumerate(zip(amps_per_sample, mu2_per_sample, strict=True)):
                al0 = amp_map[left["orderings"][0]]
                al1 = amp_map[left["orderings"][1]]
                ar0 = amp_map[right["orderings"][0]]
                ar1 = amp_map[right["orderings"][1]]
                X[row_idx, 0] = al0 * ar0 * mu2
                X[row_idx, 1] = al0 * ar1 * mu2
                X[row_idx, 2] = al1 * ar0 * mu2
                X[row_idx, 3] = al1 * ar1 * mu2

            X = np.clip(X, -1e12, 1e12)
            if not np.isfinite(X).all():
                continue

            linear_mse = linear_val_mse_no_intercept(X, y_arr, train_idx, val_idx)
            Xs = StandardScaler().fit_transform(X)
            rank, cond = matrix_rank_and_cond(Xs)
            C = np.corrcoef(Xs, rowvar=False)
            max_corr = float(np.max(np.abs(C - np.eye(C.shape[0])))) if C.size else 1.0

            screened.append(
                {
                    "left": left,
                    "right": right,
                    "X": X,
                    "rank": rank,
                    "cond": cond,
                    "max_corr": max_corr,
                    "linear_mse": linear_mse,
                    "tier_score": int(left.get("tier", 0)) + int(right.get("tier", 0)),
                }
            )

    if not screened:
        raise RuntimeError("No valid non-overlapping basis pair found.")

    screened.sort(
        key=lambda d: (
            int(d["tier_score"]) if PRIORITIZE_LEGACY_CANDIDATES else 0,
            d["linear_mse"],
            d["cond"],
            d["max_corr"],
            -d["rank"],
        )
    )
    finalists = screened[: min(int(topk_pairs), len(screened))] if topk_pairs and topk_pairs > 0 else screened

    best = None
    report_rows = []
    for item in finalists:
        scores = fit_and_score(item["X"])
        pair_key = (item["left"]["orderings"], item["right"]["orderings"])
        report_rows.append(
            {
                "decision": "",
                "left_orderings": format_orderings(item["left"]["orderings"]),
                "right_orderings": format_orderings(item["right"]["orderings"]),
                "linear_val_mse": item["linear_mse"],
                "train_mse": scores["train_mse"],
                "val_mse": scores["val_mse"],
                "rank": int(item["rank"]),
                "cond": item["cond"],
                "max_abs_corr": item["max_corr"],
                "tier_score": int(item["tier_score"]),
                "_pair_key": pair_key,
            }
        )

        key = (
            int(item["tier_score"]) if PRIORITIZE_LEGACY_CANDIDATES else 0,
            scores["val_mse"],
            item["cond"],
            item["max_corr"],
            -item["rank"],
        )
        if best is None or key < best["key"]:
            best = {
                "key": key,
                "left": item["left"],
                "right": item["right"],
                "val_mse": scores["val_mse"],
                "train_mse": scores["train_mse"],
                "rank": item["rank"],
                "cond": item["cond"],
                "max_corr": item["max_corr"],
                "tier_score": item["tier_score"],
                "_pair_key": pair_key,
            }

    if best is None:
        raise RuntimeError("No basis pair survived final scoring.")

    if print_report:
        for row in report_rows:
            if row["_pair_key"] == best["_pair_key"]:
                row["decision"] = "SELECT"

        report_rows.sort(
            key=lambda row: (
                0 if row["decision"] == "SELECT" else 1,
                float(row["val_mse"]),
                float(row["cond"]),
                float(row["max_abs_corr"]),
            )
        )
        if report_max_rows is not None and int(report_max_rows) > 0:
            report_rows = report_rows[: min(int(report_max_rows), len(report_rows))]

        printable_rows = [
            {
                "decision": row["decision"],
                "left_orderings": row["left_orderings"],
                "right_orderings": row["right_orderings"],
                "tier": int(row.get("tier_score", 0)),
                "linear_val_mse": row["linear_val_mse"],
                "train_mse": row["train_mse"],
                "val_mse": row["val_mse"],
                "rank": row["rank"],
                "cond": row["cond"],
                "max_abs_corr": row["max_abs_corr"],
            }
            for row in report_rows
        ]
        print_table(
            "XGBoost basis-pair search report",
            [
                "decision",
                "left_orderings",
                "right_orderings",
                "tier",
                "linear_val_mse",
                "train_mse",
                "val_mse",
                "rank",
                "cond",
                "max_abs_corr",
            ],
            printable_rows,
        )

    return best["left"], best["right"], best


def build_dataset(
    nsamples,
    left_basis,
    right_order_pool,
    neg=(1, 2),
    seed=0,
    int_range=4,
    nu2=1e8,
    max_abs_value=1e12,
    s_median_max=1e6,
    enforce_gluon_amp_magnitude_window=False,
    gluon_amp_abs_min=1.0,
    gluon_amp_abs_max=1e4,
    max_attempts=None,
):
    seed_i = normalize_seed(seed)
    random.seed(seed_i)
    np.random.seed(seed_i)

    unique_orderings = sorted(set(left_basis + right_order_pool))

    rows = []
    y_vals = []
    attempts = 0
    while len(rows) < nsamples:
        attempts += 1
        if (max_attempts is not None) and attempts > int(max_attempts):
            raise RuntimeError(
                f"Could not build dataset: accepted {len(rows)}/{nsamples} "
                f"after {int(max_attempts)} attempts. "
                f"Try relaxing test constraints (e.g. TEST_INT_RANGE, TEST_MAX_ABS_VALUE, TEST_S_MEDIAN_MAX)."
            )

        sp = generate_exact_spinors(n=5, int_range=int_range)

        if enforce_gluon_amp_magnitude_window:
            ok = True
            for ordering in unique_orderings:
                try:
                    amp = float(parke_taylor_mhv(sp, ordering, neg=neg))
                except ZeroDivisionError:
                    ok = False
                    break
                amp_abs = abs(amp)
                if (not np.isfinite(amp_abs)) or (amp_abs < gluon_amp_abs_min) or (amp_abs > gluon_amp_abs_max):
                    ok = False
                    break
            if not ok:
                continue

        try:
            y_raw = float(hodges_gravity_mhv_5pt(sp, neg=neg))
            s_feats = all_pairwise_mandelstams(sp)

            al = [float(parke_taylor_mhv(sp, o, neg=neg)) for o in left_basis]
            ar = [float(parke_taylor_mhv(sp, o, neg=neg)) for o in right_order_pool]
        except ZeroDivisionError:
            continue

        raw_feats = dict(s_feats)
        for i, o_left in enumerate(left_basis):
            for j, o_right in enumerate(right_order_pool):
                name = f"B_L{ordering_text(o_left)}_R{ordering_text(o_right)}"
                raw_feats[name] = al[i] * ar[j]

        s_med = float(np.median([abs(raw_feats[k]) for k in raw_feats if k.startswith("s_")]))
        if (not np.isfinite(s_med)) or s_med == 0.0:
            continue
        if (s_median_max is not None) and s_med > float(s_median_max):
            continue

        if (not np.isfinite(y_raw)) or (not all(np.isfinite(v) for v in raw_feats.values())):
            continue

        if max(abs(v) for v in raw_feats.values()) > max_abs_value:
            continue

        feats_scaled, y_scaled = scale_features(raw_feats, y_raw, nu2)
        if (not np.isfinite(y_scaled)) or (not all(np.isfinite(v) for v in feats_scaled.values())):
            continue

        rows.append(feats_scaled)
        y_vals.append(y_scaled)

    return rows, np.asarray(y_vals, dtype=float)


def subset_design_matrix(rows, s_subset, bilinear_cols):
    X_s = rows_to_matrix(rows, s_subset)
    X_b = rows_to_matrix(rows, bilinear_cols)
    pairs = list(itertools.combinations(range(len(s_subset)), 2))
    Q = np.empty((X_s.shape[0], len(pairs)), dtype=float)
    for idx, (i, j) in enumerate(pairs):
        Q[:, idx] = X_s[:, i] * X_s[:, j]
    return (X_b[:, :, None] * Q[:, None, :]).reshape(X_s.shape[0], X_b.shape[1] * len(pairs))


def select_mandelstam_subset(rows, y, s_cols, bilinear_cols, k, seed, val_fraction):
    if k <= 0:
        raise ValueError("k must be positive.")
    if k > len(s_cols):
        raise ValueError(f"Requested k={k} Mandelstams, but only {len(s_cols)} available.")

    X_b = rows_to_matrix(rows, bilinear_cols)
    X_s_all = rows_to_matrix(rows, s_cols)
    C_all = np.corrcoef(StandardScaler().fit_transform(X_s_all), rowvar=False)
    s_col_to_idx = {name: idx for idx, name in enumerate(s_cols)}

    idx_all = np.arange(len(rows))
    train_idx, val_idx = train_test_split(
        idx_all,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
    )

    y_train = y[train_idx]
    y_val = y[val_idx]
    best_subset = None
    best_key = None
    best_mse = None
    for subset in itertools.combinations(s_cols, k):
        subset = list(subset)
        subset_idx = [s_col_to_idx[name] for name in subset]
        X_s_sub = X_s_all[:, subset_idx]

        pairs = list(itertools.combinations(range(len(subset)), 2))
        Q = np.empty((X_s_sub.shape[0], len(pairs)), dtype=float)
        for t, (i, j) in enumerate(pairs):
            Q[:, t] = X_s_sub[:, i] * X_s_sub[:, j]

        Z = (X_b[:, :, None] * Q[:, None, :]).reshape(X_b.shape[0], X_b.shape[1] * len(pairs))

        scaler = StandardScaler()
        Z_train = scaler.fit_transform(Z[train_idx])
        Z_val = scaler.transform(Z[val_idx])

        model = Ridge(alpha=float(SUBSET_RIDGE_ALPHA), fit_intercept=True, random_state=seed)
        model.fit(Z_train, y_train)
        pred_val = model.predict(Z_val)
        mse = float(np.mean((pred_val - y_val) ** 2))

        Z_std = StandardScaler().fit_transform(Z)
        _, cond_z = matrix_rank_and_cond(Z_std)
        max_abs_corr = (
            float(np.max(np.abs(C_all[np.ix_(subset_idx, subset_idx)] - np.eye(len(subset_idx)))))
            if len(subset_idx) > 1
            else 0.0
        )

        key = (mse, float(cond_z), max_abs_corr, ",".join(subset))
        if best_key is None or key < best_key:
            best_key = key
            best_subset = subset
            best_mse = mse

    if best_subset is None:
        raise RuntimeError("Failed to select Mandelstam subset.")
    return best_subset, float(best_mse)


def save_dataset_csv(path, rows, y, columns):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([*columns, "y"])
        for i, row in enumerate(rows):
            writer.writerow([*(row[c] for c in columns), float(y[i])])


def run_pysr(
    X,
    y,
    feature_names,
    seed=0,
    niterations=2000,
    enforce_dimensional_constraints=False,
    dimensional_constraint_penalty=None,
    dimensionless_constants_only=None,
    X_units=None,
    y_units=None,
    **_kwargs,
):
    if "JULIA_DEPOT_PATH" not in os.environ:
        os.environ["JULIA_DEPOT_PATH"] = str(Path(__file__).with_name(".julia_depot"))
    if "JULIA_EXE" not in os.environ:
        julia_exe = shutil.which("julia")
        if julia_exe:
            os.environ["JULIA_EXE"] = julia_exe

    try:
        from pysr import PySRRegressor
    except Exception as e:
        print("\n[PySR unavailable]")
        print("Install with: pip install pysr")
        print(f"Import error: {e}")
        return None

    model_kwargs = dict(
        niterations=int(niterations),
        populations=int(PYSR_POPULATIONS),
        population_size=int(PYSR_POPULATION_SIZE),
        maxsize=int(PYSR_MAXSIZE),
        binary_operators=list(PYSR_BINARY_OPERATORS),
        unary_operators=list(PYSR_UNARY_OPERATORS),
        parsimony=float(PYSR_PARSIMONY),
        nested_constraints=dict(PYSR_NESTED_CONSTRAINTS),
        random_state=int(seed),
        parallelism="multithreading",
        turbo=True,
        batching=True,
        batch_size=256,
    )
    if enforce_dimensional_constraints:
        if dimensional_constraint_penalty is None:
            dimensional_constraint_penalty = 1000.0
        model_kwargs["dimensional_constraint_penalty"] = float(dimensional_constraint_penalty)
        if dimensionless_constants_only is not None:
            model_kwargs["dimensionless_constants_only"] = bool(dimensionless_constants_only)

    model = PySRRegressor(**model_kwargs)

    fit_kwargs = {"variable_names": list(feature_names)}
    if X_units is not None:
        fit_kwargs["X_units"] = list(X_units)
    if y_units is not None:
        fit_kwargs["y_units"] = str(y_units)

    model.fit(X, y, **fit_kwargs)
    return model


def select_best_equation_row(model):
    eq_df = getattr(model, "equations_", None)
    if eq_df is None or len(eq_df) == 0:
        raise RuntimeError("PySR returned no equations.")

    if "score" in eq_df.columns:
        scores = np.asarray(eq_df["score"], dtype=float)
        finite = np.isfinite(scores)
        if np.any(finite):
            idx = int(np.argmax(np.where(finite, scores, -np.inf)))
            return idx, eq_df.iloc[idx], "max(score)"

    if "loss" in eq_df.columns:
        losses = np.asarray(eq_df["loss"], dtype=float)
        finite = np.isfinite(losses)
        if np.any(finite):
            idx = int(np.argmin(np.where(finite, losses, np.inf)))
            return idx, eq_df.iloc[idx], "min(loss)"

    idx = len(eq_df) - 1
    return idx, eq_df.iloc[idx], "last_row"


def simplify_equation(eq_row):
    expr = eq_row.get("sympy_format", None)
    if expr is None or str(expr) == "nan":
        expr = eq_row.get("equation", None)
    if expr is None:
        return None, "No expression text found."

    try:
        import sympy as sp
    except Exception as e:
        return None, f"SymPy import failed: {e}"

    try:
        simplified = sp.simplify(sp.sympify(expr))
    except Exception as e:
        return None, f"SymPy simplify failed: {e}"
    return simplified, None


def predict_equation(model, X, eq_idx, eq_row, feature_names):
    try:
        y_hat = np.asarray(model.predict(X, index=int(eq_idx)), dtype=float).reshape(-1)
        if y_hat.shape[0] == X.shape[0]:
            return y_hat, "model.predict(index=...)"
    except Exception:
        pass

    lam = eq_row.get("lambda_format", None)
    if callable(lam):
        try:
            cols = [X[:, i] for i in range(X.shape[1])]
            y_hat = np.asarray(lam(*cols), dtype=float).reshape(-1)
            if y_hat.shape[0] == X.shape[0]:
                return y_hat, "lambda_format"
        except Exception:
            pass

    try:
        y_hat = np.asarray(model.predict(X), dtype=float).reshape(-1)
        if y_hat.shape[0] == X.shape[0]:
            return y_hat, "model.predict(default)"
    except Exception:
        pass

    raise RuntimeError("Could not evaluate selected equation on test set.")


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch in regression_metrics.")

    err = y_pred - y_true
    mse = float(np.mean(err * err))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))

    denom = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    r2 = float("nan") if denom == 0 else float(1.0 - np.sum(err * err) / denom)

    rel_denom = np.maximum(np.abs(y_true), 1e-12)
    rel_abs = np.abs(err) / rel_denom
    mean_rel = float(np.mean(rel_abs))
    median_rel = float(np.median(rel_abs))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mean_relative_abs_error": mean_rel,
        "median_relative_abs_error": median_rel,
    }


def main():
    neg = (1, 2)
    seed_i = normalize_seed(SEED)
    random.seed(seed_i)
    np.random.seed(seed_i)

    candidates_left = generate_bcj_basis_candidates_5pt(first=1)
    candidates_right = generate_bcj_basis_candidates_5pt(first=1)
    print(f"BCJ candidate count: LEFT={len(candidates_left)}, RIGHT={len(candidates_right)}")

    print(f"Sampling {BASIS_SELECT_NSAMPLES} points for basis selection...")
    spinors_pool = [generate_exact_spinors(n=5, int_range=INT_RANGE) for _ in range(BASIS_SELECT_NSAMPLES)]

    best_left, best_right, basis_info = auto_select_bcj_bases_xgb(
        candidates_left=candidates_left,
        candidates_right=candidates_right,
        spinors_pool=spinors_pool,
        neg=neg,
        seed=seed_i,
        n_estimators=BASIS_SELECT_N_ESTIMATORS,
        candidates_per_group=BASIS_CANDIDATES_PER_GROUP,
        topk_pairs=BASIS_SELECT_TOPK_PAIRS,
        print_report=BASIS_PRINT_REPORT,
        report_max_rows=BASIS_REPORT_MAX_ROWS,
    )

    print("\nSelected basis pair:")
    print(f"  LEFT : {best_left['scheme']} fixed={best_left['fixed_leg']} tag={best_left['tag']} orderings={best_left['orderings']}")
    print(f"  RIGHT: {best_right['scheme']} fixed={best_right['fixed_leg']} tag={best_right['tag']} orderings={best_right['orderings']}")
    print(f"  basis val_mse={basis_info['val_mse']:.3e}, rank={basis_info['rank']}/4, cond={basis_info['cond']:.3e}")

    print(f"\nGenerating final dataset ({NSAMPLES} samples)...")
    rows, y = build_dataset(
        nsamples=NSAMPLES,
        left_basis=best_left["orderings"],
        right_order_pool=best_right["orderings"],
        neg=neg,
        seed=seed_i,
        int_range=INT_RANGE,
        nu2=NU2,
        max_abs_value=MAX_ABS_VALUE,
        s_median_max=S_MEDIAN_MAX,
        enforce_gluon_amp_magnitude_window=ENFORCE_GLUON_AMP_MAGNITUDE_WINDOW,
        gluon_amp_abs_min=GLUON_AMP_ABS_MIN,
        gluon_amp_abs_max=GLUON_AMP_ABS_MAX,
    )

    s_cols = sorted([k for k in rows[0].keys() if k.startswith("s_")], key=lambda n: (int(n[2]), int(n[3])))
    bilinear_cols = sorted([k for k in rows[0].keys() if k.startswith("B_")])

    k_eff = min(MAX_MANDELSTAMS, len(s_cols))
    selected_s, subset_mse = select_mandelstam_subset(
        rows=rows,
        y=y,
        s_cols=s_cols,
        bilinear_cols=bilinear_cols,
        k=k_eff,
        seed=seed_i,
        val_fraction=SUBSET_VAL_FRACTION,
    )

    final_cols = selected_s + bilinear_cols
    X_final = rows_to_matrix(rows, final_cols)
    X_final_std = StandardScaler().fit_transform(X_final)
    rank, cond = matrix_rank_and_cond(X_final_std)

    print("\nSelected Mandelstam subset:")
    print(f"  {selected_s}")
    print(f"  subset val_mse={subset_mse:.3e}")

    print("\nFinal feature set:")
    print(f"  columns={final_cols}")
    print(f"  matrix shape={X_final.shape[0]} x {X_final.shape[1]}")
    print(f"  rank={rank}/{X_final.shape[1]}, cond={cond:.3e}")
    print(f"  median(|y|)={float(np.median(np.abs(y))):.3e}, rms(y)={float(np.sqrt(np.mean(y * y))):.3e}")

    if SAVE_CSV_PATH:
        save_dataset_csv(SAVE_CSV_PATH, rows, y, final_cols)
        print(f"\nSaved dataset: {SAVE_CSV_PATH}")

    if not RUN_PYSR:
        print("\nRUN_PYSR is False; stopping before symbolic regression.")
        return

    X_pysr = X_final
    y_pysr = y
    if PYSR_ENFORCE_DIMENSIONAL_CONSTRAINTS:
        x_units = [feature_unit(name) for name in final_cols]
        y_units = PYSR_TARGET_UNIT
    else:
        x_units = None
        y_units = None

    print("\nRunning PySR...")
    model = run_pysr(
        X=X_pysr,
        y=y_pysr,
        feature_names=final_cols,
        seed=seed_i,
        niterations=PYSR_NITERATIONS,
        enforce_dimensional_constraints=PYSR_ENFORCE_DIMENSIONAL_CONSTRAINTS,
        dimensional_constraint_penalty=PYSR_DIMENSIONAL_CONSTRAINT_PENALTY,
        dimensionless_constants_only=PYSR_DIMENSIONLESS_CONSTANTS_ONLY,
        X_units=x_units,
        y_units=y_units,
    )
    if model is None:
        return

    print("\nPySR hall of fame:")
    print(model.equations_)

    best_idx, best_row, criterion = select_best_equation_row(model)
    print(f"\nBest equation criterion: {criterion}")
    if "score" in model.equations_.columns:
        print(f"  score={float(best_row.get('score', np.nan)):.6e}")
    if "loss" in model.equations_.columns:
        print(f"  train_loss={float(best_row.get('loss', np.nan)):.6e}")
    if "complexity" in model.equations_.columns:
        print(f"  complexity={best_row.get('complexity')}")
    print(f"  equation={best_row.get('equation')}")

    simplified, err = simplify_equation(best_row)
    if err is None:
        print(f"  simplified={simplified}")
    else:
        print(f"  simplified unavailable: {err}")

    print("\nGenerating fresh test set for selected best equation...")
    test_seed = normalize_seed(seed_i + TEST_SEED_OFFSET)
    rows_test, y_test = build_dataset(
        nsamples=TEST_NSAMPLES,
        left_basis=best_left["orderings"],
        right_order_pool=best_right["orderings"],
        neg=neg,
        seed=test_seed,
        int_range=TEST_INT_RANGE,
        nu2=TEST_NU2,
        max_abs_value=TEST_MAX_ABS_VALUE,
        s_median_max=TEST_S_MEDIAN_MAX,
        max_attempts=TEST_MAX_ATTEMPTS,
    )
    X_test = rows_to_matrix(rows_test, final_cols)
    y_pred, _backend = predict_equation(model, X_test, best_idx, best_row, final_cols)
    test_stats = regression_metrics(y_test, y_pred)
    metrics_rows = [
        {"metric": "mse", "value": f"{test_stats['mse']:.6e}"},
        {"metric": "rmse", "value": f"{test_stats['rmse']:.6e}"},
        {"metric": "mae", "value": f"{test_stats['mae']:.6e}"},
        {"metric": "r2", "value": f"{test_stats['r2']:.6e}"},
        {"metric": "mean_relative_abs_error", "value": f"{test_stats['mean_relative_abs_error']:.6e}"},
        {"metric": "median_relative_abs_error", "value": f"{test_stats['median_relative_abs_error']:.6e}"},
    ]
    print_table("Best-expression fresh-test metrics", ["metric", "value"], metrics_rows)


if __name__ == "__main__":
    main()
