# Amplitudes and Symbolic Regression

Code for data-driven rediscovery of the KLT relations using CPQR feature selection and symbolic regression.

This repository loosely accompanies the paper "Learning the S-matrix from data: Rediscovering gravity from gauge theory via symbolic regression". The scripts here implement the computational pipeline discussed there, in particular the 5-point KLT experiments and Mandelstam rank analysis.

## What is included

- `klt_pysr_5pt.py`  
  5-point pipeline: basis selection (XGBoost), Mandelstam subset selection, dataset construction, PySR search, and fresh test-set validation. Several seeds reproduce the results (with different KLT's discovered for each!).
- `mandelstam_rank.py`  
  CPQR-based rank discovery for a minimum set of Mandelstam invariants.
- `rdklt/`  
  This contains simple amplitude generation stuff used in the main scripts, e.g.
  - `kinematics.py`: exact rational spinor generation in split signature
  - `amplitudes.py`: Parke–Taylor and 5pt Hodges/KLT validation amplitudes
  - `linear_algebra.py`: CPQR and rank/conditioning helpers
  - `mandelstams.py`, `orderings.py`: invariant labeling and ordering combinatorics

## Quick start

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy scikit-learn xgboost pysr
```

Run Mandelstam rank discovery (example):

```bash
python mandelstam_rank.py -n 5 -N 200 --print-relations
```

Run the 5-point KLT symbolic-regression pipeline:

```bash
python klt_pysr_5pt.py
```

## Notes

- `pysr` requires a working Julia installation.
- Main runtime knobs are near the top of `klt_pysr_5pt.py` (sample counts, seeds, operator set, etc.). See comments.
- This code is research-oriented and tuned for reproducibility/experimentation rather than fancy packaging. Please email me if you have any questions: n.moynihan@qmul.ac.uk.
