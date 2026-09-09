# Contributing to EEGProc

## Setup

```bash
git clone https://github.com/VitorInserra/EEGProc.git
cd EEGProc
python -m venv .venv && source .venv/bin/activate
pip install -e ".[deep-learning,dev]"
pytest src/tests
```

For featurization-only work, `pip install -e ".[dev]"` is enough and much faster;
the tests that need TensorFlow skip themselves.

## Ground rules

**The base install must not require TensorFlow.** Only
`deep_learning/cross_validation/`, `deep_learning/training_outputs.py` and
`deep_learning/domain_generalization/` may import it. CI fails the build if
TensorFlow becomes importable in a base environment.

**The cross-validation package is layered.** Every import inside
`cross_validation/` must point strictly downward:

```
L0  constants, arrays, workers
L1  probabilities, splits
L2  aggregation, metrics
L3  callbacks, reporting
L4  search, evaluation
L5  loso_fold, calibration_subject, nested_fold
L6  loso, calibration, nested
```

This replaced a three-way import cycle that had been worked around by deferring
imports into function bodies. `test_cross_validation_layering.py` enforces it,
including a check that no intra-package import is deferred again. Adding a module
means adding it to that test's `LEVELS` map.

**Behaviour is pinned by golden data.** `test_cross_validation.py` compares metric
values against `src/tests/data/loso_cv_golden.json`, and
`test_windowing_equivalence.py` checks the windowing assembler byte-for-byte
against the reference implementation it replaced. Regenerate a golden file only
when you intend to change behaviour, and say so in the changelog — never to make a
red test go green.

**Docstring examples must run.** The v2 cleanup removed documentation for a
function that never existed and a Quick Start that could not execute. If you add an
example, execute it first.

## Pull requests

- Keep refactors move-only where you can; a diff that both moves and edits code is
  hard to review.
- Note anything user-visible in `CHANGELOG.md` under "unreleased".
