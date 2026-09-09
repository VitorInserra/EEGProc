# Changelog

All notable changes to EEGProc are documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] — unreleased

v2 turns EEGProc from a research tree into a library: a smaller, documented public
surface, an install that does not drag in TensorFlow, and CI that actually runs.

### Added

- **`eegproc.data`** — a tidy-frame data layer, importable without TensorFlow.
  `EEGFrame` describes a table of signal or feature columns alongside `subject`,
  `trial` and label columns; `to_supervised_arrays` windows it into the rank-3
  channels-last tensor the cross-validators consume, returning features, labels,
  subject ids **and trial ids** together. Sessions need no new machinery:
  `trial_columns=("session", "trial")` scopes trials per session and
  `subject_columns=("subject", "session")` gives leave-one-session-out.
- **`cross_validate_dataframe`** — run any cross-validation strategy directly from
  a tidy table. Subject codes are mapped back to your own identifiers, so results
  report `"P07"` rather than `0`.
- **`parse_feature_column`** — parse `{channel}_{band}_{feature}` column names
  instead of matching them with substring regexes.
- Characterization tests for the cross-validation entry points, layering guards
  for the package, and a bit-for-bit equivalence test for the windowing assembler.

### Changed — breaking

- **`shannons_entropy` now consumes a PSD table and returns one value per channel.**
  It previously took a raw signal plus `fs`, `window_sec`, and `overlap`, and emitted
  `{channel}_{band}_entropy`. It now takes the output of `psd_bandpowers` and emits
  `{channel}_entropy`, measuring how evenly a channel's energy is spread *across* its
  bands rather than within each band. A channel with fewer than two band columns
  yields `NaN`. This aligns it with `wavelet_entropy` and `imf_entropy`, which
  already consumed their corresponding energy tables.

  ```python
  # before
  entropy_df = shannons_entropy(clean, fs, FREQUENCY_BANDS, window_sec=4.0, overlap=0.5)
  # after
  psd = psd_bandpowers(clean, fs, FREQUENCY_BANDS)
  entropy_df = shannons_entropy(psd)
  ```

- **TensorFlow is no longer a base dependency.** `pip install eegproc` installs the
  preprocessing and featurization stack only. The cross-validation stack now needs
  `pip install eegproc[deep-learning]`, which adds TensorFlow, scikit-learn, and joblib.
- **`requires-python` is now `>=3.10`** (was `>=3`). The package already used PEP 604
  unions in function signatures, so earlier versions installed but could not import.
- **Removed from the distribution:** the SIC model and all of
  `deep_learning/joint_architectures/`, `model_explainability/`,
  `deep_learning/supervised/` (MTLFuseNet, STSNet, CMHFE-DAN), and
  `deep_learning/unsupervised/` (encoders).
- **Removed from `eegproc.plotting`:** `results_io`, `report`, `result_figures`, and
  `plot_calibration_curve`. These encoded the result-JSON schema of an unpublished
  model. `plot_eeg_features` is now the package's only public export.
- **The wheel no longer installs a top-level `tests` package** into `site-packages`.
- **`cross_val.py` was split into `eegproc.deep_learning.cross_validation`.** The
  7,555-line module is now 18 layered modules whose imports point strictly
  downward. `from eegproc.deep_learning.cross_val import loso_cv` still works via
  a deprecation shim, which re-exports the public surface only — private helpers
  moved and are not re-exported. Anything that pickled a private `cross_val._*`
  reference (multiprocessing artifacts) will not unpickle.
- **`generalize_optimization_strats` was renamed to `domain_generalization`**, its
  modules to `meta_learning` / `alternating_group_learning`, and its SIC-named
  symbols to `MLDGEpisodeSequence`, `fit_mldg` and `run_mldg_train_step`. These
  strategies are generic; the old names referred to a model that does not ship.
- **`plot_eeg_features` now returns `(fig, axes)`** instead of `None`, so callers
  can close or customize the figure. `end_row` now defaults to the whole frame
  rather than a single row, and a new `overlap` argument makes the time axis
  correct for overlapping windows.

### Fixed

- **An import-time monkeypatch is gone.** `training_outputs` rebound a global in
  `cross_val` so that leave-one-subject-out would pick up the consolidated
  held-out-user callback. The fold runner now imports it explicitly. Left as-is,
  the split would have silently downgraded LOSO to the base callback — different
  metrics, different history keys, and no error.
- **`generate_all_features` returned only its last group** (the accumulator was
  reassigned inside the loop), raised `NameError` whenever `channels=` was passed,
  and called `shannons_entropy` with its pre-v2 signature. All three fixed.
- **`plot_eeg_features` defects:** it leaked a matplotlib figure on both the error
  path and whenever `save_path` was omitted; its channel filter matched by
  unanchored regex, so `channels=["F3"]` also selected `AF3`; user-supplied names
  were compiled as regexes and raised on metacharacters; an empty selection died
  inside matplotlib rather than saying which channel or band was unknown; and a
  band filter on band-less columns silently selected nothing.
- **The documentation build had 13 warnings** and runs with `-W`, so it could not
  have deployed. Fixed the malformed docstrings behind them, stopped documenting
  re-exported names twice, and excluded the deprecation shim from the reference.
- Removed three private helpers with no callers.
- **pandas 3 compatibility.** pandas 3 returns read-only arrays from
  `to_numpy(copy=False)` under copy-on-write, and the Cython kernels in PyWavelets and
  PyEMD reject read-only buffers — every wavelet and IMF feature raised
  `ValueError: buffer source array is read-only` on a fresh install. Arrays are now
  made writable at those call sites, copying only when necessary.
- **`feature_grouped_by_metadata` dropped the metadata columns** whenever a group
  produced no features, returning a frame with no columns at all instead of an empty
  frame with a usable schema.
- **The test workflow had never run.** `.github/workflows/tests.yml` was malformed
  (`pull request:` with a space, and no `jobs:` key). CI now runs a TensorFlow-free
  base matrix on 3.10–3.13, a separate deep-learning job, and a job asserting the
  wheel ships only `eegproc`.
- Corrected the `shannons_entropy` docstring, which described the wrong input, the
  wrong output columns, and exceptions the function does not raise.
- Corrected the documented Quick Start, which called a nonexistent `plot_per_channel`
  and passed `fs` to `shannons_entropy` as its `bands` argument.

### Removed

- Eleven declared dependencies that nothing in the package imports: `dill`,
  `multiprocess`, `pathos`, `pox`, `ppft`, `python-dateutil`, `pytz`, `six`,
  `threadpoolctl`, `tqdm`, and `tzdata`.
- Experiment output (`runs/`) and build artifacts are no longer tracked.
