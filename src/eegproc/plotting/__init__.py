"""Plotting helpers for EEG feature frames.

The results-reporting modules that used to live here (``results_io``, ``report``,
``result_figures``, ``plot_calibration_curve``) encoded the result-JSON schema of
an unpublished model and were removed in v2.
"""

from .plots import plot_eeg_features

__all__ = ["plot_eeg_features"]
