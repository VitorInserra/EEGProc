import numpy as np
import pytest
from eegproc.preprocessing import bandpass_filter, FREQUENCY_BANDS


def trim_edges(arr, fs, seconds=1):
    """Drop 'seconds' from both ends to avoid filtfilt transients."""
    n = int(fs * seconds)
    if 2 * n >= len(arr):
        return arr  # too short to trim; fallback
    return arr[n:-n]


ORDERED_BANDS = [
    ("delta", FREQUENCY_BANDS["delta"]),
    ("theta", FREQUENCY_BANDS["theta"]),
    ("alpha", FREQUENCY_BANDS["alpha"]),
    ("betaL", FREQUENCY_BANDS["betaL"]),
    ("betaH", FREQUENCY_BANDS["betaH"]),
    ("gamma", FREQUENCY_BANDS["gamma"]),
]

PASS_CASES = [
    (2.0, "delta"),
    (6.0, "theta"),
    (10.0, "alpha"),
    (15.0, "betaL"),
    (25.0, "betaH"),
    (35.0, "gamma"),
]

STOP_CASES = [
    (2.0, "betaH"),
    (6.0, "gamma"),
    (10.0, "delta"),
    (15.0, "delta"),
    (25.0, "theta"),
    (35.0, "alpha"),
]

FS = 128


@pytest.mark.parametrize("sine_df", [2.0], indirect=True)
@pytest.mark.filterwarnings("ignore:reref=True ignored:RuntimeWarning")
def test_bandpass_filter_column_output(sine_df):
    out = bandpass_filter(
        sine_df, FS, bands=FREQUENCY_BANDS, reref=False, detrend=False
    )

    in_cols = list(sine_df.columns)
    for c in in_cols:
        for band_name in FREQUENCY_BANDS.keys():
            expected_col = f"{c}_{band_name}"
            assert expected_col in out.columns, f"Missing column {expected_col}"


@pytest.mark.parametrize(
    "sine_df,freq,band", [(f, f, b) for f, b in PASS_CASES], indirect=["sine_df"]
)
@pytest.mark.filterwarnings("ignore:reref=True ignored:RuntimeWarning")
def test_bandpass_filter_passband_gain(sine_df, freq, band):
    out = bandpass_filter(
        sine_df, FS, bands=FREQUENCY_BANDS, reref=False, detrend=False
    )

    x = trim_edges(sine_df["A1"].to_numpy(), FS, seconds=1)
    y = trim_edges(out[f"A1_{band}"].to_numpy(), FS, seconds=1)

    in_std = x.std()
    out_std = y.std()
    gain = out_std / (in_std + 1e-12)

    # Allow moderate ripple
    assert gain > 0.80, f"{band} should pass {freq} Hz (gain={gain:.3f})"


@pytest.mark.parametrize(
    "sine_df,freq,band", [(f, f, b) for f, b in STOP_CASES], indirect=["sine_df"]
)
@pytest.mark.filterwarnings("ignore:reref=True ignored:RuntimeWarning")
def test_bandpass_filter_stopband_attenuation(sine_df, freq, band):
    out = bandpass_filter(
        sine_df, FS, bands=FREQUENCY_BANDS, reref=False, detrend=False
    )

    x = trim_edges(sine_df["A1"].to_numpy(), FS, seconds=1)
    y = trim_edges(out[f"A1_{band}"].to_numpy(), FS, seconds=1)

    in_std = x.std()
    out_std = y.std()
    gain = out_std / (in_std + 1e-12)

    # -10 dB ≈ 0.316×. Tighten to 0.1 (≈ -20 dB) if your filter order is higher.
    assert gain < 0.316, f"{band} should attenuate {freq} Hz (gain={gain:.3f})"


@pytest.mark.parametrize(
    "sine_df,freq,band", [(f, f, b) for f, b in PASS_CASES], indirect=["sine_df"]
)
@pytest.mark.filterwarnings("ignore:reref=True ignored:RuntimeWarning")
def test_bandpass_optional_flags_do_not_distort_gain(sine_df, freq, band):
    """
    Sanity-check that enabling reref/detrend doesn't collapse the signal. Only one signal (A1) should not allow reref.
    """
    out = bandpass_filter(sine_df, FS, bands=FREQUENCY_BANDS, reref=True, detrend=True)

    x = trim_edges(sine_df["A1"].to_numpy(), FS, seconds=1)
    y = trim_edges(out[f"A1_{band}"].to_numpy(), FS, seconds=1)

    gain = y.std() / (x.std() + 1e-12)
    assert (
        gain > 0.75
    ), f"{band} with reref/detrend should still pass {freq} Hz (gain={gain:.3f})"


