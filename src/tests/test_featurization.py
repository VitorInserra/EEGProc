# tests/test_featurization.py
import numpy as np
import pandas as pd
import pytest

from eegproc.featurization import psd_bandpowers
from eegproc.preprocessing import FREQUENCY_BANDS

FS = 128

def make_sine(freq_hz: float, fs=FS, dur_sec=8.0, amp=1.0, phase=0.0):
    n = int(dur_sec * fs)
    t = np.arange(n) / fs
    return amp * np.sin(2 * np.pi * freq_hz * t + phase)

@pytest.fixture
def df_single_alpha():
    """Single column named A1_alpha containing a pure 10 Hz sine."""
    x = make_sine(10.0)
    return pd.DataFrame({"A1_alpha": x})

@pytest.fixture
def df_mismatch_alpha_as_theta():
    """Column is named A1_theta but contains a 10 Hz sine (should integrate ~0 in theta band)."""
    x = make_sine(10.0)
    return pd.DataFrame({"A1_theta": x})

@pytest.fixture
def df_all_bands_one_each():
    """
    One column per band; each column contains a sine that sits
    comfortably inside its band's pass range.
    """
    demo = {
        "delta": 2.0,
        "theta": 6.0,
        "alpha": 10.0,
        "betaL": 15.0,
        "betaH": 25.0,
        "gamma": 35.0,
    }
    data = {}
    for band, f in demo.items():
        data[f"A1_{band}"] = make_sine(f)
    return pd.DataFrame(data)



'''TESTS'''
def test_psd_bandpowers_columns_and_rows(df_all_bands_one_each):
    """
    Ensures the output has the same columns and the correct number of rows
    based on window_sec and overlap.
    """
    window_sec = 4.0
    overlap = 0.5
    n = len(df_all_bands_one_each)
    nperseg = int(round(window_sec * FS))
    hop = int(round(nperseg * (1.0 - overlap)))
    expected_rows = 1 + (n - nperseg) // hop

    out = psd_bandpowers(
        df_all_bands_one_each, FS, FREQUENCY_BANDS, window_sec=window_sec, overlap=overlap
    )

    assert list(out.columns) == list(df_all_bands_one_each.columns)
    assert len(out) == expected_rows


def test_band_energy_concentrates_in_matching_band(df_single_alpha):
    """
    A 10 Hz sine named A1_alpha should produce non-trivial power in alpha,
    and near-zero if we lie about the band suffix (theta).
    """
    out_alpha = psd_bandpowers(df_single_alpha, FS, FREQUENCY_BANDS, window_sec=4.0, overlap=0.5)
    # Power should be positive and stable across windows
    assert (out_alpha["A1_alpha"] > 0).all()

def test_band_energy_is_near_zero_if_suffix_band_does_not_cover_freq(df_mismatch_alpha_as_theta):
    """
    If we name the column A1_theta but put a 10 Hz sine into it, the theta
    integration (4–8 Hz) should capture ~0 power.
    """
    out = psd_bandpowers(df_mismatch_alpha_as_theta, FS, FREQUENCY_BANDS, window_sec=4.0, overlap=0.5)
    # Allow tiny numerical noise
    assert np.all(out["A1_theta"].to_numpy() < 1e-6)


def test_amplitude_scaling_is_quadratic_in_amp():
    """
    Power should scale ~amplitude^2. Compare amp=2 vs amp=1 on the same 10 Hz alpha column.
    """
    df1 = pd.DataFrame({"A1_alpha": make_sine(10.0, amp=1.0)})
    df2 = pd.DataFrame({"A1_alpha": make_sine(10.0, amp=2.0)})

    p1 = psd_bandpowers(df1, FS, FREQUENCY_BANDS, window_sec=4.0, overlap=0.5)["A1_alpha"].median()
    p2 = psd_bandpowers(df2, FS, FREQUENCY_BANDS, window_sec=4.0, overlap=0.5)["A1_alpha"].median()

    ratio = p2 / (p1 + 1e-12)
    # Hann/Welch/windowing cause small deviations—tolerate ~±10–15%
    assert 3.3 < ratio < 4.7, f"Expected ~4× power, got {ratio:.2f}×"


def test_returns_empty_when_window_longer_than_signal():
    """
    If nperseg > n_samples, function should return empty DataFrame with the same columns.
    """
    short = pd.DataFrame({"A1_alpha": make_sine(10.0, dur_sec=1.0)})  # 128 samples
    out = psd_bandpowers(short, FS, FREQUENCY_BANDS, window_sec=4.0, overlap=0.5)
    assert out.empty
    assert list(out.columns) == ["A1_alpha"]


def test_invalid_overlap_raises():
    df = pd.DataFrame({"A1_alpha": make_sine(10.0)})
    with pytest.raises(ValueError, match="overlap must be in"):
        psd_bandpowers(df, FS, FREQUENCY_BANDS, window_sec=4.0, overlap=1.1)

def test_too_small_window_raises():
    df = pd.DataFrame({"A1_alpha": make_sine(10.0)})
    # nperseg = round(window_sec*fs) <= 8 triggers the guard
    tiny_win = 8.0 / FS  # exactly 8 samples worth
    with pytest.raises(ValueError, match="window_sec too small"):
        psd_bandpowers(df, FS, FREQUENCY_BANDS, window_sec=tiny_win, overlap=0.0)
