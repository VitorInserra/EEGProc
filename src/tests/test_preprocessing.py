import pytest
import pandas as pd
import numpy as np
from eegproc import *


def test_bandpass_filter_column_output(sine_df):
    fs = 128
    bands = FREQUENCY_BANDS
    out = bandpass_filter(sine_df, fs, bands=bands, reref=False, detrend=False)

    for band in FREQUENCY_BANDS.keys():
        if f"A1_{band}" not in out.keys():
            raise AssertionError("Bands not outputted.")
        
def rms(z): 
    return float(np.sqrt(np.mean(z**2)))

def test_bandpass_filter_values(sine_df):
    fs = 128
    bands = FREQUENCY_BANDS
    out = bandpass_filter(sine_df, fs, bands=bands, reref=False, detrend=False)

    print(out)
    for i, j in zip(sine_df["A1"], out["A1_delta"]):
        print(i, j)
        assert abs(j) < 0.05 * abs(i) 