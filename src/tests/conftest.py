# tests/conftest.py
import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def sine_df(request):
    fs = 128
    dur = 8.0  # seconds -> 1024 samples; enough cycles even for 2 Hz
    t = np.arange(0, int(dur * fs)) / fs
    freq = float(request.param)
    x = np.sin(2 * np.pi * freq * t)
    df = pd.DataFrame({"A1": x})
    return df