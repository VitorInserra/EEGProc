# tests/conftest.py
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sine_df(fs: int = 128, sinusoid_freq: int = 10):
    t = np.arange(0, 2, 1/fs)
    x = np.sin(2 * np.pi * sinusoid_freq * t)
    return pd.DataFrame({"A1": x})