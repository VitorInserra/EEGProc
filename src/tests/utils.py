import numpy as np

FS = 128

def make_sine(freq_hz: float, fs=FS, dur_sec=8.0, amp=1.0, phase=0.0):
    """Make 2pi*freq*t hz sine function"""
    n = int(dur_sec * fs)  # dur_sec gives duration of sine wave
    t = np.arange(n) / fs
    return amp * np.sin(2 * np.pi * freq_hz * t + phase)

def make_white_noise(fs=FS, dur_sec=12.0, amp=1.0, seed=0):
    rng = np.random.RandomState(seed)
    n = int(dur_sec * fs)
    return amp * rng.randn(n)

def expected_mobility(freq_hz: float, fs=FS):
    # mobility = ~2 * sin(pi * f / fs)
    return 2.0 * np.sin(np.pi * freq_hz / fs)

def window_rows(n_samples: int, fs=FS, window_sec=4.0, overlap=0.5):
    win = int(round(window_sec * fs))
    hop = int(round(win * (1.0 - overlap)))
    return 1 + (n_samples - win) // hop