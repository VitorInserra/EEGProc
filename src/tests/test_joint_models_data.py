"""Tests for joint-model signal representations."""

import numpy as np

from eegproc.deep_learning.joint_architectures.joint_models_data import (
    bandpass_trial_signal,
)


def test_trial_bandpass_produces_channel_major_theta_alpha_beta_features():
    fs = 128.0
    time = np.arange(int(4 * fs), dtype=np.float32) / fs
    frequencies = (6.0, 10.0, 20.0)
    components = [
        np.sin(2.0 * np.pi * frequency * time) for frequency in frequencies
    ]
    channel = np.sum(components, axis=0)
    trial = np.stack((channel, 2.0 * channel), axis=0)

    filtered = bandpass_trial_signal(
        trial,
        fs=fs,
        bands=((4.0, 8.0), (8.0, 13.0), (13.0, 30.0)),
    )

    assert filtered.shape == (6, len(time))
    central = slice(64, -64)
    for band_index, component in enumerate(components):
        correlation = np.corrcoef(
            filtered[band_index, central], component[central]
        )[0, 1]
        assert correlation > 0.95
        np.testing.assert_allclose(
            filtered[3 + band_index, central],
            2.0 * filtered[band_index, central],
            rtol=1e-5,
            atol=1e-5,
        )
