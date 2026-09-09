Getting Started
===============

EEGProc is a fully vectorized library designed for preprocessing and extracting features from EEG(Electroencephalogram) data. This library is optimized for performance and ease of use, making it suitable for researchers and developers working in the field of neuroscience, biomedical engineering, and machine learning.

Installation
------------

Install from PyPI:

```bash
pip install eegproc
```

or, for the latest development version:

```bash
pip install git+https://github.com/VitorInserra/EEGProc.git
```

Dependencies
------------

EEGProc relies on:

- **NumPy**, **Pandas**, **SciPy** – numerical processing
- **PyWavelets** – wavelet features
- **PyEMD** – empirical mode decomposition
- **Matplotlib** – plotting utilities

Quick Start
-----------

1. **Import and load your EEG data:**

```python
import pandas as pd
from eegproc import bandpass_filter, FREQUENCY_BANDS

df = pd.read_csv("my_eeg_data.csv")
fs = 128  # Hz
```

2. **Filter into frequency bands:**

```python
from eegproc import bandpass_filter, FREQUENCY_BANDS

clean = bandpass_filter(df, fs, bands=FREQUENCY_BANDS)
# -> columns named {channel}_{band}, e.g. AF3_alpha
```

3. **Extract features:**

```python
from eegproc import psd_bandpowers, shannons_entropy, hjorth_params

psd = psd_bandpowers(clean, fs, bands=FREQUENCY_BANDS)   # {channel}_{band}
entropy_df = shannons_entropy(psd)                        # {channel}_entropy
hjorth_df = hjorth_params(clean, fs)                      # {channel}_activity, ...
```

`shannons_entropy` consumes the **PSD table**, not the raw signal, and returns one
value per channel describing how evenly that channel's energy is spread across
bands. The other entropy featurizers follow the same shape:
`wavelet_entropy` consumes `wavelet_band_energy`, and `imf_entropy` consumes
`imf_band_energy`.

4. **Visualize results:**

```python
from eegproc.plotting import plot_eeg_features

plot_eeg_features(entropy_df, title="Shannon Entropy per Channel", seconds=4.0)
```

Documentation Structure
-----------------------

```{toctree}
:maxdepth: 2

api/modules
```
