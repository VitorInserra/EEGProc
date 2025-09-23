import numpy as np
import pandas as pd
import pywt
from math import log2, floor
from scipy.signal import welch
from parameters import detrend_df, _numeric_interp, bandpass_filter, FREQUENCY_BANDS

'''SPECTRAL ENTROPY'''
def psd_bandpowers(
    df: pd.DataFrame,
    fs: float,
    bands: dict[str, tuple[float, float]] = FREQUENCY_BANDS,
    window_sec: float = 4.0,
    overlap: float = 0.5,
    detrend: str | None = "constant",
    relative: bool = False,
) -> pd.DataFrame:
    """
    Windowed Welch PSD band-powers per channel.
    Returns a DataFrame with one row per window and columns "<channel>_<band>" only.
    """
    if detrend in {"constant", "linear"}:
        X = detrend_df(df, kind=detrend)
    elif detrend is None:
        X = _numeric_interp(df)
    else:
        raise ValueError("detrend must be 'constant', 'linear', or None")

    X = X.select_dtypes(include=[np.number]).copy()
    if X.empty:
        raise ValueError("No numeric columns after cleaning.")

    channels = list(X.columns)
    n_samples = len(X)

    nperseg = int(round(window_sec * fs))
    if nperseg <= 8:
        raise ValueError("window_sec too small for given fs; increase window_sec.")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0.0, 1.0).")
    hop = int(round(nperseg * (1.0 - overlap)))
    if hop <= 0:
        raise ValueError("overlap too large; hop size must be >= 1 sample.")

    if nperseg > n_samples:
        band_cols = [f"{ch}_{name}" for ch in channels for name in bands]
        return pd.DataFrame(columns=band_cols)

    lo_all = min(lo for lo, _ in bands.values())
    hi_all = max(hi for _, hi in bands.values())

    band_cols = [f"{ch}_{name}" for ch in channels for name in bands]

    rows = []
    for start in range(0, n_samples - nperseg + 1, hop):
        end = start + nperseg
        seg = X.iloc[start:end]

        row = {}
        for ch in channels:
            y = seg[ch].to_numpy(dtype=float, copy=False)
            f_win, Pxx = welch(
                y,
                fs=fs,
                window="hann",
                nperseg=nperseg,
                noverlap=0,
                detrend=False,
                scaling="density",
                return_onesided=True,
            )

            if relative:
                total_m = (f_win >= lo_all) & (f_win <= hi_all)
                total_power = np.trapezoid(Pxx[total_m], f_win[total_m])
                total_power = float(total_power) if total_power > 0 else np.nan

            for name, (lo, hi) in bands.items():
                m = (f_win >= lo) & (f_win <= hi)
                bp = np.trapezoid(Pxx[m], f_win[m]) if m.any() else 0.0
                if relative:
                    bp = bp / total_power if (total_power and np.isfinite(total_power)) else np.nan
                row[f"{ch}_{name}"] = float(bp)

        rows.append(row)

    out = pd.DataFrame(rows)
    return out[[c for c in band_cols if c in out.columns]]


'''HJORTH PARAMETRIZATION'''
def hjorth_params(
    df: pd.DataFrame,
    fs: float,
    window_sec: float = 4.0,
    step_sec: float | None = None,
    detrend: str | None = "constant",
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Compute Hjorth parameters per window for each numeric column in a band-passed EEG DataFrame.

    Returns a DataFrame with multiple rows (one per window) and wide columns:
      <col>_activity, <col>_mobility, <col>_complexity, plus t0,t1 (seconds).

    Parameters
    df : DataFrame (samples x channels/bands)
    fs : float            sampling rate (e.g., 128 for DREAMER)
    window_sec : float    window length in seconds
    step_sec : float      step between windows in seconds; defaults to window_sec (no overlap)
    detrend : {"constant","linear",None}
    eps : float           numerical guard
    """
    if detrend in {"constant", "linear"}:
        X = detrend_df(df, kind=detrend)
    elif detrend is None:
        X = _numeric_interp(df)
    else:
        raise ValueError("detrend must be 'constant', 'linear', or None")
    
    cols = list(X.columns)
    data = X.to_numpy(dtype=float)
    n_samples, n_cols = data.shape

    win = int(round(window_sec * fs))
    if win < 3:
        raise ValueError("window_sec too small (need >= 3 samples for second differences).")
    step = int(round((step_sec if step_sec is not None else window_sec) * fs))
    step = max(step, 1)

    

    rows = []
    starts = range(0, max(n_samples - win + 1, 0), step)
    for i0 in starts:
        i1 = i0 + win
        seg = data[i0:i1, :]
        if seg.shape[0] < 3:
            continue

        act = np.nanvar(seg, axis=0, ddof=0) 

        dx = np.diff(seg, n=1, axis=0)
        ddx = np.diff(seg, n=2, axis=0)

        var_dx = np.nanvar(dx,  axis=0, ddof=0)
        var_ddx = np.nanvar(ddx, axis=0, ddof=0)

        mob = np.sqrt((var_dx  + eps) / (act + eps))
        mob_dx = np.sqrt((var_ddx + eps) / (var_dx + eps))
        comp = mob_dx / (mob + eps)

        row = {}

        for k, c in enumerate(cols):
            row[f"{c}_activity"] = float(act[k]) if np.isfinite(act[k]) else np.nan
            row[f"{c}_mobility"] = float(mob[k]) if np.isfinite(mob[k]) else np.nan
            row[f"{c}_complexity"] = float(comp[k]) if np.isfinite(comp[k]) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


'''WAVELET FEATURES'''
def _choose_dwt_level(n_samples: int, fs: float, wavelet: str, min_freq: float) -> int:
    """Pick a DWT level so the lowest-approx band goes below min_freq (if possible)."""
    max_lvl = pywt.dwt_max_level(n_samples, pywt.Wavelet(wavelet).dec_len)
    target = max(1, floor(log2(fs / max(min_freq, 1e-6)) - 1))
    return max(1, min(max_lvl, target))

def _dwt_subband_ranges(fs: float, level: int):
    """
    Return dict of DWT subband -> (f_lo, f_hi):
      D1..DL: (fs/2^(j+1), fs/2^j), A_L: (0, fs/2^(L+1))
    """
    bands = {}
    for j in range(1, level + 1):
        f_hi = fs / (2 ** j)
        f_lo = fs / (2 ** (j + 1))
        bands[f"D{j}"] = (f_lo, f_hi)
    bands[f"A{level}"] = (0.0, fs / (2 ** (level + 1)))
    return bands

def _overlap(a, b):
    """Length of overlap between intervals a=(lo,hi), b=(lo,hi), assumes lo<hi."""
    lo = max(a[0], b[0]); hi = min(a[1], b[1])
    return max(0.0, hi - lo)

def wavelet_band_energy(
    df: pd.DataFrame,
    fs: float,
    bands: dict[str, tuple[float, float]],
    *,
    wavelet: str = "db4",
    level: int | None = None,
    mode: str = "periodization",
    relative: bool = False,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Wavelet band energy per channel using DWT with proportional frequency-overlap mapping.

    Returns a single-df DataFrame with columns like "<channel>_<band>_wenergy".
    Set `relative=True` to normalize energies by the channel's total energy.
    """
    X = df.select_dtypes(include=[np.number])
    if X.empty:
        raise ValueError("No numeric columns in df.")

    min_band_lo = min(lo for lo, _ in bands.values())
    L = level or _choose_dwt_level(len(X), fs, wavelet, min_band_lo)

    out = {}
    for ch in X.columns:
        x = X[ch].to_numpy()
        coeffs = pywt.wavedec(x, wavelet=wavelet, level=L, mode=mode)
        cA = coeffs[0]
        cDs = coeffs[1:]

        sub_eng = {}
        for idx, c in enumerate(cDs):
            j = L - idx
            sub_eng[f"D{j}"] = float(np.sum(c.astype(float) ** 2))
        sub_eng[f"A{L}"] = float(np.sum(cA.astype(float) ** 2))

        sub_ranges = _dwt_subband_ranges(fs, L)

        band_energy = {name: 0.0 for name in bands}
        for sub_name, e_sub in sub_eng.items():
            f_lo, f_hi = sub_ranges[sub_name]
            width = (f_hi - f_lo) or 1.0
            for bname, (blo, bhi) in bands.items():
                olap = _overlap((f_lo, f_hi), (blo, bhi))
                if olap > 0:
                    band_energy[bname] += e_sub * (olap / width)

        total = sum(band_energy.values()) + eps
        for bname, e in band_energy.items():
            val = e / total if relative else e
            out[f"{ch}_{bname}_wenergy"] = float(val)

    return pd.DataFrame([out])


def wavelet_entropy(
    df: pd.DataFrame,
    fs: float,
    bands: dict[str, tuple[float, float]],
    *,
    wavelet: str = "db4",
    level: int | None = None,
    mode: str = "periodization",
    normalize: bool = True,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Wavelet (spectral) entropy per channel:
      - Compute DWT band energies (same mapping as wavelet_band_energy_df, absolute).
      - Convert to probabilities across bands p_k = E_k / sum(E_k).
      - Shannon entropy H = -sum p_k log(p_k). If `normalize=True`, divide by log(K) to get [0,1].

    Returns a single-df DataFrame with columns "<channel>_wwentropy".
    """
    X = df.select_dtypes(include=[np.number])
    if X.empty:
        raise ValueError("No numeric columns in df.")

    # get absolute band energies
    eng = wavelet_band_energy_df(
        df=X, fs=fs, bands=bands, wavelet=wavelet, level=level, mode=mode, relative=False, eps=eps
    )

    out = {}
    K = len(bands)
    logK = np.log(K) if normalize else 1.0

    for ch in X.columns:
        # extract that channel's band energies in the canonical band order
        vals = [float(eng[f"{ch}_{b}_wenergy"].iloc[0]) for b in bands.keys()]
        total = sum(vals) + eps
        p = np.array(vals) / total
        H = -np.sum(p * np.log(p + eps))
        out[f"{ch}_wwentropy"] = float(H / (logK or 1.0))

    return pd.DataFrame([out])


if __name__ == "__main__":
    FS = 128
    csv_path = "DREAMER.csv"
    chunk_iter = pd.read_csv(csv_path, chunksize=1)
    first_chunk = next(chunk_iter)
    sensor_columns = [col for col in first_chunk.columns if col[len(col)-1].isdigit()]
    print(f"Detected sensor columns: {sensor_columns}")

    dreamer_df = []

    for chunk in pd.read_csv(csv_path, chunksize=10000):
        sensor_df = chunk[sensor_columns]
        dreamer_df.append(sensor_df)

    dreamer_df = pd.concat(dreamer_df, ignore_index=True)

    clean = bandpass_filter(dreamer_df, FS, bands=FREQUENCY_BANDS, low=0.5, high=45.0, notch_hz=60)
    print("Bandpass filtering\n", clean.head())

    hj = hjorth_params(clean, FS)
    print("Hjorth Parameters\n", hj.head())

    psd_df = psd_bandpowers(clean, FS, bands=FREQUENCY_BANDS)
    print("PSD\n", psd_df.head())
