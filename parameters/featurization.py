import numpy as np
import pandas as pd
import pywt
from math import log2, floor
from scipy.signal import welch
from parameters import bandpass_filter, apply_detrend, FREQUENCY_BANDS


'''SPECTRAL ENTROPY'''
def psd_bandpowers(
    df: pd.DataFrame,
    fs: float,
    bands: dict[str, tuple[float, float]] = FREQUENCY_BANDS,
    window_sec: float = 4.0,
    overlap: float = 0.5,
    detrend: str | None = "constant",
) -> pd.DataFrame:
    df = apply_detrend(detrend, df)

    band_keys = set(bands.keys())
    col_band, col_chan = {}, {}
    for col in df.columns:
        parts = col.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in band_keys:
            col_band[col] = parts[1]
            col_chan[col] = parts[0]
    if not col_band:
        raise ValueError("No columns named like '{channel}_{band}' with band in FREQUENCY_BANDS.")
    df = df[list(col_band.keys())]

    data = df.to_numpy(dtype=float, copy=False)
    n_samples, n_cols = data.shape
    nperseg = int(round(window_sec * fs))
    if nperseg <= 8:
        raise ValueError("window_sec too small for given fs; increase window_sec.")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0.0, 1.0).")
    hop = int(round(nperseg * (1.0 - overlap)))
    if hop <= 0:
        raise ValueError("overlap too large; hop size must be >= 1 sample.")
    if nperseg > n_samples:
        return pd.DataFrame(columns=list(df.columns))

    band_to_idx = {}
    for i, col in enumerate(df.columns):
        band_to_idx.setdefault(col_band[col], []).append(i)

    rows = []
    for start in range(0, n_samples - nperseg + 1, hop):
        seg = data[start:start + nperseg, :]

        f, psd = welch(
            seg,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=0,
            detrend=False,
            scaling="density",
            return_onesided=True,
            axis=0,
        )

        row = {}
        for band, idxs in band_to_idx.items():
            lo, hi = bands[band]
            m = (f >= lo) & (f <= hi)
            if not m.any():
                for j in idxs:
                    row[df.columns[j]] = 0.0
                continue

            band_power = np.trapezoid(psd[m][:, idxs], f[m], axis=0)
            for k, j in enumerate(idxs):
                row[df.columns[j]] = float(band_power[k])


        rows.append(row)

    return pd.DataFrame(rows, columns=list(df.columns))


def shannons_entropy(
    df: pd.DataFrame,
    fs: float,
    bands: dict[str, tuple[float, float]] = FREQUENCY_BANDS,
    window_sec: float = 4.0,
    overlap: float = 0.5,
    eps: float = 1e-300,
    detrend: str | None = "constant",
) -> pd.DataFrame:
    df = apply_detrend(detrend, df)
    band_keys = set(bands.keys())
    col_band = {}
    for col in df.columns:
        parts = col.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in band_keys:
            col_band[col] = parts[1]
    if not col_band:
        raise ValueError("No columns named like '{channel}_{band}' with band in FREQUENCY_BANDS.")
    df = df[list(col_band.keys())]

    data = df.to_numpy(dtype=float, copy=False)
    n_samples, n_cols = data.shape
    nperseg = int(round(window_sec * fs))
    if nperseg <= 8:
        raise ValueError("window_sec too small for given fs; increase window_sec.")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0.0, 1.0).")
    hop = int(round(nperseg * (1.0 - overlap)))
    if hop <= 0:
        raise ValueError("overlap too large; hop size must be >= 1 sample.")
    if nperseg > n_samples:
        return pd.DataFrame(columns=[f"{c}_entropy" for c in df.columns])

    band_to_idx = {}
    for i, col in enumerate(df.columns):
        band_to_idx.setdefault(col_band[col], []).append(i)

    rows = []
    for start in range(0, n_samples - nperseg + 1, hop):
        seg = data[start:start + nperseg, :]

        f, psd = welch(
            seg,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=0,
            detrend=False,
            scaling="density",
            return_onesided=True,
            axis=0,
        ) 

        row = {}
        for band, idxs in band_to_idx.items():
            lo, hi = bands[band]
            m = (f >= lo) & (f <= hi)
            count = int(np.count_nonzero(m))
            if count < 2:
                for j in idxs:
                    row[f"{df.columns[j]}_entropy"] = np.nan
                continue

            band_power = psd[m][:, idxs]
            totals = np.sum(band_power, axis=0)
            valid = np.isfinite(totals) & (totals > 0)

            p = np.empty_like(band_power)
            p[:, valid] = band_power[:, valid] / totals[valid]
            p[:, ~valid] = np.nan
            p = np.clip(p, eps, 1.0)

            H = -np.nansum(p * np.log2(p), axis=0)
            H /= np.log2(count)

            for k, j in enumerate(idxs):
                row[f"{df.columns[j]}_entropy"] = float(H[k]) if np.isfinite(H[k]) else np.nan

        rows.append(row)

    return pd.DataFrame(rows, columns=[f"{c}_entropy" for c in df.columns])




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
      <col>_activity, <col>_mobility, <col>_compledfity,.

    Parameters
    df : DataFrame (samples df channels/bands)
    fs : float            sampling rate (e.g., 128 for DREAMER)
    window_sec : float    window length in seconds
    step_sec : float      step between windows in seconds; defaults to window_sec (no overlap)
    detrend : {"constant","linear",None}
    eps : float           numerical guard
    """
    df = apply_detrend(detrend, df)    

    cols = list(df.columns)
    data = df.to_numpy(dtype=float)
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
    bands = {}
    for j in range(1, level + 1):
        f_hi = fs / (2 ** j)
        f_lo = fs / (2 ** (j + 1))
        bands[f"D{j}"] = (f_lo, f_hi)
    bands[f"A{level}"] = (0.0, fs / (2 ** (level + 1)))
    return bands

def _overlap(a, b):
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
    df = df.select_dtypes(include=[np.number])
    if df.empty:
        raise ValueError("No numeric columns in df.")

    min_band_lo = min(lo for lo, _ in bands.values())
    L = level or _choose_dwt_level(len(df), fs, wavelet, min_band_lo)

    out = {}
    for ch in df.columns:
        df = df[ch].to_numpy()
        coeffs = pywt.wavedec(df, wavelet=wavelet, level=L, mode=mode)
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
    df = df.select_dtypes(include=[np.number])
    if df.empty:
        raise ValueError("No numeric columns in df.")

    # get absolute band energies
    eng = wavelet_band_energy_df(
        df=df, fs=fs, bands=bands, wavelet=wavelet, level=level, mode=mode, relative=False, eps=eps
    )

    out = {}
    K = len(bands)
    logK = np.log(K) if normalize else 1.0

    for ch in df.columns:
        # edftract that channel's band energies in the canonical band order
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

    import time

    start_time = time.time()
    psd_df = psd_bandpowers(clean, FS, bands=FREQUENCY_BANDS)
    print("PSD\n", psd_df.head())
    
    shannons_df = shannons_entropy(clean, FS, bands=FREQUENCY_BANDS)
    print("Shannons\n", shannons_df.head())
    print(time.time() - start_time, "seconds")
