import numpy as np
import pandas as pd
import pywt
from math import log2, floor
from scipy.signal import welch
from eegproc import bandpass_filter, apply_detrend, FREQUENCY_BANDS
from PyEMD import EMD


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
    eps: float = 1e-300, # avoids 0 denominators and log(0)
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
    overlap: float = 0.5,
    detrend: str | None = "constant",
    eps: float = 1e-300,
) -> pd.DataFrame:
    """
    Compute Hjorth parameters per window for each numeric column in a band-passed EEG DataFrame.

    Returns a DataFrame with multiple rows (one per window):
      <col>_activity, <col>_mobility, <col>_complexity

    Parameters
    df : DataFrame (samples df channels/bands)
    fs : float            sampling rate in Hz
    window_sec : float    window length in seconds
    step_sec : float      step between windows in seconds; defaults to window_sec (no overlap)
    detrend : {"constant", "linear" ,None}
    eps : float           numerical guard
    """
    df = apply_detrend(detrend, df)    

    cols = list(df.columns)
    data = df.to_numpy(dtype=float)
    n_samples, n_cols = data.shape

    win = int(round(window_sec * fs))
    if win < 3:
        raise ValueError("window_sec too small (need >= 3 samples for second differences).")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0.0, 1.0).")

    hop = int(round(win * (1.0 - overlap)))
    if hop <= 0:
        raise ValueError("overlap too large; hop size must be >= 1 sample.")

    rows = []
    starts = range(0, n_samples - win + 1, hop)
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
    max_lvl = pywt.dwt_max_level(n_samples, pywt.Wavelet(wavelet).dec_len)
    target = max(1, floor(log2(fs / max(min_freq, 1e-6)) - 1))
    return max(1, min(max_lvl, target))

def _dwt_subband_ranges(fs: float, level: int) -> dict[str, tuple[float, float]]:
    bands: dict[str, tuple[float, float]] = {}
    for j in range(1, level + 1):
        f_hi = fs / (2 ** j)
        f_lo = fs / (2 ** (j + 1))
        bands[f"D{j}"] = (f_lo, f_hi)
    bands[f"A{level}"] = (0.0, fs / (2 ** (level + 1)))
    return bands

def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    lo = max(a[0], b[0]); hi = min(a[1], b[1])
    return max(0.0, hi - lo)

def wavelet_band_energy(
    df: pd.DataFrame,
    fs: float,
    bands: dict[str, tuple[float, float]],
    wavelet: str = "db4",
    mode: str = "periodization",
    window_sec: float = 4.0,
    overlap: float = 0.5,
) -> pd.DataFrame:
    df = df.select_dtypes(include=[np.number])

    n_samples = len(df)
    nperseg = int(round(window_sec * fs))
    if nperseg <= 8:
        raise ValueError("window_sec too small for given fs; increase window_sec.")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0.0, 1.0).")
    hop = int(round(nperseg * (1.0 - overlap)))
    if hop <= 0:
        raise ValueError("overlap too large; hop size must be >= 1 sample.")
    if nperseg > n_samples:
        return pd.DataFrame(columns=[f"{ch}_{b}_wenergy" for ch in df.columns for b in bands])

    min_band_lo = min(lo for lo, _ in bands.values())
    L = _choose_dwt_level(n_samples=nperseg, fs=fs, wavelet=wavelet, min_freq=min_band_lo)
    sub_ranges = _dwt_subband_ranges(fs, L)

    cols = [f"{ch}_{b}_wenergy" for ch in df.columns for b in bands]
    rows = []

    for start in range(0, n_samples - nperseg + 1, hop):
        win = df.iloc[start:start + nperseg]
        row: dict[str, float] = {}

        for ch in df.columns:
            y = win[ch].to_numpy(dtype=float, copy=False)

            coeffs = pywt.wavedec(y, wavelet=wavelet, level=L, mode=mode) # applies wavelet transform function
            wv_coeff_approx = coeffs[0]
            wv_coeff_details = coeffs[1:]

            sub_eng: dict[str, float] = {}
            for idx, c in enumerate(wv_coeff_details):
                j = L - idx
                sub_eng[f"D{j}"] = float(np.sum(c.astype(float) ** 2))
            sub_eng[f"A{L}"] = float(np.sum(wv_coeff_approx.astype(float) ** 2))

            band_energy = {name: 0.0 for name in bands}
            for sub_name, e_sub in sub_eng.items():
                f_lo, f_hi = sub_ranges[sub_name]
                width = (f_hi - f_lo) or 1.0
                if width <= 0:
                    continue
                for band_name, (blo, bhi) in bands.items():
                    olap = _overlap((f_lo, f_hi), (blo, bhi))
                    if olap > 0:
                        band_energy[band_name] += e_sub * (olap / width)

            for band_name, e in band_energy.items():
                row[f"{ch}_{band_name}_wenergy"] = float(e)

        rows.append(row)

    return pd.DataFrame(rows, columns=cols)

def wavelet_entropy(
    wv_band_energy_df: pd.DataFrame,
    bands: dict[str, tuple[float, float]],
    normalize: bool = True,
    eps: float = 1e-300,
) -> pd.DataFrame:
    df = wv_band_energy_df.select_dtypes(include=[np.number]).copy()

    band_list = list(bands.keys())
    K = len(band_list)
    norm = (np.log(K) if (normalize and K > 1) else 1.0)

    channel_to_cols: dict[str, list[str]] = {}
    for col in df.columns:
        if not col.endswith("_wenergy"):
            continue
        core = col[:-8]
        if "_" not in core:
            continue
        ch, b = core.rsplit("_", 1)
        if b in bands:
            channel_to_cols.setdefault(ch, [None] * K)

    if not channel_to_cols:
        raise ValueError("No columns with pattern '{channel}_{band}_wenergy' matching provided bands.")

    for col in df.columns:
        if not col.endswith("_wenergy"):
            continue
        core = col[:-8]
        if "_" not in core:
            continue
        ch, b = core.rsplit("_", 1)
        if ch in channel_to_cols and b in bands:
            idx = band_list.index(b)
            channel_to_cols[ch][idx] = col

    out_cols = [f"{ch}_wentropy" for ch in channel_to_cols.keys()]
    rows = []

    for i in range(len(df)):
        row_out = {}
        for ch, cols_in_order in channel_to_cols.items():
            vals = []
            for c in cols_in_order:
                if c is None:
                    vals.append(0.0)
                else:
                    v = df.iat[i, df.columns.get_loc(c)]
                    vals.append(float(v) if np.isfinite(v) else 0.0)
            
            total = float(np.nansum(vals))
            total = total if (np.isfinite(total) and total > 0) else eps

            p = np.asarray(vals, dtype=float) / total
            p = np.clip(p, eps, 1.0)
            p /= p.sum()

            H = -np.sum(p * np.log(p))
            row_out[f"{ch}_wentropy"] = float(H / (norm or 1.0))
        rows.append(row_out)

    return pd.DataFrame(rows, columns=out_cols)


'''IMF FEATURES'''
def imf_band_energy(
    df: pd.DataFrame,
    fs: float,
    imf_to_band: list[str] = ["delta", "theta", "alpha", "betaL", "betaH", "gamma"],
    window_sec: float = 4.0,
    overlap: float = 0.5,
    EMD_kwargs: dict = {},
) -> pd.DataFrame:

    df = df.select_dtypes(include=[np.number])

    n_samples = len(df)
    nperseg = int(round(window_sec * fs))

    if nperseg <= 8:
        raise ValueError("window_sec too small for given fs; increase window_sec.")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0.0, 1.0).")
    hop = int(round(nperseg * (1.0 - overlap)))
    if hop <= 0:
        raise ValueError("overlap too large; hop size must be >= 1 sample.")
    if nperseg > n_samples:
        cols = [f"{ch}_{band}_imfenergy" for ch in df.columns for band in imf_to_band]
        return pd.DataFrame(columns=cols)

    max_imf_needed = int(len(imf_to_band))
    cols = [f"{ch}_{band}_imfenergy" for ch in df.columns for band in imf_to_band]

    emd = EMD(**(EMD_kwargs))

    rows = []
    row: dict[str, float] = {}

    emd._imf_cumsums = {}
    
    for ch in df.columns:
        y_full = df[ch].to_numpy(dtype=float, copy=False).astype(float, copy=False)
        imfs_full = emd.emd(y_full, max_imf=max_imf_needed)
        sq = imfs_full ** 2
        cumsq = np.hstack([np.zeros((sq.shape[0], 1), dtype=sq.dtype),
                        np.cumsum(sq, axis=1)])
        
        emd._imf_cumsums[ch] = (imfs_full, cumsq)


    for start in range(0, n_samples - nperseg + 1, hop):
        end = start + nperseg
        row: dict[str, float] = {}

        for ch in df.columns:
            e_win = emd._imf_cumsums[ch][1][:, end] - emd._imf_cumsums[ch][1][:, start]

            for imf_idx in range(len(imf_to_band)):
                e = float(e_win[imf_idx]) if imf_idx < e_win.shape[0] else 0.0
                row[f"{ch}_{imf_to_band[imf_idx]}_imfenergy"] = e

        rows.append(row)


    return pd.DataFrame(rows, columns=cols)

def imf_entropy(
    imf_energy_df: pd.DataFrame,
    bands: list[str] = ["delta", "theta", "alpha", "betaL", "betaH", "gamma"],
    normalize: bool = True,
    eps: float = 1e-300,
) -> pd.DataFrame:
    df = imf_energy_df.select_dtypes(include=[np.number])

    k = len(bands)
    norm = (np.log(k) if (normalize and k > 1) else 1.0)

    channel_to_cols: dict[str, list[str]] = {}
    suffix = "_imfenergy"

    for col in df.columns:
        if not col.endswith(suffix):
            continue
        ch_band = col[:-len(suffix)]
        if "_" not in ch_band:
            continue
        ch, band = ch_band.rsplit("_", 1)
        if band in bands:
            channel_to_cols.setdefault(ch, [None] * k)

    for col in df.columns:
        if not col.endswith(suffix):
            continue
        core = col[:-len(suffix)]
        if "_" not in core:
            continue
        ch, band = core.rsplit("_", 1)
        if ch in channel_to_cols and band in bands:
            channel_to_cols[ch][bands.index(band)] = col

    out_cols = [f"{ch}_imfentropy" for ch in channel_to_cols.keys()]
    rows: list[dict[str, float]] = []

    for i in range(len(df)):
        row_out: dict[str, float] = {}
        for ch, cols_in_order in channel_to_cols.items():
            vals = []
            for c in cols_in_order:
                v = df.iat[i, df.columns.get_loc(c)]
                vals.append(float(v) if np.isfinite(v) else 0.0)

            total = float(np.nansum(vals))
            if not (np.isfinite(total) and total > 0):
                row_out[f"{ch}_imfentropy"] = np.nan
                continue

            p = np.asarray(vals, dtype=float) / total
            p = np.clip(p, eps, 1.0)
            p /= p.sum()
            H = -np.sum(p * np.log(p))
            row_out[f"{ch}_imfentropy"] = float(H / (norm or 1.0))

        rows.append(row_out)

    return pd.DataFrame(rows, columns=out_cols)


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
    print("Bandpass filtering\n", clean)

    hj = hjorth_params(clean, FS)
    print("Hjorth Parameters\n", hj)

    psd_df = psd_bandpowers(clean, FS, bands=FREQUENCY_BANDS)
    print("PSD\n", psd_df)
    
    shannons_df = shannons_entropy(clean, FS, bands=FREQUENCY_BANDS)
    print("Shannons\n", shannons_df)

    wt_df = wavelet_band_energy(dreamer_df, FS, bands=FREQUENCY_BANDS)
    print("WT Energy\n", wt_df)

    wt_df = wavelet_entropy(wt_df, bands=FREQUENCY_BANDS)
    print("WT Entropy\n", wt_df)

    imf_df = imf_band_energy(dreamer_df, FS)
    print("IMF Energy\n", imf_df)

    imf_df = imf_entropy(imf_df)
    print("IMF Entropy\n", imf_df)