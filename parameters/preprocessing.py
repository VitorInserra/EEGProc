import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
from scipy.signal import detrend as scipy_detrend

FREQUENCY_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "betaL": (13.0, 20.0),
    "betaH": (20.0, 30.0),
    "gamma": (30.0, 45.0),
}


def apply_detrend(detrend: str | None, df: pd.DataFrame) -> pd.DataFrame:
    if detrend in {"constant", "linear"}:
        df = detrend_df(df, kind=detrend)
    elif detrend is None:
        df = _numeric_interp(df)
    else:
        raise ValueError("detrend must be 'constant', 'linear', or None")
    
    return df

def _numeric_interp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.select_dtypes(include=[np.number]).astype(float).copy()
    return df.apply(lambda s: s.interpolate(limit_direction="both"))

def detrend_df(df: pd.DataFrame, kind: str = "linear") -> pd.DataFrame:
    df = _numeric_interp(df)
    arr = df.to_numpy(copy=False)
    arr = scipy_detrend(arr, type=kind, axis=0)
    return pd.DataFrame(arr, index=df.index, columns=df.columns)

def bandpass_filter(
    df: pd.DataFrame,
    fs: float,
    bands: dict[str, tuple[float, float]] = FREQUENCY_BANDS,
    low: float | None = None,
    high: float | None = None,
    *,
    order: int = 4,
    notch_hz: float | int | list | tuple | None = None,
    notch_q: float = 30.0,
    reref: bool = True,
    detrend: bool = True,
) -> pd.DataFrame:
    df = _numeric_interp(df).apply(pd.to_numeric, errors="coerce")
    df = df.astype("float64")
    nyq = fs / 2.0
    cols = list(df.columns)

    def _sosfiltfilt_safe(sos, y):
        if np.all(np.isnan(y)):
            return y
        y = y.copy()
        nans = np.isnan(y)
        if nans.any():
            iddf = np.where(~nans)[0]
            if iddf.size >= 2:
                y[nans] = np.interp(np.flatnonzero(nans), iddf, y[iddf])
            else:
                y[nans] = 0.0
        if y.size < 15:
            return y
        return sosfiltfilt(sos, y)

    def _apply_notch_once(dfin: pd.DataFrame) -> pd.DataFrame:
        if notch_hz is None:
            return dfin
        freqs = notch_hz if isinstance(notch_hz, (list, tuple)) else [notch_hz]
        edfpanded = []
        for f0 in freqs:
            edfpanded.append(float(f0))
            if 2 * f0 < nyq - 1.0:
                edfpanded.append(float(2 * f0))
        out = dfin.copy()
        for f0 in edfpanded:
            w0 = f0 / nyq
            if 0 < w0 < 1:
                b, a = iirnotch(w0, notch_q)
                for c in cols:
                    y = out[c].to_numpy()
                    if y.size >= max(15, 3 * max(len(a), len(b))):
                        out[c] = filtfilt(b, a, y, method="gust")
        return out

    if reref:
        car = df.mean(axis=1)
        for c in cols:
            df[c] = df[c] - car

    df = _apply_notch_once(df)

    if bands is None:
        if low is None or high is None:
            raise ValueError("Provide (low, high) or a bands dict.")
        if not (0 < low < high < nyq):
            raise ValueError(f"Cutoffs must satisfy 0 < low < high < fs/2={nyq:.3f}.")
        sos = butter(order, [low/nyq, high/nyq], btype="bandpass", output="sos")
        Y = pd.DataFrame(index=df.index)
        for c in cols:
            Y[c] = _sosfiltfilt_safe(sos, df[c].to_numpy())
        if detrend:
            for c in cols:
                Y[c] = scipy_detrend(Y[c].to_numpy(), type="constant")
        return Y

    out = {}
    band_sos = {}
    for name, (lo, hi) in bands.items():
        if not (0 < lo < hi < nyq):
            raise ValueError(f"Bad band {name}: {lo}-{hi} vs fs/2={nyq:.3f}")
        band_sos[name] = butter(order, [lo/nyq, hi/nyq], btype="bandpass", output="sos")

    for c in cols:
        y0 = df[c].to_numpy()
        for name, sos in band_sos.items():
            yb = _sosfiltfilt_safe(sos, y0)
            if detrend:
                yb = scipy_detrend(yb, type="constant")
            out[f"{c}_{name}"] = yb

    return pd.DataFrame(out, index=df.index)
