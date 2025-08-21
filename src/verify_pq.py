#!/usr/bin/env python3

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf, pacf

CSV_PATH = "/home/daniel/data/stonks/GOOG_data.csv"


def pick_close(df: pd.DataFrame) -> pd.Series:
    cols_lower = {c.lower(): c for c in df.columns}
    if "close" in cols_lower:
        s = df[cols_lower["close"]]
    else:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            raise SystemExit("No numeric columns found in CSV")
        s = df[num_cols[0]]
    return pd.to_numeric(s, errors="coerce").dropna()


def choose_d(y: pd.Series, alpha: float = 0.05, max_d: int = 2) -> int:
    d = 0
    x = y.copy()
    while d < max_d:
        try:
            _stat, pval, *_ = adfuller(x, autolag="AIC")
        except Exception:
            break
        if pval < alpha:
            break
        x = x.diff().dropna()
        d += 1
        if len(x) < 10:
            break
    return d


def last_significant(series_vals: np.ndarray, band: float) -> int:
    # series_vals expected for lags 0..K; we scan 1..K
    last = 0
    for k in range(1, len(series_vals)):
        if abs(series_vals[k]) > band:
            last = k
    return last


def main():
    df = pd.read_csv(CSV_PATH)
    s = pick_close(df).astype("float64").dropna()
    if len(s) < 20:
        raise SystemExit("Not enough data to estimate")

    # Determine d and get differenced series used for ACF/PACF
    d = choose_d(s, alpha=0.05, max_d=2)
    y = s.copy()
    for _ in range(d):
        y = y.diff().dropna()

    T = len(y)
    K = max(1, int(np.sqrt(T)))
    band = 1.96 / np.sqrt(T)

    # ACF (biased) and PACF (Yule-Walker-M) up to K
    acf_vals = acf(y, nlags=K, fft=True, adjusted=False)
    pacf_vals = pacf(y, nlags=K, method="ywm")

    p = last_significant(pacf_vals, band)
    q = last_significant(acf_vals, band)

    print(f"p={p} d={d} q={q}")

    show = min(10, K)
    print(f"T={T}, K={K}, band={band:.4f}")
    print("lag  ACF      PACF     >band")
    for k in range(1, show + 1):
        a = acf_vals[k]
        pcf = pacf_vals[k]
        flag = "*" if (abs(a) > band or abs(pcf) > band) else ""
        print(f"{k:>3}  {a:>+7.4f}  {pcf:>+7.4f}  {flag}")


if __name__ == "__main__":
    main()

