from __future__ import annotations
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from numpy.linalg import multi_dot
from scipy.stats import norm

def parametric_var_series(R: pd.DataFrame, w: pd.Series, window: int, alpha: float, V: float) -> pd.Series:
    z = norm.ppf(alpha)
    w_all = w.reindex(R.columns).astype(float)
    out = {}
    for i in range(window, len(R)):
        X = R.iloc[i-window:i]
        # drop cols with too many missing values
        col_mask = X.notna().mean(axis=0) >= 0.8
        X = X.loc[:, col_mask].dropna()
        if X.empty:
            continue
        ww = w_all[col_mask]
        ww = ww / ww.sum()
        mu = X.mean().values
        S = X.cov().values
        mp = float(ww.values @ mu)
        sp = float(np.sqrt(multi_dot([ww.values, S, ww.values])))
        out[X.index[-1]] = V * (z * sp - mp)
    return pd.Series(out, dtype=float)

def historical_var_series(Rp: pd.Series, window: int, alpha: float, V: float) -> pd.Series:
    out = {}
    for i in range(window, len(Rp)):
        w = Rp.iloc[i-window:i].dropna().values
        if len(w) < int(0.8 * window):
            continue
        q = np.quantile(w, 1 - alpha)
        out[Rp.index[i]] = -V * q
    return pd.Series(out, dtype=float)

def marginal_var_series(R: pd.DataFrame, w: pd.Series, window: int, alpha: float, V: float, names: List[str]) -> Dict[str, pd.Series]:
    z = norm.ppf(alpha)
    w_all = w.reindex(R.columns).astype(float)
    series = {n: {} for n in names}
    for i in range(window, len(R)):
        X = R.iloc[i-window:i]
        col_mask = X.notna().mean(axis=0) >= 0.8
        X = X.loc[:, col_mask].dropna()
        if X.empty:
            continue
        cols = list(X.columns)
        ww = w_all[col_mask]
        ww = ww / ww.sum()
        S = X.cov().values
        s = float(np.sqrt(ww.values @ S @ ww.values))
        g = S @ ww.values
        t = X.index[-1]
        for n in names:
            if n in cols:
                j = cols.index(n)
                series[n][t] = V * z * g[j] / s
    return {k: pd.Series(v, dtype=float) for k, v in series.items()}

def bootstrap_ci_param_and_hs(R: pd.DataFrame, w: pd.Series, Rp: pd.Series, alpha: float, V: float, window: int, dates: pd.DatetimeIndex, B: int, rng: np.random.Generator) -> Tuple[pd.DataFrame, pd.DataFrame]:
    z = norm.ppf(alpha)
    w_all = w.reindex(R.columns).astype(float)
    ci_param = {"lo": {}, "hi": {}}
    ci_hs = {"lo": {}, "hi": {}}
    for t in dates:
        j = R.index.get_loc(t)
        X = R.iloc[j-window:j].dropna()
        if X.shape[0] < window:
            continue
        # align weights to available columns
        ww = w_all.reindex(X.columns).astype(float)
        ww = ww / ww.sum()
        Rp_win = (X.values @ ww.values).astype(float)
        n = len(Rp_win)
        boot_param = np.empty(B)
        boot_hs = np.empty(B)
        for b in range(B):
            idxb = rng.integers(0, n, size=n)
            Xb = X.values[idxb, :]
            mub = Xb.mean(axis=0)
            Sb = np.cov(Xb, rowvar=False, ddof=1)
            mp = float(ww.values @ mub)
            sp = float(np.sqrt(ww.values @ Sb @ ww.values))
            boot_param[b] = V * (z * sp - mp)
            boot_hs[b] = -V * np.quantile(Rp_win[idxb], 1 - alpha)
        ci_param["lo"][t] = np.quantile(boot_param, 0.025)
        ci_param["hi"][t] = np.quantile(boot_param, 0.975)
        ci_hs["lo"][t] = np.quantile(boot_hs, 0.025)
        ci_hs["hi"][t] = np.quantile(boot_hs, 0.975)
    return pd.DataFrame(ci_param), pd.DataFrame(ci_hs)
