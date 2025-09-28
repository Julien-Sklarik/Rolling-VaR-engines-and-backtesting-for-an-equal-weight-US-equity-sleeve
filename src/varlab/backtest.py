from __future__ import annotations
import pandas as pd

def backtest_fixed_threshold(loss: pd.Series, var_series: pd.Series, end_date: pd.Timestamp, window_eval: int = 500) -> dict:
    end_date = pd.to_datetime(end_date)
    loss_full = (-loss).dropna()
    loss_eval = loss_full.loc[loss_full.index < end_date].iloc[-window_eval:]
    star = float(var_series.loc[var_series.index <= end_date].iloc[-1])
    exc = int((loss_eval > star).sum())
    return {"exceptions": exc, "rate": exc / len(loss_eval), "days": len(loss_eval)}
