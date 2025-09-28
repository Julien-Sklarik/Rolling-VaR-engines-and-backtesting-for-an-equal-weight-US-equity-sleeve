from __future__ import annotations
import os
from typing import Iterable
import numpy as np
import pandas as pd

DEFAULT_UNIVERSE = ["GME","PAG","TTC","KRTX","OLPX","RRC","FND","MANH","WSC","PSTG"]

def ensure_dirs() -> None:
    for d in ["output", "figures", ".data_cache"]:
        os.makedirs(d, exist_ok=True)

def download_prices_yf(tickers: Iterable[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(
        tickers=list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    # Normalize to a wide frame of Adj Close
    if isinstance(df.columns, pd.MultiIndex):
        prices = pd.concat({t: df[t]["Adj Close"] for t in tickers if t in df.columns.levels[0]}, axis=1)
    else:
        prices = df.rename(columns={"Adj Close": tickers[0]})
    return prices.sort_index()

def download_prices_wrds(tickers: Iterable[str], start: str, end: str) -> pd.DataFrame:
    import wrds
    db = wrds.Connection(wrds_username=os.getenv("WRDS_USERNAME", "").strip() or None)
    tickers_sql = ",".join("'" + t.replace("'", "''") + "'" for t in tickers)
    sql = f"""
    select d.date::date as date, sn.ticker as ticker, d.prc, d.ret
    from crsp.dsf as d
    inner join crsp.stocknames as sn
      on d.permno = sn.permno
     and d.date between sn.namedt and coalesce(sn.nameenddt, date '9999-12-31')
    where sn.ticker in ({tickers_sql})
      and d.date between date '{start}' and date '{end}'
    order by d.date, sn.ticker
    """
    df = db.raw_sql(sql, date_cols=["date"])
    db.close()
    df["prc"] = df["prc"].abs()
    pivot_prc = df.pivot(index="date", columns="ticker", values="prc").sort_index()
    return pivot_prc

def get_prices(tickers: Iterable[str], start: str, end: str) -> pd.DataFrame:
    if os.getenv("WRDS_USERNAME"):
        try:
            return download_prices_wrds(tickers, start, end)
        except Exception:
            pass
    return download_prices_yf(tickers, start, end)

def pct_returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().replace([np.inf, -np.inf], np.nan)

def adaptive_portfolio_returns(R: pd.DataFrame, w: pd.Series) -> pd.Series:
    out = []
    idx = []
    for t, r in R.iterrows():
        mask = r.notna()
        if not mask.any():
            out.append(np.nan); idx.append(t); continue
        w_eff = w.reindex(R.columns)[mask].astype(float)
        w_eff = w_eff / w_eff.sum()
        out.append(float(np.dot(r[mask].values, w_eff.values)))
        idx.append(t)
    return pd.Series(out, index=pd.Index(idx, name="date"), dtype=float).dropna()
