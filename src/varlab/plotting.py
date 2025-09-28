from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt

def save_series(ts: pd.Series, title: str, ylab: str, fname: str) -> None:
    os.makedirs("figures", exist_ok=True)
    plt.figure()
    ts.dropna().plot()
    plt.title(title)
    plt.xlabel("date")
    plt.ylabel(ylab)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

def save_series_with_bands(ts: pd.Series, bands: pd.DataFrame, title: str, ylab: str, fname: str) -> None:
    os.makedirs("figures", exist_ok=True)
    plt.figure()
    ts.dropna().plot()
    bands["lo"].dropna().plot(style=":")
    bands["hi"].dropna().plot(style=":")
    plt.title(title)
    plt.xlabel("date")
    plt.ylabel(ylab)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
