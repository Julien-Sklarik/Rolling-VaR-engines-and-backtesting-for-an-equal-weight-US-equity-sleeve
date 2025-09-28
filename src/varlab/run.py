from __future__ import annotations
import sys
import os
import numpy as np
import pandas as pd

from .data import ensure_dirs, DEFAULT_UNIVERSE, get_prices, pct_returns_from_prices, adaptive_portfolio_returns
from .risk import parametric_var_series, historical_var_series, marginal_var_series, bootstrap_ci_param_and_hs
from .plotting import save_series, save_series_with_bands
from .backtest import backtest_fixed_threshold

def main():
    # args: start end forecast_start as YYYYMMDD
    if len(sys.argv) < 4:
        print("usage: python run.py START END FORECAST_START  dates as YYYYMMDD")
        sys.exit(1)

    start = sys.argv[1]
    end = sys.argv[2]
    forecast_start = pd.to_datetime(sys.argv[3])
    forecast_end = pd.to_datetime(end)

    portfolio_value = float(os.getenv("PORTFOLIO_VALUE", 10_000_000))
    alpha = float(os.getenv("ALPHA", 0.95))
    window_short = int(os.getenv("WINDOW_SHORT", 100))
    window_long = int(os.getenv("WINDOW_LONG", 500))

    tickers = os.getenv("TICKERS", ",".join(DEFAULT_UNIVERSE)).split(",")
    weights = pd.Series(1.0 / len(tickers), index=tickers, dtype=float)

    ensure_dirs()

    prices = get_prices(tickers, start, end)
    rets = pct_returns_from_prices(prices).sort_index()

    # Adaptive portfolio returns that auto renormalize when some names have gaps
    rp = adaptive_portfolio_returns(rets[tickers], weights.reindex(tickers))

    # Parametric VaR and plots
    VaR_param = parametric_var_series(rets.loc[rets.index >= forecast_start], weights, window_short, alpha, portfolio_value)
    VaR_param.to_csv("output/q2_VaR_param.csv")
    save_series(VaR_param, "VaR parametric", "USD", "figures/q2_var_param.png")

    # Historical VaR windows
    VaR_hs_100 = historical_var_series(rp.reindex(rets.index), window_short, alpha, portfolio_value)
    VaR_hs_100.to_csv("output/q3_VaR_hs_100.csv")
    save_series(VaR_hs_100, "VaR one hundred day window", "USD", "figures/q3_var_hs100.png")

    # Marginal VaR two names
    tickA = tickers[0]
    tickB = tickers[1] if len(tickers) > 1 else tickers[0]
    mvars_100 = marginal_var_series(rets, weights, window_short, alpha, portfolio_value, [tickA, tickB])
    mvars_100[tickA].to_csv("output/q4_mvar_A_100.csv")
    mvars_100[tickB].to_csv("output/q4_mvar_B_100.csv")
    from matplotlib import pyplot as plt
    plt.figure()
    mvars_100[tickA].plot(linestyle="-")
    mvars_100[tickB].plot(linestyle="--")
    plt.title(f"Marginal VaR {tickA} and {tickB} one hundred day window")
    plt.xlabel("date")
    plt.ylabel("USD")
    plt.tight_layout()
    plt.savefig("figures/q4_mvar_100.png", dpi=150)
    plt.close()

    # Bootstrap CIs for parametric and hs one hundred
    rng = np.random.default_rng(7)
    datesI = VaR_param.dropna().index.intersection(VaR_hs_100.dropna().index)
    ci_param, ci_hs100 = bootstrap_ci_param_and_hs(rets, weights, rp, alpha, portfolio_value, window_short, datesI, 200, rng)
    ci_param.to_csv("output/q5_ci_param.csv")
    ci_hs100.to_csv("output/q5_ci_hs100.csv")
    save_series_with_bands(VaR_param, ci_param, "VaR parametric with ninety five percent CI", "USD", "figures/q5_param_ci.png")
    save_series_with_bands(VaR_hs_100, ci_hs100, "VaR one hundred day window with ninety five percent CI", "USD", "figures/q5_hs100_ci.png")

    # Larger window five hundred
    VaR_hs_500 = historical_var_series(rp.reindex(rets.index), window_long, alpha, portfolio_value)
    VaR_hs_500.to_csv("output/q6_VaR_hs_500.csv")
    # simple bootstrap bands
    ci500_lo = {}
    ci500_hi = {}
    for t in VaR_hs_500.dropna().index:
        j = rp.index.get_loc(t)
        w = rp.iloc[j-window_long:j].dropna().values
        n = len(w)
        boot = np.empty(200)
        for b in range(200):
            idxb = rng.integers(0, n, size=n)
            boot[b] = -portfolio_value * np.quantile(w[idxb], 1 - alpha)
        ci500_lo[t] = np.quantile(boot, 0.025)
        ci500_hi[t] = np.quantile(boot, 0.975)
    ci_hs500 = pd.DataFrame({"lo": pd.Series(ci500_lo), "hi": pd.Series(ci500_hi)})
    ci_hs500.to_csv("output/q6_ci_hs500.csv")
    save_series_with_bands(VaR_hs_500, ci_hs500, "VaR five hundred day window with ninety five percent CI", "USD", "figures/q6_hs500_ci.png")

    # Marginal VaR five hundred
    mvars_500 = marginal_var_series(rets, weights, window_long, alpha, portfolio_value, [tickA, tickB])
    mvars_500[tickA].to_csv("output/q6_mvar_A_500.csv")
    mvars_500[tickB].to_csv("output/q6_mvar_B_500.csv")
    plt.figure()
    mvars_500[tickA].plot(linestyle="-")
    mvars_500[tickB].plot(linestyle="--")
    plt.title(f"Marginal VaR {tickA} and {tickB} five hundred day window")
    plt.xlabel("date")
    plt.ylabel("USD")
    plt.tight_layout()
    plt.savefig("figures/q6_mvar_500.png", dpi=150)
    plt.close()

    # Backtest with a fixed threshold at the last forecast date before end
    end_date = pd.to_datetime(end)
    loss = (-adaptive_portfolio_returns(rets[tickers], weights.reindex(tickers)) * portfolio_value).dropna()
    bt_param = backtest_fixed_threshold(loss, VaR_param, end_date)
    bt_h100  = backtest_fixed_threshold(loss, VaR_hs_100, end_date)
    bt_h500  = backtest_fixed_threshold(loss, VaR_hs_500, end_date)
    bt = pd.DataFrame(
        {"exceptions": [bt_param["exceptions"], bt_h100["exceptions"], bt_h500["exceptions"]],
         "rate": [bt_param["rate"], bt_h100["rate"], bt_h500["rate"]],
         "days": [bt_param["days"], bt_h100["days"], bt_h500["days"]]},
        index=["parametric", "hs_100", "hs_500"]
    )
    bt.to_csv("output/q7_backtest_fixed.csv")
    print(bt)

if __name__ == "__main__":
    main()
