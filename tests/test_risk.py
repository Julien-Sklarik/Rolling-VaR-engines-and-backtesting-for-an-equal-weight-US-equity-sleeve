import numpy as np
import pandas as pd
from varlab.risk import parametric_var_series, historical_var_series
from varlab.data import adaptive_portfolio_returns

def test_var_shapes_and_nans():
    np.random.seed(0)
    T = 600
    N = 5
    R = pd.DataFrame(np.random.normal(0.0005, 0.02, size=(T, N)), columns=[f"A{i}" for i in range(N)])
    w = pd.Series(1.0 / N, index=R.columns)
    rp = adaptive_portfolio_returns(R, w)
    VaR_p = parametric_var_series(R, w, window=100, alpha=0.95, V=1e6)
    VaR_h = historical_var_series(rp, window=100, alpha=0.95, V=1e6)
    assert VaR_p.dropna().shape[0] > 0
    assert VaR_h.dropna().shape[0] > 0
