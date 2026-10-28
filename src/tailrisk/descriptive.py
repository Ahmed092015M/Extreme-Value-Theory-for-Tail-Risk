"""
Descriptive statistics of the tails, and a test of remaining volatility clustering.

Sections
--------
1. Tails by era        ERAS, tail_stats, tail_table     (Step 2.1)
2. Autocorrelation     ljung_box                         (Step 7.3)
"""
import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["ERAS", "tail_stats", "tail_table", "ljung_box"]

ERAS = {"1928-1945": ("1928", "1945"), "1946-1986": ("1946", "1986"), "1987-2026": ("1987", "2026")}


# =========================================================================== #
# 1. Tails by era                                                             #
# =========================================================================== #
def tail_stats(x):
    """Tail statistics of one series of losses (positive = loss), compared with a normal of the same mean and std.

    Parameters
    ----------
    x : pd.Series or array
        Daily losses in %.

    Returns
    -------
    dict
        'days', 'std %', 'skew', 'excess kurtosis', 'q99 %', 'q99.9 %',
        'q99 / normal', 'q99.9 / normal' (centred quantile / normal quantile with the same std),
        'days > 5 sd', 'normal days > 5 sd' (expected under the normal), 'max (sd)'.
    """
    x = np.asarray(x, dtype=float)
    m, s = x.mean(), x.std(ddof=1)
    q99, q999 = np.quantile(x, [0.99, 0.999])
    return {"days": len(x), "std %": s, "skew": stats.skew(x), "excess kurtosis": stats.kurtosis(x),
            "q99 %": q99, "q99.9 %": q999,
            "q99 / normal": (q99 - m) / (s * stats.norm.ppf(0.99)),
            "q99.9 / normal": (q999 - m) / (s * stats.norm.ppf(0.999)),
            "days > 5 sd": int(((x - m) / s > 5).sum()),
            "normal days > 5 sd": len(x) * stats.norm.sf(5),
            "max (sd)": (x.max() - m) / s}


def tail_table(x, eras=ERAS):
    """``tail_stats`` of the loss tail and the gain tail, for the full sample and each era.

    Parameters
    ----------
    x : pd.Series
        Daily losses in %, indexed by date.
    eras : dict of {name: (start, end)}
        Sub-periods.

    Returns
    -------
    pd.DataFrame
        Index = (period, tail) with tail 'loss' or 'gain'; columns as in ``tail_stats``.
    """
    periods = {"full sample": (None, None), **eras}
    rows = {}
    for name, (a, b) in periods.items():
        sub = x.loc[a:b]
        rows[(name, "loss")] = tail_stats(sub)
        rows[(name, "gain")] = tail_stats(-sub)              # the gains are the losses of a short position
    return pd.DataFrame(rows).T


# =========================================================================== #
# 2. Autocorrelation                                                          #
# =========================================================================== #
def ljung_box(x, lags=20):
    """Ljung-Box test of no autocorrelation up to `lags`: Q = n(n+2) sum rho_k^2 / (n - k) ~ chi2(lags).

    Applied to squared losses / residuals, it tests for remaining volatility clustering.

    Returns
    -------
    (float, float)
        Q statistic and p-value.
    """
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    denom = np.sum(x ** 2)
    rho = np.array([np.sum(x[k:] * x[:-k]) / denom for k in range(1, lags + 1)])
    q = n * (n + 2) * np.sum(rho ** 2 / (n - np.arange(1, lags + 1)))
    return q, stats.chi2.sf(q, lags)
