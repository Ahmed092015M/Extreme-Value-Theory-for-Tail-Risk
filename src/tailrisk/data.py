"""
Market data: the S&P 500 index since 1927 (Yahoo Finance), data-error detection and daily losses.

Sections
--------
1. Download            load_yahoo                          (Step 1.2)
2. Data errors         BAD_CLOSES, suspect_reversals,
                       clean_closes                        (Steps 1.4 - 1.5, 7.4 - 7.6)
3. Losses              losses                              (Step 1.6, Equation 1)
"""
import numpy as np
import pandas as pd

__all__ = ["BAD_CLOSES", "load_yahoo", "suspect_reversals", "clean_closes", "losses"]

# Closes identified as data errors: a 6-8 sigma move in a calm market, fully reversed the next day,
# with both neighbouring closes consistent. Steps 1.4 (moves > 5%) and 7.5 (second pass, moves > 3%).
BAD_CLOSES = ["1935-04-16", "1935-08-16", "1936-09-16", "1961-04-17"]


# =========================================================================== #
# 1. Download                                                                 #
# =========================================================================== #
def load_yahoo(ticker="^GSPC", start="1927-01-01"):
    """Download the daily close of one Yahoo Finance ticker as a pd.Series.

    ``yfinance`` is imported inside the function so that the rest of the package
    (and the unit tests) work without it.

    Parameters
    ----------
    ticker : str
        Yahoo ticker, e.g. "^GSPC" (S&P 500 price index).
    start : str
        First date requested (YYYY-MM-DD); Yahoo's ^GSPC history begins on 1927-12-30.

    Returns
    -------
    pd.Series
        Index = date (no time zone), name = ticker; closing level, no missing values.
    """
    import yfinance as yf

    df = yf.download(ticker, start=start, auto_adjust=False, progress=False)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):          # newer yfinance: one column per ticker
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.index.name = "date"
    return close.dropna().astype(float).rename(ticker)


# =========================================================================== #
# 2. Data errors                                                              #
# =========================================================================== #
def suspect_reversals(close, min_move=0.05, min_undo=0.8, window=60, min_z=6.0):
    """Large daily moves that are almost reversed the next day, with their size in units of recent volatility.

    A data error (a mistyped close) is large compared with the volatility of the time and is undone
    the next day; a real crisis move happens when volatility is already high and is not undone.

    Parameters
    ----------
    close : pd.Series
        Daily closes.
    min_move : float
        Smallest absolute return considered (0.05 = 5%).
    min_undo : float
        The next day must undo at least this share of the move (0.8 = 80%), in the opposite direction.
    window : int
        Days of past returns used for the volatility before the move.
    min_z : float
        A reversed move larger than ``min_z`` times the past volatility is flagged as suspect.

    Returns
    -------
    pd.DataFrame
        One row per reversed move: 'return %', 'next day %', 'vol before %' (daily),
        'z' and 'z next' (moves in units of that volatility), 'suspect' (True / False).
    """
    r = close.pct_change()
    vol = r.rolling(window).std().shift(1)                  # volatility up to the day before
    nxt = r.shift(-1)
    rev = (r.abs() > min_move) & (np.sign(nxt) == -np.sign(r)) & (nxt.abs() > min_undo * r.abs())
    out = pd.DataFrame({"return %": 100 * r, "next day %": 100 * nxt, "vol before %": 100 * vol,
                        "z": r / vol, "z next": nxt / vol})[rev]
    out["suspect"] = out["z"].abs() > min_z
    return out


def clean_closes(close, bad=None):
    """Remove closes identified as data errors; the return over the gap becomes one correct return.

    Parameters
    ----------
    close : pd.Series
        Daily closes.
    bad : list of str, optional
        Dates of the erroneous closes (default: BAD_CLOSES, read when the function is called,
        not when it is defined).

    Returns
    -------
    pd.Series
        The closes without the erroneous days.
    """
    bad = pd.to_datetime(BAD_CLOSES if bad is None else bad)
    missing = bad.difference(close.index)
    if len(missing):
        raise KeyError(f"dates not in the series: {list(missing.date)}")
    return close.drop(bad)


# =========================================================================== #
# 3. Losses                                                                   #
# =========================================================================== #
def losses(close):
    """Daily losses in %: x_t = -100 ln(S_t / S_{t-1}) (Equation 1); positive = the index fell.

    Parameters
    ----------
    close : pd.Series
        Daily closes.

    Returns
    -------
    pd.Series
        Index = date (from the second close), name 'loss'; the gains of a long position are -loss.
    """
    return (-100 * np.log(close).diff()).dropna().rename("loss")
