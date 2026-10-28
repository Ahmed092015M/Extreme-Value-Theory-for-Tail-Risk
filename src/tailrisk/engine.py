"""
Out-of-sample VaR / ES engine: conditional (GJR-GARCH filtered) and unconditional methods.

Sections
--------
1. Residual tail models     LEVELS, residual_var_es             (Step 8.1)
2. Rolling engine           METHODS, rolling_tail_var           (Steps 8.2 - 8.3, Equation 26)
3. Expanding residual tail  expanding_tail_var                  (Steps 9.3 - 9.4)

No look-ahead: every forecast for day t uses losses up to t - 1 only.
"""
import numpy as np
import pandas as pd

from .benchmarks import fit_t, var_es_hist, var_es_normal, var_es_t
from .evt import fit_gpd, gpd_var_es
from .garch import fit_gjr_qml, gjr_variance

__all__ = ["LEVELS", "METHODS", "residual_var_es", "rolling_tail_var", "expanding_tail_var"]

LEVELS = (0.975, 0.99, 0.995, 0.999)
METHODS = ["McNeil-Frey (GPD)", "GARCH-normal", "GARCH-t", "FHS", "EVT unconditional", "HS"]


# =========================================================================== #
# 1. Residual tail models                                                     #
# =========================================================================== #
def residual_var_es(z, levels=LEVELS, k_frac=0.10):
    """Standardised VaR and ES of residuals z under four shock models (in units of sigma).

    Parameters
    ----------
    z : array
        Standardised residuals of one window.
    levels : sequence of float
        VaR / ES levels.
    k_frac : float
        Share of residuals in the GPD tail (McNeil-Frey: 0.10, i.e. 100 of 1,000).

    Returns
    -------
    dict of {method: (var array, es array)}
        Methods 'McNeil-Frey (GPD)', 'GARCH-normal', 'GARCH-t', 'FHS'.
    """
    z = np.asarray(z, dtype=float)
    q = np.asarray(levels, dtype=float)
    u = np.quantile(z, 1 - k_frac)
    y = z[z > u] - u
    g = fit_gpd(y)
    t = fit_t(z)
    return {"McNeil-Frey (GPD)": gpd_var_es(u, g["xi"], g["beta"], len(z), len(y), q),
            "GARCH-normal": var_es_normal(0.0, 1.0, q),
            "GARCH-t": var_es_t(t["mu"], t["scale"], t["nu"], q),
            "FHS": var_es_hist(z, q)}


# =========================================================================== #
# 2. Rolling engine                                                           #
# =========================================================================== #
def rolling_tail_var(x, window=1000, refit_every=20, levels=LEVELS, k_frac=0.10, start=None, end=None):
    """Out-of-sample daily VaR and ES of losses with six methods (Equation 26 for the conditional ones).

    Every `refit_every` days: GJR-GARCH by QML on the previous `window` losses, standardised residuals,
    residual tail models; sigma_t is updated every day with losses up to t - 1.

    Parameters
    ----------
    x : pd.Series
        Daily losses (%), indexed by date.
    window, refit_every : int
        Estimation window and days between re-estimations.
    levels : sequence of float
        VaR / ES levels.
    k_frac : float
        Share of the window in the GPD tails (0.10).
    start, end : int, optional
        First and last+1 positions forecast (default: window, len(x)).

    Returns
    -------
    var_es : pd.DataFrame
        Index = forecast date; columns (method, 'VaR' / 'ES', level).
    sigma : pd.Series
        GJR volatility forecast for each day.
    params : pd.DataFrame
        GJR estimates of each block (index = first day of the block).
    """
    xv = np.asarray(x, dtype=float)
    start = window if start is None else start
    end = len(xv) if end is None else end
    q = np.asarray(levels, dtype=float)
    res = {(m, meas, lv): np.full(end - start, np.nan) for m in METHODS for meas in ("VaR", "ES") for lv in q}
    sigma = np.full(end - start, np.nan)
    params, prev = [], None
    for j in range(start, end, refit_every):
        b_end = min(j + refit_every, end)
        w = xv[j - window:j]
        g = fit_gjr_qml(w, start=prev)
        prev = (g["mu"], g["omega"], g["alpha"], g["gamma"], g["beta"])
        s2 = gjr_variance(xv[j - window:b_end], *prev, s2_0=w.var())       # window + block
        z_w = (w - g["mu"]) / np.sqrt(s2[:window])
        sig_b = np.sqrt(s2[window:])                                       # forecasts for the block days
        u_r = np.quantile(w, 1 - k_frac)
        g_r = fit_gpd(w[w > u_r] - u_r)
        cond = residual_var_es(z_w, q, k_frac)
        unc = {"EVT unconditional": gpd_var_es(u_r, g_r["xi"], g_r["beta"], window, int((w > u_r).sum()), q),
               "HS": var_es_hist(w, q)}
        sl = slice(j - start, b_end - start)
        sigma[sl] = sig_b
        for m, (v, e) in cond.items():
            for i, lv in enumerate(q):
                res[(m, "VaR", lv)][sl] = g["mu"] + sig_b * v[i]
                res[(m, "ES", lv)][sl] = g["mu"] + sig_b * e[i]
        for m, (v, e) in unc.items():
            for i, lv in enumerate(q):
                res[(m, "VaR", lv)][sl] = v[i]
                res[(m, "ES", lv)][sl] = e[i]
        params.append({"date": x.index[j], **{k: g[k] for k in ("mu", "omega", "alpha", "gamma", "beta",
                                                                  "persistence", "converged")}})
    idx = x.index[start:end]
    out = pd.DataFrame(res, index=idx)
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["method", "measure", "level"])
    return out, pd.Series(sigma, index=idx, name="sigma"), pd.DataFrame(params).set_index("date")



# =========================================================================== #
# 3. Expanding residual tail                                                  #
# =========================================================================== #
def expanding_tail_var(x, var_es, sigma, params, z_init, levels=LEVELS, k_frac=0.10, refit_every=20):
    """McNeil-Frey and FHS with an expanding residual pool: sigma_t from the rolling GJR, the tail from all past residuals.

    Parameters
    ----------
    x : pd.Series
        Losses on the forecast days (same index as ``var_es``).
    var_es : pd.DataFrame
        Output of ``rolling_tail_var`` (used only for the index).
    sigma : pd.Series
        Rolling GJR volatility forecasts.
    params : pd.DataFrame
        Rolling GJR estimates (for mu), indexed by the first day of each block.
    z_init : array
        Residuals available before the first forecast day (the first window).
    levels, k_frac, refit_every
        As in ``rolling_tail_var``.

    Returns
    -------
    pd.DataFrame
        Columns (method, measure, level) for 'McNeil-Frey expanding' and 'FHS expanding'.
    """
    idx = var_es.index
    mu = params["mu"].reindex(idx, method="ffill").values
    sig = sigma.reindex(idx).values
    z_oos = (x.reindex(idx).values - mu) / sig                         # out-of-sample residuals
    q = np.asarray(levels, dtype=float)
    names = ["McNeil-Frey expanding", "FHS expanding"]
    res = {(m, meas, lv): np.full(len(idx), np.nan) for m in names for meas in ("VaR", "ES") for lv in q}
    for j in range(0, len(idx), refit_every):
        pool = np.concatenate([z_init, z_oos[:j]])                     # residuals before day j only
        u = np.quantile(pool, 1 - k_frac)
        g = fit_gpd(pool[pool > u] - u)
        tails = {"McNeil-Frey expanding": gpd_var_es(u, g["xi"], g["beta"], len(pool), int((pool > u).sum()), q),
                 "FHS expanding": var_es_hist(pool, q)}
        sl = slice(j, min(j + refit_every, len(idx)))
        for m, (v, e) in tails.items():
            for i, lv in enumerate(q):
                res[(m, "VaR", lv)][sl] = mu[sl] + sig[sl] * v[i]
                res[(m, "ES", lv)][sl] = mu[sl] + sig[sl] * e[i]
    out = pd.DataFrame(res, index=idx)
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["method", "measure", "level"])
    return out
