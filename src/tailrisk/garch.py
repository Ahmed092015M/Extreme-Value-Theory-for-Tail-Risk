"""
GJR-GARCH(1,1) for daily losses, estimated by quasi-maximum likelihood (the McNeil-Frey filter).

Sections
--------
1. Model and likelihood     gjr_variance, gjr_negloglik     (Equations 24-25)
2. Estimation               fit_gjr_qml                     (Step 7.2)
3. Residuals                standardised_residuals          (Step 7.3)

Model (losses x_t in %): x_t = mu + sigma_t z_t,
sigma2_t = omega + (alpha + gamma 1[x_{t-1} > mu]) (x_{t-1} - mu)^2 + beta sigma2_{t-1}.
The leverage term reacts to a loss above its mean, i.e. a fall of the index.
"""
import numpy as np
import pandas as pd
from scipy import optimize

__all__ = ["gjr_variance", "gjr_negloglik", "fit_gjr_qml", "standardised_residuals"]


# =========================================================================== #
# 1. Model and likelihood                                                     #
# =========================================================================== #
def gjr_variance(x, mu, omega, alpha, gamma, beta, s2_0=None):
    """Conditional variance of losses (Equation 24): leverage term on losses above the mean.

    Parameters
    ----------
    x : array
        Losses (%), in time order.
    mu, omega, alpha, gamma, beta : float
        Mean and GJR parameters.
    s2_0 : float, optional
        Variance of the first day (default: sample variance of x - mu).

    Returns
    -------
    np.ndarray
        sigma2_t for every day, using data up to t - 1.
    """
    e = np.asarray(x, dtype=float) - mu
    s2 = np.empty(len(e))
    s2[0] = e.var() if s2_0 is None else s2_0
    for t in range(1, len(e)):
        s2[t] = omega + (alpha + gamma * (e[t - 1] > 0)) * e[t - 1] ** 2 + beta * s2[t - 1]
    return s2


def gjr_negloglik(params, x):
    """Negative normal (quasi) log-likelihood of GJR-GARCH(1,1) with a constant mean (Equation 25)."""
    mu, omega, alpha, gamma, beta = params
    s2 = gjr_variance(x, mu, omega, alpha, gamma, beta)
    if np.any(s2 <= 0) or not np.all(np.isfinite(s2)):
        return 1e10
    e = np.asarray(x, dtype=float) - mu
    return 0.5 * np.sum(np.log(2 * np.pi) + np.log(s2) + e ** 2 / s2)


# =========================================================================== #
# 2. Estimation                                                               #
# =========================================================================== #
def fit_gjr_qml(x, start=None):
    """Quasi-maximum-likelihood fit of GJR-GARCH(1,1) to losses (SLSQP with bounds and constraints).

    QML (normal likelihood) estimates the variance equation consistently even with fat-tailed shocks
    (Bollerslev-Wooldridge); the tail of the shocks is left to EVT, as in McNeil and Frey (2000).

    Parameters
    ----------
    x : array
        Losses (%).
    start : sequence, optional
        Starting values (mu, omega, alpha, gamma, beta); warm starts make rolling fits fast.

    Returns
    -------
    dict
        'mu', 'omega', 'alpha', 'gamma', 'beta', 'persistence' (alpha + gamma / 2 + beta),
        'loglik', 'converged'.
    """
    x = np.asarray(x, dtype=float)
    if start is None:
        start = (x.mean(), 0.05 * x.var(), 0.02, 0.10, 0.88)
    bounds = [(None, None), (1e-8, None), (0.0, 1.0), (-1.0, 1.0), (0.0, 1.0)]
    cons = [{"type": "ineq", "fun": lambda p: 0.9999 - p[2] - p[3] / 2 - p[4]},     # stationarity
            {"type": "ineq", "fun": lambda p: p[2] + p[3]}]                          # alpha + gamma >= 0
    res = optimize.minimize(gjr_negloglik, start, args=(x,), method="SLSQP", bounds=bounds,
                            constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
    mu, omega, alpha, gamma, beta = res.x
    return {"mu": mu, "omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta,
            "persistence": alpha + gamma / 2 + beta, "loglik": -res.fun, "converged": bool(res.success)}


# =========================================================================== #
# 3. Residuals                                                                #
# =========================================================================== #
def standardised_residuals(x, fit):
    """Standardised residuals z_t = (x_t - mu) / sigma_t and the conditional volatility sigma_t.

    Parameters
    ----------
    x : pd.Series
        Losses (%), indexed by date.
    fit : dict
        Output of ``fit_gjr_qml``.

    Returns
    -------
    z, sigma : pd.Series
    """
    s2 = gjr_variance(x.values, fit["mu"], fit["omega"], fit["alpha"], fit["gamma"], fit["beta"])
    sigma = pd.Series(np.sqrt(s2), index=x.index, name="sigma")
    return ((x - fit["mu"]) / sigma).rename("z"), sigma
