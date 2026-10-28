"""
Benchmark VaR / ES models compared with EVT: normal, Student-t and historical.

Sections
--------
1. Student-t fit       t_negloglik, fit_t                        (Step 5.2)
2. VaR / ES            var_es_normal, var_es_t, var_es_hist      (Step 5.3, Equations 17-19)
"""
import numpy as np
from scipy import optimize, special, stats

__all__ = ["t_negloglik", "fit_t", "var_es_normal", "var_es_t", "var_es_hist"]


# =========================================================================== #
# 1. Student-t fit                                                            #
# =========================================================================== #
def t_negloglik(params, x):
    """Negative log-likelihood of a Student-t with location mu, scale s and nu degrees of freedom."""
    mu, s, nu = params
    z = (x - mu) / s
    c = special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - np.log(s)
    return -np.sum(c - (nu + 1) / 2 * np.log1p(z ** 2 / nu))


def fit_t(x, nu_bounds=(2.05, 200.0)):
    """Maximum-likelihood Student-t fit (location, scale, degrees of freedom), L-BFGS-B with bounds.

    Parameters
    ----------
    x : array
        Observations (losses).
    nu_bounds : tuple
        Bounds for nu; nu > 2 so that the variance exists.

    Returns
    -------
    dict
        'mu', 'scale', 'nu', 'sd' (= scale sqrt(nu / (nu - 2))), 'loglik', 'converged'.
    """
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med))                        # robust scale (normal-consistent)
    res = optimize.minimize(t_negloglik, [med, mad, 5.0], args=(x,), method="L-BFGS-B",
                            bounds=[(None, None), (1e-6, None), nu_bounds])
    mu, s, nu = res.x
    return {"mu": mu, "scale": s, "nu": nu, "sd": s * np.sqrt(nu / (nu - 2)),
            "loglik": -res.fun, "converged": bool(res.success)}


# =========================================================================== #
# 2. VaR / ES                                                                 #
# =========================================================================== #
def var_es_normal(mu, sd, q):
    """Normal VaR and ES (Equation 17): mu + sd z_q and mu + sd phi(z_q) / (1 - q)."""
    z = stats.norm.ppf(np.asarray(q, dtype=float))
    return mu + sd * z, mu + sd * stats.norm.pdf(z) / (1 - np.asarray(q, dtype=float))


def var_es_t(mu, scale, nu, q):
    """Student-t VaR and ES for location mu, scale s and nu degrees of freedom (Equation 18; nu > 1).

    For a t with standard deviation sd, use scale = sd * sqrt((nu - 2) / nu).
    """
    q = np.asarray(q, dtype=float)
    tq = stats.t.ppf(q, nu)
    es_std = stats.t.pdf(tq, nu) / (1 - q) * (nu + tq ** 2) / (nu - 1)
    return mu + scale * tq, mu + scale * es_std


def var_es_hist(x, q):
    """Historical VaR (empirical quantile) and ES (mean of the losses beyond it) (Equation 19)."""
    x = np.asarray(x, dtype=float)
    q = np.atleast_1d(np.asarray(q, dtype=float))
    var = np.quantile(x, q)
    es = np.array([x[x > v].mean() for v in var])
    return var, es
