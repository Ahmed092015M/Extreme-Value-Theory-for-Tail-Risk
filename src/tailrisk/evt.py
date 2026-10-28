"""
Extreme value theory: tail index (Hill), peaks over threshold (GPD), uncertainty and clustering.

Sections
--------
1. Hill estimator          hill, hill_var_es                              (Steps 3.1 - 3.4, Eq. 6-9)
2. Choosing the threshold  mean_excess, threshold_stability               (Steps 4.1 - 4.5, Eq. 15)
3. GPD fit                 gpd_negloglik, fit_gpd                         (Step 4.3, Eq. 10-11)
4. GPD VaR / ES            gpd_var_es                                     (Step 5.1, Eq. 12-14)
5. Uncertainty             gpd_var_ci, gpd_var_profile_ci,
                           bootstrap_var_es                               (Section 6, Eq. 20-22)
6. Clustering              extremal_index_runs                            (Step 7.1, Eq. 23)

Convention: x are losses (positive = loss); the tail studied is the upper tail of x.
The gain tail is analysed by passing -x.
"""
import numpy as np
import pandas as pd
from scipy import optimize, stats

__all__ = ["hill", "hill_var_es", "mean_excess", "gpd_negloglik", "fit_gpd", "threshold_stability",
           "gpd_var_es", "gpd_var_ci", "gpd_var_profile_ci", "bootstrap_var_es", "extremal_index_runs"]


# =========================================================================== #
# 1. Hill estimator                                                           #
# =========================================================================== #
def hill(x, k):
    """Hill estimator of the tail index xi from the k largest values (Equation 6), with its standard error (Equation 7).

    xi_k = (1/k) sum_{i=1..k} [ln X_(i) - ln X_(k+1)], computed for every k at once with a cumulative sum.

    Parameters
    ----------
    x : array
        Observations (losses); only the largest k + 1 are used, and they must be positive.
    k : int or array of int
        Number of tail observations; the threshold is the (k+1)-th largest value.

    Returns
    -------
    xi : float or np.ndarray
        Estimate(s) of xi (alpha = 1 / xi).
    se : float or np.ndarray
        Asymptotic standard error(s), xi / sqrt(k).
    """
    k = np.asarray(k, dtype=int)
    k_max = int(k.max())
    top = np.sort(np.asarray(x, dtype=float))[::-1][:k_max + 1]        # the k_max + 1 largest, descending
    if top[-1] <= 0:
        raise ValueError("the (k+1)-th largest value must be positive")
    logs = np.log(top)
    mean_top = np.cumsum(logs)[:-1] / np.arange(1, k_max + 1)          # mean of the first k logs, k = 1..k_max
    xi_all = mean_top - logs[1:]                                       # minus log of the (k+1)-th
    xi = xi_all[k - 1]
    return xi, xi / np.sqrt(k)


def hill_var_es(x, k, q):
    """VaR and ES from the Hill estimate (Equations 8-9): Weissman extrapolation from the (k+1)-th largest value.

    VaR_q = X_(k+1) (k / (n (1 - q)))^xi;  ES_q = VaR_q / (1 - xi).

    Parameters
    ----------
    x : array
        Observations (losses).
    k : int
        Number of tail observations used by Hill.
    q : float or array
        Confidence level(s), e.g. 0.999; the tail probability 1 - q should be below k / n.

    Returns
    -------
    var, es : float or np.ndarray
        VaR and ES at level q (same units as x); ES is infinite if xi >= 1.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    q = np.asarray(q, dtype=float)
    xi, _ = hill(x, k)
    threshold = np.sort(x)[::-1][k]                        # X_(k+1)
    var = threshold * (k / (n * (1 - q))) ** xi
    es = var / (1 - xi) if xi < 1 else np.full_like(var, np.inf)
    return var, es


# =========================================================================== #
# 2. Choosing the threshold                                                   #
# =========================================================================== #
def mean_excess(x, thresholds):
    """Empirical mean excess function e(u) = mean of (x - u) over x > u (Equation 15), with its standard error.

    For a GPD tail, e(u) = (beta + xi u) / (1 - xi) is a straight line in u.

    Parameters
    ----------
    x : array
        Observations (losses).
    thresholds : array
        Thresholds u.

    Returns
    -------
    pd.DataFrame
        Index = u; columns 'n above', 'mean excess', 'se'.
    """
    x = np.asarray(x, dtype=float)
    rows = []
    for u in np.asarray(thresholds, dtype=float):
        y = x[x > u] - u
        n_u = len(y)
        se = y.std(ddof=1) / np.sqrt(n_u) if n_u > 1 else np.nan
        rows.append({"u": u, "n above": n_u, "mean excess": y.mean() if n_u else np.nan, "se": se})
    return pd.DataFrame(rows).set_index("u")


def threshold_stability(x, thresholds):
    """GPD fits over a grid of thresholds: xi and the modified scale beta* = beta - xi u, with standard errors.

    For a valid threshold u0, xi and beta* are constant for every u > u0 (up to noise).

    Parameters
    ----------
    x : array
        Observations (losses).
    thresholds : array
        Thresholds u (round them, e.g. np.round(np.arange(...), 1), to use them as labels).

    Returns
    -------
    pd.DataFrame
        Index = u; columns 'N_u', 'xi', 'se_xi', 'beta', 'beta_star', 'se_beta_star', 'converged'.
    """
    x = np.asarray(x, dtype=float)
    rows = []
    for u in np.asarray(thresholds, dtype=float):
        f = fit_gpd(x[x > u] - u)
        c = f["cov"]
        var_bs = c[1, 1] + u ** 2 * c[0, 0] - 2 * u * c[0, 1]          # delta method for beta - xi u
        rows.append({"u": u, "N_u": f["n"], "xi": f["xi"], "se_xi": f["se_xi"], "beta": f["beta"],
                     "beta_star": f["beta"] - f["xi"] * u, "se_beta_star": np.sqrt(var_bs),
                     "converged": f["converged"]})
    return pd.DataFrame(rows).set_index("u")


# =========================================================================== #
# 3. GPD fit                                                                  #
# =========================================================================== #
def gpd_negloglik(params, y):
    """Negative GPD log-likelihood of the excesses y (Equation 11); +inf outside the valid region."""
    xi, beta = params
    if beta <= 0:
        return np.inf
    z = 1 + xi * y / beta
    if np.any(z <= 0):
        return np.inf
    if abs(xi) < 1e-8:                                         # exponential limit (xi -> 0)
        return len(y) * np.log(beta) + y.sum() / beta
    return len(y) * np.log(beta) + (1 + 1 / xi) * np.log(z).sum()


def _hessian(f, p, rel=1e-4):
    """Numerical Hessian of f at p by central differences (step relative to each parameter)."""
    p = np.asarray(p, dtype=float)
    h = rel * np.maximum(np.abs(p), 1e-2)
    k = len(p)
    H = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            e_i, e_j = np.eye(k)[i] * h[i], np.eye(k)[j] * h[j]
            H[i, j] = (f(p + e_i + e_j) - f(p + e_i - e_j) - f(p - e_i + e_j) + f(p - e_i - e_j)) / (4 * h[i] * h[j])
    return H


def fit_gpd(y):
    """Maximum-likelihood fit of the GPD to excesses y > 0, with standard errors from the inverse Hessian.

    Nelder-Mead from method-of-moments starting values; points outside the valid region have
    -loglik = +inf (the "invalid value" warnings this creates inside scipy are silenced: they are harmless).

    Parameters
    ----------
    y : array
        Excesses over the threshold (x - u for x > u).

    Returns
    -------
    dict
        'xi', 'beta', 'se_xi', 'se_beta', 'cov' (2 x 2 covariance of (xi, beta)), 'loglik', 'n', 'converged'.
    """
    y = np.asarray(y, dtype=float)
    m, v = y.mean(), y.var()
    start = [0.5 * (1 - m ** 2 / v), 0.5 * m * (1 + m ** 2 / v)]           # method of moments
    with np.errstate(invalid="ignore"):                                    # inf - inf inside Nelder-Mead
        res = optimize.minimize(gpd_negloglik, start, args=(y,), method="Nelder-Mead",
                                options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000})
    cov = np.linalg.inv(_hessian(lambda p: gpd_negloglik(p, y), res.x))
    se = np.sqrt(np.diag(cov))
    return {"xi": res.x[0], "beta": res.x[1], "se_xi": se[0], "se_beta": se[1], "cov": cov,
            "loglik": -res.fun, "n": len(y), "converged": bool(res.success)}


# =========================================================================== #
# 4. GPD VaR / ES                                                             #
# =========================================================================== #
def gpd_var_es(u, xi, beta, n, n_u, q):
    """VaR and ES from the GPD tail (Equations 13-14).

    VaR_q = u + (beta / xi) [((n / N_u)(1 - q))^(-xi) - 1];  ES_q = (VaR_q + beta - xi u) / (1 - xi).

    Parameters
    ----------
    u : float
        Threshold.
    xi, beta : float
        GPD shape and scale fitted to the excesses over u.
    n, n_u : int
        Sample size and number of observations above u.
    q : float or array
        Confidence level(s); 1 - q must be below n_u / n.

    Returns
    -------
    var, es : float or np.ndarray
        VaR and ES at level q; ES is infinite if xi >= 1.
    """
    q = np.asarray(q, dtype=float)
    ratio = (n / n_u) * (1 - q)                                # P(X > VaR) / P(X > u)
    if abs(xi) < 1e-8:                                         # exponential limit
        var = u - beta * np.log(ratio)
        return var, var + beta
    var = u + beta / xi * (ratio ** (-xi) - 1)
    es = (var + beta - xi * u) / (1 - xi) if xi < 1 else np.full_like(var, np.inf)
    return var, es


# =========================================================================== #
# 5. Uncertainty                                                              #
# =========================================================================== #
def gpd_var_ci(u, fit, n, q, level=0.95, measure="var"):
    """Delta-method confidence interval for the GPD VaR or ES (Equation 20); N_u / n treated as known.

    Parameters
    ----------
    u : float
        Threshold.
    fit : dict
        Output of ``fit_gpd`` (uses 'xi', 'beta', 'cov', 'n').
    n : int
        Sample size.
    q : float or array
        Confidence level(s) of the VaR / ES.
    level : float
        Confidence level of the interval (0.95).
    measure : str
        "var" or "es".

    Returns
    -------
    pd.DataFrame
        Index = q; columns 'estimate', 'se', 'low', 'high'.
    """
    idx = 0 if measure == "var" else 1
    p0 = np.array([fit["xi"], fit["beta"]])
    h = 1e-5 * np.maximum(np.abs(p0), 1e-2)
    z = stats.norm.ppf(0.5 + level / 2)
    rows = []
    for qq in np.atleast_1d(q):
        f = lambda p: float(gpd_var_es(u, p[0], p[1], n, fit["n"], qq)[idx])
        grad = np.array([(f(p0 + h * e) - f(p0 - h * e)) / (2 * h[i]) for i, e in enumerate(np.eye(2))])
        est, se = f(p0), float(np.sqrt(grad @ fit["cov"] @ grad))
        rows.append({"q": qq, "estimate": est, "se": se, "low": est - z * se, "high": est + z * se})
    return pd.DataFrame(rows).set_index("q")


def _g(xi, r):
    """(r^-xi - 1) / xi, with its limit -ln r at xi = 0."""
    return -np.log(r) if abs(xi) < 1e-8 else (r ** -xi - 1) / xi


def gpd_var_profile_ci(y, u, n, q, level=0.95):
    """Profile-likelihood confidence interval for the GPD VaR at level q (Equation 21).

    The GPD is re-parametrised by (xi, VaR): beta = (VaR - u) / g(xi, r) with r = (n / N_u)(1 - q);
    for each VaR the likelihood is maximised over xi, and the interval keeps the VaR values with
    2 (loglik_max - profile) <= chi2_1 quantile. The interval can be asymmetric.

    Parameters
    ----------
    y : array
        Excesses over the threshold u.
    u : float
        Threshold.
    n : int
        Sample size.
    q : float
        VaR level (1 - q must be below N_u / n).
    level : float
        Confidence level of the interval.

    Returns
    -------
    dict
        'estimate', 'low', 'high', 'loglik_max', 'profile_at_estimate'.
    """
    y = np.asarray(y, dtype=float)
    fit = fit_gpd(y)
    r = (n / len(y)) * (1 - q)
    v_hat = u + fit["beta"] * _g(fit["xi"], r)

    def profile(v):
        def nll(xi):
            val = gpd_negloglik([xi, (v - u) / _g(xi, r)], y)
            return val if np.isfinite(val) else 1e12
        return -optimize.minimize_scalar(nll, bounds=(-0.45, 1.5), method="bounded",
                                         options={"xatol": 1e-8}).fun

    target = fit["loglik"] - stats.chi2.ppf(level, 1) / 2
    lo = optimize.brentq(lambda v: profile(v) - target, u + 0.01 * (v_hat - u), v_hat)
    b = v_hat * 1.5
    while profile(b) > target:                                         # widen until outside the interval
        b *= 1.5
    hi = optimize.brentq(lambda v: profile(v) - target, v_hat, b)
    return {"estimate": v_hat, "low": lo, "high": hi, "loglik_max": fit["loglik"],
            "profile_at_estimate": profile(v_hat)}


def bootstrap_var_es(x, u, q, n_boot=500, block=20, seed=0):
    """Moving-block bootstrap of the GPD tail: resample the losses in blocks, refit above u, compute VaR and ES.

    Parameters
    ----------
    x : array
        Losses in time order.
    u : float
        Threshold (kept fixed; the number of excesses varies across samples).
    q : array
        VaR / ES levels.
    n_boot : int
        Number of bootstrap samples.
    block : int
        Block length in days (1 = ordinary i.i.d. bootstrap).
    seed : int
        Random seed.

    Returns
    -------
    dict of np.ndarray
        'xi' (n_boot,), 'n_u' (n_boot,), 'var' and 'es' (n_boot, len(q)).
    """
    x = np.asarray(x, dtype=float)
    q = np.atleast_1d(np.asarray(q, dtype=float))
    n = len(x)
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    out = {"xi": np.empty(n_boot), "n_u": np.empty(n_boot, dtype=int),
           "var": np.empty((n_boot, len(q))), "es": np.empty((n_boot, len(q)))}
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]          # consecutive days inside each block
        s = x[idx]
        y = s[s > u] - u
        f = fit_gpd(y)
        v, e = gpd_var_es(u, f["xi"], f["beta"], n, len(y), q)
        out["xi"][b] = f["xi"]
        out["n_u"][b] = len(y)
        out["var"][b] = v
        out["es"][b] = e
    return out


# =========================================================================== #
# 6. Clustering                                                               #
# =========================================================================== #
def extremal_index_runs(x, u, r=5):
    """Runs estimator of the extremal index (Equation 23): clusters per exceedance of u.

    A new cluster starts when more than r days separate two exceedances. Even independent data give
    theta about (1 - p)^r with this estimator, so compare with shuffled data.

    Parameters
    ----------
    x : array
        Observations in time order.
    u : float
        Threshold.
    r : int
        Run length (days below u that end a cluster).

    Returns
    -------
    dict
        'theta', 'n_exceed', 'n_clusters', 'mean_cluster_size' (= 1 / theta).
    """
    pos = np.flatnonzero(np.asarray(x, dtype=float) > u)          # days with an exceedance
    if len(pos) == 0:
        return {"theta": np.nan, "n_exceed": 0, "n_clusters": 0, "mean_cluster_size": np.nan}
    n_clusters = 1 + int(np.sum(np.diff(pos) > r))                 # gap of more than r days -> new cluster
    theta = n_clusters / len(pos)
    return {"theta": theta, "n_exceed": len(pos), "n_clusters": n_clusters, "mean_cluster_size": 1 / theta}
