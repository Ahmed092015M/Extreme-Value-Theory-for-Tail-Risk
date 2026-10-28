"""
Unit tests for the tailrisk package. Run from the repository root:  pytest -q

Every test uses synthetic data with a known answer, so no download is needed.
Groups: data, Hill, GPD, uncertainty, clustering, benchmarks, GARCH, engine (look-ahead), backtests, plots.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tailrisk import backtest as bt
from tailrisk import benchmarks as bm
from tailrisk import data, descriptive, engine, evt, garch, plots


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture
def pareto():
    """50,000 Pareto draws with tail index alpha = 3 (xi = 1/3), minimum 1."""
    return stats.pareto.rvs(3.0, size=50_000, random_state=np.random.default_rng(0))


@pytest.fixture
def gjr_path():
    """3,000 losses simulated from a known GJR-GARCH(1,1) with Student-t(6) shocks, as a dated Series."""
    rng = np.random.default_rng(1)
    n, mu, om, a, g, b = 3000, -0.05, 0.02, 0.03, 0.10, 0.90
    z = rng.standard_t(6, n) / np.sqrt(6 / 4)
    x, s2 = np.empty(n), np.empty(n)
    s2[0] = om / (1 - a - g / 2 - b)
    for t in range(n):
        if t:
            e = x[t - 1] - mu
            s2[t] = om + (a + g * (e > 0)) * e ** 2 + b * s2[t - 1]
        x[t] = mu + np.sqrt(s2[t]) * z[t]
    return pd.Series(x, index=pd.bdate_range("2000-01-03", periods=n), name="loss")


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #
def test_losses_sign_and_value():
    close = pd.Series([100.0, 110.0, 99.0], index=pd.bdate_range("2020-01-01", periods=3))
    x = data.losses(close)
    assert len(x) == 2
    assert x.iloc[0] == pytest.approx(-100 * np.log(1.1))          # index up -> negative loss
    assert x.iloc[1] == pytest.approx(-100 * np.log(0.9))          # index down -> positive loss


def test_clean_closes_drops_dates_and_rejects_unknown():
    idx = pd.bdate_range("1935-04-15", periods=3)                   # 15, 16, 17 April 1935
    close = pd.Series([10.0, 12.0, 10.1], index=idx)
    out = data.clean_closes(close, bad=["1935-04-16"])
    assert list(out.index) == [idx[0], idx[2]]
    with pytest.raises(KeyError):
        data.clean_closes(close, bad=["1999-01-01"])


def test_suspect_reversals_flags_spike_in_calm_market():
    rng = np.random.default_rng(2)
    r = rng.normal(0, 0.005, 200)
    r[150], r[151] = 0.06, -0.06 / 1.06                             # +6% then exactly back
    close = pd.Series(100 * np.cumprod(1 + r), index=pd.bdate_range("2000-01-03", periods=200))
    s = data.suspect_reversals(close)
    assert close.index[150] in s.index
    assert bool(s.loc[close.index[150], "suspect"])


# --------------------------------------------------------------------------- #
# Hill                                                                        #
# --------------------------------------------------------------------------- #
def test_hill_toy_example():
    x = np.array([8.0, 4.0, 2.0, 1.0])                              # logs relative to X_(3) = 2: ln4, ln2
    xi, se = evt.hill(x, 2)
    assert xi == pytest.approx((np.log(4) + np.log(2)) / 2)
    assert se == pytest.approx(xi / np.sqrt(2))


def test_hill_recovers_pareto_index(pareto):
    xi, se = evt.hill(pareto, 2000)
    assert abs(xi - 1 / 3) < 3 * se


def test_hill_vector_equals_scalar(pareto):
    ks = np.array([50, 200, 1000])
    xi_v, _ = evt.hill(pareto, ks)
    assert np.allclose(xi_v, [evt.hill(pareto, k)[0] for k in ks])


def test_hill_var_matches_pareto_quantile(pareto):
    var, es = evt.hill_var_es(pareto, 2000, 0.999)
    true = 0.001 ** (-1 / 3)                                        # Pareto(3) quantile, minimum 1
    assert var == pytest.approx(true, rel=0.08)
    assert es == pytest.approx(var / (1 - evt.hill(pareto, 2000)[0]))


# --------------------------------------------------------------------------- #
# GPD                                                                         #
# --------------------------------------------------------------------------- #
def test_fit_gpd_matches_scipy():
    y = stats.genpareto.rvs(0.25, scale=1.5, size=4000, random_state=np.random.default_rng(3))
    f = evt.fit_gpd(y)
    c, _, s = stats.genpareto.fit(y, floc=0)
    assert f["converged"]
    assert f["xi"] == pytest.approx(c, abs=2e-3)
    assert f["beta"] == pytest.approx(s, rel=2e-3)
    assert f["se_xi"] > 0 and f["se_beta"] > 0


def test_gpd_negloglik_invalid_region_and_exponential_limit():
    y = np.array([0.5, 1.0, 3.0])
    assert evt.gpd_negloglik([0.2, -1.0], y) == np.inf             # beta <= 0
    assert evt.gpd_negloglik([-0.5, 1.0], y) == np.inf             # 1 + xi y / beta <= 0
    assert evt.gpd_negloglik([0.0, 2.0], y) == pytest.approx(-stats.expon.logpdf(y, scale=2.0).sum())


def test_gpd_var_es_against_simulation():
    rng = np.random.default_rng(4)
    xi, beta, u, n, n_u = 0.2, 0.8, 2.0, 1_000_000, 50_000
    sim = u + stats.genpareto.rvs(xi, scale=beta, size=n_u, random_state=rng)
    var, es = evt.gpd_var_es(u, xi, beta, n, n_u, 0.999)
    p_cond = 1 - (1 - 0.999) * n / n_u                               # quantile level inside the tail
    assert var == pytest.approx(np.quantile(sim, p_cond), rel=0.02)
    assert es == pytest.approx(sim[sim > var].mean(), rel=0.03)


def test_gpd_var_es_exponential_limit_is_continuous():
    v0, e0 = evt.gpd_var_es(1.0, 0.0, 0.5, 10_000, 500, 0.999)
    v1, e1 = evt.gpd_var_es(1.0, 1e-6, 0.5, 10_000, 500, 0.999)
    assert v0 == pytest.approx(v1, rel=1e-5) and e0 == pytest.approx(e1, rel=1e-5)


def test_threshold_stability_beta_star_constant_for_gpd():
    y = stats.genpareto.rvs(0.2, scale=1.0, size=20_000, random_state=np.random.default_rng(5))
    st = evt.threshold_stability(y, [0.0, 0.5, 1.0])
    assert np.all(np.abs(st["xi"] - 0.2) < 3 * st["se_xi"])
    assert np.all(np.abs(st["beta_star"] - 1.0) < 3 * st["se_beta_star"])


def test_mean_excess_exponential_is_flat():
    x = stats.expon.rvs(scale=2.0, size=200_000, random_state=np.random.default_rng(6))
    me = evt.mean_excess(x, [0.0, 1.0, 2.0])
    assert np.allclose(me["mean excess"], 2.0, rtol=0.03)
    assert me["n above"].is_monotonic_decreasing


# --------------------------------------------------------------------------- #
# Uncertainty                                                                 #
# --------------------------------------------------------------------------- #
@pytest.fixture
def gpd_sample():
    """2,000 losses with a GPD tail above u = 1: 10% exceed u."""
    rng = np.random.default_rng(7)
    body = rng.uniform(0, 1, 1800)
    tail = 1 + stats.genpareto.rvs(0.25, scale=0.6, size=200, random_state=rng)
    return np.concatenate([body, tail])


def test_delta_ci_matches_analytic_gradient(gpd_sample):
    u, n = 1.0, len(gpd_sample)
    y = gpd_sample[gpd_sample > u] - u
    f = evt.fit_gpd(y)
    ci = evt.gpd_var_ci(u, f, n, [0.999])
    xi, b, r = f["xi"], f["beta"], (n / f["n"]) * 0.001
    grad = np.array([b * (-(r ** -xi) * np.log(r) / xi - (r ** -xi - 1) / xi ** 2), (r ** -xi - 1) / xi])
    assert ci["se"].iloc[0] == pytest.approx(np.sqrt(grad @ f["cov"] @ grad), rel=1e-4)
    assert ci["low"].iloc[0] < ci["estimate"].iloc[0] < ci["high"].iloc[0]


def test_profile_ci_contains_estimate_and_is_right_skewed(gpd_sample):
    u, n = 1.0, len(gpd_sample)
    y = gpd_sample[gpd_sample > u] - u
    p = evt.gpd_var_profile_ci(y, u, n, 0.999)
    assert p["low"] < p["estimate"] < p["high"]
    assert p["profile_at_estimate"] == pytest.approx(p["loglik_max"], abs=1e-3)
    assert p["high"] - p["estimate"] > p["estimate"] - p["low"]        # longer upper side


def test_bootstrap_shapes_and_reproducibility(gpd_sample):
    a = evt.bootstrap_var_es(gpd_sample, 1.0, [0.99, 0.999], n_boot=15, block=10, seed=3)
    b = evt.bootstrap_var_es(gpd_sample, 1.0, [0.99, 0.999], n_boot=15, block=10, seed=3)
    assert a["var"].shape == (15, 2) and a["es"].shape == (15, 2)
    assert np.array_equal(a["var"], b["var"])
    assert np.all(a["es"] > a["var"])


# --------------------------------------------------------------------------- #
# Clustering                                                                  #
# --------------------------------------------------------------------------- #
def test_extremal_index_toy_and_iid():
    x = np.zeros(30)
    x[[2, 3, 4, 20]] = 5.0                                           # one cluster of 3, one of 1
    r = evt.extremal_index_runs(x, 1.0, r=5)
    assert r["n_clusters"] == 2 and r["theta"] == pytest.approx(0.5)
    iid = np.random.default_rng(8).standard_normal(200_000)
    u = np.quantile(iid, 0.99)
    assert evt.extremal_index_runs(iid, u, r=5)["theta"] == pytest.approx(0.99 ** 5, abs=0.02)


# --------------------------------------------------------------------------- #
# Benchmarks                                                                  #
# --------------------------------------------------------------------------- #
def test_fit_t_matches_scipy():
    x = stats.t.rvs(4, loc=0.1, scale=0.8, size=5000, random_state=np.random.default_rng(9))
    f = bm.fit_t(x)
    nu, loc, sc = stats.t.fit(x)
    assert f["nu"] == pytest.approx(nu, rel=0.02)
    assert f["scale"] == pytest.approx(sc, rel=0.01)
    assert f["mu"] == pytest.approx(loc, abs=0.01)


def test_normal_and_t_var_es_closed_forms():
    v, e = bm.var_es_normal(0, 1, 0.99)
    assert v == pytest.approx(2.326348, rel=1e-6) and e == pytest.approx(2.665214, rel=1e-6)
    sims = stats.t.rvs(5, size=2_000_000, random_state=np.random.default_rng(10))
    vt, et = bm.var_es_t(0, 1, 5, 0.99)
    assert vt == pytest.approx(stats.t.ppf(0.99, 5))
    assert et == pytest.approx(sims[sims > vt].mean(), rel=0.01)


def test_var_es_hist():
    x = np.arange(1, 1001, dtype=float)
    v, e = bm.var_es_hist(x, [0.99])
    assert v[0] == pytest.approx(np.quantile(x, 0.99))
    assert e[0] == pytest.approx(x[x > v[0]].mean())


# --------------------------------------------------------------------------- #
# GARCH and descriptive                                                       #
# --------------------------------------------------------------------------- #
def test_gjr_variance_recursion():
    x = np.array([1.0, -2.0, 3.0])
    s2 = garch.gjr_variance(x, 0.0, 0.1, 0.05, 0.1, 0.8, s2_0=1.0)
    assert s2[1] == pytest.approx(0.1 + (0.05 + 0.1) * 1.0 + 0.8 * 1.0)     # loss above mean: leverage on
    assert s2[2] == pytest.approx(0.1 + 0.05 * 4.0 + 0.8 * s2[1])           # gain: no leverage


def test_fit_gjr_recovers_parameters(gjr_path):
    f = garch.fit_gjr_qml(gjr_path.values)
    assert f["converged"]
    assert f["gamma"] > 0.03 and 0.8 < f["beta"] < 0.97
    assert f["persistence"] < 1
    z, sigma = garch.standardised_residuals(gjr_path, f)
    assert z.std() == pytest.approx(1.0, abs=0.08)
    assert (sigma > 0).all()


def test_ljung_box_iid_vs_garch(gjr_path):
    iid = np.random.default_rng(11).standard_normal(3000)
    assert descriptive.ljung_box(iid ** 2)[1] > 0.01
    assert descriptive.ljung_box(gjr_path.values ** 2)[1] < 0.01


def test_tail_stats_normal():
    s = descriptive.tail_stats(np.random.default_rng(12).standard_normal(100_000))
    assert s["q99 / normal"] == pytest.approx(1.0, abs=0.03)
    assert abs(s["excess kurtosis"]) < 0.1


# --------------------------------------------------------------------------- #
# Engine: no look-ahead                                                       #
# --------------------------------------------------------------------------- #
def test_rolling_engine_has_no_lookahead(gjr_path):
    kw = dict(window=500, refit_every=50, end=700)
    a, sa, _ = engine.rolling_tail_var(gjr_path, **kw)
    shocked = gjr_path.copy()
    shocked.iloc[650:] *= 10                                         # change the future only
    b, sb, _ = engine.rolling_tail_var(shocked, **kw)
    assert np.allclose(a.iloc[:150].values, b.iloc[:150].values)    # forecasts up to day 649 unchanged
    assert np.allclose(sa.iloc[:151], sb.iloc[:151])                # sigma for day 650 uses data to 649
    assert a.shape[1] == len(engine.METHODS) * 2 * len(engine.LEVELS)


def test_expanding_engine_has_no_lookahead(gjr_path):
    ve, sig, par = engine.rolling_tail_var(gjr_path, window=500, refit_every=50, end=800)
    z0 = np.random.default_rng(13).standard_normal(500)
    oos = gjr_path.iloc[500:800]
    a = engine.expanding_tail_var(oos, ve, sig, par, z0, refit_every=50)
    shocked = oos.copy()
    shocked.iloc[200:] *= 10
    b = engine.expanding_tail_var(shocked, ve, sig, par, z0, refit_every=50)
    assert np.allclose(a.iloc[:200].values, b.iloc[:200].values)
    assert (a[("McNeil-Frey expanding", "ES", 0.999)] > a[("McNeil-Frey expanding", "VaR", 0.999)]).all()


# --------------------------------------------------------------------------- #
# Backtests                                                                   #
# --------------------------------------------------------------------------- #
def test_kupiec_exact_rate_and_too_many():
    hits = np.zeros(1000, dtype=int)
    hits[::100] = 1                                                  # 10 breaches at p = 1%
    assert bt.kupiec_test(hits, 0.01) == (10, pytest.approx(1.0))
    hits[:40] = 1
    assert bt.kupiec_test(hits, 0.01)[1] < 0.001


def test_christoffersen_detects_clustering():
    spread = np.zeros(1000, dtype=int)
    spread[::100] = 1
    clustered = np.zeros(1000, dtype=int)
    clustered[500:510] = 1                                           # same count, all in a row
    assert bt.christoffersen_test(spread, 0.01)[1] > 0.5
    n11, p = bt.christoffersen_test(clustered, 0.01)
    assert n11 == 9 and p < 0.001


def test_binomial_pvalue_symmetric_cases():
    hits = np.zeros(1000, dtype=int)
    hits[:10] = 1
    assert bt.binomial_pvalue(hits, 0.01) == pytest.approx(1.0)
    hits[:30] = 1
    assert bt.binomial_pvalue(hits, 0.01) < 0.001


def test_z2_zero_when_correct_and_negative_when_es_too_low():
    rng = np.random.default_rng(14)
    p, x = 0.01, rng.standard_normal(500_000)
    v, e = bm.var_es_normal(0, 1, 1 - p)
    assert bt.acerbi_szekely_z2(x, v, e, p) == pytest.approx(0.0, abs=0.05)
    assert bt.acerbi_szekely_z2(2 * x, v, e, p) < -1                  # true risk twice the model


def test_z2_critical_and_backtest_table(gjr_path):
    crit, mean = bt.z2_critical(1000, 0.01, n_sim=300)
    assert crit < 0 and abs(mean) < 0.1
    ve, _, _ = engine.rolling_tail_var(gjr_path, window=1000, refit_every=100, end=2000)
    tab = bt.backtest_table(gjr_path.iloc[1000:2000], ve, ["HS", "GARCH-normal"], [0.99], {0.99: crit})
    assert list(tab.index) == [(0.99, "HS"), (0.99, "GARCH-normal")]
    assert tab["expected"].iloc[0] == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# Plots                                                                       #
# --------------------------------------------------------------------------- #
def test_plots_return_figures(gjr_path):
    import matplotlib.pyplot as plt
    x = gjr_path
    u = float(np.quantile(x, 0.95))
    f = evt.fit_gpd(x[x > u].values - u)
    stab = evt.threshold_stability(x.values, [0.5, 1.0])
    ve, _, _ = engine.rolling_tail_var(x, window=1000, refit_every=200, end=1400)
    tab = bt.backtest_table(x.iloc[1000:1400], ve, ["HS", "FHS"], [0.99, 0.999], {0.99: -0.5, 0.999: -0.5})
    figs = [plots.plot_qq(x, 5.0),
            plots.plot_hill(x.values, k_max=500),
            plots.plot_mean_excess(evt.mean_excess(x.values, np.arange(0.5, 2.5, 0.1)), fit=(0.5, 2, 0.2, 0.5)),
            plots.plot_stability(stab, hill_xi=[0.3, 0.3], u_chosen=1.0),
            plots.plot_gpd_fit(x.values, u, f["xi"], f["beta"]),
            plots.plot_return_levels({"GPD": ([1, 10], [3, 5])}, [0.5, 1.0], [2.5, 3.0], {"event": 4.0}),
            plots.plot_backtest_summary(tab, {0.99: -0.5, 0.999: -0.5}, highlight=["HS"]),
            plots.plot_var_through_crises(x.iloc[1000:1400], {"HS": ve[("HS", "VaR", 0.99)]},
                                          [("2003-11", "2004-06", "test")], "title")]
    assert all(isinstance(fg, matplotlib.figure.Figure) for fg in figs)
    plt.close("all")
