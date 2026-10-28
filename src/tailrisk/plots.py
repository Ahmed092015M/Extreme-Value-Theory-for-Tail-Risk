"""
Charts for the tail-risk project. Every function draws one chart and returns the Figure,
so the notebook can save it (fig.savefig) before showing it.

Sections
--------
1. The data        plot_qq                                         (Step 2.2)
2. Tail estimation plot_hill, plot_mean_excess, plot_stability,
                   plot_gpd_fit                                    (Steps 3.2, 4.2, 4.5, 4.6)
3. Tail risk       plot_return_levels                              (Step 5.4)
4. Backtests       plot_backtest_summary, plot_var_through_crises  (Steps 9.5 - 9.6)
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from .evt import hill

__all__ = ["plot_qq", "plot_hill", "plot_mean_excess", "plot_stability", "plot_gpd_fit",
           "plot_return_levels", "plot_backtest_summary", "plot_var_through_crises"]



def _clean(ax):
    """Remove the top and right frame lines."""
    ax.spines[["top", "right"]].set_visible(False)



# =========================================================================== #
# 1. The data                                                                 #
# =========================================================================== #
def plot_qq(x, nu, label_top=1):
    """QQ plots of standardised losses against the normal (left) and a standardised Student-t (right).

    Parameters
    ----------
    x : pd.Series
        Daily losses in %, indexed by date (both tails are shown: losses top right, gains bottom left).
    nu : float
        Degrees of freedom of the Student-t, scaled to variance 1.
    label_top : int
        Number of largest losses labelled with their date.

    Returns
    -------
    matplotlib.figure.Figure
    """
    z = ((x - x.mean()) / x.std()).sort_values()
    n = len(z)
    p = (np.arange(1, n + 1) - 0.5) / n                     # plotting positions
    theo = {"Normal": stats.norm.ppf(p),
            f"Student-t (nu = {nu:.1f}, variance 1)": stats.t.ppf(p, nu) * np.sqrt((nu - 2) / nu)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, (name, q) in zip(axes, theo.items()):
        ax.scatter(q, z.values, s=4, color="#1f77b4")
        lim = max(abs(z).max(), abs(q).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], color="black", lw=0.8)
        for i in range(1, label_top + 1):
            ax.annotate(z.index[-i].date(), (q[-i], z.values[-i]), xytext=(-70, -4),
                        textcoords="offset points", fontsize=8)
        ax.set_xlabel(f"{name} quantile")
        ax.set_ylabel("standardised loss")
        ax.set_title(f"Losses vs {name.split(' (')[0]}", fontsize=11)
        _clean(ax)
    fig.tight_layout()
    return fig



# =========================================================================== #
# 2. Tail estimation                                                          #
# =========================================================================== #
def plot_hill(x, k_max=2500, k_min=20, k_marks=(100, 250, 500)):
    """Hill plot of the loss tail and the gain tail: xi_k with its 95% band, for k = k_min..k_max.

    Parameters
    ----------
    x : pd.Series or array
        Daily losses in % (the gain tail is -x).
    k_max, k_min : int
        Range of tail sizes shown.
    k_marks : tuple of int
        Values of k marked with a vertical line.

    Returns
    -------
    matplotlib.figure.Figure
    """
    ks = np.arange(k_min, k_max + 1)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for data, label, color in [(x, "losses", "#d62728"), (-np.asarray(x), "gains", "#1f77b4")]:
        xi, se = hill(data, ks)
        ax.plot(ks, xi, color=color, lw=1.2, label=label)
        ax.fill_between(ks, xi - 1.96 * se, xi + 1.96 * se, color=color, alpha=0.15)
    for k in k_marks:
        ax.axvline(k, color="grey", lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("k = number of tail observations (log scale)")
    ax.set_ylabel("Hill estimate of xi")
    ax.set_title("Hill plot of S&P 500 daily losses and gains, 1928-2026 (95% band)", fontsize=11)
    ax.legend(fontsize=9)
    _clean(ax)
    fig.tight_layout()
    return fig


def plot_mean_excess(me, fit=None, u_max=None):
    """Mean excess plot with a 95% band, and optionally a fitted straight line.

    Parameters
    ----------
    me : pd.DataFrame
        Output of ``mean_excess`` (index u; columns 'mean excess', 'se', 'n above').
    fit : tuple (u_from, u_to, slope, intercept), optional
        Straight line drawn over [u_from, u_to].
    u_max : float, optional
        Largest threshold shown.

    Returns
    -------
    matplotlib.figure.Figure
    """
    m = me if u_max is None else me.loc[:u_max]
    u, e, se = m.index.values, m["mean excess"].values, m["se"].values
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(u, e, color="#d62728", lw=1.3, label="mean excess e(u)")
    ax.fill_between(u, e - 1.96 * se, e + 1.96 * se, color="#d62728", alpha=0.15, label="95% band")
    if fit is not None:
        a, b, slope, icpt = fit
        uu = np.linspace(a, b, 50)
        ax.plot(uu, icpt + slope * uu, color="black", lw=1.2, ls="--",
                label=f"weighted line {a:g}-{b:g}%: slope {slope:.3f}, xi = {slope / (1 + slope):.3f}")
    ax.set_xlabel("threshold u (daily loss, %)")
    ax.set_ylabel("mean excess (%)")
    ax.set_title("Mean excess plot of S&P 500 daily losses, 1928-2026", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    _clean(ax)
    fig.tight_layout()
    return fig


def plot_stability(stab, hill_xi=None, u_chosen=None):
    """Threshold stability: GPD xi (with Hill for comparison) and the modified scale beta*, with 95% bands.

    Parameters
    ----------
    stab : pd.DataFrame
        Output of ``threshold_stability``.
    hill_xi : array, optional
        Hill estimates at k = N_u for each threshold, drawn dashed on the xi panel.
    u_chosen : float, optional
        Chosen threshold, marked with a vertical line.

    Returns
    -------
    matplotlib.figure.Figure
    """
    u = stab.index.values
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, col, se, title in [(axes[0], "xi", "se_xi", "Shape xi"),
                               (axes[1], "beta_star", "se_beta_star", "Modified scale beta* = beta - xi u")]:
        v, s = stab[col].values, stab[se].values
        ax.plot(u, v, color="#d62728", lw=1.3, label=f"GPD {col}")
        ax.fill_between(u, v - 1.96 * s, v + 1.96 * s, color="#d62728", alpha=0.15, label="95% band")
        if u_chosen is not None:
            ax.axvline(u_chosen, color="black", lw=0.8, ls=":", label=f"chosen u = {u_chosen:g}%")
        ax.set_xlabel("threshold u (daily loss, %)")
        ax.set_title(title, fontsize=11)
        _clean(ax)
    if hill_xi is not None:
        axes[0].plot(u, hill_xi, color="#1f77b4", lw=1.2, ls="--", label="Hill at k = N_u")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle("GPD parameter stability, S&P 500 daily losses 1928-2026", fontsize=11)
    fig.tight_layout()
    return fig


def plot_gpd_fit(x, u, xi, beta):
    """GPD goodness of fit: QQ plot of the excesses (left) and the tail plot P(loss > x) on log-log axes (right).

    Parameters
    ----------
    x : array
        All losses (%).
    u, xi, beta : float
        Threshold and fitted GPD parameters.

    Returns
    -------
    matplotlib.figure.Figure
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    tail = np.sort(x[x > u])
    n_u = len(tail)
    p = (np.arange(1, n_u + 1) - 0.5) / n_u
    q_gpd = stats.genpareto.ppf(p, c=xi, scale=beta)                      # model quantiles of the excesses

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    ax.scatter(q_gpd, tail - u, s=6, color="#d62728")
    lim = max(q_gpd.max(), (tail - u).max()) * 1.05
    ax.plot([0, lim], [0, lim], color="black", lw=0.8)
    ax.set_xlabel("GPD quantile of the excess")
    ax.set_ylabel("observed excess (loss - u)")
    ax.set_title(f"QQ plot of the {n_u} excesses over u = {u:g}%", fontsize=11)
    _clean(ax)

    ax = axes[1]
    emp = (n_u - np.arange(n_u)) / n                                        # P(X >= each sorted loss)
    grid = np.linspace(u, tail.max() * 1.1, 400)
    gpd_tail = (n_u / n) * (1 + xi * (grid - u) / beta) ** (-1 / xi)       # Equation 12
    norm_tail = stats.norm.sf(grid, x.mean(), x.std())
    ax.scatter(tail, emp, s=6, color="#d62728", label="data")
    ax.plot(grid, gpd_tail, color="black", lw=1.2, label=f"GPD tail (xi = {xi:.2f})")
    ax.plot(grid, norm_tail, color="#1f77b4", lw=1.2, ls="--", label="normal")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.3 / n, 2 * n_u / n)
    ax.set_xlabel("daily loss x (%, log scale)")
    ax.set_ylabel("P(loss > x) (log scale)")
    ax.set_title("Tail plot", fontsize=11)
    ax.legend(fontsize=8)
    _clean(ax)
    fig.tight_layout()
    return fig



# =========================================================================== #
# 3. Tail risk                                                                #
# =========================================================================== #
def plot_return_levels(curves, emp_years, emp_var, events=None):
    """Daily VaR against the return period (years) for several models, with the empirical quantiles as points.

    Parameters
    ----------
    curves : dict of {label: (years, var)}
        One curve per model.
    emp_years, emp_var : array
        Return periods and empirical quantiles (inside the data only).
    events : dict of {label: loss}, optional
        Historical losses drawn as horizontal lines (e.g. 1987).

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    for label, (yrs, v) in curves.items():
        ax.plot(yrs, v, lw=1.4, label=label)
    ax.scatter(emp_years, emp_var, s=10, color="black", zorder=3, label="historical (data)")
    for label, loss in (events or {}).items():
        ax.axhline(loss, color="grey", lw=0.8, ls=":")
        ax.text(min(emp_years), loss, f" {label}", va="bottom", fontsize=8, color="grey")
    ax.set_xscale("log")
    ax.set_xlabel("return period: once every ... years (log scale)")
    ax.set_ylabel("daily loss (%)")
    ax.set_title("Daily VaR of the S&P 500 by return period, 1928-2026", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    _clean(ax)
    fig.tight_layout()
    return fig



# =========================================================================== #
# 4. Backtests                                                                #
# =========================================================================== #
def plot_backtest_summary(bt, z2_crit, highlight=()):
    """Breach ratio (left) and ES test Z2 (right) by VaR level, one line per method.

    Parameters
    ----------
    bt : pd.DataFrame
        Output of ``backtest_table`` (index (level, method); columns 'ratio', 'Z2').
    z2_crit : dict of {level: float}
        5% critical values of Z2.
    highlight : sequence of str
        Methods drawn with thicker lines.

    Returns
    -------
    matplotlib.figure.Figure
    """
    levels = sorted(bt.index.get_level_values(0).unique())
    methods = list(dict.fromkeys(bt.index.get_level_values(1)))
    xpos = np.arange(len(levels))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for m in methods:
        lw = 2.6 if m in highlight else 1.2
        r = [bt.loc[(q, m), "ratio"] for q in levels]
        z2 = [max(bt.loc[(q, m), "Z2"], -2.5) for q in levels]
        axes[0].plot(xpos, r, marker="o", lw=lw, label=m)
        axes[1].plot(xpos, z2, marker="o", lw=lw, label=m)
    axes[0].axhline(1, color="black", lw=0.8)
    axes[0].set_ylabel("breaches / expected")
    axes[0].set_title("VaR: breach ratio (1 = correct)", fontsize=11)
    axes[1].plot(xpos, [z2_crit[q] for q in levels], color="black", ls="--", lw=1, label="5% critical value")
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_ylabel("Z2 (clipped at -2.5)")
    axes[1].set_title("ES: Acerbi-Szekely Z2 (reject below the dashed line)", fontsize=11)
    for ax in axes:
        ax.set_xticks(xpos, [f"{q:.1%}" for q in levels])
        ax.set_xlabel("VaR / ES level")
        _clean(ax)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_var_through_crises(loss, var_lines, periods, title):
    """Daily losses (bars, positive = loss) against several VaR forecasts (lines), one panel per period.

    Parameters
    ----------
    loss : pd.Series
        Realised daily losses (%).
    var_lines : dict of {label: pd.Series}
        VaR forecasts (positive losses).
    periods : list of (start, end, label)
        Periods to show, one panel each.
    title : str
        Overall title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, len(periods), figsize=(6.2 * len(periods), 4.6))
    for ax, (a, b, lab) in zip(np.atleast_1d(axes), periods):
        l = loss.loc[a:b]
        ax.bar(l.index, l.values, width=1.0, color=np.where(l.values > 0, "#d62728", "#7f7f7f"), alpha=0.55)
        for label, v in var_lines.items():
            ax.plot(v.loc[a:b].index, v.loc[a:b].values, lw=1.4, label=label)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(lab, fontsize=11)
        ax.set_ylabel("daily loss (%)")
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
        _clean(ax)
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig
