"""
tailrisk – extreme value theory for the tail risk of the S&P 500.

Project 7 of the Financial Risk Forecasting series: 99 years of daily S&P 500
losses, the Hill estimator, peaks over threshold with the Generalized Pareto
Distribution (GPD), 99%-99.9% VaR and ES against normal, Student-t and
historical models, confidence intervals (delta method, profile likelihood,
block bootstrap), clustering of extremes, and the McNeil-Frey conditional EVT
model (GJR-GARCH + GPD on the residuals) backtested out of sample.

Modules
-------
data         download, data-error detection and cleaning, daily losses
descriptive  tail statistics by era, Ljung-Box test
evt          Hill, mean excess, GPD fit, threshold stability, GPD VaR/ES,
             confidence intervals, block bootstrap, extremal index
benchmarks   normal, Student-t and historical VaR / ES; Student-t fit
garch        GJR-GARCH(1,1) by quasi-maximum likelihood, standardised residuals
engine       out-of-sample engines: rolling (six methods) and expanding residual tail
backtest     Kupiec, Christoffersen, exact binomial, Acerbi-Szekely Z2, backtest table
plots        every chart of the project
"""

__version__ = "1.0.0"
