# Project 7 – Extreme Value Theory for Tail Risk – Theory Notes

*Methods follow Daníelsson, Financial Risk Forecasting, Chapter 9 (extreme value theory), with Chapter 2 (GARCH) and Chapter 8 (backtesting), used as a guide only. This file goes in the repo as `notes/theory.md`.*

These notes use the same frameworks as Projects 1–6:

- **Part A** applies the problem-solving framework to the project as a whole.
- **Part B** applies the deep-learning framework to each technique: (1) what it is, (2) why it was developed, (3) the maths, with a definition for every equation and the meaning of every variable, (4) when it is valid, (5) its limitations, (6) how to derive, implement and validate it. Each technique ends with a small real-life example.
- **Part C** is a one-page summary of all the equations.
- **Part D** is the development plan, step by step.

The worked examples use round, illustrative numbers (a daily volatility of 1.2%, a tail index of about 4, a portfolio of USD 10m). Our own results will come from our data.

**Units and conventions.**

- Daily **losses** in % of the portfolio value: $x_t = -100 \ln(S_t / S_{t-1})$, so a large positive number is a large loss. Gains are studied as the losses of a short position, $-x_t$.
- VaR and ES are reported as **positive losses** in % (and in USD for a USD 10m long position in the S&P 500).
- "99.9% VaR" means the loss exceeded on 1 day in 1,000 — about **once every four years** of trading days.
- As in every project, a forecast for day *t* uses information up to day *t − 1* only.

---

## Note – Which sections are VaR methods?

Not every section of Part B is a way of computing VaR. Each section has one of four roles:

| Section | What it is | Role |
|---|---|---|
| **B0. Losses and the tail** | Defines the loss, VaR and ES | **Definitions** (what we measure) |
| **B1. Extreme value theory** | The theorem: the tail can only have one of three shapes | **Theory** behind the method |
| **B2. Hill estimator** | Measures how fat the tail is; Equation 8 also gives a VaR | **VaR method** (EVT, simple version) |
| **B3. Peaks over threshold / GPD** | Fits the tail beyond a threshold, then gives VaR and ES | **VaR method** (the main EVT one) |
| **B4. Normal, Student-t, historical** | The usual VaR methods | **VaR methods** (the benchmarks) |
| **B5. Confidence intervals** | How sure we are of a VaR number | **Accuracy check** (not a VaR method) |
| **B6. Extremal index** | Whether extreme days come in clusters | **Diagnostic** (not a VaR method) |
| **B7. McNeil–Frey** | GARCH volatility + EVT on the residuals | **VaR method** (the most advanced, conditional) |
| **B8. Backtesting** | Kupiec, Christoffersen, Z2 | **Validation** (tests VaR methods, does not produce one) |

The project therefore compares **six VaR / ES methods**:

1. Normal.
2. Student-t.
3. Historical (empirical quantile).
4. Hill (EVT).
5. GPD / peaks over threshold (EVT).
6. McNeil–Frey (GARCH + EVT, conditional).

Out of sample, GARCH-normal, GARCH-t and FHS are added as competitors, as in Projects 3–6.

**To remember:** B2, B3, B4 and B7 **produce** a VaR; B5, B6 and B8 **judge** a VaR; B0 and B1 explain what we measure and why EVT works.

---

## Decisions for this project

| Topic | Decision |
|---|---|
| Data | **S&P 500 price index** (Yahoo, `^GSPC`), daily closes from **30 December 1927** to today (about 24,800 days), downloaded fresh in Step 1 |
| Losses | Negative **log** returns in %: additive over time, symmetric treatment of both tails, the standard in the EVT literature; VaR converted back to a % loss of value at the end |
| Tail estimators | **Hill** estimator of the tail index; **peaks-over-threshold (POT)** with the **Generalized Pareto Distribution (GPD)**, fitted by maximum likelihood |
| Benchmarks | Normal, Student-t (fitted by maximum likelihood), historical (empirical quantile) |
| Risk levels | VaR at **99%, 99.5% and 99.9%**; ES at 97.5%, 99% and 99.9% |
| Uncertainty | Standard errors and **confidence intervals** of ξ, VaR and ES (delta method, profile likelihood, bootstrap) |
| Conditional model | **McNeil–Frey**: GJR-GARCH(1,1) estimated by quasi-maximum likelihood, then GPD on the standardised residuals; rolling 1,000-day window, refit every 20 days |
| Competitors out of sample | GARCH-normal, GARCH-t, FHS (filtered historical simulation), unconditional EVT, historical simulation |
| Evaluation | Kupiec, Christoffersen, Acerbi–Szekely Z2, Basel traffic light; at 99%, 99.5% and 99.9% |
| Main message to test | Normal (and, at 99.9%, even Student-t) models **understate extreme losses**, and EVT works best **after** removing volatility clustering |

---

## Part A – The project's problem

### A1. What do I need to determine?

Projects 3–6 measured VaR at 99% and ES at 97.5%. A risk team also needs numbers **much further in the tail**: economic capital (99.9% at one year in Basel's credit framework, 99.5% in Solvency II for insurers), stress scenarios, and the answer to questions such as "how big is a once-in-ten-years day?". At those levels there are very few observations, and the usual models are no longer trustworthy. This project answers:

> **How large are the extreme daily losses of the S&P 500, how should a risk team estimate 99.9% VaR and ES, and how wrong are the normal and Student-t models that far in the tail?**

More precisely:

1. How fat is the tail of S&P 500 losses — what is its **tail index**, and is it the same in every era and in both tails?
2. What are the 99%, 99.5% and 99.9% VaR and ES according to EVT (Hill and GPD), and how different are they from normal, Student-t and historical estimates? How **uncertain** are these numbers?
3. Does EVT work better on raw returns or on **GARCH-standardised residuals** (the McNeil–Frey approach)?
4. Which method passes **out-of-sample backtests** at these extreme levels, over almost a century of data?

### A2. What information and assumptions do I have?

- **Data:** about 24,800 daily closes of the S&P 500 since 1928, including the 1929 crash, the 1930s, 1987 (−20.5% in a day), 2008 and 2020.
- **Results from earlier projects:**
  - S&P 500 returns have fat tails (kurtosis about 14 since 1990) and volatility clusters (Project 1).
  - GJR-GARCH-t is the best volatility forecast (Project 2); FHS is the best one-day VaR/ES method at 99% / 97.5% (Projects 3–6).
- **Assumptions:**
  - The tail of the loss distribution is approximately a **power law** (Part B1–B3) — to be checked, not assumed blindly.
  - For unconditional EVT, the losses are treated as identically distributed; for McNeil–Frey, only the **standardised residuals** are.
- **Known difficulties:**
  - Before 1957 Yahoo's `^GSPC` is the older S&P 90 composite; until 1952 the market traded on Saturdays. These change the data, not the method, and are checked in Step 1.
  - Extremes are rare by definition: estimates rest on a few hundred points, so uncertainty must be reported.
  - Losses are **not independent**: they cluster in crises.

### A3. Which method is appropriate?

| Question | Method |
|---|---|
| How fat is the tail? | Hill estimator and Hill plot; QQ plots against normal and t |
| Where does the tail start? | Mean excess plot, parameter-stability plots |
| Shape of the tail beyond the threshold | Generalized Pareto Distribution, maximum likelihood |
| Extreme VaR / ES | GPD tail formulas (and Hill–Weissman); benchmarks normal, t, empirical |
| How sure are we? | Standard errors (delta method), profile-likelihood intervals, bootstrap |
| Clustering of extremes | Extremal index; then filter with GJR-GARCH (McNeil–Frey) |
| Does it work in practice? | Rolling out-of-sample forecasts and backtests |

### A4. Why should the method work?

- **Extreme value theory** is to the tail what the central limit theorem is to the average: whatever the exact distribution, the tail beyond a high threshold has (approximately) **one universal shape**, the GPD, described by just two parameters. We therefore model the tail with the shape the theory says it must have, instead of hoping that a distribution fitted to the whole sample (normal, t) also fits the extremes.
- **Filtering first** makes the EVT assumption (independent, identically distributed data) much more realistic, and makes the VaR react to today's volatility.

### A5. How can I verify the answer?

- **Code:** on simulated data with a known tail (Student-t with ν = 4 has ξ = 0.25), the Hill and GPD estimators must recover the truth; our GPD fit must match `scipy.stats.genpareto`; VaR/ES formulas must match simulated quantiles.
- **Fit:** QQ plots of the excesses against the fitted GPD; stability of ξ across thresholds.
- **Out of sample:** no look-ahead test; breach counts and the three tests at each level; Basel traffic light.

---

## Part B – The techniques

### B0. Losses and the tail

**1. What is it?** The **loss** on day *t* and its distribution's **tail** — the region of rare, large losses.

**2. Why?** Risk management cares about large losses, which are determined by the tail, not by the centre of the distribution where most data lie.

**3. Maths.**

**Variables used in this section:**

- $S_t$: S&P 500 close on day *t*.
- $x_t$: daily loss in %; $F$: its distribution function, $F(x) = P(X \le x)$.
- $\bar F(x) = 1 - F(x)$: the **tail (survival) function**, the probability of a loss larger than $x$.
- $q$: confidence level (0.99, 0.999); $p = 1 - q$: tail probability.

**Equation 1 – Loss:**
$$x_t = -100\,\ln\left(\frac{S_t}{S_{t-1}}\right).$$
*Definition:* the daily log loss in %; positive when the index falls.

**Equation 2 – VaR and ES:**
$$\text{VaR}_q = F^{-1}(q), \qquad \text{ES}_q = E\left[X \mid X > \text{VaR}_q\right].$$
*Definition:* VaR is the loss exceeded with probability $p = 1 - q$; ES is the average loss when VaR is exceeded.

**Equation 3 – From log loss to % loss of value:**
$$\text{loss of value (\%)} = 100\left(1 - e^{-\text{VaR}/100}\right).$$
*Definition:* converts a log-loss VaR into the % of the portfolio lost; for small numbers the two are almost equal (a 5% log loss = 4.88% of value).

**4. Valid when.** Always (definitions).

**5. Limitations.** At $q = 0.999$, a sample of 1,000 days has **one** observation beyond VaR on average: the empirical quantile is just the largest or second-largest loss, extremely noisy. This is the problem EVT solves.

**6. Implement, validate.** `losses(S)`; check against known days: 19 October 1987 (−20.47% simple, 22.9% log loss), 16 March 2020 (−11.98%), and the late-October 1929 crash days.

**Example.** A USD 10m S&P 500 position with a 99.9% VaR of 6.6% (log) loses more than 100 × (1 − e^(−0.066)) = 6.4% of its value, i.e. **USD 640k**, on about 1 day in 1,000.

---

### B1. Extreme value theory: the three tail types

**1. What is it?** The theory of the distribution of the **largest values** of a sample. Its main theorem (Fisher–Tippett–Gnedenko) says that, after rescaling, the maximum of many independent losses can only follow **one family** of distributions, the **Generalized Extreme Value (GEV)** distribution.

**2. Why was it developed?** For engineering (the highest flood a dyke must resist — Gumbel, 1958), insurance and hydrology: to estimate events **larger than any in the sample**. Finance adopted it in the 1990s (Embrechts, Klüppelberg and Mikosch, 1997; Daníelsson and de Vries, 1997; McNeil and Frey, 2000).

**3. Maths.**

**Variables:**

- $M_n = \max(X_1, \dots, X_n)$: the largest loss in a block of $n$ days.
- $a_n > 0$, $b_n$: rescaling constants.
- $\xi$: **shape parameter** (tail index of the theory); $\alpha = 1/\xi$: **tail index** (the power of the tail).

**Equation 4 – GEV distribution (Fisher–Tippett–Gnedenko):**
$$P\left(\frac{M_n - b_n}{a_n} \le x\right) \xrightarrow[n\to\infty]{} H_\xi(x) = \exp\left(-(1 + \xi x)^{-1/\xi}\right), \quad 1 + \xi x > 0.$$
*Definition:* the only possible limit of rescaled maxima. Three cases:

- $\xi > 0$ — **Fréchet** type: **fat (power-law) tails**: Student-t, Pareto, and financial returns.
- $\xi = 0$ — **Gumbel** type (the limit $\exp(-e^{-x})$): **thin tails** that decay exponentially: normal, log-normal.
- $\xi < 0$ — **Weibull** type: a **finite** maximum (uniform, beta).

**Equation 5 – Power-law tail:**
$$\bar F(x) = P(X > x) \approx C\, x^{-\alpha} = C\, x^{-1/\xi} \quad \text{for large } x.$$
*Definition:* in the Fréchet case the probability of a loss larger than *x* falls like a power of *x*, much more slowly than the normal's $e^{-x^2/2}$.

- Moments: $E[X^k]$ is finite only for $k < \alpha$. With α = 4, the variance exists but the kurtosis is infinite (on paper) — one reason sample kurtosis is so unstable (Project 1).
- Student-t with ν degrees of freedom: $\alpha = \nu$, $\xi = 1/\nu$.

**4. Valid when.** The data are independent and identically distributed (i.i.d.) or weakly dependent (then an **extremal index** corrects for clustering, B6), and the sample is large.

**5. Limitations.** It is an **asymptotic** result: it tells the shape far in the tail, not where "far" begins (B3). It says nothing about the centre of the distribution.

**6. Implement, validate.** Not estimated directly in this project (we use the POT form, B3, which uses more data than block maxima), but used to interpret ξ.

**Example.** If daily losses have α = 4, then a loss **twice as large** is $2^{-4} = 1/16$ as likely. For the normal, going from a 3σ to a 6σ loss divides the probability by about **7 million**. That is the whole difference between "1987 cannot happen" and "1987 happens about once a century".

---

### B2. The Hill estimator

**1. What is it?** A simple estimator of the tail index ξ (or α) from the *k* largest losses.

**2. Why was it developed?** Hill (1975) derived it as the maximum-likelihood estimator of α when the tail beyond a threshold is exactly Pareto. It needs no optimisation and is the standard first look at a tail.

**3. Maths.**

**Variables:**

- $X_{(1)} \ge X_{(2)} \ge \dots \ge X_{(n)}$: the losses sorted from largest to smallest (order statistics).
- $k$: number of tail observations used; the threshold is $X_{(k+1)}$.
- $\hat\xi_k$: Hill estimate; $\hat\alpha_k = 1/\hat\xi_k$.

**Equation 6 – Hill estimator:**
$$\hat\xi_k = \frac{1}{k}\sum_{i=1}^{k}\left[\ln X_{(i)} - \ln X_{(k+1)}\right], \qquad \hat\alpha_k = 1/\hat\xi_k.$$
*Definition:* the average log distance of the *k* largest losses above the threshold. Fat tail → the largest losses are far above the threshold → large $\hat\xi$.

**Equation 7 – Standard error:**
$$\text{s.e.}(\hat\xi_k) \approx \frac{\hat\xi_k}{\sqrt{k}}.$$
*Definition:* asymptotic standard error under a Pareto tail; a 95% interval is $\hat\xi_k \pm 1.96\,\hat\xi_k/\sqrt k$.

**Equation 8 – Hill (Weissman) quantile, VaR:**
$$\widehat{\text{VaR}}_q = X_{(k+1)}\left(\frac{k}{n\,(1-q)}\right)^{\hat\xi_k}.$$
*Definition:* extrapolates from the threshold to the level *q* along the fitted power law; $k/n$ is the empirical probability of exceeding the threshold.

**Equation 9 – Hill ES (for ξ < 1):**
$$\widehat{\text{ES}}_q = \frac{\widehat{\text{VaR}}_q}{1 - \hat\xi_k}.$$

**The Hill plot.** $\hat\xi_k$ against *k*. Small *k*: few points, high variance. Large *k*: points from the centre of the distribution enter, bias. We look for a **stable region** in between.

**4. Valid when.** ξ > 0 (fat tail), i.i.d. data, losses positive beyond the threshold, *k* small relative to *n* but large in absolute terms.

**5. Limitations.**

- **Choice of k** is a bias–variance trade-off with no exact rule.
- Not **location-invariant**: shifting the data changes $\hat\xi$ (the GPD does not have this problem).
- Only valid for ξ > 0.
- Assumes a pure power law beyond the threshold; slow convergence for some distributions (e.g. the Student-t).

**6. Implement, validate.** `hill(losses, k)` and `hill_plot`. Validation: simulate 25,000 Student-t(4) losses (true ξ = 0.25) and Pareto losses with ξ = 0.5; the Hill plot must sit near the truth over a range of *k*; Weissman VaR must match the true quantile.

**Example (toy).** The five largest losses are 9, 7, 6, 5 and 4.5%, and the 6th largest (the threshold) is 4%. Logs of the ratios: ln(9/4) = 0.811, ln(7/4) = 0.560, ln(6/4) = 0.405, ln(5/4) = 0.223, ln(4.5/4) = 0.118. Average: $\hat\xi = 0.423$, i.e. $\hat\alpha = 2.36$, with a standard error of 0.423/√5 = 0.19 — far too uncertain with *k* = 5. With 25,000 days we will use *k* in the hundreds.

---

### B3. Peaks over threshold and the Generalized Pareto Distribution

**1. What is it?** Choose a high threshold *u* and model the **excesses** $y = x - u$ of all losses above it with the **Generalized Pareto Distribution** (GPD). Together with the frequency of exceeding *u*, this gives the whole tail.

**2. Why was it developed?** Block maxima (B1) throw away all but one point per block. The **Pickands–Balkema–de Haan theorem** (1974–75) showed that excesses over a high threshold have a GPD limit, so all large losses can be used. Davison and Smith (1990) made POT the standard method; McNeil (1997) and McNeil and Frey (2000) brought it to finance.

**3. Maths.**

**Variables:**

- $u$: threshold; $N_u$: number of losses above *u*; $n$: sample size.
- $y_i = x_i - u > 0$: excesses.
- $\xi$: shape (the same ξ as in B1–B2); $\beta > 0$: scale (in % of loss).

**Equation 10 – Excess distribution and its limit (Pickands–Balkema–de Haan):**
$$F_u(y) = P(X - u \le y \mid X > u) \approx G_{\xi,\beta}(y) = 1 - \left(1 + \frac{\xi y}{\beta}\right)^{-1/\xi}, \quad y \ge 0.$$
*Definition:* for a high enough threshold, the distribution of the excess is a GPD. ξ > 0: Pareto-type power tail; ξ = 0: exponential, $1 - e^{-y/\beta}$; ξ < 0: bounded.

**Equation 11 – Log-likelihood of the excesses:**
$$\ell(\xi, \beta) = -N_u \ln\beta - \left(1 + \frac{1}{\xi}\right)\sum_{i=1}^{N_u}\ln\left(1 + \frac{\xi y_i}{\beta}\right).$$
*Definition:* maximised over ξ and β (numerically) to fit the GPD. Standard errors come from the inverse Hessian (the observed information).

**Equation 12 – Tail estimator:**
$$\hat{\bar F}(x) = \frac{N_u}{n}\left(1 + \hat\xi\,\frac{x - u}{\hat\beta}\right)^{-1/\hat\xi}, \quad x > u.$$
*Definition:* probability of a loss larger than *x*: (probability of exceeding *u*, estimated empirically) × (GPD probability of exceeding *x* given that *u* is exceeded).

**Equation 13 – GPD VaR:**
$$\widehat{\text{VaR}}_q = u + \frac{\hat\beta}{\hat\xi}\left[\left(\frac{n}{N_u}(1-q)\right)^{-\hat\xi} - 1\right].$$
*Definition:* Equation 12 solved for the loss whose exceedance probability is $1 - q$.

**Equation 14 – GPD ES (for ξ < 1):**
$$\widehat{\text{ES}}_q = \frac{\widehat{\text{VaR}}_q}{1 - \hat\xi} + \frac{\hat\beta - \hat\xi u}{1 - \hat\xi}.$$
*Definition:* uses the fact that the excesses over a higher threshold are again GPD with the same ξ.

**Equation 15 – Mean excess function:**
$$e(u) = E[X - u \mid X > u] = \frac{\beta + \xi u}{1 - \xi}.$$
*Definition:* for a GPD tail the mean excess is a **straight line** in *u* with slope ξ/(1 − ξ) (rising if ξ > 0). The **mean excess plot** (empirical mean excess against *u*) is therefore a tool to choose *u*: the GPD is plausible from the point where the plot becomes linear.

**Equation 16 – Return level ("once in N years"):**
$$x_N = \widehat{\text{VaR}}_{q}, \quad q = 1 - \frac{1}{252\,N}.$$
*Definition:* the daily loss exceeded on average once every *N* years (252 trading days a year).

**Choosing the threshold.** Too low: the GPD approximation does not hold yet (bias). Too high: few excesses (variance). We combine:

- the **mean excess plot** (linear from *u* onwards),
- **parameter stability**: $\hat\xi$ and the modified scale $\hat\beta - \hat\xi u$ should be constant above a valid threshold,
- **QQ plots** of the excesses against the fitted GPD,
- a practical rule used in the literature (the top 5–10% of losses), checked against the three diagnostics.

**4. Valid when.** The threshold is high enough, the excesses are (close to) i.i.d., and ξ < 1 for ES to exist (ξ < 0.5 for a finite variance of the losses).

**5. Limitations.**

- Threshold choice remains partly judgement.
- Maximum likelihood is irregular for ξ < −0.5 (not a concern for financial losses).
- Clustering of extremes violates independence (B6).
- Extrapolation far beyond the data (a 1-in-100-years loss from 97 years of data) carries very wide confidence intervals.

**6. Implement, validate.** `fit_gpd(excesses)` by maximum likelihood with `scipy.optimize`; validated against `scipy.stats.genpareto.fit` and on simulated GPD and Student-t data; `gpd_var_es` checked against the empirical quantiles of a very large simulated sample; `mean_excess` and `threshold_stability` for the plots.

**Example.** Suppose *n* = 10,000 days, threshold *u* = 2.0% exceeded on $N_u$ = 500 days (5%), fitted ξ = 0.25 and β = 0.7.

- 99% VaR: $(n/N_u)(1-q) = 20 \times 0.01 = 0.2$; $0.2^{-0.25} = 1.495$; VaR = 2.0 + (0.7/0.25)(1.495 − 1) = **3.39%**.
- 99.9% VaR: $20 \times 0.001 = 0.02$; $0.02^{-0.25} = 2.659$; VaR = 2.0 + 2.8 × 1.659 = **6.65%**; ES = (6.65 + 0.7 − 0.25 × 2.0) / 0.75 = **9.13%**.
- Normal with the same daily volatility of 1.2%: 99% VaR 2.79%, 99.9% VaR **3.71%** (ES 4.04%).
- On USD 10m: EVT says a once-in-four-years day costs about **USD 640k**; the normal says USD 360k.
- The 1987 crash (a 20.5% loss): probability per day 1.5 × 10⁻⁵ under this GPD (about **once in 260 years**), but 10⁻⁶⁵ under the normal (never in the life of the universe).

---

### B4. The benchmarks: normal, Student-t and historical

**1. What is it?** The three standard ways of computing VaR/ES that EVT is compared with.

**2. Why?** They are what most risk systems use; the project measures how wrong they are far in the tail.

**3. Maths.**

**Variables:**

- $\mu$, $\sigma$: mean and standard deviation of the losses.
- $z_q = \Phi^{-1}(q)$, $\varphi$: standard normal quantile and density.
- $\nu$: Student-t degrees of freedom; $t_\nu^{-1}$, $g_\nu$: quantile and density of the standard t.

**Equation 17 – Normal VaR and ES:**
$$\text{VaR}_q = \mu + \sigma z_q, \qquad \text{ES}_q = \mu + \sigma\,\frac{\varphi(z_q)}{1-q}.$$

**Equation 18 – Student-t VaR and ES (scaled to standard deviation σ):**
$$\text{VaR}_q = \mu + \sigma\sqrt{\tfrac{\nu-2}{\nu}}\; t^{-1}_\nu(q), \qquad \text{ES}_q = \mu + \sigma\sqrt{\tfrac{\nu-2}{\nu}}\;\frac{g_\nu\!\left(t^{-1}_\nu(q)\right)}{1-q}\,\frac{\nu + \left(t^{-1}_\nu(q)\right)^2}{\nu - 1}.$$
*Definition:* the factor $\sqrt{(\nu-2)/\nu}$ makes the t have variance $\sigma^2$, so the comparison with the normal is fair.

**Equation 19 – Historical (empirical) VaR and ES:**
$$\widehat{\text{VaR}}_q = \text{empirical } q\text{-quantile}, \qquad \widehat{\text{ES}}_q = \text{average of losses beyond it}.$$

**4. Valid when.** Normal: never in the tail of daily returns (Projects 1, 3). Student-t: a fat tail with ξ = 1/ν, but a **single** parameter governs both the centre and the tail, and it is symmetric. Historical: only within the range of the data and with enough points beyond the quantile.

**5. Limitations.** The normal has ξ = 0 (thin tail). The t is fitted mostly by the many ordinary days, so its tail is whatever fits the centre. Historical VaR at 99.9% from 1,000 days is the largest or second-largest loss.

**6. Implement, validate.** `var_es_normal`, `var_es_t` (closed forms, checked against simulation), `var_es_hist`; `fit_t` by maximum likelihood checked against `scipy.stats.t.fit`.

**Example.** Daily σ = 1.2%: 99.9% VaR normal 3.71%, Student-t(4) **6.09%** (ES 8.22%), GPD example 6.65% (ES 9.13%). At 99% the t (3.18%) is closer to the GPD (3.39%) than the normal (2.79%); at 99.9% the gap between normal and the others is 65–80%.

---

### B5. Uncertainty of extreme quantiles

**1. What is it?** Confidence intervals for ξ, VaR and ES.

**2. Why?** A model validator will ask: "how sure are you of your 99.9% VaR?" With a few hundred tail points, the answer can be ±30% or more. Reporting a point estimate alone would be unprofessional.

**3. Maths.**

**Variables:** $\theta = (\xi, \beta)$; $\hat\Sigma$: estimated covariance of $\hat\theta$ (inverse Hessian of −ℓ); $\nabla \text{VaR}$: gradient of Equation 13 with respect to θ; $\ell_p$: profile log-likelihood.

**Equation 20 – Delta method:**
$$\text{Var}\left(\widehat{\text{VaR}}_q\right) \approx \nabla\text{VaR}^{\top}\,\hat\Sigma\,\nabla\text{VaR}.$$
*Definition:* a first-order approximation of the variance of a function of estimated parameters.

**Equation 21 – Profile-likelihood interval:**
$$\left\{v : 2\left[\ell(\hat\theta) - \ell_p(v)\right] \le \chi^2_{1;0.95} = 3.84\right\},$$
*Definition:* re-parametrise the GPD by its VaR *v* (Equation 13 solved for β), maximise the likelihood over ξ for each *v*, and keep the values of *v* that the data do not reject. This interval is **asymmetric** (longer on the high side), as it should be for a tail quantile.

**Equation 22 – Bootstrap:** resample the losses (or, for dependence, blocks of days) many times, refit, and take the 2.5% and 97.5% quantiles of the VaR estimates.

**4. Valid when.** Delta method: large samples, estimates far from the boundary. Profile likelihood: better in small samples. Block bootstrap: robust to dependence.

**5. Limitations.** All three assume the GPD model is right above *u*; none captures threshold-choice uncertainty, which is shown separately with the stability plot.

**6. Implement, validate.** `gpd_var_ci(..., method="delta" | "profile")`, `bootstrap_var_ci`. Validation: on simulated data, about 95% of the intervals must contain the true VaR (coverage check).

**Example (illustrative).** With 500 excesses and ξ = 0.25 ± 0.05, the 99.9% VaR of 6.65% may have a 95% interval of roughly 5.9–7.7%: about **−11% / +16%**, asymmetric.

---

### B6. Clustering of extremes: the extremal index

**1. What is it?** The **extremal index** θ ∈ (0, 1] measures how much extremes come in clusters: 1/θ is roughly the average size of a cluster of exceedances.

**2. Why?** EVT assumes independence. If extremes cluster (they do: October 1987, October 2008, March 2020), the number of **independent** extreme episodes is smaller than the number of extreme days, and a 99.9% VaR that is right on average will be breached in bursts.

**3. Maths.**

**Variables:** $N_u$ exceedances of *u*; $C_u$ number of clusters (runs of exceedances separated by at least *r* days below *u*).

**Equation 23 – Runs estimator of the extremal index:**
$$\hat\theta = \frac{C_u}{N_u}.$$
*Definition:* clusters per exceedance; θ = 1 means no clustering, θ = 0.5 means extremes come in pairs on average.

**4. Valid when.** High threshold, long sample; the run length *r* is a choice.

**5. Limitations.** Sensitive to *r* and *u*. It describes the problem; McNeil–Frey (B7) removes it at the source.

**6. Implement, validate.** `extremal_index_runs`; validated on i.i.d. data (θ ≈ 1) and on a GARCH simulation (θ < 1).

**Example.** If 500 exceedances fall into 250 clusters, θ = 0.5: the daily probability of entering an extreme episode is half the daily exceedance probability.

---

### B7. McNeil–Frey: EVT on GARCH-standardised residuals

**1. What is it?** A two-step **conditional** EVT model (McNeil and Frey, 2000):

1. fit a GARCH model to the losses and compute the **standardised residuals** $z_t = (x_t - \mu_t)/\sigma_t$;
2. fit a GPD to the tail of the residuals; the VaR for tomorrow is today's volatility forecast × the EVT quantile of the residuals.

**2. Why was it developed?** Unconditional EVT ignores that volatility changes: its VaR is the same in 2017 (VIX 10) and in October 2008, so its breaches cluster in crises. GARCH alone captures volatility but assumes a normal or t distribution for the shocks, which is wrong in the far tail. McNeil–Frey combines the two: **GARCH for the "when", EVT for the "how big"**. The residuals are much closer to i.i.d., which is what EVT needs.

**3. Maths.**

**Variables:**

- $\mu_t$, $\sigma_t$: conditional mean and volatility of the loss for day *t*, known on day *t − 1*.
- $z_t$: standardised residual; $\omega, \alpha, \gamma, \beta$: GJR-GARCH parameters (Project 2).
- $z_q^{EVT}$, $\text{ES}_q^{EVT}(z)$: GPD VaR and ES of the residuals (Equations 13–14 applied to $z$).

**Equation 24 – GJR-GARCH(1,1) for the losses:**
$$x_t = \mu + \sigma_t z_t, \qquad \sigma_t^2 = \omega + \left(\alpha + \gamma\,\mathbb 1[x_{t-1} > \mu]\right)(x_{t-1} - \mu)^2 + \beta\,\sigma_{t-1}^2.$$
*Definition:* the volatility reacts more to a **loss** (a fall of the index) than to a gain — the leverage effect of Projects 2–6, written here in loss terms.

**Equation 25 – Quasi-maximum likelihood (QML):**
$$\ell(\omega,\alpha,\gamma,\beta,\mu) = -\frac12\sum_t\left[\ln\sigma_t^2 + \frac{(x_t-\mu)^2}{\sigma_t^2}\right].$$
*Definition:* the normal likelihood, used even though the residuals are not normal; it still estimates the volatility parameters consistently (Bollerslev–Wooldridge). McNeil and Frey use QML because the tail is then modelled by EVT, not by the likelihood.

**Equation 26 – Conditional VaR and ES:**
$$\text{VaR}_{q,t} = \mu_t + \sigma_t\, z_q^{EVT}, \qquad \text{ES}_{q,t} = \mu_t + \sigma_t\, \text{ES}_q^{EVT}(z).$$
*Definition:* tomorrow's VaR = tomorrow's mean loss + tomorrow's volatility × the extreme quantile of the standardised shocks.

**Out-of-sample implementation.** Rolling window of 1,000 days (as in McNeil and Frey, and Projects 4–6), GARCH refitted every 20 days, GPD fitted on the 100 largest residuals of the window (the top 10%, McNeil and Frey's choice), and the volatility updated every day.

**4. Valid when.** The GARCH model captures the volatility dynamics well enough that the residuals are close to i.i.d. (checked with Ljung–Box tests on $z_t^2$ and with the extremal index of the residuals).

**5. Limitations.**

- Two-step estimation: the GPD ignores the uncertainty of the GARCH parameters.
- A 1,000-day window gives only 100 tail residuals: the 99.9% residual quantile is still an extrapolation.
- The leverage effect makes the loss and gain tails of the residuals different — we fit each tail separately.

**6. Implement, validate.** `fit_gjr_qml`, `standardised_residuals`, `mcneil_frey_var_es`, and the rolling engine `rolling_tail_var`. Validation: QML estimates against the `arch` package; on a simulated GJR-GARCH-t process with known parameters, the model's 99.9% VaR breach rate must be close to 0.1%; look-ahead test with a fake crash.

**Example (illustrative).** Today's GARCH volatility is 1.0% (a calm day). The GPD of the residuals gives $z^{EVT}_{0.999}$ = 4.2 (a normal would give 3.09). The 99.9% VaR is 1.0 × 4.2 = **4.2%**. After a crash, volatility jumps to 3.5% and the VaR becomes **14.7%**, while unconditional EVT would still say about 6.6%.

---

### B8. Backtesting extreme VaR and ES

**1. What is it?** The same tests as Projects 3–6 — Kupiec, Christoffersen, Acerbi–Szekely Z2 — applied at 99%, 99.5% and 99.9%, plus the Basel traffic light.

**2. Why?** At 99.9% a correct model is breached once in 1,000 days. With 1,000 days of data the test has almost no power; with our ~23,800 out-of-sample days (1932–2026) we expect about **24 breaches**, enough for a meaningful test — one of the advantages of the long history.

**3. Maths.**

**Variables:** $I_t = \mathbb 1[x_t > \text{VaR}_{q,t}]$ breach indicator; $T$ number of forecasts; $N = \sum I_t$; $p = 1 - q$.

**Equation 27 – Kupiec likelihood ratio:**
$$LR_{uc} = -2\ln\left[\frac{p^{N}(1-p)^{T-N}}{\hat\pi^{N}(1-\hat\pi)^{T-N}}\right] \sim \chi^2_1, \quad \hat\pi = N/T.$$

**Equation 28 – Christoffersen conditional coverage:**
$$LR_{cc} = LR_{uc} + LR_{ind} \sim \chi^2_2,$$
where $LR_{ind}$ tests whether the probability of a breach depends on yesterday's breach.

**Equation 29 – Acerbi–Szekely Z2 (losses written positive):**
$$Z_2 = 1 - \frac{1}{T\,p}\sum_{t=1}^{T}\frac{x_t\, I_t}{\text{ES}_{q,t}}.$$
*Definition:* 0 on average if ES is right; **negative if realised tail losses are larger than ES predicts (ES too low)** — the same reading as in Projects 3–6. Critical values by simulation.

**Equation 30 – Exact binomial p-value:** with few expected breaches, the χ² approximation is poor; the exact probability $P(N \ge n_{obs})$ under Binomial(T, p) is reported as well.

**4. Valid when.** Forecasts are truly out of sample; breaches are measured on the same losses the model forecasts.

**5. Limitations.** Low power at 99.9% on short samples; tests check frequency and size of breaches, not the **ranking** of models (use loss functions too: the quantile "tick" loss).

**6. Implement, validate.** Reuse the Project 3–6 functions; add `binomial_pvalue` and `quantile_loss`; check on artificial hit sequences.

**Example (illustrative).** 23,800 days at 99.9%: 23.8 breaches expected. A normal-GARCH model with 70 breaches is rejected (p ≈ 0); an EVT model with 27 breaches passes (Kupiec p ≈ 0.5).

---

## Part C – Summary of the equations

| # | Equation | What it gives |
|---|---|---|
| 1 | $x_t = -100\ln(S_t/S_{t-1})$ | daily loss in % |
| 2 | $\text{VaR}_q = F^{-1}(q)$, $\text{ES}_q = E[X \mid X > \text{VaR}_q]$ | the risk measures |
| 3 | $100(1 - e^{-\text{VaR}/100})$ | % loss of value |
| 4 | $H_\xi(x) = \exp(-(1+\xi x)^{-1/\xi})$ | GEV limit of maxima |
| 5 | $\bar F(x) \approx C x^{-1/\xi}$ | power-law tail |
| 6 | $\hat\xi_k = \frac1k\sum \ln(X_{(i)}/X_{(k+1)})$ | Hill estimator |
| 7 | s.e. $= \hat\xi/\sqrt k$ | Hill standard error |
| 8 | $X_{(k+1)}(k/(n(1-q)))^{\hat\xi}$ | Hill–Weissman VaR |
| 9 | $\text{VaR}/(1-\xi)$ | Hill ES |
| 10 | $G_{\xi,\beta}(y) = 1-(1+\xi y/\beta)^{-1/\xi}$ | GPD of excesses |
| 11 | $-N_u\ln\beta - (1+1/\xi)\sum\ln(1+\xi y_i/\beta)$ | GPD log-likelihood |
| 12 | $\frac{N_u}{n}(1+\xi\frac{x-u}{\beta})^{-1/\xi}$ | tail probability |
| 13 | $u + \frac\beta\xi[(\frac n{N_u}(1-q))^{-\xi} - 1]$ | GPD VaR |
| 14 | $\frac{\text{VaR}}{1-\xi} + \frac{\beta-\xi u}{1-\xi}$ | GPD ES |
| 15 | $e(u) = (\beta + \xi u)/(1-\xi)$ | mean excess (linear) |
| 16 | $q = 1 - 1/(252N)$ | N-year return level |
| 17 | $\mu + \sigma z_q$, $\mu + \sigma\varphi(z_q)/(1-q)$ | normal VaR, ES |
| 18 | scaled Student-t VaR, ES | t benchmark |
| 19 | empirical quantile and tail mean | historical benchmark |
| 20 | $\nabla^\top\hat\Sigma\nabla$ | delta-method variance |
| 21 | $2[\ell(\hat\theta) - \ell_p(v)] \le 3.84$ | profile-likelihood CI |
| 22 | resample and refit | bootstrap CI |
| 23 | $\hat\theta = C_u/N_u$ | extremal index |
| 24 | GJR-GARCH(1,1) on losses | conditional volatility |
| 25 | normal (quasi) likelihood | GARCH estimation |
| 26 | $\mu_t + \sigma_t z_q^{EVT}$ | McNeil–Frey VaR, ES |
| 27 | $LR_{uc} \sim \chi^2_1$ | Kupiec |
| 28 | $LR_{cc} \sim \chi^2_2$ | Christoffersen |
| 29 | $Z_2$ | ES backtest |
| 30 | $P(N \ge n)$, Binomial(T, p) | exact breach p-value |

---

## Part D – Development plan (step by step, one function at a time)

As in Projects 1–6, each step is one function or one cell, with the five questions, a line-by-line explanation and a validation; you run it and send the output before the next step.

| Step | Content | Functions (package `tailrisk`) |
|---|---|---|
| **1. Data** | Download ^GSPC since 1927; save; check the calendar (Saturday sessions to 1952), zero returns, known crash days; losses (Eq. 1) | `data.load_yahoo`, `data.losses` |
| **2. The tails in the data** | Statistics by era (1928–45, 1946–86, 1987–2026); QQ plots against normal and t; the 20 worst days; both tails | `descriptive.tail_table`, `plots.plot_qq` |
| **3. Hill estimator** | Hill function; validation on simulated t(4) and Pareto; Hill plots of both tails; ξ by era | `evt.hill`, `evt.hill_var_es`, `plots.plot_hill` |
| **4. POT and GPD** | Mean excess plot; GPD maximum likelihood (vs scipy); parameter stability; QQ of excesses; threshold choice | `evt.mean_excess`, `evt.fit_gpd`, `evt.threshold_stability` |
| **5. Extreme VaR and ES** | 99%, 99.5%, 99.9% by GPD, Hill, normal, t, historical; return levels; the 1987 crash under each model | `evt.gpd_var_es`, `benchmarks.var_es_normal`, `var_es_t`, `var_es_hist` |
| **6. Uncertainty** | Delta-method and profile-likelihood intervals; block bootstrap; coverage check on simulated data | `evt.gpd_var_ci`, `evt.bootstrap_var_ci` |
| **7. Clustering and the GARCH filter** | Extremal index of losses; GJR-GARCH by QML (vs `arch`); residual diagnostics; extremal index of residuals | `evt.extremal_index_runs`, `garch.fit_gjr_qml` |
| **8. McNeil–Frey out of sample** | Rolling engine (1,000 days, refit every 20): McNeil–Frey, GARCH-normal, GARCH-t, FHS, unconditional EVT, historical; look-ahead test | `engine.rolling_tail_var` |
| **9. Backtests** | Kupiec, Christoffersen, Z2, exact binomial, traffic light at 99%, 99.5%, 99.9%; charts through 1929, 1987, 2008, 2020 | `backtest.*`, `plots.plot_backtest`, `plots.plot_var_through_crises` |
| **10. Conclusions** | Findings, recommendation, limitations; package, tests, notebook, report, README | — |
