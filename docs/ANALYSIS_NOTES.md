# Analysis notes — methodology defence

Every design decision in the engine, with the reasoning, the alternatives considered, and
the case against each choice. The second half is the ten questions I would expect to be
asked about this repository in an interview, answered.

Companion documents: [CV_CLAIMS.md](CV_CLAIMS.md) for what this repo does and does not
support as a claim; [`analysis/metric_validation.ipynb`](../analysis/metric_validation.ipynb)
for the evidence behind everything asserted here.

---

## 1. Why the repository contains no real trading data

This is the decision that shapes everything else, so it goes first.

The original version of this project advertised a Sharpe ratio of 1.35 from twelve months
of my own Interactive Brokers account. That claim is now gone, for three reasons, in
increasing order of importance.

**It was unverifiable.** No reader could check it. A number that a reader must take on
faith is not evidence in a portfolio piece; it is an assertion with a chart next to it.

**The code computing it was wrong.** The Sortino denominator, the alpha regression, the
VaR quantile lookup and the FIFO matcher all contained errors documented below. Whatever
the true figure was, 1.35 was not it.

**It was self-refuting in context.** Section 2 of the validation notebook measures the
standard error of a one-year Sharpe estimate at approximately 1.0. Publishing an analysis
which establishes that single-year Sharpe ratios are mostly noise, and then citing my own
single-year Sharpe as a credential, is a contradiction. If the analysis is right, the
claim is meaningless; if the claim is meaningful, the analysis is wrong.

Removing it costs a headline number and buys a defensible position. Given the choice
between a repository that claims performance and one that demonstrates competence, the
second is both more honest and — for the roles this is aimed at — more relevant. A risk
team does not need to know what my personal portfolio returned. It needs to know whether
I can implement a risk metric correctly and describe its uncertainty.

`private_data/` is gitignored as a standing guard, and nothing in the package can open a
broker connection.

---

## 2. Sharpe ratio

**Implemented as** `mean(r - rf_daily) / std(r, ddof=1) × √252`, with the Lo (2002)
standard error `√((1 + SR_daily²/2)/n)` scaled by √252, and a 95% normal interval.

**Arithmetic, not geometric.** The alternative headline — annualised compound return
minus the risk-free rate, over annualised volatility — is what the original code used and
what many retail tools report. It mixes a geometric numerator with an arithmetic
denominator, and it is not the estimator the Lo standard error describes. Since the whole
point here is to attach honest error bars, the estimator has to be the one the theory
covers. The geometric annualised return is still reported separately as a return figure,
just not inside the Sharpe.

**Risk-free conversion.** `rf_daily = rf / 252`, arithmetic. The geometric conversion
`(1 + rf)^(1/252) − 1` differs by well under a basis point on the daily rate at realistic
rates, and the arithmetic form is consistent with the arithmetic excess returns the ratio
is built from.

**ddof = 1.** Sample standard deviation. With n in the hundreds the difference from the
population version is immaterial, but the sample estimator is the unbiased one for
variance and it is what `scipy` and `pandas` default to, so consistency wins.

**Why the interval is everywhere.** A Sharpe ratio quoted to two decimal places implies a
precision that a one-year sample does not contain. The engine makes it structurally hard
to report the point estimate alone: `sharpe_ratio()` returns a `SharpeEstimate` carrying
the standard error and both interval bounds, and `summary()` emits all four fields.

**Known limitation.** This is the IID form of the Lo standard error. Lo's paper also
gives an autocorrelation-corrected version, and positive autocorrelation — from illiquid
holdings, stale marks, or monthly-valued positions — inflates the true standard error.
**Every interval in this repository is therefore a lower bound on the real uncertainty.**
Implementing the Newey-West correction is the single most valuable extension to this code.

---

## 3. Sortino ratio

**Implemented as** annualised mean excess return over
`√(mean(min(r − τ, 0)²)) × √252`, with τ = the daily risk-free rate.

**The bug this replaces.** The original computed `returns[returns < 0].std()` — the
standard deviation of the negative subset. That is wrong twice over. It divides by the
count of losing days rather than all days, and it re-centres the squared deviations on
the mean of the losses rather than on the target. Both push the denominator down and the
ratio up. On the notebook's four-observation example the inflation is 11.8%; on a real
series with many small gains and few large losses, where the negative subset is a small
fraction of the sample, it is larger.

**Target choice.** τ = daily risk-free rate, so Sortino and Sharpe answer the same
question and differ only in how they measure risk. τ = 0 is the more common retail
convention and produces a higher number. Either is defensible; mixing them across a
comparison is how people accidentally cherry-pick, so the engine fixes one and documents
it.

---

## 4. Alpha and beta

**Implemented as** OLS of `(r_p − rf_d)` on `(r_b − rf_d)`, intercept annualised ×252,
with the intercept's standard error annualised the same way and reported as a 95%
interval.

**Why excess returns.** This is the definition of CAPM alpha. Regressing raw returns
gives an intercept of `alpha + (1 − β)·rf_d`, which coincides with alpha only when β = 1.
For a leveraged book with β around 1.3 and a 2% risk-free rate the discrepancy is small
in absolute terms but conceptually wrong, and it is exactly the sort of thing that gets
picked up in a technical interview. The raw variant is retained as `raw_alpha_beta` for
comparison, and a test asserts the two differ by precisely `(1 − β)·rf_d·252`.

**Why the alpha interval matters more than the alpha.** Section 4 of the notebook builds
a portfolio with zero alpha by construction and shows the estimate landing at +0.12% per
year with a 95% interval of roughly ±7 points. The point estimate looks like a small
positive edge. The interval says there is nothing there. Alpha without an interval is
close to meaningless over any realistic sample.

**Single-factor limitation.** CAPM against SPY is the wrong model for a concentrated
growth book. Momentum and size exposure will show up as "alpha" that a Fama-French or
Carhart model would correctly attribute to factors. The right reading of the alpha figure
here is "return not explained by the market", not "skill". Multi-factor attribution is
future work.

**Constant beta.** The headline regression uses the full sample. `rolling_beta` exists
and is tested, but a book that changed its leverage midway would have that averaged away
in the headline number.

---

## 5. Value at risk

**Two estimators, both reported.** Parametric: `−(μ + z_(1−c)·σ)` using
`scipy.stats.norm.ppf`, mean term included. Historical: the empirical `(1−c)` quantile.

**The bug this replaces.** The original used a two-entry dictionary `{0.95: 1.645, 0.99:
2.326}` with the 95% z-score as the fallback, so any other confidence level silently
returned the wrong number. It also dropped the mean term, which for a strongly trending
series is not negligible.

**Why both.** They agree when returns really are normal and diverge when they are not,
and the direction of the divergence is diagnostic. On the notebook's fat-tailed example
(same mean, same variance, excess kurtosis 11.4) the parametric figure is the *larger* at
95% and the *smaller* at 99% — the signature of a distribution with more mass in both the
centre and the extreme tail. Reporting the parametric number alone would understate
exactly the losses a risk limit exists to control.

**What neither does.** Both are one-day measures on the unconditional distribution.
Neither is a conditional VaR (expected shortfall), neither accounts for volatility
clustering, and neither survives a regime break. Expected shortfall is the natural next
addition and is what post-Basel-III risk teams actually use.

---

## 6. FIFO profit and loss

**Implemented as** a per-ticker deque of open lots. A buy first covers open shorts oldest
first, then opens a long with any remainder; a sell mirrors it. Commissions are allocated
per share on both legs, so `net = gross − entry_commission_share − exit_commission_share`.

**The two bugs this replaces.** The original appended buys to a queue and matched sells
against it — so a sell arriving with no open lot fell through the `while remaining > 0 and
queue` loop and vanished, taking the entire short leg of the book with it. And it loaded
a `commission` column that it never referenced, so every P&L figure was gross.

**Why FIFO.** It matches the tax treatment in most jurisdictions, it is what brokers
report, and it is deterministic — LIFO or average-cost would give different realised
figures from identical executions, and any comparison across sources needs a fixed
convention. FIFO is the one that reconciles with a broker statement.

**Why commissions are not a rounding error.** On the sample's round-trips they take one
or two percent off gross P&L. On a high-turnover book they are the difference between a
positive and a negative strategy. A backtest that ignores them is not a backtest of
anything tradeable.

**Position flips.** A single execution larger than the open position closes it and opens
one in the opposite direction, producing one realised record and one new lot. This is
tested explicitly because it is where naive implementations either crash or silently drop
the remainder.

**What it does not model.** Corporate actions, splits, dividends, currency conversion,
margin interest, borrow cost on shorts, slippage. On a real leveraged book financing cost
is a material drag that this engine does not capture at all.

---

## 7. Position sizing

**Kelly.** `f* = (b·p − q)/b` with `b = avg_win/avg_loss`, clipped at zero when the edge
is non-positive, then scaled by a multiplier defaulting to 0.25.

**Why quarter-Kelly.** Full Kelly maximises the expected log of terminal wealth, which
sounds like what you want until you look at the path. It produces drawdowns that are
brutal in practice, and — more importantly — it is optimal only if `p`, `avg_win` and
`avg_loss` are known exactly. They never are; they are estimates from a small sample, and
Kelly is highly sensitive to overestimating the edge. Fractional Kelly is the standard
response: quarter-Kelly gives roughly 75% of the log-growth benefit at a fraction of the
volatility, and it degrades gracefully when the inputs turn out to be optimistic. Given
that a one-year win rate has an enormous standard error, being conservative about the
inputs matters more than being optimal given them.

**Volatility targeting is not equal risk contribution.** The original code claimed ERC
and implemented `target_vol / ticker_vol`, which scales by standalone volatility and never
looks at the covariance matrix. These coincide only when assets are uncorrelated. The
method is now named `vol_target_size`, and true ERC — weights whose covariance-based risk
contributions are equal, solved by SLSQP — is implemented separately as `erc_weights`.
The notebook shows three assets with identical volatilities where volatility targeting
assigns equal weights and wildly unequal risk shares, while ERC tilts toward the
diversifier. Both are legitimate tools. They are not the same tool, and calling one by
the other's name is the kind of thing that ends an interview early.

---

## 8. Determinism and provenance

The demo has a fixed seed and a fixed date range. The original called
`datetime.today()`, so its output moved every day and no printed figure could stay true.
A test now asserts the demo period is literally `2024-01-02 to 2024-12-31`.

CI regenerates `data/samples/`, re-runs the demo, rebuilds the report, and fails on any
diff. So the committed data provably matches its generator, and the committed report
provably matches the engine — the README's numbers cannot silently drift from what the
code produces.

---

# The ten questions

### 1. Why is there no Sharpe ratio for your actual portfolio?

Because it would be unverifiable and, on one year of data, statistically meaningless.
Section 2 of the notebook measures the standard error of a one-year Sharpe at
approximately 1.0 — the same size as the estimate. A reported 1.35 would carry a 95%
interval of roughly −0.6 to +3.3, which is consistent with no skill and with excellent
skill simultaneously. I removed the claim rather than defend a number I had also just
proved was noise. The repository argues that estimates need error bars; it would be
incoherent to exempt my own.

### 2. Why quarter-Kelly rather than full Kelly?

Full Kelly is optimal only when the edge parameters are known exactly. They are estimated
from a small sample with substantial error, and Kelly is asymmetrically punishing when
you overestimate your edge — overbetting degrades log growth far faster than underbetting
does. Quarter-Kelly retains most of the growth benefit with much smaller drawdowns and
tolerates parameter error gracefully. Given that a win rate estimated from a few dozen
trades has a standard error of several percentage points, robustness to bad inputs
matters more than optimality conditional on good ones.

### 3. What does a Sharpe confidence interval actually mean?

It is a frequentist statement about the procedure, not about this particular number: if
I repeated the experiment many times, intervals constructed this way would contain the
true Sharpe about 95% of the time. Section 2 tests exactly that and gets 95.6% coverage
over 2,000 simulated years, so the procedure is calibrated. What it does *not* mean is
"there is a 95% probability the true Sharpe is in this range" — that is a Bayesian
statement requiring a prior. And the caveat: this is the IID interval, so under
autocorrelated returns it is too narrow.

### 4. Why report both parametric and historical VaR?

Because their disagreement is information. Parametric VaR assumes normality; historical
does not. When they agree, the normal approximation is adequate at that quantile. When
they diverge, the data has told you something about its tails. In the notebook's
fat-tailed example the parametric 99% VaR understates the empirical one — and it is
precisely the severe tail that a risk limit exists to control, so reporting only the
parametric figure would be wrong in the dangerous direction.

### 5. Why FIFO and not LIFO or average cost?

Three reasons. It matches tax treatment in most jurisdictions, so realised P&L
reconciles with what actually gets reported. It matches what brokers produce, so figures
can be checked against a statement. And it is deterministic — identical executions
always give identical realised P&L, which matters because the alternative conventions
would produce different numbers from the same trades and make any cross-source
comparison meaningless.

### 6. Why would most of a leveraged portfolio's return be beta rather than alpha?

Because leverage multiplies market exposure, and in a rising market that alone produces
large returns. Section 4 constructs a portfolio with exactly zero alpha — 1.8× leveraged
market exposure plus zero-mean idiosyncratic noise — and it beats its benchmark by ten
percentage points over three years. The CAPM decomposition attributes 98.9% of the excess
return to beta and returns an alpha interval containing zero. The chart looks like skill;
the regression shows there is none. This is why raw return comparisons between a
leveraged portfolio and an unleveraged index are close to meaningless, and why the
risk-adjusted view is the only fair one.

### 7. How long a track record would you need to prove you have skill?

At a true Sharpe of 1.0, about 5.8 years of daily data to reject "Sharpe = 0" at 95%
confidence. At 0.5 it is over 17 years. The derivation is in section 3.1: setting the
expected estimate equal to 1.96 standard errors and solving for the sample length. This
is why fund track records are quoted in years and why a strong single year is weak
evidence — and it applies to institutions with far more data than I have.

### 8. Someone shows you a strategy with a backtested Sharpe of 2. What do you ask?

How many strategies were tried. Section 3.2 simulates 1,000 traders with exactly zero
skill and finds the best one posts a one-year Sharpe above 3.2, with roughly 16% of them
clearing 1.0. The maximum of many draws from a zero-centred distribution is large by
construction. Without knowing the number of specifications tested, a single reported
Sharpe carries almost no information. I would also ask about transaction costs,
survivorship in the universe, and whether the parameters were chosen on the same data
being reported.

### 9. What is the biggest weakness in this repository?

That validation on IID normal simulated data proves the arithmetic is right, not that the
assumptions hold. Real returns are neither IID nor normal. The concrete consequence is
the Lo standard error: I use the IID form, so every interval here is a lower bound on the
true uncertainty for an autocorrelated series. Section 1.3 partially addresses the
normality half by breaking it deliberately and showing the VaR estimators diverge, but
the autocorrelation half is unaddressed. The Newey-West-corrected standard error is the
first thing I would add.

### 10. What would you build next?

In order: the autocorrelation-adjusted Lo standard error, because it is the stated
limitation and it changes every interval the engine reports. Then expected shortfall
alongside VaR, since that is what post-Basel-III risk teams actually use. Then
multi-factor attribution — Fama-French at minimum — because single-factor alpha against
SPY misattributes factor exposure as skill for any concentrated book. Then financing
costs in the FIFO engine, since on a leveraged portfolio margin interest and borrow are a
material drag the engine currently ignores entirely.
