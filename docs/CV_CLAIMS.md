# CV claims this repository supports

Every bullet below is defensible verbatim, with a pointer to the code or notebook cell
that proves it. Anything not on this list is not supported by this repository and should
not be claimed on the strength of it.

Last verified: 2026-07-28, against commit on `main`. Reproduce everything with
`pytest && jupyter nbconvert --to notebook --execute analysis/metric_validation.ipynb`.

---

## Defensible bullets

### Primary (use this one)

> **Built and statistically validated a portfolio-analytics engine in Python** — CAPM
> attribution, Sharpe ratios with Lo (2002) confidence intervals, parametric and
> historical VaR, and long/short FIFO P&L — **verifying every estimator against simulated
> data with known ground truth.** Monte Carlo analysis showed nominal 95% Sharpe
> intervals achieve 95.6% coverage, and that a one-year Sharpe estimate carries a
> standard error near 1.0, i.e. single-year performance figures are largely noise.

| Clause | Proof |
|---|---|
| CAPM attribution | [`trademetrics/benchmark.py`](../trademetrics/benchmark.py) · `tests/test_benchmark.py::test_beta_recovered_exactly_for_a_noiseless_leveraged_clone` |
| Sharpe with Lo (2002) CI | [`trademetrics/metrics.py`](../trademetrics/metrics.py) `sharpe_with_ci` · `tests/test_metrics.py::test_sharpe_with_ci_hand_computed` |
| Parametric and historical VaR | `metrics.py` `var_parametric` / `var_historical` · notebook §1.3 |
| Long/short FIFO P&L | [`trademetrics/fifo.py`](../trademetrics/fifo.py) · `tests/test_fifo.py` (13 tests) |
| Validated against known ground truth | Notebook §1 — bias near zero at every sample length |
| 95.6% interval coverage | Notebook §2, coverage experiment over 2,000 simulated years |
| One-year Sharpe SE ≈ 1.0 | Notebook §2, empirical SD of estimates = 0.994 |

### Secondary (risk-focused roles)

> **Demonstrated quantitatively why leveraged market exposure is mistaken for skill:**
> simulated a 1.8× leveraged portfolio with zero alpha by construction, which outperformed
> its benchmark by 10 percentage points over three years while CAPM decomposition
> attributed 98.9% of excess return to beta and returned an alpha confidence interval
> containing zero.

Proof: notebook §4.

### Secondary (statistical rigour)

> **Quantified the selection effect in performance reporting:** showed that the best of
> 1,000 simulated zero-skill traders posts a one-year Sharpe above 3.2, and derived that
> a true Sharpe of 1.0 requires ~5.8 years of daily data to distinguish from zero at 95%
> confidence.

Proof: notebook §3.1 and §3.2.

### Secondary (software engineering)

> **Rewrote a portfolio analytics library, fixing six methodology defects** — including a
> Sortino denominator computed over the wrong sample, an alpha regression on raw rather
> than excess returns, a hardcoded VaR quantile lookup, and a FIFO matcher that dropped
> short sales and ignored commissions — **backed by a 79-test suite and CI that verifies
> determinism end to end.**

Proof: [`docs/ANALYSIS_NOTES.md`](ANALYSIS_NOTES.md) §2–7 documents each defect;
`tests/` covers each fix; [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)
regenerates the sample data, re-runs the demo, rebuilds the report and fails on any diff.

---

## Claims this repository does NOT support

Listed explicitly so they are not made by accident.

| Retired claim | Why it is gone |
|---|---|
| **"Achieved Sharpe ratio 1.35"** | Unverifiable from a private account; computed by code that has since been shown to contain errors in the Sortino, alpha, VaR and FIFO paths; and — by this repository's own §2 analysis — statistically indistinguishable from zero over one year. **This claim should be removed from the CV**, not merely from the repo. Keeping an unverifiable single-year Sharpe on a CV while the repository teaches that single-year Sharpes are noise recreates the exact contradiction this work removes. |
| **"12 months of IBKR data via API"** | No live NAV pull was ever implemented, and the repository now works exclusively on committed synthetic data. `examples/ibkr_loader.py` is an untested sketch, excluded from CI and from the package. |
| Any statement about personal trading performance | The data is private and not in the repository. Nothing here is evidence about it in either direction. |
| "Backtested strategy" / "generated alpha" | The engine measures; it does not implement or test a strategy. |
| Validation under real-world return distributions | Validation is on IID normal simulated data. §1.3 deliberately breaks normality to show the VaR estimators diverge, but autocorrelation is unaddressed — see the limitations in §6 of the notebook. |

---

## If asked about the retraction in an interview

The honest version, which is also the strongest version:

> I originally published a Sharpe of 1.35 from my own account. When I went back to
> validate the engine, I found four separate calculation errors in the code that produced
> it — and then, when I quantified the estimator's uncertainty, I found the standard error
> of a one-year Sharpe is around 1.0, so even a correct 1.35 would have had a confidence
> interval spanning zero. I removed the claim and rebuilt the repository around what can
> actually be verified: that the estimators are unbiased and their error bars are
> calibrated.

The point being demonstrated is not that the first attempt was flawed. It is that the
errors were found by systematic self-auditing, and that the response was to retract
publicly rather than quietly restate.
