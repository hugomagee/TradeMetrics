# TradeMetrics — a portfolio analytics engine, validated against known truth

[![CI](https://github.com/hugomagee/TradeMetrics/actions/workflows/ci.yml/badge.svg)](https://github.com/hugomagee/TradeMetrics/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

A Python engine that computes risk-adjusted performance metrics, CAPM attribution and
long/short FIFO profit and loss — and a notebook that proves the engine is correct by
running it against simulated data whose true parameters are known in advance.

**The repository contains no real trading data.** My own brokerage account is private and
deliberately absent. The claim this repo makes is not "look at my returns" — it is
"here is the estimator, and here is the evidence it is unbiased and that its error bars
are honest." The second claim is one you can check line by line from a fresh clone.

![TradeMetrics report](docs/screenshots/report-overview.png)

---

## Why this exists

I built the first version of this to analyse my own leveraged equity portfolio. Reading
it back later, it had the failure mode that a lot of retail performance code has: it
reported a confident-looking Sharpe ratio, and almost every number feeding that ratio was
computed incorrectly. Sortino used the wrong downside deviation. Alpha regressed raw
rather than excess returns. VaR had a hardcoded z-score lookup. The FIFO matcher loaded
commissions and never subtracted them, and silently dropped every short sale.

So the project was rebuilt around a different question. Not *what did my portfolio do*,
which is unverifiable by a reader and — as the notebook demonstrates — barely
distinguishable from noise over a single year anyway. Instead: **is this code right?**
That question has a checkable answer.

Full methodology defence and interview questions: [docs/ANALYSIS_NOTES.md](docs/ANALYSIS_NOTES.md).
The defensible CV claims and where each is proved: [docs/CV_CLAIMS.md](docs/CV_CLAIMS.md).

---

## Quickstart

Requires Python 3.12+.

```bash
git clone https://github.com/hugomagee/TradeMetrics.git
cd TradeMetrics
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Run the demo

```bash
python -m trademetrics.demo
```

Deterministic — fixed seed, fixed date range, no `datetime.today()` anywhere — so it
prints exactly this on every machine:

```
================================================================
  TradeMetrics — engine demo on committed synthetic sample data
  Period 2024-01-02 to 2024-12-31  ·  rf = 2.0% (annualised)
================================================================

  Performance
--------------------------------------------
  Total return            -6.46%
  Annualised return       -6.26%
  Annualised volatility   16.21%
  Sharpe ratio            -0.44  (95% CI -2.37 to 1.49, Lo SE 0.98, n=260)
  Sortino ratio           -0.63
  Calmar ratio            -0.36
  Max drawdown            -17.43%

  One-day Value at Risk (fraction of NAV)
--------------------------------------------
  95% parametric          1.70%
  95% historical          1.66%
  99% parametric          2.40%
  99% historical          2.03%

  Benchmark comparison (CAPM, excess-return regression)
--------------------------------------------
  SPY   alpha -5.35%/yr (95% CI -21.11% to +10.42%)  beta 1.25  R² 0.75  IR -0.66
  QQQ   alpha -7.48%/yr (95% CI -26.55% to +11.59%)  beta 0.90  R² 0.63  IR -0.75

  Realised FIFO P&L (net of commissions)
--------------------------------------------
  Round-trips             13
  Win rate                61.5%
  Profit factor           5.86
  Avg holding period      169.0 days
```

That Sharpe of −0.44 is the sample portfolio's, and it means nothing: the sample data
exists to exercise every code path (longs, shorts, partial fills, multi-lot FIFO,
commissions), not to demonstrate performance. Its 95% confidence interval runs from
−2.37 to +1.49, which is the honest width of any one-year Sharpe estimate.

### Build the report

```bash
python tools/build_dashboard.py
```

Writes `trademetrics.html` — a single self-contained page with no network dependency,
built from `outputs/results.json`. Open it in any browser.

### Run the tests

```bash
pytest
```

79 tests. Every metric is checked against a hand-computed fixture, and the FIFO engine
is checked against trade sequences whose expected P&L is worked out by hand in the test
comments.

### Run the validation notebook

```bash
jupyter notebook analysis/metric_validation.ipynb
```

Executes end to end in about seven seconds. CI runs it on every push.

---

## What the validation notebook shows

[`analysis/metric_validation.ipynb`](analysis/metric_validation.ipynb) runs entirely on
simulated data with known ground truth.

| Section | Finding |
|---|---|
| 1 · Estimator correctness | Bias is near zero at every sample length, and the empirical spread of estimates matches the analytical Lo (2002) standard error. Each metric also agrees with a hand calculation to machine precision. |
| 2 · Uncertainty calibration | Nominal 95% Sharpe intervals cover the true value **95.6%** of the time across 2,000 simulated years. The standard error of a one-year Sharpe is **≈ 1.0** — the same magnitude as the estimate itself. |
| 3 · Skill vs luck | A true Sharpe of 1.0 needs **≈ 5.8 years** of daily data to separate from zero at 95% confidence. The best of 1,000 traders with *zero* skill posts a one-year Sharpe above 3.2. |
| 4 · The beta illusion | A 1.8× leveraged portfolio with **zero alpha by construction** beats its benchmark by 10 points over three years. The CAPM decomposition attributes **98.9%** of the excess return to beta and returns an alpha interval containing zero. |
| 5 · FIFO validation | Six hand-computed trade sequences — longs, shorts, partial fills, multi-lot ordering, a long-to-short flip — all match to the cent. |
| 6 · Limitations | Written as the skeptical reviewer, including what the IID Lo standard error does *not* cover. |
| 7 · Conclusion | What the engine can and cannot claim. |

---

## Methodology

Every choice below is defended at length in [docs/ANALYSIS_NOTES.md](docs/ANALYSIS_NOTES.md).

**Sharpe ratio.** Mean daily excess return over daily standard deviation, annualised by
√252 — arithmetic, not geometric, because that is the quantity the Lo (2002) standard
error applies to. **Every Sharpe the code reports carries its standard error and 95%
confidence interval**, in the terminal, in the JSON, on the report page and in this
README. A point estimate without an interval is the thing this project exists to argue
against.

**Sortino ratio.** Downside deviation is `sqrt(mean(min(r - target, 0)^2))` computed over
*all* observations, not the standard deviation of the negative subset. Days above target
contribute zero to the numerator but still count in the denominator's sample size. Target
is the daily risk-free rate. On the notebook's worked example the common buggy form
overstates the ratio by 11.8%.

**Alpha and beta.** CAPM: excess portfolio returns regressed on excess benchmark returns,
with the OLS intercept standard error annualised and reported as a confidence interval.
The raw-return variant is kept and labelled for comparison.

**Value at risk.** Parametric VaR uses `scipy.stats.norm.ppf` for any confidence level and
includes the mean term. Historical-simulation VaR is reported alongside it, because the
gap between the two is the diagnostic — on fat-tailed returns the parametric figure
overstates moderate losses and understates severe ones.

**FIFO P&L.** Full long *and* short matching, oldest lot first, with commissions allocated
per share across both legs so a partial close carries only its proportional share. A sell
with no open lot opens a short rather than being discarded; a single oversized execution
can close a position and open one in the opposite direction.

**Position sizing.** Fractional Kelly (quarter-Kelly by default) and volatility targeting.
Volatility targeting is named honestly: it scales by standalone volatility and ignores
correlation, so it is *not* equal risk contribution. True covariance-based ERC is
implemented separately as `erc_weights`, and the notebook shows where the two diverge.

---

## Repository layout

```
TradeMetrics/
├── trademetrics/              # the engine
│   ├── metrics.py             # Sharpe (with Lo CI), Sortino, VaR, drawdown
│   ├── fifo.py                # long/short FIFO with commission-adjusted P&L
│   ├── benchmark.py           # CAPM alpha/beta on excess returns
│   ├── attribution.py         # realised P&L by ticker, sector, period
│   ├── sizing.py              # fractional Kelly, vol targeting, true ERC
│   ├── simulate.py            # deterministic generators with known parameters
│   ├── loaders.py             # CSV loading and trade-log cleaning
│   └── demo.py                # end-to-end demo, writes outputs/results.json
├── analysis/
│   └── metric_validation.ipynb
├── data/samples/              # synthetic; regenerate with tools/make_sample_data.py
│   ├── nav_sample.csv
│   ├── trades_sample.csv
│   └── benchmarks_sample.csv
├── tests/                     # 79 tests
├── tools/
│   ├── make_sample_data.py    # regenerates data/samples/
│   └── build_dashboard.py     # renders trademetrics.html from results.json
├── examples/
│   └── ibkr_loader.py         # optional broker adapter — untested, unused
├── docs/
│   ├── ANALYSIS_NOTES.md
│   └── CV_CLAIMS.md
├── outputs/results.json       # engine output, the report's only data source
└── trademetrics.html          # generated report
```

### On the data

Everything in `data/samples/` is synthetic, generated from fixed seeds by
`tools/make_sample_data.py`. A test asserts the committed CSVs are byte-identical to what
the generator produces, so the data cannot drift away from its stated provenance.

`examples/ibkr_loader.py` sketches what a broker adapter would look like. It is not
imported by the package, not tested, and not run in CI — it is marked as such in its own
docstring. **No part of this repository requires or establishes a broker connection.**
`private_data/` is gitignored as a standing guard.

---

## Portfolio context

This repository is one of a pair, sharing a thesis: **I audit my own numbers — and I
don't publish ones I can't back.**

- **TradeMetrics** (this repo) — a quantitative engine, validated against simulated data
  with known ground truth. The personal performance figure it used to advertise has been
  removed, because a one-year Sharpe from a private account is both unverifiable and, by
  this repo's own analysis, statistically indistinguishable from noise.
- **[OptimalAthlete](https://github.com/hugomagee/OptimalAthlete)** — an n=1 training-data
  measurement methodology, where I publicly retracted my own inflated R² after finding
  the leakage in my evaluation protocol that produced it.

Both repos are CI-tested, deterministic, and reproducible from a fresh clone.

---

## Tech stack

| Tool | Purpose |
|---|---|
| pandas, numpy | data manipulation, time series |
| scipy | OLS regression, normal quantiles, ERC optimisation |
| pytest | 79-test suite |
| ruff | linting |
| Jupyter | validation notebook, executed in CI |

The report page is hand-written HTML with inline SVG charts and no JavaScript
dependencies, so it stays a single self-contained file that works offline.

---

## License

MIT — see [LICENSE](LICENSE).
