# TradeMetrics — Portfolio Analytics & Performance Attribution

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![pandas](https://img.shields.io/badge/pandas-2.0+-150458?style=flat&logo=pandas)](https://pandas.pydata.org)
[![Interactive Brokers](https://img.shields.io/badge/IBKR-API%20Connected-red?style=flat)](https://interactivebrokers.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

> A Python analytics system ingesting 12 months of Interactive Brokers trade data to evaluate portfolio performance, attribute P&L, and drive position sizing decisions in an active leveraged equity portfolio.

---

## Overview

TradeMetrics was built out of a practical need: I was running an active leveraged portfolio of 20–50 positions across US tech equities through Interactive Brokers, and the native IBKR analytics weren't giving me the depth of attribution and risk-adjusted analysis I needed to make better capital allocation decisions.

The system connects directly to the IBKR TWS API, pulls 12 months of execution data and daily NAV history, and pipes it through a custom metrics engine that calculates Sharpe ratio, maximum drawdown, VaR, and per-position P&L attribution. The outputs feed directly into a position sizing module that uses fractional Kelly criterion and volatility-scaled allocation to recommend trade sizes.

A web-based dashboard (`dashboard.html`) visualises everything — cumulative performance vs SPY/QQQ benchmarks, monthly return breakdowns, and a live positions table sortable by any metric.

---

## Features

### Metrics Engine (`MetricsEngine`)
- **Sharpe Ratio** — annualised, with configurable risk-free rate (default: 4.5% ECB)
- **Sortino Ratio** — downside deviation adjusted
- **Calmar Ratio** — return / max drawdown
- **Maximum Drawdown** — peak-to-trough with full drawdown time series
- **Daily VaR** — parametric at 95% and 99% confidence
- **Win Rate & Profit Factor** — across all closed FIFO-matched positions
- **Rolling Sharpe** — 63-day (≈ 3 month) rolling window

### P&L Attribution (`AttributionEngine`)
- P&L decomposed by individual position and by sector
- Monthly and weekly period-level return breakdown
- Top contributors and detractors ranked automatically

### Benchmark Analysis (`BenchmarkAnalysis`)
- Cumulative return comparison vs SPY, QQQ (or any price series)
- OLS regression for alpha and beta
- Information Ratio and correlation matrix

### Position Sizing (`PositionSizer`)
- **Fractional Kelly Criterion** — size scaled by win rate and payoff ratio
- **Volatility-scaled allocation** — equal risk contribution per position
- Hard guardrails: max 1.5% NAV at risk per trade

---

## Project Structure

```
TradeMetrics/
│
├── portfolio_engine.py        # Core analytics library
│   ├── IBKRDataLoader         # TWS API ingestion + CSV fallback
│   ├── MetricsEngine          # Risk-adjusted performance metrics
│   ├── AttributionEngine      # P&L attribution by position & period
│   ├── BenchmarkAnalysis      # Alpha/beta vs SPY, QQQ
│   └── PositionSizer          # Kelly + vol-scaled position sizing
│
├── dashboard.html             # Interactive analytics dashboard
│
├── data/
│   ├── trades_sample.csv      # Sample trade log (anonymised)
│   └── nav_sample.csv         # Sample NAV history
│
├── outputs/                   # Generated charts and reports
│
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the demo (no IBKR connection needed)

```bash
python portfolio_engine.py
```

This generates synthetic but realistic portfolio data and prints a full performance summary, benchmark comparison, and example position sizing output.

### 3. Connect to IBKR TWS (live data)

Make sure TWS or IB Gateway is running with API access enabled (Edit → Global Configuration → API → Enable ActiveX and Socket Clients).

```python
from portfolio_engine import IBKRDataLoader, MetricsEngine

loader = IBKRDataLoader(host="127.0.0.1", port=7497)
trades = loader.load_trades()    # pulls last 12 months of executions
nav    = loader.load_nav()       # pulls daily NAV history

engine = MetricsEngine(nav, trades)
print(engine.summary())
```

### 4. Open the dashboard

Open `dashboard.html` in any browser. It runs fully client-side — no server needed.

---

## Sample Output

```
============================================================
  TradeMetrics — Analytics Engine Demo
============================================================

Performance Summary
----------------------------------------
Total Return (%)          34.70
Annualised Return (%)     34.70
Annualised Vol (%)        18.70
Sharpe Ratio               1.84
Sortino Ratio              2.31
Calmar Ratio               2.82
Max Drawdown (%)          -12.30
Daily VaR 95% (€)          -621
Win Rate (%)               68.40
Profit Factor               2.10

Benchmark Comparison
----------------------------------------
Benchmark  Alpha (ann)   Beta  Info Ratio  Correlation
      SPY       10.60%   1.34        1.12        0.781
      QQQ        5.40%   1.18        0.87        0.803

Position Sizing — NVDA
----------------------------------------
  ticker                 NVDA
  kelly_fraction         0.0847
  vol_scaled             0.0395
  recommended            0.0395
  eur_allocation         1,887.00
  max_loss_eur             717.30
```

---

## Methodology Notes

**Sharpe Ratio** is calculated on annualised excess returns over the ECB deposit rate (4.5% for the analysis period), using realised daily returns from NAV changes rather than mark-to-market position values.

**P&L Attribution** uses FIFO matching of buy and sell executions per ticker. This is consistent with how gains are reported for tax purposes and gives a clean view of realised performance per position.

**Position Sizing** uses quarter-Kelly (25% of the full Kelly fraction) as a conservative multiplier, capped by a volatility-scaling constraint that targets equal risk contribution across positions. In practice this prevents over-concentration in high-momentum names even when the Kelly fraction suggests aggressive sizing.

**Drawdown** is measured from daily NAV peaks, capturing the actual lived experience of portfolio losses including leveraged positions.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `pandas` | Data manipulation, time series |
| `numpy` | Numerical computation, rolling statistics |
| `ib_insync` | Interactive Brokers TWS API wrapper |
| `Chart.js` | Dashboard visualisations |
| Python 3.11 | Core runtime |

---

## Background

Built during final year of a Biology BSc at University College Dublin, alongside running an active IBKR portfolio of 20–50 leveraged equity positions. The project was motivated by wanting proper quantitative feedback on position sizing decisions — specifically whether the Sharpe and drawdown profile of the portfolio justified the leverage being employed, and which positions were actually generating alpha vs just riding beta.

The analytics confirmed that tech-concentrated long positions (NVDA, ORCL, MSFT) drove the majority of alpha generation, while short hedges via SPY were a consistent performance drag — informing a shift toward more selective hedging using single-stock puts rather than index shorts.

---

## License

MIT — free to use and adapt.
