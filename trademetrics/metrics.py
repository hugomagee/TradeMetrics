"""Risk-adjusted performance metrics from a daily NAV series.

Conventions (stated once, applied everywhere):

- Daily returns are arithmetic: ``r_t = nav_t / nav_{t-1} - 1``.
- The annualised risk-free rate ``rf`` is converted to a daily rate
  arithmetically, ``rf_daily = rf / 252``. Over one year the difference vs the
  geometric conversion ``(1+rf)**(1/252) - 1`` is below half a basis point on
  the daily rate; we take the arithmetic form because Sharpe/Sortino are built
  from arithmetic daily excess returns.
- Sharpe is the *arithmetic* daily Sharpe annualised by sqrt(252):
  ``mean(r - rf_daily) / std(r) * sqrt(252)``. The alternative "geometric"
  headline (annualised compound return minus rf, over annualised vol) is what
  the original version of this project reported; it mixes a compound numerator
  with an arithmetic-vol denominator and is not what the Lo (2002) standard
  error applies to, so it is not used here.
- Every Sharpe estimate is reported with the Lo (2002) IID standard error
  ``SE(SR_daily) = sqrt((1 + SR_daily^2 / 2) / n)`` scaled by sqrt(252), and a
  95% normal CI. Under autocorrelated returns the true SE is larger, so these
  intervals are a lower bound on the uncertainty.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


@dataclass
class SharpeEstimate:
    """Annualised Sharpe ratio with Lo (2002) standard error and 95% CI."""

    sharpe: float
    se: float
    ci_low: float
    ci_high: float
    n_obs: int

    def __format__(self, spec: str) -> str:
        return f"{self.sharpe:.2f} (95% CI {self.ci_low:.2f} to {self.ci_high:.2f}, n={self.n_obs})"


def sharpe_with_ci(returns: pd.Series | np.ndarray, rf: float = 0.0) -> SharpeEstimate:
    """Annualised Sharpe from daily returns, with Lo (2002) SE and 95% CI.

    Parameters
    ----------
    returns : daily arithmetic returns
    rf      : annualised risk-free rate (converted to daily as rf/252)
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 2:
        raise ValueError("need at least 2 return observations")
    rf_daily = rf / TRADING_DAYS
    sd = r.std(ddof=1)
    # A constant series does not give std exactly 0 in floating point (the mean
    # is not representable), so compare against the scale of the data.
    if sd <= 1e-12 * max(np.abs(r).max(), 1e-12):
        raise ValueError("zero return volatility — Sharpe undefined")
    sr_daily = (r.mean() - rf_daily) / sd
    se_daily = math.sqrt((1 + 0.5 * sr_daily**2) / n)
    sr_ann = sr_daily * math.sqrt(TRADING_DAYS)
    se_ann = se_daily * math.sqrt(TRADING_DAYS)
    z = 1.959963984540054  # norm.ppf(0.975)
    return SharpeEstimate(
        sharpe=sr_ann,
        se=se_ann,
        ci_low=sr_ann - z * se_ann,
        ci_high=sr_ann + z * se_ann,
        n_obs=n,
    )


class MetricsEngine:
    """Compute performance and risk metrics from a daily NAV series.

    Parameters
    ----------
    nav : pd.Series
        Daily portfolio value, date-indexed. Gaps (weekends/holidays) are fine;
        annualisation assumes the observations are trading days.
    realised_pnl : pd.DataFrame, optional
        Output of :meth:`trademetrics.fifo.FifoEngine.realised_pnl`. Needed only
        for the trade-level statistics (win rate, profit factor, holding period).
    rf : float
        Annualised risk-free rate. The demo passes 0.02; pass the rate that
        matches your currency and period — every metric that uses it takes it
        from this single place.
    """

    def __init__(self, nav: pd.Series, realised_pnl: pd.DataFrame | None = None, rf: float = 0.0):
        nav = nav.sort_index().dropna()
        if len(nav) < 3:
            raise ValueError("need at least 3 NAV observations")
        self.nav = nav
        self.rf = rf
        self.rf_daily = rf / TRADING_DAYS
        self.returns = nav.pct_change().dropna()
        self.realised = realised_pnl

    # ── return / volatility ────────────────────────────────────────────────
    def total_return(self) -> float:
        return float(self.nav.iloc[-1] / self.nav.iloc[0] - 1)

    def annualised_return(self) -> float:
        """Geometric annualisation of the total return over the sample."""
        n = len(self.returns)
        return float((1 + self.total_return()) ** (TRADING_DAYS / n) - 1)

    def annualised_volatility(self) -> float:
        return float(self.returns.std(ddof=1) * np.sqrt(TRADING_DAYS))

    # ── risk-adjusted ratios ───────────────────────────────────────────────
    def sharpe_ratio(self) -> SharpeEstimate:
        return sharpe_with_ci(self.returns, rf=self.rf)

    def sortino_ratio(self) -> float:
        """Sortino with target = daily risk-free rate.

        Downside deviation is computed over ALL observations:
        ``sqrt(mean(min(r - target, 0)^2))`` — days above target contribute
        zero, they are not dropped from the denominator's sample size.
        """
        shortfall = np.minimum(self.returns.values - self.rf_daily, 0.0)
        downside_daily = math.sqrt(np.mean(shortfall**2))
        if downside_daily == 0:
            return float("nan")
        excess_ann = (self.returns.mean() - self.rf_daily) * TRADING_DAYS
        return float(excess_ann / (downside_daily * math.sqrt(TRADING_DAYS)))

    def calmar_ratio(self) -> float:
        mdd = abs(self.max_drawdown())
        return float(self.annualised_return() / mdd) if mdd != 0 else float("nan")

    # ── drawdown ───────────────────────────────────────────────────────────
    def drawdown_series(self) -> pd.Series:
        roll_max = self.nav.cummax()
        return (self.nav - roll_max) / roll_max

    def max_drawdown(self) -> float:
        return float(self.drawdown_series().min())

    # ── value at risk ──────────────────────────────────────────────────────
    def var_parametric(self, confidence: float = 0.95) -> float:
        """One-day parametric (normal) VaR as a positive loss fraction of NAV.

        ``VaR = -(mu + z_(1-c) * sigma)`` with mu the mean daily return — the
        mean term is included, so a strongly positive drift reduces the VaR.
        """
        mu = self.returns.mean()
        sigma = self.returns.std(ddof=1)
        z = stats.norm.ppf(1 - confidence)
        return float(-(mu + z * sigma))

    def var_historical(self, confidence: float = 0.95) -> float:
        """One-day historical-simulation VaR: the empirical (1-c) quantile."""
        return float(-np.quantile(self.returns.values, 1 - confidence))

    # ── rolling ────────────────────────────────────────────────────────────
    def rolling_sharpe(self, window: int = 63) -> pd.Series:
        """Rolling annualised Sharpe over `window` trading days (default ≈ 3 months)."""
        mean = self.returns.rolling(window).mean()
        std = self.returns.rolling(window).std(ddof=1)
        return (mean - self.rf_daily) / std * np.sqrt(TRADING_DAYS)

    def rolling_volatility(self, window: int = 21) -> pd.Series:
        return self.returns.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)

    # ── trade-level statistics (need realised FIFO P&L) ───────────────────
    def _require_realised(self) -> pd.DataFrame:
        if self.realised is None or self.realised.empty:
            raise ValueError("no realised P&L supplied — pass FifoEngine.realised_pnl()")
        return self.realised

    def win_rate(self) -> float:
        """Fraction of realised round-trips with positive net (post-commission) P&L."""
        realised = self._require_realised()
        return float((realised["net_pnl"] > 0).mean())

    def profit_factor(self) -> float:
        """Gross winnings / gross losses on net realised P&L."""
        realised = self._require_realised()
        wins = realised.loc[realised["net_pnl"] > 0, "net_pnl"].sum()
        losses = abs(realised.loc[realised["net_pnl"] <= 0, "net_pnl"].sum())
        return float(wins / losses) if losses != 0 else float("nan")

    def avg_holding_days(self) -> float:
        realised = self._require_realised()
        held = (realised["exit_date"] - realised["entry_date"]).dt.days
        return float(held.mean())

    # ── summary ────────────────────────────────────────────────────────────
    def summary(self) -> dict:
        """All headline metrics. Sharpe is a dict carrying its CI — always."""
        sharpe = self.sharpe_ratio()
        out = {
            "total_return": self.total_return(),
            "annualised_return": self.annualised_return(),
            "annualised_volatility": self.annualised_volatility(),
            "sharpe": sharpe.sharpe,
            "sharpe_se": sharpe.se,
            "sharpe_ci_low": sharpe.ci_low,
            "sharpe_ci_high": sharpe.ci_high,
            "n_obs": sharpe.n_obs,
            "sortino": self.sortino_ratio(),
            "calmar": self.calmar_ratio(),
            "max_drawdown": self.max_drawdown(),
            "var_95_parametric": self.var_parametric(0.95),
            "var_95_historical": self.var_historical(0.95),
            "var_99_parametric": self.var_parametric(0.99),
            "var_99_historical": self.var_historical(0.99),
            "rf": self.rf,
        }
        if self.realised is not None and not self.realised.empty:
            out["win_rate"] = self.win_rate()
            out["profit_factor"] = self.profit_factor()
            out["avg_holding_days"] = self.avg_holding_days()
            out["n_round_trips"] = int(len(self.realised))
        return out
