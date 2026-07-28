"""CAPM alpha/beta and benchmark comparison.

The headline regression is on EXCESS returns — that is CAPM alpha:

    (r_p - r_f) = alpha + beta * (r_b - r_f) + eps

``alpha`` is a daily intercept, annualised by * 252, and reported with the OLS
standard error of the intercept (also annualised) and a 95% CI. A raw-return
regression (no rf subtraction) is kept as ``raw_alpha_beta`` for comparison —
with a constant rf the beta is identical and only the intercept shifts by
``(1 - beta) * rf_daily``.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


@dataclass
class CapmEstimate:
    """CAPM regression result: annualised alpha with CI, beta, and fit stats."""

    alpha_ann: float
    alpha_se_ann: float
    alpha_ci_low: float
    alpha_ci_high: float
    beta: float
    beta_se: float
    r_squared: float
    n_obs: int


class BenchmarkAnalysis:
    """Compare daily portfolio returns against benchmark price series.

    Parameters
    ----------
    portfolio_returns : pd.Series — daily arithmetic returns, date-indexed
    benchmark_prices  : pd.DataFrame — one price column per benchmark, date-indexed
    rf : float — annualised risk-free rate (converted to daily as rf/252)
    """

    def __init__(
        self,
        portfolio_returns: pd.Series,
        benchmark_prices: pd.DataFrame,
        rf: float = 0.0,
    ):
        bench_returns = benchmark_prices.sort_index().pct_change().dropna()
        idx = portfolio_returns.index.intersection(bench_returns.index)
        if len(idx) < 30:
            raise ValueError(
                f"only {len(idx)} overlapping dates between portfolio and benchmarks — "
                "need at least 30 for a meaningful regression"
            )
        self.port = portfolio_returns.loc[idx]
        self.bench = bench_returns.loc[idx]
        self.bench_prices = benchmark_prices.loc[benchmark_prices.index.intersection(idx)]
        self.rf = rf
        self.rf_daily = rf / TRADING_DAYS

    def capm(self, benchmark: str) -> CapmEstimate:
        """OLS of daily excess portfolio returns on daily excess benchmark returns."""
        excess_p = self.port.values - self.rf_daily
        excess_b = self.bench[benchmark].values - self.rf_daily
        fit = stats.linregress(excess_b, excess_p)
        n = len(excess_p)
        z = 1.959963984540054  # norm.ppf(0.975); n>30 so normal ≈ t
        alpha_ann = fit.intercept * TRADING_DAYS
        alpha_se_ann = fit.intercept_stderr * TRADING_DAYS
        return CapmEstimate(
            alpha_ann=float(alpha_ann),
            alpha_se_ann=float(alpha_se_ann),
            alpha_ci_low=float(alpha_ann - z * alpha_se_ann),
            alpha_ci_high=float(alpha_ann + z * alpha_se_ann),
            beta=float(fit.slope),
            beta_se=float(fit.stderr),
            r_squared=float(fit.rvalue**2),
            n_obs=n,
        )

    def raw_alpha_beta(self, benchmark: str) -> tuple[float, float]:
        """Raw-return regression (no rf): annualised intercept and beta.

        Kept for comparison with the CAPM version; not a headline number.
        """
        fit = stats.linregress(self.bench[benchmark].values, self.port.values)
        return float(fit.intercept * TRADING_DAYS), float(fit.slope)

    def rolling_beta(self, benchmark: str, window: int = 63) -> pd.Series:
        """Rolling OLS beta over `window` trading days."""
        b = self.bench[benchmark]
        cov = self.port.rolling(window).cov(b)
        var = b.rolling(window).var()
        return cov / var

    def information_ratio(self, benchmark: str) -> float:
        """Annualised mean active return over its standard deviation.

        Returns NaN when the tracking error is numerically zero (a portfolio
        that clones the benchmark has no meaningful information ratio).
        """
        active = self.port - self.bench[benchmark]
        sd = active.std(ddof=1)
        if sd <= 1e-12 * max(self.port.abs().max(), 1e-12):
            return float("nan")
        return float(active.mean() / sd * np.sqrt(TRADING_DAYS))

    def cumulative_indexed(self) -> pd.DataFrame:
        """Portfolio and benchmarks as growth of 100 from the common start date."""
        df = pd.DataFrame({"Portfolio": self.port})
        for col in self.bench.columns:
            df[col] = self.bench[col]
        return (1 + df).cumprod() * 100

    def correlation_matrix(self) -> pd.DataFrame:
        df = pd.DataFrame({"Portfolio": self.port})
        for col in self.bench.columns:
            df[col] = self.bench[col]
        return df.corr()

    def summary(self) -> pd.DataFrame:
        """One row per benchmark: CAPM alpha (with CI), beta, IR, correlation."""
        rows = []
        for bm in self.bench.columns:
            est = self.capm(bm)
            rows.append(
                {
                    "benchmark": bm,
                    "capm_alpha_ann": est.alpha_ann,
                    "alpha_ci_low": est.alpha_ci_low,
                    "alpha_ci_high": est.alpha_ci_high,
                    "beta": est.beta,
                    "r_squared": est.r_squared,
                    "info_ratio": self.information_ratio(bm),
                    "correlation": float(self.port.corr(self.bench[bm])),
                }
            )
        return pd.DataFrame(rows)
