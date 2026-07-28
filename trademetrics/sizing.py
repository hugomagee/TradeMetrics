"""Position sizing: fractional Kelly, volatility targeting, and true ERC.

Naming is deliberate. ``vol_target_size`` scales each position so its
STANDALONE volatility contribution hits a target — it ignores correlations and
is NOT equal risk contribution. True ERC (:func:`erc_weights`) solves for
weights whose covariance-based risk contributions are equal, via SLSQP. With an
identity-correlation covariance matrix the two coincide; with correlated assets
they do not — the validation notebook shows the difference.
"""

import numpy as np
from scipy.optimize import minimize


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Full Kelly fraction for a binary-outcome bet.

    f* = (b*p - q) / b  with b = avg_win/avg_loss, p = win_rate, q = 1 - p.
    Returns 0 when the edge is non-positive. ``avg_win``/``avg_loss`` are
    positive magnitudes (e.g. 0.21 and 0.09 for +21% / -9%).
    """
    if not 0 <= win_rate <= 1:
        raise ValueError("win_rate must be in [0, 1]")
    if avg_loss <= 0 or avg_win <= 0:
        raise ValueError("avg_win and avg_loss must be positive magnitudes")
    b = avg_win / avg_loss
    f_full = (b * win_rate - (1 - win_rate)) / b
    return max(0.0, f_full)


def erc_weights(cov: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """True equal-risk-contribution weights for a covariance matrix.

    Solves min Σ_ij (RC_i - RC_j)^2 subject to w ≥ 0, Σw = 1, where
    RC_i = w_i (Σw)_i / (wᵀΣw). For assets with equal correlations this
    reduces to inverse-volatility weighting.
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    if cov.shape != (n, n):
        raise ValueError("cov must be square")

    def objective(w: np.ndarray) -> float:
        port_var = w @ cov @ w
        rc = w * (cov @ w) / port_var
        return float(np.sum((rc[:, None] - rc[None, :]) ** 2))

    w0 = np.full(n, 1.0 / n)
    res = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 500, "ftol": tol},
    )
    if not res.success:
        raise RuntimeError(f"ERC optimisation failed: {res.message}")
    return res.x


class PositionSizer:
    """Translate engine statistics into a recommended position size.

    Parameters
    ----------
    nav : float — current portfolio value
    max_risk_per_trade : float — cap on the fraction of NAV risked per position
    kelly_multiplier : float — fraction of full Kelly used (0.25 = quarter Kelly)
    """

    def __init__(
        self,
        nav: float,
        max_risk_per_trade: float = 0.015,
        kelly_multiplier: float = 0.25,
    ):
        self.nav = nav
        self.max_risk = max_risk_per_trade
        self.kelly_multiplier = kelly_multiplier

    def kelly_size(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Fractional Kelly position size as a fraction of NAV."""
        return kelly_fraction(win_rate, avg_win, avg_loss) * self.kelly_multiplier

    def vol_target_size(self, ticker_vol: float, target_vol: float = 0.15) -> float:
        """Volatility-targeted size: weight such that weight * ticker_vol = target_vol.

        Standalone vol scaling — ignores correlations (see module docstring).
        Capped at 5x the per-trade risk limit.
        """
        if ticker_vol <= 0:
            raise ValueError("ticker_vol must be positive")
        return min(target_vol / ticker_vol, self.max_risk * 5)

    def recommended_size(
        self,
        ticker: str,
        ticker_vol: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> dict:
        """Most-conservative-of(Kelly, vol target, hard cap), sized in currency."""
        kelly = self.kelly_size(win_rate, avg_win, avg_loss)
        vol_t = self.vol_target_size(ticker_vol)
        size = min(kelly, vol_t, self.max_risk * 4)
        return {
            "ticker": ticker,
            "kelly_fraction": kelly,
            "vol_target": vol_t,
            "recommended": size,
            "allocation": size * self.nav,
            "max_loss_at_risk_cap": self.max_risk * self.nav,
        }
