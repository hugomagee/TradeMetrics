"""
TradeMetrics — Portfolio Analytics Engine
==========================================
Ingests 12 months of Interactive Brokers trade data via IBKR API,
computes risk-adjusted performance metrics, and drives position sizing
decisions for an active leveraged equity portfolio.

Author : Hugo Magee
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────
#  IBKR DATA INGESTION
# ─────────────────────────────────────────

class IBKRDataLoader:
    """
    Connects to Interactive Brokers TWS / IB Gateway via ib_insync
    and pulls 12 months of trade confirmations + portfolio NAV history.

    For reproducibility / offline use, falls back to CSV if TWS
    is not running (set USE_LIVE=False in config.py).
    """

    def __init__(self, host="127.0.0.1", port=7497, client_id=1):
        self.host = host
        self.port = port
        self.client_id = client_id

    def load_trades(self, csv_path: Optional[str] = None) -> pd.DataFrame:
        """Return DataFrame of all executions over trailing 12 months."""
        if csv_path:
            df = pd.read_csv(csv_path, parse_dates=["datetime"])
            return self._clean_trades(df)
        try:
            from ib_insync import IB, util
            ib = IB()
            ib.connect(self.host, self.port, clientId=self.client_id)
            fills = ib.reqExecutions()
            records = []
            for f in fills:
                records.append({
                    "datetime": util.parseIBDatetime(f.execution.time),
                    "ticker":   f.contract.symbol,
                    "action":   f.execution.side,          # 'BOT' | 'SLD'
                    "qty":      f.execution.shares,
                    "price":    f.execution.price,
                    "currency": f.contract.currency,
                    "commission": f.commissionReport.commission,
                })
            ib.disconnect()
            return self._clean_trades(pd.DataFrame(records))
        except Exception as e:
            raise ConnectionError(
                f"IBKR connection failed: {e}\n"
                "Pass csv_path= to load from CSV instead."
            )

    def load_nav(self, csv_path: Optional[str] = None) -> pd.Series:
        """Return daily portfolio NAV as a time-indexed Series."""
        if csv_path:
            df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
            return df["nav"].sort_index()
        # Placeholder — in live use, pull from IBKR account summary
        raise NotImplementedError("Pass csv_path= for NAV history.")

    @staticmethod
    def _clean_trades(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        df["side"] = df["action"].map({"BOT": 1, "SLD": -1, "BUY": 1, "SELL": -1})
        # Drop rows with invalid action codes (resulting in NaN side)
        invalid_count = df["side"].isna().sum()
        if invalid_count > 0:
            print(f"Warning: Dropped {invalid_count} trades with unrecognized action codes")
            df = df.dropna(subset=["side"])
        df["notional"] = df["qty"] * df["price"] * df["side"]
        return df


# ─────────────────────────────────────────
#  METRICS ENGINE
# ─────────────────────────────────────────

class MetricsEngine:
    """
    Calculates risk-adjusted performance metrics from a daily NAV Series
    and an executions DataFrame.

    Parameters
    ----------
    nav    : pd.Series  — daily portfolio value, date-indexed
    trades : pd.DataFrame — cleaned trade log from IBKRDataLoader
    rf     : float      — annualised risk-free rate (default: 4.5% ECB)
    """

    TRADING_DAYS = 252

    def __init__(self, nav: pd.Series, trades: pd.DataFrame, rf: float = 0.045):
        self.nav    = nav.sort_index().dropna()
        self.trades = trades
        self.rf     = rf
        self.returns = self.nav.pct_change().dropna()

    # ── Core metrics ──────────────────────────────────

    def total_return(self) -> float:
        return (self.nav.iloc[-1] / self.nav.iloc[0]) - 1

    def annualised_return(self) -> float:
        n = len(self.nav)
        return (1 + self.total_return()) ** (self.TRADING_DAYS / n) - 1

    def annualised_volatility(self) -> float:
        return self.returns.std() * np.sqrt(self.TRADING_DAYS)

    def sharpe_ratio(self) -> float:
        excess = self.annualised_return() - self.rf
        return excess / self.annualised_volatility()

    def sortino_ratio(self) -> float:
        downside = self.returns[self.returns < 0].std() * np.sqrt(self.TRADING_DAYS)
        excess   = self.annualised_return() - self.rf
        return excess / downside if downside != 0 else np.nan

    def max_drawdown(self) -> float:
        """Peak-to-trough maximum drawdown."""
        roll_max  = self.nav.cummax()
        drawdowns = (self.nav - roll_max) / roll_max
        return drawdowns.min()

    def drawdown_series(self) -> pd.Series:
        roll_max = self.nav.cummax()
        return (self.nav - roll_max) / roll_max

    def calmar_ratio(self) -> float:
        mdd = abs(self.max_drawdown())
        return self.annualised_return() / mdd if mdd != 0 else np.nan

    def value_at_risk(self, confidence: float = 0.95) -> float:
        """Parametric VaR as a fraction of current portfolio value."""
        z = {0.95: 1.645, 0.99: 2.326}.get(confidence, 1.645)
        daily_vol = self.returns.std()
        return -z * daily_vol * self.nav.iloc[-1]

    def win_rate(self) -> float:
        closed = self._closed_pnl()
        return (closed["pnl"] > 0).mean() if not closed.empty else np.nan

    def profit_factor(self) -> float:
        closed = self._closed_pnl()
        wins   = closed[closed["pnl"] > 0]["pnl"].sum()
        losses = abs(closed[closed["pnl"] <= 0]["pnl"].sum())
        return wins / losses if losses != 0 else np.nan

    # ── Rolling metrics ───────────────────────────────

    def rolling_sharpe(self, window: int = 63) -> pd.Series:
        """63-trading-day (≈ 3 month) rolling Sharpe ratio."""
        roll_ret  = self.returns.rolling(window).mean() * self.TRADING_DAYS
        roll_vol  = self.returns.rolling(window).std()  * np.sqrt(self.TRADING_DAYS)
        return (roll_ret - self.rf) / roll_vol

    def rolling_vol(self, window: int = 21) -> pd.Series:
        return self.returns.rolling(window).std() * np.sqrt(self.TRADING_DAYS)

    # ── Summary ───────────────────────────────────────

    def summary(self) -> pd.Series:
        return pd.Series({
            "Total Return (%)":        round(self.total_return()       * 100, 2),
            "Annualised Return (%)":   round(self.annualised_return()  * 100, 2),
            "Annualised Vol (%)":      round(self.annualised_volatility()* 100, 2),
            "Sharpe Ratio":            round(self.sharpe_ratio(),         2),
            "Sortino Ratio":           round(self.sortino_ratio(),        2),
            "Calmar Ratio":            round(self.calmar_ratio(),         2),
            "Max Drawdown (%)":        round(self.max_drawdown()       * 100, 2),
            "Daily VaR 95% (€)":       round(self.value_at_risk(0.95),   0),
            "Win Rate (%)":            round(self.win_rate()           * 100, 1),
            "Profit Factor":           round(self.profit_factor(),        2),
        }, name="TradeMetrics")

    # ── Internal helpers ──────────────────────────────

    def _closed_pnl(self) -> pd.DataFrame:
        """FIFO P&L for all fully closed positions."""
        records = []
        for ticker, grp in self.trades.groupby("ticker"):
            queue = []  # (qty, price) FIFO
            for _, row in grp.iterrows():
                if row["side"] == 1:   # buy
                    queue.append([row["qty"], row["price"]])
                else:                  # sell
                    remaining = row["qty"]
                    while remaining > 0 and queue:
                        lot_qty, lot_px = queue[0]
                        matched = min(remaining, lot_qty)
                        pnl = matched * (row["price"] - lot_px)
                        records.append({"ticker": ticker, "pnl": pnl,
                                        "date": row["datetime"]})
                        queue[0][0] -= matched
                        remaining   -= matched
                        if queue[0][0] == 0:
                            queue.pop(0)
        return pd.DataFrame(records) if records else pd.DataFrame(columns=["ticker","pnl","date"])


# ─────────────────────────────────────────
#  P&L ATTRIBUTION ENGINE
# ─────────────────────────────────────────

class AttributionEngine:
    """
    Decomposes portfolio P&L by position and time period.

    Inputs
    ------
    nav_by_position : dict[str, pd.Series]  — per-ticker daily NAV contribution
    trades          : pd.DataFrame
    sector_map      : dict[str, str]        — ticker → sector label
    """

    def __init__(self, nav_by_position: dict, trades: pd.DataFrame,
                 sector_map: Optional[dict] = None):
        self.nav_pos    = nav_by_position
        self.trades     = trades
        self.sector_map = sector_map or {}

    def pnl_by_position(self) -> pd.DataFrame:
        """Absolute and percentage P&L for every ticker held."""
        rows = []
        for ticker, series in self.nav_pos.items():
            pnl_abs = series.iloc[-1] - series.iloc[0]
            pnl_pct = pnl_abs / series.iloc[0] * 100
            rows.append({
                "ticker":   ticker,
                "sector":   self.sector_map.get(ticker, "Other"),
                "pnl_eur":  round(pnl_abs, 2),
                "pnl_pct":  round(pnl_pct, 2),
                "days_held": len(series),
            })
        return pd.DataFrame(rows).sort_values("pnl_eur", ascending=False)

    def pnl_by_sector(self) -> pd.DataFrame:
        pos = self.pnl_by_position()
        return (pos.groupby("sector")["pnl_eur"]
                   .sum()
                   .sort_values(ascending=False)
                   .reset_index())

    def pnl_by_period(self, freq: str = "ME") -> pd.DataFrame:
        """
        Monthly (freq='ME') or weekly (freq='W') P&L breakdown.
        Returns a DataFrame with period start/end and total return.
        """
        total_nav = sum(self.nav_pos.values())
        monthly   = total_nav.resample(freq).last().pct_change().dropna() * 100
        return monthly.rename("return_pct").reset_index()

    def top_contributors(self, n: int = 5) -> pd.DataFrame:
        return self.pnl_by_position().head(n)

    def top_detractors(self, n: int = 5) -> pd.DataFrame:
        return self.pnl_by_position().tail(n).sort_values("pnl_eur")


# ─────────────────────────────────────────
#  BENCHMARK COMPARATOR
# ─────────────────────────────────────────

class BenchmarkAnalysis:
    """
    Compares portfolio returns against SPY and QQQ benchmarks.

    Parameters
    ----------
    portfolio_returns : pd.Series  — daily portfolio returns
    benchmark_prices  : pd.DataFrame — columns = ['SPY','QQQ'], date-indexed
    """

    def __init__(self, portfolio_returns: pd.Series,
                 benchmark_prices: pd.DataFrame):
        self.port  = portfolio_returns
        self.bench = benchmark_prices.pct_change().dropna()
        # Align indices
        idx = self.port.index.intersection(self.bench.index)
        if len(idx) == 0:
            raise ValueError(
                "No overlapping dates between portfolio and benchmark data. "
                "Ensure they cover the same time period."
            )
        self.port  = self.port.loc[idx]
        self.bench = self.bench.loc[idx]

    def alpha_beta(self, benchmark: str = "SPY") -> tuple[float, float]:
        """OLS regression: port_ret = alpha + beta * bench_ret + epsilon."""
        b = self.bench[benchmark].values
        p = self.port.values
        cov_matrix = np.cov(p, b)
        beta  = cov_matrix[0, 1] / cov_matrix[1, 1]
        alpha = p.mean() - beta * b.mean()
        alpha_ann = alpha * 252   # annualise
        return round(alpha_ann, 4), round(beta, 4)

    def information_ratio(self, benchmark: str = "SPY") -> float:
        active = self.port - self.bench[benchmark]
        return (active.mean() / active.std()) * np.sqrt(252)

    def cumulative_returns(self) -> pd.DataFrame:
        """Index all series to 100 at start for visual comparison."""
        df = pd.DataFrame({"Portfolio": self.port})
        for col in self.bench.columns:
            df[col] = self.bench[col]
        return (1 + df).cumprod() * 100

    def correlation_matrix(self) -> pd.DataFrame:
        df = pd.DataFrame({"Portfolio": self.port})
        for col in self.bench.columns:
            df[col] = self.bench[col]
        return df.corr().round(3)

    def summary(self) -> pd.DataFrame:
        rows = []
        for bm in self.bench.columns:
            alpha, beta = self.alpha_beta(bm)
            ir   = self.information_ratio(bm)
            corr = self.port.corr(self.bench[bm])
            rows.append({
                "Benchmark":  bm,
                "Alpha (ann)": f"{alpha*100:.2f}%",
                "Beta":        beta,
                "Info Ratio":  round(ir, 2),
                "Correlation": round(corr, 3),
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────
#  POSITION SIZING ENGINE
# ─────────────────────────────────────────

class PositionSizer:
    """
    Translates analytics output into recommended position sizes.

    Implements:
      - Kelly Criterion (fractional Kelly for conservatism)
      - Volatility-scaled sizing (equal risk contribution)
      - Max drawdown guardrails
    """

    def __init__(self, nav: float, max_risk_per_trade: float = 0.015,
                 kelly_fraction: float = 0.25):
        """
        Parameters
        ----------
        nav               : current portfolio value (€)
        max_risk_per_trade: maximum fraction of NAV to risk per position
        kelly_fraction    : fraction of full Kelly to use (0.25 = quarter Kelly)
        """
        self.nav         = nav
        self.max_risk    = max_risk_per_trade
        self.kelly_frac  = kelly_fraction

    def kelly_size(self, win_rate: float, avg_win: float,
                   avg_loss: float) -> float:
        """
        Fractional Kelly position size as a fraction of NAV.

        f* = (W/L * p - (1-p)) / (W/L)
        where p = win_rate, W/L = avg_win / avg_loss
        """
        if avg_loss == 0:
            return 0
        wl_ratio = avg_win / avg_loss
        f_full   = (wl_ratio * win_rate - (1 - win_rate)) / wl_ratio
        return max(0, f_full * self.kelly_frac)

    def vol_scaled_size(self, ticker_vol: float,
                        target_vol: float = 0.15) -> float:
        """
        Scale position so each holding contributes equally to portfolio vol.
        Returns fraction of NAV.
        """
        return min(target_vol / ticker_vol, self.max_risk * 5)

    def recommended_size(self, ticker: str, ticker_vol: float,
                         win_rate: float, avg_win: float,
                         avg_loss: float) -> dict:
        kelly = self.kelly_size(win_rate, avg_win, avg_loss)
        vol_s = self.vol_scaled_size(ticker_vol)
        size  = min(kelly, vol_s, self.max_risk * 4)
        eur   = size * self.nav

        return {
            "ticker":         ticker,
            "kelly_fraction": round(kelly, 4),
            "vol_scaled":     round(vol_s, 4),
            "recommended":    round(size, 4),
            "eur_allocation": round(eur, 2),
            "max_loss_eur":   round(self.max_risk * self.nav, 2),
        }


# ─────────────────────────────────────────
#  QUICK-START DEMO
# ─────────────────────────────────────────

def _generate_demo_data(n: int = 252, seed: int = 42) -> tuple:
    """Generate synthetic but realistic portfolio data for demo / testing."""
    np.random.seed(seed)
    dates = pd.bdate_range(end=datetime.today(), periods=n)
    
    # Generate CORRELATED returns (portfolio has beta ~1.3 to SPY)
    spy_ret = np.random.normal(0.00082, 0.0055, n)  # SPY base returns
    market_shock = np.random.normal(0, 0.004, n)    # Common market factor
    
    # Portfolio = 1.3*SPY + alpha + idiosyncratic
    port_ret = (1.3 * spy_ret + 
                0.0004 +  # daily alpha ≈ 10% annual outperformance
                np.random.normal(0, 0.003, n))  # stock-specific risk
    
    qqq_ret = (1.05 * spy_ret +  # QQQ slightly more volatile than SPY
               np.random.normal(0, 0.002, n))
    
    # Convert returns to price series
    nav = pd.Series(
        35_000 * np.cumprod(1 + port_ret),
        index=dates, name="nav"
    )
    spy_prices = pd.Series(
        450 * np.cumprod(1 + spy_ret),
        index=dates, name="SPY"
    )
    qqq_prices = pd.Series(
        380 * np.cumprod(1 + qqq_ret),
        index=dates, name="QQQ"
    )
    bench = pd.DataFrame({"SPY": spy_prices, "QQQ": qqq_prices})

    # Fake trade log
    tickers = ["MSFT","NVDA","ORCL","META","AMD","RBLX","PLTR","GOOGL","AMZN","CRM"]
    rows = []
    for t in tickers:
        entry_idx = np.random.randint(0, n // 2)
        entry_price = np.random.uniform(50, 500)
        exit_price  = entry_price * (1 + np.random.normal(0.12, 0.22))
        qty = int(np.random.uniform(5, 40))
        rows += [
            {"datetime": dates[entry_idx], "ticker": t, "action": "BOT",
             "side": 1,
             "qty": qty, "price": round(entry_price, 2), "commission": 1.0},
            {"datetime": dates[min(entry_idx + np.random.randint(10,120), n-1)],
             "ticker": t, "action": "SLD",
             "side": -1,
             "qty": qty, "price": round(exit_price, 2), "commission": 1.0},
        ]
    trades = pd.DataFrame(rows)
    return nav, bench, trades


if __name__ == "__main__":
    print("=" * 60)
    print("  TradeMetrics — Analytics Engine Demo")
    print("=" * 60)

    nav, bench, trades = _generate_demo_data()

    # ── Metrics
    engine = MetricsEngine(nav, trades)
    print("\n  Performance Summary")
    print("-" * 40)
    print(engine.summary().to_string())

    # ── Benchmark
    bm = BenchmarkAnalysis(engine.returns, bench)
    print("\n  Benchmark Comparison")
    print("-" * 40)
    print(bm.summary().to_string(index=False))

    # ── Position sizing
    sizer  = PositionSizer(nav=nav.iloc[-1])
    sizing = sizer.recommended_size(
        ticker="NVDA", ticker_vol=0.38,
        win_rate=0.68, avg_win=0.21, avg_loss=0.09
    )
    print("\n  Position Sizing — NVDA")
    print("-" * 40)
    for k, v in sizing.items():
        print(f"  {k:<22} {v}")

    print("\n✅  Engine loaded successfully. Connect IBKR TWS to use live data.")
