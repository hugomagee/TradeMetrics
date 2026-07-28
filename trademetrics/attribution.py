"""Realised P&L attribution by ticker, sector and calendar period.

Attribution here is on REALISED (closed, commission-adjusted) FIFO P&L from
:class:`trademetrics.fifo.FifoEngine` — not on mark-to-market position values,
which the sample data does not carry. Period attribution of total performance
uses the NAV series directly.
"""

import pandas as pd


class AttributionEngine:
    """Slice realised P&L and NAV returns for reporting.

    Parameters
    ----------
    realised_pnl : pd.DataFrame — output of FifoEngine.realised_pnl()
    nav          : pd.Series   — daily portfolio NAV, date-indexed
    sector_map   : dict[str, str], optional — ticker → sector label
    """

    def __init__(
        self,
        realised_pnl: pd.DataFrame,
        nav: pd.Series,
        sector_map: dict[str, str] | None = None,
    ):
        self.realised = realised_pnl
        self.nav = nav.sort_index()
        self.sector_map = sector_map or {}

    def pnl_by_ticker(self) -> pd.DataFrame:
        """Net realised P&L per ticker with sector labels, best first."""
        if self.realised.empty:
            return pd.DataFrame(columns=["ticker", "sector", "net_pnl", "commission", "trades"])
        grouped = (
            self.realised.groupby("ticker")
            .agg(
                net_pnl=("net_pnl", "sum"),
                commission=("commission", "sum"),
                trades=("net_pnl", "size"),
            )
            .reset_index()
        )
        grouped["sector"] = grouped["ticker"].map(lambda t: self.sector_map.get(t, "Other"))
        return grouped.sort_values("net_pnl", ascending=False).reset_index(drop=True)

    def pnl_by_sector(self) -> pd.DataFrame:
        by_ticker = self.pnl_by_ticker()
        if by_ticker.empty:
            return pd.DataFrame(columns=["sector", "net_pnl"])
        return (
            by_ticker.groupby("sector")["net_pnl"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

    def returns_by_period(self, freq: str = "ME") -> pd.Series:
        """NAV return per calendar period (freq 'ME' monthly, 'W' weekly), in %."""
        period_nav = self.nav.resample(freq).last()
        # prepend the series start so the first period measures from inception
        first = self.nav.iloc[[0]]
        period_nav = pd.concat([first, period_nav]).sort_index()
        returns = period_nav.pct_change().dropna() * 100
        return returns.rename("return_pct")

    def top_contributors(self, n: int = 5) -> pd.DataFrame:
        return self.pnl_by_ticker().head(n)

    def top_detractors(self, n: int = 5) -> pd.DataFrame:
        return self.pnl_by_ticker().tail(n).sort_values("net_pnl").reset_index(drop=True)
