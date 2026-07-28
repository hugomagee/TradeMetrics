"""Regenerate the committed sample data in data/samples/ from fixed seeds.

    python tools/make_sample_data.py

Deterministic by construction (seeded numpy default_rng, fixed 2024 date
range) — running this twice produces byte-identical CSVs, and CI verifies the
committed files match the generator. Ground-truth parameters live in
trademetrics.simulate.SAMPLE_MARKET.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from trademetrics.simulate import sample_trade_log, simulate_market  # noqa: E402

OUT = REPO_ROOT / "data" / "samples"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    nav, bench = simulate_market()

    nav.round(2).to_csv(OUT / "nav_sample.csv", header=["nav"], date_format="%Y-%m-%d")
    bench.round(2).to_csv(OUT / "benchmarks_sample.csv", date_format="%Y-%m-%d")
    sample_trade_log().to_csv(OUT / "trades_sample.csv", index=False)

    print(f"Wrote {OUT / 'nav_sample.csv'} ({len(nav)} rows)")
    print(f"Wrote {OUT / 'benchmarks_sample.csv'} ({len(bench)} rows)")
    print(f"Wrote {OUT / 'trades_sample.csv'} ({len(sample_trade_log())} rows)")


if __name__ == "__main__":
    main()
