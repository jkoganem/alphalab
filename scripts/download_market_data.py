#!/usr/bin/env python3
"""Download and cache FRED macro and FMP fundamental data.

This script fetches data from external APIs and caches it locally in
MarketDataDatabase to reduce API calls and enable offline usage.

Usage:
    python scripts/download_market_data.py --symbols AAPL MSFT GOOGL --start 2020-01-01
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

from alphalab.data.collectors.aggregator import DataAggregator
from alphalab.data.collectors.fred import FREDCollector
from alphalab.data.collectors.fmp import FinancialModelingPrepCollector
from alphalab.data.collectors.yfinance_collector import YFinanceCollector
from alphalab.data.market_data_db import MarketDataDatabase


def download_macro_data(
    db: MarketDataDatabase,
    fred_api_key: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    """Download FRED macro economic indicators and cache them.

    Args:
        db: MarketDataDatabase instance
        fred_api_key: FRED API key (or use FRED_API_KEY env var)
        start_date: Start date (ISO format)
        end_date: End date (ISO format)
    """
    print("\n" + "=" * 80)
    print("DOWNLOADING MACRO ECONOMIC DATA (FRED)")
    print("=" * 80)

    try:
        fred = FREDCollector(api_key=fred_api_key)
    except ValueError as e:
        print(f"[ERROR] FRED API key not found: {e}")
        print("   Skipping macro data download")
        return

    # Fetch common economic indicators
    print("\nFetching common FRED indicators...")
    macro_df = fred.fetch_common_indicators(start_date, end_date)

    if macro_df.empty:
        print("[WARNING] No macro data fetched")
        return

    # Store each series in database
    print(f"\nCaching {len(macro_df.columns)} macro series:")
    for series_id in macro_df.columns:
        series_df = macro_df[[series_id]].rename(columns={series_id: "value"})
        series_df = series_df.dropna()

        if not series_df.empty:
            db.store_macro_data(series_id, series_df, source="FRED")
            print(f"  [OK] {series_id}: {len(series_df)} data points")
        else:
            print(f"  [SKIP] {series_id}: No data")

    print("\n[SUCCESS] Macro data cached successfully")


def download_fundamental_data(
    db: MarketDataDatabase,
    symbols: list[str],
    source: str = "yfinance",
    fmp_api_key: str | None = None,
    period: str = "quarter",
    limit: int = 20,
) -> None:
    """Download fundamental metrics and cache them.

    Args:
        db: MarketDataDatabase instance
        symbols: List of stock tickers
        source: Data source - "yfinance" (free) or "fmp" (requires API key)
        fmp_api_key: FMP API key (or use FMP_API_KEY env var), only for source="fmp"
        period: "annual" or "quarter" (note: yfinance is less granular)
        limit: Number of periods to fetch (default: 20)
    """
    print("\n" + "=" * 80)
    print(f"DOWNLOADING FUNDAMENTAL DATA ({source.upper()})")
    print("=" * 80)

    # Initialize collector based on source
    if source == "fmp":
        try:
            collector = FinancialModelingPrepCollector(api_key=fmp_api_key)
            print(f"\nNote: FMP free tier has 250 calls/day limit")
        except ValueError as e:
            print(f"[ERROR] FMP API key not found: {e}")
            print("   Skipping fundamental data download")
            return
    elif source == "yfinance":
        collector = YFinanceCollector()
        print(f"\nNote: Using yfinance (free, no API key required)")
    else:
        print(f"[ERROR] Unknown source: {source}. Use 'yfinance' or 'fmp'")
        return

    print(f"\nFetching fundamentals for {len(symbols)} symbols...\n")

    success_count = 0
    fail_count = 0

    for i, symbol in enumerate(symbols, 1):
        try:
            print(f"  [{i}/{len(symbols)}] Fetching {symbol}...", end=" ")

            # Fetch fundamental metrics
            fund_df = collector.fetch_key_metrics(symbol, period=period, limit=limit)

            if fund_df.empty:
                print("[ERROR] No data")
                fail_count += 1
                continue

            # Store in database
            db.store_fundamental_data(
                symbol, fund_df, period=period, source="FMP"
            )
            print(f"[OK] {len(fund_df)} periods cached")
            success_count += 1

        except Exception as e:
            print(f"[ERROR] Error: {e}")
            fail_count += 1

    print(f"\n[SUCCESS] Fundamental data: {success_count} symbols cached, {fail_count} failed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and cache FRED macro and FMP fundamental data"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Stock symbols to fetch fundamental data for",
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Start date for macro data (YYYY-MM-DD, default: 10 years ago)",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date for macro data (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="quarter",
        choices=["quarter", "annual"],
        help="Fundamental data period (default: quarter)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of fundamental periods to fetch (default: 20)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=".alphalab/market_data.db",
        help="Database path (default: .alphalab/market_data.db)",
    )
    parser.add_argument(
        "--fred-key",
        type=str,
        help="FRED API key (or use FRED_API_KEY env var)",
    )
    parser.add_argument(
        "--fmp-key",
        type=str,
        help="FMP API key (or use FMP_API_KEY env var)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="yfinance",
        choices=["yfinance", "fmp"],
        help="Fundamental data source: 'yfinance' (free) or 'fmp' (requires API key, default: yfinance)",
    )
    parser.add_argument(
        "--skip-macro",
        action="store_true",
        help="Skip downloading macro data",
    )
    parser.add_argument(
        "--skip-fundamentals",
        action="store_true",
        help="Skip downloading fundamental data",
    )

    args = parser.parse_args()

    # Initialize database
    print("\n" + "=" * 80)
    print("MARKET DATA DOWNLOAD SCRIPT")
    print("=" * 80)
    print(f"\nDatabase: {args.db_path}")

    db = MarketDataDatabase(db_path=args.db_path)

    # Set default dates if not provided
    if not args.start:
        args.start = (datetime.now() - timedelta(days=10 * 365)).strftime("%Y-%m-%d")
    if not args.end:
        args.end = datetime.now().strftime("%Y-%m-%d")

    print(f"Date range: {args.start} to {args.end}")

    # Download macro data
    if not args.skip_macro:
        download_macro_data(
            db,
            fred_api_key=args.fred_key,
            start_date=args.start,
            end_date=args.end,
        )
    else:
        print("\n[SKIP] Skipping macro data download")

    # Download fundamental data
    if not args.skip_fundamentals:
        if not args.symbols:
            print("\n[WARNING] No symbols provided, skipping fundamental data")
        else:
            download_fundamental_data(
                db,
                symbols=args.symbols,
                source=args.source,
                fmp_api_key=args.fmp_key,
                period=args.period,
                limit=args.limit,
            )
    else:
        print("\n[SKIP] Skipping fundamental data download")

    # Print statistics
    print("\n" + "=" * 80)
    print("DATABASE STATISTICS")
    print("=" * 80)

    stats = db.get_statistics()
    print(f"\nMacro Data:")
    print(f"  Series count: {stats['macro_series_count']}")
    print(f"  Data points: {stats['macro_data_points']}")
    if stats["macro_date_range"][0]:
        print(f"  Date range: {stats['macro_date_range'][0]} to {stats['macro_date_range'][1]}")

    print(f"\nFundamental Data:")
    print(f"  Symbols: {stats['fundamental_symbols_count']}")
    print(f"  Metrics: {stats['fundamental_metrics_count']}")
    print(f"  Data points: {stats['fundamental_data_points']}")
    if stats["fundamental_date_range"][0]:
        print(f"  Date range: {stats['fundamental_date_range'][0]} to {stats['fundamental_date_range'][1]}")

    print("\n" + "=" * 80)
    print("[SUCCESS] DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"\nData cached in: {args.db_path}")
    print("You can now use this cached data in your feature pipelines!")


if __name__ == "__main__":
    main()
