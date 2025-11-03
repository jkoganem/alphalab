"""Test script for free data source collectors.

This script demonstrates how to fetch data from free APIs:
- FRED: Economic indicators (unlimited free)
- FMP: Fundamentals (250 calls/day free)
- yfinance: Stock prices (unlimited free)

Requirements:
    export FRED_API_KEY="your-fred-key"
    export FMP_API_KEY="your-fmp-key"

Usage:
    python examples/test_data_sources.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alphalab.data.collectors import (
    DataAggregator,
    FinancialModelingPrepCollector,
    FREDCollector,
)
from alphalab.utils.config import get_config


def test_fred():
    """Test FRED economic data collection."""
    print("\n" + "=" * 80)
    print("TESTING FRED (Federal Reserve Economic Data)")
    print("=" * 80)

    config = get_config()
    if not config.fred_api_key:
        print("[WARNING] FRED_API_KEY not set - skipping FRED tests")
        print("   Get your free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return

    try:
        fred = FREDCollector()
        print("[OK] FREDCollector initialized")

        # Test single series
        print("\nFetching GDP data...")
        gdp = fred.fetch("GDP", start_date="2020-01-01", end_date="2023-12-31")
        print(f"  [OK] Got {len(gdp)} GDP observations")
        print(f"    Latest GDP: ${gdp.iloc[-1]['value']:.1f}B")

        # Test common indicators
        print("\nFetching common economic indicators...")
        indicators = fred.fetch_common_indicators(
            start_date="2020-01-01", end_date="2023-12-31"
        )
        print(f"  [OK] Got {len(indicators.columns)} indicators over {len(indicators)} days")
        print(f"    Available: {list(indicators.columns)}")

        print("\n[SUCCESS] FRED tests passed!")

    except Exception as e:
        print(f"\n[FAILED] FRED test failed: {e}")


def test_fmp():
    """Test Financial Modeling Prep fundamental data collection."""
    print("\n" + "=" * 80)
    print("TESTING FMP (Financial Modeling Prep)")
    print("=" * 80)

    config = get_config()
    if not config.fmp_api_key:
        print("[WARNING] FMP_API_KEY not set - skipping FMP tests")
        print("   Get your free key at: https://site.financialmodelingprep.com/developer/docs/")
        return

    try:
        fmp = FinancialModelingPrepCollector()
        print("[OK] FinancialModelingPrepCollector initialized")

        # Test company profile
        print("\nFetching AAPL company profile...")
        profile = fmp.fetch_company_profile("AAPL")
        print(f"  [OK] Company: {profile.get('companyName', 'N/A')}")
        print(f"    Sector: {profile.get('sector', 'N/A')}")
        print(f"    Market Cap: ${profile.get('mktCap', 0) / 1e9:.1f}B")

        # Test key metrics
        print("\nFetching AAPL key metrics (last 4 quarters)...")
        metrics = fmp.fetch_key_metrics("AAPL", period="quarter", limit=4)
        print(f"  [OK] Got {len(metrics)} quarters of data")
        if not metrics.empty and "peRatio" in metrics.columns:
            print(f"    Latest P/E: {metrics.iloc[0]['peRatio']:.2f}")

        # Test fundamentals panel for multiple stocks
        print("\nFetching fundamentals panel for [AAPL, MSFT, GOOGL]...")
        print("  (This uses 3 API calls)")
        panel = fmp.fetch_fundamentals_panel(
            symbols=["AAPL", "MSFT", "GOOGL"], period="quarter", limit=4
        )
        print(f"  [OK] Got data for {len(panel.index.get_level_values('symbol').unique())} stocks")
        print(f"    Columns: {list(panel.columns[:5])}...")

        print("\n[SUCCESS] FMP tests passed!")
        print(f"   API calls used: ~4-5 out of 250 daily limit")

    except Exception as e:
        print(f"\n[FAILED] FMP test failed: {e}")


def test_aggregator():
    """Test DataAggregator combining all sources."""
    print("\n" + "=" * 80)
    print("TESTING DATA AGGREGATOR (All Sources)")
    print("=" * 80)

    config = get_config()
    has_fred = bool(config.fred_api_key)
    has_fmp = bool(config.fmp_api_key)

    if not has_fred and not has_fmp:
        print("[WARNING] No API keys set - aggregator will only fetch price data")
        print("   Set FRED_API_KEY and FMP_API_KEY in .env for full functionality")

    try:
        aggregator = DataAggregator()
        print("[OK] DataAggregator initialized")

        # Test with small universe
        symbols = ["AAPL", "MSFT", "GOOGL"]
        print(f"\nFetching all data for {symbols}...")
        print(f"  Date range: 2020-01-01 to 2023-12-31")

        data_package = aggregator.fetch_all(
            symbols=symbols,
            start_date="2020-01-01",
            end_date="2023-12-31",
            include_fundamentals=has_fmp,
            include_macro=has_fred,
        )

        print("\n[DATA] Data Package Contents:")
        print(f"  [OK] Prices: {len(data_package['prices'])} rows")
        print(f"    Symbols: {data_package['prices'].index.get_level_values('symbol').unique().tolist()}")

        if data_package["fundamentals"] is not None:
            print(f"  [OK] Fundamentals: {len(data_package['fundamentals'])} rows")
            print(f"    Columns: {list(data_package['fundamentals'].columns[:5])}...")
        else:
            print(f"  [SKIP] Fundamentals: Not fetched (FMP_API_KEY not set)")

        if data_package["macro"] is not None:
            print(f"  [OK] Macro: {len(data_package['macro'])} rows")
            print(f"    Indicators: {list(data_package['macro'].columns)}")
        else:
            print(f"  [SKIP] Macro: Not fetched (FRED_API_KEY not set)")

        print(f"\n  Metadata:")
        for key, value in data_package["metadata"].items():
            print(f"    {key}: {value}")

        # Test caching
        print("\nTesting cache functionality...")
        cache_dir = Path(__file__).parent.parent / "data" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        aggregator_with_cache = DataAggregator(cache_dir=str(cache_dir))
        aggregator_with_cache.save_to_cache(data_package, name="test_data")
        print(f"  [OK] Saved to cache: {cache_dir / 'test_data.pkl'}")

        loaded = aggregator_with_cache.load_from_cache(name="test_data")
        print(f"  [OK] Loaded from cache: {len(loaded['prices'])} price rows")

        print("\n[SUCCESS] Aggregator tests passed!")

    except Exception as e:
        print(f"\n[FAILED] Aggregator test failed: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Run all data source tests."""
    print("\n" + "=" * 80)
    print("FREE DATA SOURCES TEST SUITE")
    print("=" * 80)

    # Load configuration
    config = get_config()

    print("\n[CONFIG] API Key Status:")
    print(f"  FRED_API_KEY: {'[OK] Set' if config.fred_api_key else '[FAIL] Not set'}")
    print(f"  FMP_API_KEY: {'[OK] Set' if config.fmp_api_key else '[FAIL] Not set'}")

    if not config.fred_api_key and not config.fmp_api_key:
        print("\n[WARNING] WARNING: No API keys found!")
        print("   You can still test yfinance (price data), but FRED and FMP will be skipped.")
        print("\n   To get free API keys:")
        print("   1. FRED: https://fred.stlouisfed.org/docs/api/api_key.html (2 min)")
        print("   2. FMP: https://site.financialmodelingprep.com/developer/docs/ (2 min)")
        print("\n   Then add to .env file in project root:")
        print("     FRED_API_KEY=your-key")
        print("     FMP_API_KEY=your-key")

    # Run tests
    test_fred()
    test_fmp()
    test_aggregator()

    print("\n" + "=" * 80)
    print("TEST SUITE COMPLETE")
    print("=" * 80)

    # Summary
    print("\n[DATA] Summary:")
    print("  - FRED: Free unlimited economic data")
    print("  - FMP: Free 250 calls/day for fundamentals")
    print("  - yfinance: Free unlimited stock prices")

    print("\n[RUN] Next Steps:")
    print("  1. Set API keys if you haven't already")
    print("  2. Run: python examples/llm_strategy_discovery.py --provider openai")
    print("  3. Let the LLM generate profitable strategies for you!")


if __name__ == "__main__":
    main()
