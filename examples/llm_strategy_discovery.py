"""Example: LLM-powered strategy discovery.

This script demonstrates how to use LLMs to automatically generate,
backtest, and evaluate trading strategies.

Requirements:
    pip install openai anthropic

Usage:
    # With OpenAI (cheaper):
    export OPENAI_API_KEY="sk-..."
    python examples/llm_strategy_discovery.py --provider openai

    # With Anthropic (better code quality):
    export ANTHROPIC_API_KEY="sk-ant-..."
    python examples/llm_strategy_discovery.py --provider anthropic

    # Optional: Use free data sources instead of cached data
    export FRED_API_KEY="your-fred-key"
    export FMP_API_KEY="your-fmp-key"
    python examples/llm_strategy_discovery.py --provider openai --use-api
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from alphalab.ai import StrategyDiscovery
from alphalab.data.collectors import DataAggregator
from alphalab.features.pipeline import StandardFeaturePipeline
from alphalab.utils.config import get_config


def main():
    """Run LLM-powered strategy discovery."""
    import argparse

    parser = argparse.ArgumentParser(description="LLM Strategy Discovery")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM provider (openai is cheaper, anthropic is better)",
    )
    parser.add_argument(
        "--n-strategies", type=int, default=5, help="Number of strategies to generate"
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=40.0,
        help="Minimum evaluation score to pass",
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Fetch data from free APIs instead of using cached data",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,JPM,V,JNJ,WMT,PG,MA,HD,DIS,PYPL,NFLX,ADBE,CRM,CSCO,PEP,KO,NKE,MCD,INTC,T,VZ,CMCSA,PFE,MRK,ABBV",
        help="Comma-separated list of stock symbols",
    )
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("LLM-POWERED STRATEGY DISCOVERY")
    print("=" * 80)
    print(f"\nProvider: {args.provider}")
    print(f"Strategies to generate: {args.n_strategies}")
    print(f"Minimum score: {args.min_score}")

    # Load configuration
    config = get_config()

    # Load or fetch data
    if args.use_api:
        print("\n[LOAD] Fetching data from free APIs...")
        symbols = [s.strip() for s in args.symbols.split(",")]
        print(f"  Symbols: {symbols[:5]}... ({len(symbols)} total)")

        # Check API keys
        has_fred = bool(config.fred_api_key)
        has_fmp = bool(config.fmp_api_key)
        print(f"  FRED API: {'[OK]' if has_fred else '[FAIL] (skipping macro data)'}")
        print(f"  FMP API: {'[OK]' if has_fmp else '[FAIL] (skipping fundamentals)'}")

        # Fetch data
        aggregator = DataAggregator(
            cache_dir=str(Path(__file__).parent.parent / "data" / "cache")
        )

        data_package = aggregator.fetch_all(
            symbols=symbols,
            start_date="2019-01-01",
            end_date="2023-12-31",
            include_fundamentals=has_fmp,
            include_macro=has_fred,
        )

        data = data_package["prices"]
        print(f"  [OK] Fetched {len(data)} price bars")

        # Save to cache
        aggregator.save_to_cache(data_package, name="llm_discovery_data")
        print(f"  [OK] Cached for future use")

    else:
        print("\n[LOAD] Loading cached market data...")
        data_file = Path(__file__).parent.parent / "data" / "stocks_7y_2025.parquet"

        if not data_file.exists():
            print(f"\n[ERROR] ERROR: Data file not found at {data_file}")
            print("\nOptions:")
            print("  1. Run: python scripts/download_data.py")
            print("  2. Use --use-api flag to fetch from free APIs")
            return

        data = pd.read_parquet(data_file)
        data = data.sort_index()
        print(
            f"  [OK] Loaded {len(data)} bars for {len(data.index.get_level_values('symbol').unique())} symbols"
        )

    # Generate features
    print("\nGenerating features...")
    pipeline = StandardFeaturePipeline()
    features = pipeline.transform(data)
    print(f"  [OK] Generated {len(features.columns)} features")

    # Market context for LLM
    market_context = """
    US equities from 2019-2023, including:
    - 2019: Bull market
    - 2020: COVID crash and recovery
    - 2021: Stimulus rally
    - 2022: Bear market (inflation shock)
    - 2023: Tech-led recovery

    Universe: 31 stocks across 6 sectors (Tech, Finance, Healthcare, Consumer, Energy, Industrials)
    """

    # Initialize discovery engine
    print(f"\nInitializing LLM strategy discovery ({args.provider})...")
    discovery = StrategyDiscovery(
        llm_provider=args.provider,
        initial_capital=1_000_000,
        costs={"fees_bps": 2.0, "slippage_bps": 5.0, "borrow_bps": 30.0},
    )

    # Run discovery
    report = discovery.discover_strategies(
        n_strategies=args.n_strategies,
        data=data,
        features=features,
        market_context=market_context,
        min_score=args.min_score,
    )

    # Print summary
    print(report.summary())

    # Save report
    output_file = Path(__file__).parent.parent / "out" / "llm_strategies.md"
    output_file.parent.mkdir(exist_ok=True)
    report.save(str(output_file))
    print(f"\n[REPORT] Full report saved to: {output_file}")

    # Show top strategy code
    if report.ranked:
        top = report.get_top_strategy()
        print("\n" + "=" * 80)
        print(f"TOP STRATEGY: {top['name']}")
        print("=" * 80)
        print(f"\n{top['idea']['hypothesis']}\n")
        print("Code:")
        print("-" * 80)
        print(top["code"][:500] + "..." if len(top["code"]) > 500 else top["code"])


if __name__ == "__main__":
    main()
