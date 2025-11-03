"""Example: Personalized strategy generation with comprehensive CLI flags.

This script demonstrates modular strategy generation with extensive CLI customization.

Usage:
    # Use a pre-defined profile
    python examples/personalized_strategy_discovery_v2.py --profile tech

    # Quick custom via CLI flags
    python examples/personalized_strategy_discovery_v2.py \\
        --assets AAPL,MSFT,GOOGL \\
        --strategy-types momentum,mean_reversion \\
        --max-drawdown 0.20 \\
        --target-sharpe 1.0

    # Crypto trading
    python examples/personalized_strategy_discovery_v2.py \\
        --assets BTC,ETH,SOL \\
        --asset-class crypto \\
        --position-type long_short \\
        --rebalancing daily \\
        --max-drawdown 0.30

    # With refinement for better results
    python examples/personalized_strategy_discovery_v2.py \\
        --profile aggressive \\
        --use-refinement \\
        --target-score 75 \\
        --max-iterations 5
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import os

import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
# Find the .env file in the project root (parent of examples/)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

from alphalab.ai import (
    AGGRESSIVE_TRADER,
    CONSERVATIVE_TRADER,
    CRYPTO_TRADER,
    MACRO_TRADER,
    TECH_STOCK_TRADER,
    StrategyDiscovery,
    StrategyPreferences,
)
from alphalab.data.collectors import DataAggregator
from alphalab.features.pipeline import StandardFeaturePipeline


def get_profile(profile_name: str) -> StrategyPreferences:
    """Get pre-defined profile by name."""
    profiles = {
        "conservative": CONSERVATIVE_TRADER,
        "aggressive": AGGRESSIVE_TRADER,
        "crypto": CRYPTO_TRADER,
        "tech": TECH_STOCK_TRADER,
        "macro": MACRO_TRADER,
    }
    return profiles[profile_name]


def build_preferences_from_args(args: argparse.Namespace) -> StrategyPreferences:
    """Build StrategyPreferences from CLI arguments."""
    # Parse assets
    assets = None
    if args.assets:
        assets = [s.strip() for s in args.assets.split(",")]

    # Parse strategy types
    strategy_types = None
    if args.strategy_types:
        strategy_types = [s.strip() for s in args.strategy_types.split(",")]

    # Create preferences
    return StrategyPreferences(
        assets=assets,
        asset_class=args.asset_class,
        strategy_types=strategy_types,
        position_type=args.position_type,
        max_drawdown=args.max_drawdown,
        target_sharpe=args.target_sharpe,
        rebalancing_frequency=args.rebalancing,
        complexity=args.complexity,
        use_fundamentals=args.use_fundamentals,
        use_macro_indicators=args.use_macro,
        use_technical_only=not (args.use_fundamentals or args.use_macro),
    )


def main():
    """Run personalized strategy discovery."""
    parser = argparse.ArgumentParser(
        description="Personalized LLM Strategy Discovery with Full CLI Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use a pre-defined profile
  %(prog)s --profile tech

  # Quick custom via CLI
  %(prog)s --assets AAPL,MSFT,GOOGL --strategy-types momentum --max-drawdown 0.20

  # Crypto trading
  %(prog)s --assets BTC,ETH --asset-class crypto --position-type long_short --rebalancing daily

  # With refinement
  %(prog)s --profile aggressive --use-refinement --target-score 75

  # Full customization
  %(prog)s --assets NVDA,AMD,INTC --asset-class stocks \\
           --strategy-types momentum,mean_reversion \\
           --position-type long_short --max-drawdown 0.25 --target-sharpe 1.2 \\
           --rebalancing weekly --complexity complex --use-fundamentals
        """,
    )

    # === PROFILE SELECTION ===
    profile_group = parser.add_mutually_exclusive_group()
    profile_group.add_argument(
        "--profile",
        choices=["conservative", "aggressive", "crypto", "tech", "macro"],
        help="Use pre-defined profile (mutually exclusive with custom CLI flags)",
    )

    # === LLM SETTINGS ===
    llm_group = parser.add_argument_group("LLM Settings")
    llm_group.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM provider (default: openai, ~$0.02 per 10 strategies)",
    )
    llm_group.add_argument(
        "--model",
        type=str,
        help="Specific model to use (e.g., gpt-4o, gpt-4o-mini, claude-3-5-sonnet-20241022). If not specified, uses provider default.",
    )
    llm_group.add_argument(
        "--n-strategies",
        type=int,
        default=5,
        help="Number of strategies to generate per iteration (default: 5)",
    )

    # === REFINEMENT OPTIONS ===
    refinement_group = parser.add_argument_group("Refinement Options")
    refinement_group.add_argument(
        "--use-refinement",
        action="store_true",
        help="Use iterative refinement (learns from failures, improves results)",
    )
    refinement_group.add_argument(
        "--target-score",
        type=float,
        default=70.0,
        help="Target evaluation score (0-100) for refinement (default: 70.0)",
    )
    refinement_group.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum refinement iterations (default: 3)",
    )
    refinement_group.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Early stopping: iterations without improvement before stopping (default: 10)",
    )
    refinement_group.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.5,
        help="Early stopping: minimum score improvement to reset patience (default: 0.5)",
    )

    # === ASSET SELECTION ===
    asset_group = parser.add_argument_group("Asset Selection")
    asset_group.add_argument(
        "--assets",
        type=str,
        help="Comma-separated assets (e.g., AAPL,MSFT,GOOGL or BTC,ETH,SOL)",
    )
    asset_group.add_argument(
        "--asset-class",
        choices=["stocks", "crypto", "forex", "commodities"],
        help="Asset class (e.g., stocks, crypto)",
    )

    # === STRATEGY PREFERENCES ===
    strategy_group = parser.add_argument_group("Strategy Preferences")
    strategy_group.add_argument(
        "--strategy-types",
        type=str,
        help="Comma-separated types (e.g., momentum,mean_reversion,fundamental,macro)",
    )
    strategy_group.add_argument(
        "--position-type",
        choices=["long_only", "short_only", "long_short"],
        default="long_short",
        help="Position type (default: long_short for market-neutral strategies)",
    )
    strategy_group.add_argument(
        "--rebalancing",
        choices=["daily", "weekly", "monthly"],
        help="Rebalancing frequency (e.g., daily for active, monthly for conservative)",
    )
    strategy_group.add_argument(
        "--complexity",
        choices=["simple", "moderate", "complex"],
        default="moderate",
        help="Strategy complexity (simple=rules, complex=ML allowed, default: moderate)",
    )

    # === RISK MANAGEMENT ===
    risk_group = parser.add_argument_group("Risk Management")
    risk_group.add_argument(
        "--max-drawdown",
        type=float,
        help="Maximum acceptable drawdown as decimal (e.g., 0.20 for 20%%)",
    )
    risk_group.add_argument(
        "--target-sharpe",
        type=float,
        help="Target Sharpe ratio (e.g., 1.0 for good risk-adjusted returns)",
    )
    risk_group.add_argument(
        "--volatility-tolerance",
        choices=["low", "medium", "high"],
        help="Volatility tolerance (low=conservative, high=aggressive)",
    )

    # === DATA USAGE ===
    data_group = parser.add_argument_group("Data Usage")
    data_group.add_argument(
        "--use-fundamentals",
        action="store_true",
        help="Use fundamental data (P/E, earnings, revenue, margins, etc.)",
    )
    data_group.add_argument(
        "--use-macro",
        action="store_true",
        help="Use macro indicators (GDP, inflation, interest rates, VIX, etc.)",
    )

    # === OUTPUT & DEBUGGING ===
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output",
        type=str,
        default="outputs/strategy_discovery/personalized_strategies.md",
        help="Output file path (default: outputs/strategy_discovery/personalized_strategies.md)",
    )
    output_group.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress and debug information",
    )

    args = parser.parse_args()

    # === BUILD PREFERENCES ===
    if args.profile:
        preferences = get_profile(args.profile)

        # Override with explicit CLI flags (CLI flags take precedence over profile defaults)
        if args.use_macro:
            preferences.use_macro_indicators = True
        if args.use_fundamentals:
            preferences.use_fundamentals = True

        # Update technical-only flag based on data source flags
        preferences.use_technical_only = not (preferences.use_macro_indicators or preferences.use_fundamentals)

        print(f"\n{'='*80}")
        print(f"USING PROFILE: {args.profile.upper()}")
        if args.use_macro or args.use_fundamentals:
            print(f"WITH CLI OVERRIDES: macro={preferences.use_macro_indicators}, fundamentals={preferences.use_fundamentals}")
        print(f"{'='*80}")
    else:
        # Build from CLI flags
        preferences = build_preferences_from_args(args)
        print(f"\n{'='*80}")
        print("USING CUSTOM PREFERENCES FROM CLI")
        print(f"{'='*80}")

    if args.verbose:
        print("\nPreferences:")
        print(preferences.to_prompt())

    # === SETUP ===
    print(f"\n{'='*80}")
    print("PERSONALIZED STRATEGY DISCOVERY")
    print(f"{'='*80}")
    print(f"\nProvider: {args.provider}")
    print(f"Strategies per iteration: {args.n_strategies}")
    if args.use_refinement:
        print(f"Using refinement: Yes (target={args.target_score}, max_iter={args.max_iterations})")
    else:
        print("Using refinement: No (single-pass generation)")

    # === LOAD DATA ===
    print("\n[LOAD] Loading market data...")
    data_file = Path(__file__).parent.parent / "data" / "stocks_7y_2025.parquet"

    if not data_file.exists():
        print(f"\n[ERROR] ERROR: Data file not found at {data_file}")
        print("\nOptions:")
        print("  1. Run: python scripts/download_data.py")
        print("  2. Use alphalab.data.collectors.DataAggregator to fetch from APIs")
        return

    data = pd.read_parquet(data_file)
    data = data.sort_index()
    print(
        f"  [OK] Loaded {len(data)} bars for {len(data.index.get_level_values('symbol').unique())} symbols"
    )

    # Filter to user's assets if specified
    if preferences.assets:
        available = data.index.get_level_values("symbol").unique()
        requested = preferences.assets
        valid = [s for s in requested if s in available]

        if not valid:
            print(
                f"\n[WARNING] Warning: None of {requested} found in data. Using all available symbols."
            )
        else:
            data = data.loc[pd.IndexSlice[:, valid], :]
            print(f"  [OK] Filtered to user's assets: {valid}")

    # === LOAD MACRO & FUNDAMENTAL DATA (if requested) ===
    macro_data = None
    fundamental_data = None

    # Debug: Print preferences
    print(f"\n[DEBUG] Preferences: use_macro={preferences.use_macro_indicators}, use_fundamentals={preferences.use_fundamentals}")

    if preferences.use_macro_indicators or preferences.use_fundamentals:
        from alphalab.data.market_data_db import MarketDataDatabase

        db_path = Path(__file__).parent.parent / ".alphalab" / "market_data.db"

        if db_path.exists():
            db = MarketDataDatabase(str(db_path))

            # Load macro data
            if preferences.use_macro_indicators:
                print("\n[LOAD] Loading macro economic data from cache...")
                from alphalab.data.collectors.fred import FREDCollector
                # Use normalized series IDs (keys) instead of FRED tickers (values)
                # because database stores with normalized IDs
                macro_series = list(FREDCollector.COMMON_SERIES.keys())

                start_date = data.index.get_level_values("date").min().strftime("%Y-%m-%d")
                end_date = data.index.get_level_values("date").max().strftime("%Y-%m-%d")

                macro_data = db.get_all_macro_series(macro_series, start_date, end_date)
                if not macro_data.empty:
                    print(f"  [OK] Loaded {len(macro_data.columns)} macro indicators")
                else:
                    print("  [WARN] No macro data in cache. Run: python scripts/download_market_data.py")
                    preferences.use_macro_indicators = False

            # Load fundamental data
            if preferences.use_fundamentals:
                print("\n[LOAD] Loading fundamental data from cache...")
                symbols = data.index.get_level_values("symbol").unique().tolist()

                fundamental_data = {}
                for symbol in symbols:
                    fund_df = db.get_fundamental_data(symbol, period="quarter")
                    if not fund_df.empty:
                        fundamental_data[symbol] = fund_df

                if fundamental_data:
                    print(f"  [OK] Loaded fundamentals for {len(fundamental_data)} symbols")
                else:
                    print("  [WARN] No fundamental data in cache. Run: python scripts/download_market_data.py --symbols ...")
                    preferences.use_fundamentals = False
        else:
            print(f"\n[WARN] Market data cache not found at {db_path}")
            print("  Run: python scripts/download_market_data.py")
            preferences.use_macro_indicators = False
            preferences.use_fundamentals = False

    # === GENERATE FEATURES ===
    # Create cache key based on feature configuration
    cache_suffix = ""
    if preferences.use_macro_indicators:
        cache_suffix += "_macro"
    if preferences.use_fundamentals:
        cache_suffix += "_fundamentals"

    features_cache_path = Path(f".alphalab/features_cache{cache_suffix}.parquet")

    if features_cache_path.exists():
        print(f"\n[CACHE] Loading cached features (config: {'technical+macro+fundamentals' if preferences.use_macro_indicators and preferences.use_fundamentals else 'technical+macro' if preferences.use_macro_indicators else 'technical+fundamentals' if preferences.use_fundamentals else 'technical-only'})...")
        features = pd.read_parquet(features_cache_path)
        print(f"  [OK] Loaded {len(features.columns)} cached features from {features_cache_path.name}")
    else:
        print(f"\n[CONFIG] Generating features (config: {'technical+macro+fundamentals' if preferences.use_macro_indicators and preferences.use_fundamentals else 'technical+macro' if preferences.use_macro_indicators else 'technical+fundamentals' if preferences.use_fundamentals else 'technical-only'})...")
        pipeline = StandardFeaturePipeline(
            include_macro=preferences.use_macro_indicators,
            include_fundamentals=preferences.use_fundamentals,
        )
        features = pipeline.transform(
            data,
            macro_data=macro_data,
            fundamental_data=fundamental_data,
        )
        print(f"  [OK] Generated {len(features.columns)} features")

        # Cache features for future runs
        print(f"  [CACHE] Saving features to cache ({features_cache_path.name})...")
        features_cache_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(features_cache_path)
        print(f"  [OK] Cached features to {features_cache_path}")

    # === BUILD MARKET CONTEXT ===
    if preferences.asset_class == "crypto":
        market_context = """
        Cryptocurrency markets (24/7 trading, high volatility):
        - Extreme price swings and momentum surges
        - Mean reversion after sharp moves
        - Volume spikes indicate trend changes
        - No fundamental anchors (pure technical)
        """
    elif preferences.asset_class == "stocks" and preferences.sectors:
        sectors_str = ", ".join(preferences.sectors)
        market_context = f"""
        {sectors_str} sector stocks:
        - Sector-specific dynamics and rotations
        - Fundamental drivers (earnings, guidance, margins)
        - Correlation with sector ETFs
        - Economic cycle sensitivity
        """
    else:
        market_context = """
        US equities 2019-2023:
        - 2019: Bull market
        - 2020: COVID crash and recovery
        - 2021: Stimulus rally
        - 2022: Bear market (inflation)
        - 2023: Tech-led recovery
        """

    # === INITIALIZE DISCOVERY ===
    model_info = f"{args.provider}"
    if args.model:
        model_info += f" ({args.model})"
    print(f"\n[AI] Initializing LLM strategy discovery ({model_info})...")
    discovery = StrategyDiscovery(
        llm_provider=args.provider,
        model=args.model,  # Pass model selection
        initial_capital=1_000_000,
        costs={"fees_bps": 2.0, "slippage_bps": 5.0, "borrow_bps": 30.0},
    )

    # === RUN DISCOVERY ===
    if args.use_refinement:
        print("\n[REFINE] Running discovery with automated refinement...")
        print(f"   Target score: {args.target_score}/100")
        print(f"   Max iterations: {args.max_iterations}")

        report = discovery.discover_with_refinement(
            initial_strategies=args.n_strategies,
            data=data,
            features=features,
            market_context=market_context,
            target_score=args.target_score,
            max_iterations=args.max_iterations,
            preferences=preferences,  # [TARGET] PERSONALIZED!
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
        )
    else:
        print("\n[RUN] Running single-pass discovery...")
        report = discovery.discover_strategies(
            n_strategies=args.n_strategies,
            data=data,
            features=features,
            market_context=market_context,
            min_score=40.0,
            preferences=preferences,  # [TARGET] PERSONALIZED!
        )

    # === RESULTS ===
    print(report.summary())

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(str(output_path))
    print(f"\n[REPORT] Full report saved to: {output_path}")

    # Show top strategy
    if report.ranked:
        top = report.get_top_strategy()
        print(f"\n{'='*80}")
        print(f"TOP STRATEGY: {top['name']}")
        print(f"{'='*80}")
        print(f"\nScore: {top['evaluation']['score']:.1f}/100")
        print(f"Sharpe: {top['metrics']['sharpe_ratio']:.2f}")
        print(f"Return: {top['metrics']['total_return']*100:.1f}%")
        print(f"Drawdown: {top['metrics']['max_drawdown']*100:.1f}%")
        print(f"\n{top['idea']['hypothesis']}\n")

        if args.verbose:
            print("Full code:")
            print("-" * 80)
            print(top["code"])
        else:
            print("Code preview:")
            print("-" * 80)
            code_preview = (
                top["code"][:500] + "..." if len(top["code"]) > 500 else top["code"]
            )
            print(code_preview)
            print("\n(Use --verbose to see full code)")


if __name__ == "__main__":
    main()
