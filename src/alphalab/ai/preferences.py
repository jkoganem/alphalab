"""User preferences for personalized strategy generation.

This module allows users to specify their trading preferences, which are then
used to create hyper-personalized prompts for LLM strategy generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class StrategyPreferences:
    """User preferences for strategy generation.

    This class captures what the user wants to trade, how they want to trade,
    and what constraints they have. The LLM uses these preferences to generate
    highly personalized strategies.

    Example:
        >>> prefs = StrategyPreferences(
        ...     assets=["BTC", "ETH", "SOL"],
        ...     asset_class="crypto",
        ...     strategy_types=["momentum", "mean_reversion"],
        ...     max_drawdown=0.20,
        ...     preferred_holding_period="1-3 days",
        ... )
        >>> prompt = prefs.to_prompt()
    """

    # What to trade
    assets: list[str] | None = None
    asset_class: str | None = None  # "stocks", "crypto", "forex", "commodities"
    sectors: list[str] | None = None  # e.g., ["Technology", "Healthcare"]

    # How to trade
    strategy_types: list[str] | None = None  # ["momentum", "mean_reversion", "ml"]
    market_regime: str | None = None  # "bull", "bear", "sideways", "volatile"

    # Strategy characteristics
    position_type: Literal["long_only", "short_only", "long_short"] = "long_short"
    rebalancing_frequency: str | None = None  # "daily", "weekly", "monthly"
    preferred_holding_period: str | None = None  # "intraday", "1-3 days", "weeks"

    # Risk preferences
    max_drawdown: float | None = None  # e.g., 0.20 for 20% max
    target_sharpe: float | None = None  # e.g., 1.0
    volatility_tolerance: str | None = None  # "low", "medium", "high"

    # Complexity preferences
    complexity: str = "moderate"  # "simple", "moderate", "complex"
    allow_machine_learning: bool = True
    allow_options: bool = False
    allow_leverage: bool = False

    # Data preferences
    use_fundamentals: bool = False
    use_macro_indicators: bool = False
    use_technical_only: bool = True

    # Trading constraints
    min_trades_per_year: int | None = None
    max_trades_per_year: int | None = None
    max_positions: int | None = None
    max_position_size: float | None = None  # e.g., 0.10 for 10% max per position

    # Additional context
    custom_requirements: list[str] = field(default_factory=list)
    avoid_strategies: list[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        """Convert preferences to a detailed LLM prompt.

        Returns:
            Formatted prompt string for LLM strategy generation.
        """
        sections = []

        # Assets section
        if self.assets or self.asset_class or self.sectors:
            asset_info = ["## ASSET UNIVERSE"]
            if self.assets:
                asset_info.append(f"Specific assets: {', '.join(self.assets)}")
            if self.asset_class:
                asset_info.append(f"Asset class: {self.asset_class}")
            if self.sectors:
                asset_info.append(f"Sectors: {', '.join(self.sectors)}")
            sections.append("\n".join(asset_info))

        # Strategy preferences
        if self.strategy_types or self.market_regime:
            strategy_info = ["## STRATEGY PREFERENCES"]
            if self.strategy_types:
                strategy_info.append(
                    f"Preferred strategy types: {', '.join(self.strategy_types)}"
                )
                strategy_info.append(
                    "Focus on these types, but feel free to combine or innovate."
                )
            if self.market_regime:
                strategy_info.append(
                    f"Expected market regime: {self.market_regime}"
                )
                strategy_info.append(
                    f"Design strategies that perform well in {self.market_regime} markets."
                )
            sections.append("\n".join(strategy_info))

        # Trading characteristics
        trading_info = ["## TRADING CHARACTERISTICS"]
        trading_info.append(f"Position type: {self.position_type}")

        if self.rebalancing_frequency:
            trading_info.append(f"Rebalancing: {self.rebalancing_frequency}")

        if self.preferred_holding_period:
            trading_info.append(f"Holding period: {self.preferred_holding_period}")

        sections.append("\n".join(trading_info))

        # Risk constraints
        if self.max_drawdown or self.target_sharpe or self.volatility_tolerance:
            risk_info = ["## RISK CONSTRAINTS"]
            if self.max_drawdown:
                risk_info.append(
                    f"Maximum acceptable drawdown: {self.max_drawdown*100:.0f}%"
                )
                risk_info.append(
                    "CRITICAL: Design strategies with strong risk management to stay within this limit."
                )
            if self.target_sharpe:
                risk_info.append(f"Target Sharpe ratio: {self.target_sharpe}")
            if self.volatility_tolerance:
                risk_info.append(f"Volatility tolerance: {self.volatility_tolerance}")
                if self.volatility_tolerance == "low":
                    risk_info.append(
                        "Favor stable, low-volatility strategies over aggressive ones."
                    )
            sections.append("\n".join(risk_info))

        # Data usage
        data_info = ["## DATA USAGE"]
        if self.use_technical_only:
            data_info.append("Use technical indicators (price, volume, momentum, etc.)")
        if self.use_fundamentals:
            data_info.append(
                "Use fundamental data (P/E, ROE, revenue growth, earnings, etc.)"
            )
        if self.use_macro_indicators:
            data_info.append(
                "Use macro indicators (GDP, inflation, interest rates, VIX, etc.)"
            )
        sections.append("\n".join(data_info))

        # Constraints
        if (
            self.min_trades_per_year
            or self.max_trades_per_year
            or self.max_positions
            or self.max_position_size
        ):
            constraint_info = ["## TRADING CONSTRAINTS"]
            if self.min_trades_per_year:
                constraint_info.append(
                    f"Minimum trades per year: {self.min_trades_per_year}"
                )
            if self.max_trades_per_year:
                constraint_info.append(
                    f"Maximum trades per year: {self.max_trades_per_year}"
                )
            if self.max_positions:
                constraint_info.append(f"Maximum concurrent positions: {self.max_positions}")
            if self.max_position_size:
                constraint_info.append(
                    f"Maximum position size: {self.max_position_size*100:.0f}% of portfolio"
                )
            sections.append("\n".join(constraint_info))

        # Complexity and features
        feature_info = ["## STRATEGY COMPLEXITY"]
        feature_info.append(f"Complexity level: {self.complexity}")

        if self.complexity == "simple":
            feature_info.append(
                "Keep strategies simple and interpretable. Avoid complex models."
            )
        elif self.complexity == "complex":
            feature_info.append(
                "You can use sophisticated techniques and complex models if they improve performance."
            )

        if not self.allow_machine_learning:
            feature_info.append("Do NOT use machine learning models.")

        if not self.allow_options:
            feature_info.append("Do NOT use options or derivatives.")

        if not self.allow_leverage:
            feature_info.append("Do NOT use leverage (keep total exposure <= 100%).")

        sections.append("\n".join(feature_info))

        # Custom requirements
        if self.custom_requirements:
            custom_info = ["## CUSTOM REQUIREMENTS"]
            for req in self.custom_requirements:
                custom_info.append(f"- {req}")
            sections.append("\n".join(custom_info))

        # What to avoid
        if self.avoid_strategies:
            avoid_info = ["## STRATEGIES TO AVOID"]
            avoid_info.append(
                "Do NOT generate strategies similar to these (they didn't work well):"
            )
            for avoid in self.avoid_strategies:
                avoid_info.append(f"- {avoid}")
            sections.append("\n".join(avoid_info))

        return "\n\n".join(sections)

    def to_dict(self) -> dict:
        """Convert preferences to dictionary for serialization."""
        return {
            "assets": self.assets,
            "asset_class": self.asset_class,
            "sectors": self.sectors,
            "strategy_types": self.strategy_types,
            "market_regime": self.market_regime,
            "position_type": self.position_type,
            "rebalancing_frequency": self.rebalancing_frequency,
            "preferred_holding_period": self.preferred_holding_period,
            "max_drawdown": self.max_drawdown,
            "target_sharpe": self.target_sharpe,
            "volatility_tolerance": self.volatility_tolerance,
            "complexity": self.complexity,
            "allow_machine_learning": self.allow_machine_learning,
            "allow_options": self.allow_options,
            "allow_leverage": self.allow_leverage,
            "use_fundamentals": self.use_fundamentals,
            "use_macro_indicators": self.use_macro_indicators,
            "use_technical_only": self.use_technical_only,
            "min_trades_per_year": self.min_trades_per_year,
            "max_trades_per_year": self.max_trades_per_year,
            "max_positions": self.max_positions,
            "max_position_size": self.max_position_size,
            "custom_requirements": self.custom_requirements,
            "avoid_strategies": self.avoid_strategies,
        }


# Pre-defined preference templates for common use cases
CONSERVATIVE_TRADER = StrategyPreferences(
    position_type="long_only",
    max_drawdown=0.15,
    target_sharpe=1.0,
    volatility_tolerance="low",
    complexity="simple",
    allow_machine_learning=False,
    allow_leverage=False,
    rebalancing_frequency="monthly",
    custom_requirements=[
        "Focus on capital preservation",
        "Prefer established, liquid assets",
        "Avoid high-frequency trading",
    ],
)

AGGRESSIVE_TRADER = StrategyPreferences(
    position_type="long_short",
    max_drawdown=0.40,
    target_sharpe=1.5,
    volatility_tolerance="high",
    complexity="complex",
    allow_machine_learning=True,
    allow_leverage=True,
    rebalancing_frequency="daily",
    custom_requirements=[
        "Maximize returns, willing to take risks",
        "Can handle volatility and drawdowns",
        "Open to sophisticated strategies",
    ],
)

CRYPTO_TRADER = StrategyPreferences(
    asset_class="crypto",
    assets=["BTC", "ETH", "SOL", "AVAX", "MATIC"],
    strategy_types=["momentum", "mean_reversion"],
    position_type="long_short",
    max_drawdown=0.30,
    volatility_tolerance="high",
    rebalancing_frequency="daily",
    preferred_holding_period="1-3 days",
    use_technical_only=True,
    custom_requirements=[
        "Crypto markets are 24/7 and highly volatile",
        "Exploit momentum surges and sharp reversals",
        "Use volume indicators heavily",
    ],
)

TECH_STOCK_TRADER = StrategyPreferences(
    asset_class="stocks",
    sectors=["Technology"],
    assets=["AAPL", "MSFT", "GOOGL", "NVDA", "META", "TSLA", "AMZN"],
    strategy_types=["cross_sectional_momentum", "mean_reversion"],
    position_type="long_short",
    max_drawdown=0.25,
    use_fundamentals=False,
    use_technical_only=True,
    rebalancing_frequency="weekly",
    custom_requirements=[
        "Focus on tech sector dynamics",
        "Use price and volume patterns",
        "Consider relative performance within tech sector",
    ],
)

MACRO_TRADER = StrategyPreferences(
    asset_class="stocks",
    strategy_types=["regime_based", "macro"],
    position_type="long_short",
    use_macro_indicators=True,
    use_fundamentals=True,
    rebalancing_frequency="monthly",
    preferred_holding_period="weeks",
    custom_requirements=[
        "Trade based on macroeconomic regime changes",
        "Use Fed policy, inflation, and GDP data",
        "Sector rotation based on economic cycle",
    ],
)
