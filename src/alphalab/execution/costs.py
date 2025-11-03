"""Transaction cost models.

This module implements various transaction cost models including
fixed fees, slippage, and borrow costs for short positions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class FixedCostModel:
    """Fixed transaction cost model.

    Applies a constant cost per trade as a percentage of notional value.

    Parameters
    ----------
    fees_bps : float, default 1.0
        Transaction fees in basis points (1 bps = 0.01%)
    min_fee : float, default 0.0
        Minimum fee per trade

    Examples
    --------
    >>> cost_model = FixedCostModel(fees_bps=1.0)
    >>> costs = cost_model.estimate(trades, prices)
    """

    def __init__(self, fees_bps: float = 1.0, min_fee: float = 0.0) -> None:
        """Initialize fixed cost model."""
        self.fees_bps = fees_bps
        self.min_fee = min_fee
        logger.info(f"Initialized FixedCostModel with fees_bps={fees_bps}")

    def estimate(self, orders: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
        """Estimate transaction costs for orders.

        Parameters
        ----------
        orders : pd.DataFrame
            Order data with 'quantity' or 'notional' column
        prices : pd.DataFrame
            Price data with 'price' or 'close' column

        Returns
        -------
        pd.Series
            Total costs indexed by date
        """
        logger.info("Estimating fixed transaction costs")

        # Get order notional values
        if "notional" in orders.columns:
            notional = orders["notional"].abs()
        elif "quantity" in orders.columns and "price" in prices.columns:
            # Align prices with orders
            order_prices = prices["price"].reindex(orders.index)
            notional = (orders["quantity"] * order_prices).abs()
        else:
            msg = "Orders must have 'notional' or ('quantity' and prices must have 'price')"
            raise ValueError(msg)

        # Calculate costs in basis points
        costs = notional * (self.fees_bps / 10000)

        # Apply minimum fee
        if self.min_fee > 0:
            costs = costs.clip(lower=self.min_fee)

        # Aggregate by date
        total_costs = costs.groupby(level="date").sum()

        logger.info(f"Total costs: ${total_costs.sum():,.2f}")

        return total_costs


class SlippageModel:
    """Slippage cost model.

    Implements both fixed and market-impact-based slippage.

    Parameters
    ----------
    model_type : str, default "fixed_bps"
        Slippage model: "fixed_bps" or "sqrt_impact"
    slippage_bps : float, default 5.0
        Fixed slippage in basis points (for fixed_bps model)
    impact_coeff : float, default 0.1
        Impact coefficient (for sqrt_impact model)
    max_slippage_bps : float, default 100.0
        Maximum slippage cap in basis points

    Examples
    --------
    >>> slippage = SlippageModel(model_type="sqrt_impact", impact_coeff=0.1)
    >>> costs = slippage.estimate(trades, prices, volumes)
    """

    def __init__(
        self,
        model_type: str = "fixed_bps",
        slippage_bps: float = 5.0,
        impact_coeff: float = 0.1,
        max_slippage_bps: float = 100.0,
    ) -> None:
        """Initialize slippage model."""
        if model_type not in {"fixed_bps", "sqrt_impact"}:
            msg = f"Unknown slippage model: {model_type}"
            raise ValueError(msg)

        self.model_type = model_type
        self.slippage_bps = slippage_bps
        self.impact_coeff = impact_coeff
        self.max_slippage_bps = max_slippage_bps

        logger.info(f"Initialized SlippageModel with model_type={model_type}")

    def estimate(
        self,
        orders: pd.DataFrame,
        prices: pd.DataFrame,
        volumes: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Estimate slippage costs.

        Parameters
        ----------
        orders : pd.DataFrame
            Order data with 'quantity' or 'notional'
        prices : pd.DataFrame
            Price data
        volumes : pd.DataFrame | None, optional
            Volume data (required for sqrt_impact model)

        Returns
        -------
        pd.Series
            Slippage costs indexed by date
        """
        logger.info(f"Estimating slippage costs using {self.model_type} model")

        # Get notional values
        if "notional" in orders.columns:
            notional = orders["notional"].abs()
        else:
            price_col = "price" if "price" in prices.columns else "close"
            order_prices = prices[price_col].reindex(orders.index)
            notional = (orders["quantity"] * order_prices).abs()

        if self.model_type == "fixed_bps":
            # Fixed slippage
            slippage = notional * (self.slippage_bps / 10000)

        else:  # sqrt_impact
            if volumes is None:
                logger.warning("No volume data for sqrt_impact model, using fixed slippage")
                slippage = notional * (self.slippage_bps / 10000)
            else:
                # Square root market impact model
                # slippage = k * notional * sqrt(quantity / ADV)
                vol_col = "volume" if "volume" in volumes.columns else volumes.columns[0]
                avg_daily_volume = volumes[vol_col].reindex(orders.index)

                price_col = "price" if "price" in prices.columns else "close"
                order_prices = prices[price_col].reindex(orders.index)

                # Estimate share quantity from notional
                quantities = notional / order_prices

                # ADV ratio
                adv_ratio = quantities / avg_daily_volume.clip(lower=1)  # Avoid div by zero

                # Impact: k * notional * sqrt(Q/ADV)
                impact = self.impact_coeff * notional * np.sqrt(adv_ratio)

                # Cap at maximum slippage
                max_slip = notional * (self.max_slippage_bps / 10000)
                slippage = impact.clip(upper=max_slip)

        # Aggregate by date
        total_slippage = slippage.groupby(level="date").sum()

        logger.info(f"Total slippage: ${total_slippage.sum():,.2f}")

        return total_slippage


class BorrowCostModel:
    """Borrow cost model for short positions.

    Calculates annualized borrow costs for short positions.

    Parameters
    ----------
    borrow_bps : float, default 30.0
        Annualized borrow rate in basis points
    days_per_year : int, default 252
        Trading days per year

    Examples
    --------
    >>> borrow = BorrowCostModel(borrow_bps=30.0)
    >>> costs = borrow.estimate(positions, prices)
    """

    def __init__(self, borrow_bps: float = 30.0, days_per_year: int = 252) -> None:
        """Initialize borrow cost model."""
        self.borrow_bps = borrow_bps
        self.daily_rate = (borrow_bps / 10000) / days_per_year
        logger.info(f"Initialized BorrowCostModel with borrow_bps={borrow_bps}")

    def estimate(self, positions: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
        """Estimate borrow costs for short positions.

        Parameters
        ----------
        positions : pd.DataFrame
            Position data with 'quantity' or 'value' column
        prices : pd.DataFrame
            Price data

        Returns
        -------
        pd.Series
            Daily borrow costs indexed by date
        """
        logger.info("Estimating borrow costs for short positions")

        # Get position values
        if "value" in positions.columns:
            position_values = positions["value"]
        elif "quantity" in positions.columns:
            price_col = "price" if "price" in prices.columns else "close"
            position_prices = prices[price_col].reindex(positions.index)
            position_values = positions["quantity"] * position_prices
        else:
            msg = "Positions must have 'value' or 'quantity' column"
            raise ValueError(msg)

        # Only charge borrow on short positions (negative value)
        short_values = position_values.clip(upper=0).abs()

        # Daily borrow cost = short_value * daily_rate
        daily_costs = short_values * self.daily_rate

        # Aggregate by date
        total_costs = daily_costs.groupby(level="date").sum()

        logger.info(f"Total borrow costs: ${total_costs.sum():,.2f}")

        return total_costs


class CompositeCostModel:
    """Composite cost model combining multiple cost components.

    Parameters
    ----------
    fees_bps : float, default 1.0
        Transaction fees in basis points
    slippage_model : str, default "fixed_bps"
        Slippage model type
    slippage_bps : float, default 5.0
        Slippage in basis points (for fixed model)
    impact_coeff : float, default 0.1
        Impact coefficient (for sqrt model)
    borrow_bps : float, default 30.0
        Annualized borrow cost for shorts

    Examples
    --------
    >>> costs = CompositeCostModel(fees_bps=1.0, slippage_bps=5.0, borrow_bps=30.0)
    >>> total = costs.estimate_all(orders, positions, prices, volumes)
    """

    def __init__(
        self,
        fees_bps: float = 1.0,
        slippage_model: str = "fixed_bps",
        slippage_bps: float = 5.0,
        impact_coeff: float = 0.1,
        borrow_bps: float = 30.0,
    ) -> None:
        """Initialize composite cost model."""
        self.fixed_costs = FixedCostModel(fees_bps=fees_bps)
        self.slippage = SlippageModel(
            model_type=slippage_model,
            slippage_bps=slippage_bps,
            impact_coeff=impact_coeff,
        )
        self.borrow = BorrowCostModel(borrow_bps=borrow_bps)

        logger.info(
            f"Initialized CompositeCostModel with fees={fees_bps}bps, "
            f"slippage={slippage_bps}bps, borrow={borrow_bps}bps"
        )

    def estimate_all(
        self,
        orders: pd.DataFrame,
        positions: pd.DataFrame,
        prices: pd.DataFrame,
        volumes: pd.DataFrame | None = None,
    ) -> dict[str, pd.Series]:
        """Estimate all cost components.

        Parameters
        ----------
        orders : pd.DataFrame
            Order data
        positions : pd.DataFrame
            Position data
        prices : pd.DataFrame
            Price data
        volumes : pd.DataFrame | None, optional
            Volume data

        Returns
        -------
        dict[str, pd.Series]
            Dictionary with keys: 'fees', 'slippage', 'borrow', 'total'
        """
        logger.info("Estimating all transaction costs")

        # Estimate each component
        fees = self.fixed_costs.estimate(orders, prices)
        slippage = self.slippage.estimate(orders, prices, volumes)
        borrow = self.borrow.estimate(positions, prices)

        # Combine all costs (align indices)
        all_dates = fees.index.union(slippage.index).union(borrow.index)

        fees_aligned = fees.reindex(all_dates, fill_value=0)
        slippage_aligned = slippage.reindex(all_dates, fill_value=0)
        borrow_aligned = borrow.reindex(all_dates, fill_value=0)

        total = fees_aligned + slippage_aligned + borrow_aligned

        logger.info(
            f"Cost breakdown - Fees: ${fees.sum():,.2f}, "
            f"Slippage: ${slippage.sum():,.2f}, "
            f"Borrow: ${borrow.sum():,.2f}, "
            f"Total: ${total.sum():,.2f}"
        )

        return {
            "fees": fees,
            "slippage": slippage,
            "borrow": borrow,
            "total": total,
        }
