"""Download market data from Yahoo Finance for experiments."""

from __future__ import annotations

import pandas as pd
import yfinance as yf
from pathlib import Path

# Define universe of stocks - Expanded to 150 symbols for better cross-sectional strategies
SYMBOLS = [
    # Technology (20 stocks)
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "ADBE", "CRM", "ORCL",
    "INTC", "AMD", "CSCO", "IBM", "QCOM", "TXN", "AVGO", "NFLX", "PYPL", "SHOP",

    # Finance (20 stocks)
    "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "BK", "TFC",
    "AXP", "SCHW", "BLK", "SPGI", "MCO", "CME", "ICE", "COF", "DFS", "SYF",

    # Healthcare (20 stocks)
    "UNH", "JNJ", "PFE", "ABBV", "TMO", "ABT", "LLY", "MRK", "BMY", "AMGN",
    "GILD", "MDT", "DHR", "CVS", "CI", "HUM", "ISRG", "SYK", "BSX", "ZTS",

    # Consumer Discretionary (15 stocks)
    "HD", "NKE", "MCD", "SBUX", "LOW", "TJX", "BKNG", "CMG",
    "MAR", "YUM", "ROST", "DG", "ULTA", "F", "GM",

    # Consumer Staples (15 stocks)
    "WMT", "PG", "KO", "PEP", "COST", "PM", "MO", "CL", "MDLZ", "KMB",
    "GIS", "K", "HSY", "CPB", "CAG",

    # Energy (10 stocks)
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL",

    # Industrials (15 stocks)
    "BA", "CAT", "GE", "MMM", "HON", "UPS", "RTX", "LMT", "DE", "EMR",
    "WM", "NSC", "UNP", "CSX", "FDX",

    # Materials (10 stocks)
    "LIN", "APD", "ECL", "SHW", "DD", "NEM", "FCX", "NUE", "VMC", "MLM",

    # Real Estate (10 stocks)
    "AMT", "PLD", "CCI", "EQIX", "PSA", "DLR", "O", "WELL", "AVB", "EQR",

    # Utilities (10 stocks)
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "WEC", "ES",

    # Communication Services (5 stocks)
    "DIS", "CMCSA", "T", "VZ", "CHTR",
]

def download_stock_data(
    symbols: list[str],
    start_date: str = "2019-01-01",
    end_date: str = "2024-01-01",
    output_path: str | Path = "data/stocks_5y.parquet",
) -> None:
    """Download OHLCV data for given symbols."""
    print(f"Downloading data for {len(symbols)} symbols from {start_date} to {end_date}...")

    # Download data
    data = yf.download(
        symbols,
        start=start_date,
        end=end_date,
        group_by="ticker",
        auto_adjust=True,  # Adjust for splits/dividends
        progress=True,
    )

    # Reshape to multiindex format
    dfs = []
    for symbol in symbols:
        try:
            if len(symbols) == 1:
                symbol_data = data
            else:
                symbol_data = data[symbol]

            # Create dataframe with symbol column
            df = symbol_data.copy()
            df["symbol"] = symbol
            df = df.reset_index()
            dfs.append(df)
        except KeyError:
            print(f"Warning: No data for {symbol}")
            continue

    # Combine all symbols
    combined = pd.concat(dfs, ignore_index=True)

    # Rename columns to lowercase
    combined.columns = combined.columns.str.lower()

    # Set multiindex
    combined["date"] = pd.to_datetime(combined["date"], utc=True)
    combined = combined.set_index(["date", "symbol"])

    # Save to parquet
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path)

    print(f"Downloaded {len(combined)} rows for {len(symbols)} symbols")
    print(f"Date range: {combined.index.get_level_values('date').min()} to {combined.index.get_level_values('date').max()}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    download_stock_data(SYMBOLS)
