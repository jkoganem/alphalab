# Examples

This directory contains example scripts demonstrating how to use Alpha Backtest Lab.

## Available Examples

### 1. simple_backtest.py

A complete end-to-end example showing:
- Data ingestion from Yahoo Finance
- Feature generation
- Alpha signal creation (cross-sectional momentum)
- Signal conversion to long-short positions
- Portfolio optimization
- Backtesting with realistic costs
- Performance metrics calculation

**How to run:**

```bash
cd "/Users/junichikoganemaru/Desktop/Alpha Generation"

# Activate virtual environment
source .venv/bin/activate

# Run the example
python examples/simple_backtest.py
```

**Expected output:**
- Data fetching progress
- Feature generation summary
- Signal statistics
- Backtest results
- Performance metrics including:
  - Returns (total, annual, CAGR)
  - Risk metrics (volatility, Sharpe, Sortino, Calmar)
  - Drawdown analysis
  - Trade statistics
- Equity curve saved to `out/equity_curve.csv`

**Customize the example:**

Edit the script to try different:
- Symbols (change the `symbols` list)
- Date ranges (modify `start` and `end`)
- Alpha models (replace `CrossSectionalMomentum` with `TimeSeriesMomentum` or `MeanReversion`)
- Signal parameters (adjust `long_pct`, `short_pct`)
- Portfolio methods (try `InverseVolatilityOptimizer`)
- Cost assumptions (modify `costs_cfg`)

## Creating Your Own Examples

### Basic Template

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from alphalab.data.yahoo import YahooDataSource
from alphalab.features.pipeline import StandardFeaturePipeline
from alphalab.alpha.momentum import TimeSeriesMomentum
from alphalab.signals.converters import RankLongShort
from alphalab.portfolio.optimizers import EqualWeightOptimizer
from alphalab.backtest.engine import VectorizedBacktester
from alphalab.backtest.metrics import calculate_all_metrics, print_metrics_summary

# 1. Get data
data_source = YahooDataSource()
data = data_source.fetch(
    symbols=["AAPL", "MSFT", "GOOGL"],
    start=pd.Timestamp("2020-01-01", tz="UTC"),
    end=pd.Timestamp("2024-12-31", tz="UTC"),
)

# 2. Generate features
features = StandardFeaturePipeline().transform(data)

# 3. Generate alpha
alpha = TimeSeriesMomentum(lookback_days=126).score(features)

# 4. Convert to signals
signals = RankLongShort(long_pct=0.2, short_pct=0.2).to_signal(alpha)

# 5. Optimize portfolio
weights = EqualWeightOptimizer().allocate(signals)

# 6. Run backtest
bt = VectorizedBacktester(initial_capital=1_000_000)
results = bt.run(weights, data[["open", "close"]])

# 7. Analyze
metrics = calculate_all_metrics(results['equity_curve'], results['returns'])
print_metrics_summary(metrics)
```

## More Examples (Coming Soon)

### mean_reversion_backtest.py
Mean reversion strategy with residual-based entries

### pairs_trading_backtest.py
Statistical arbitrage using cointegration

### ml_alpha_backtest.py
Machine learning-based alpha signals

### walk_forward_validation.py
Walk-forward testing with model retraining

### parameter_sensitivity.py
Grid search over strategy parameters

## Tips

1. **Start small**: Use a small universe and short date range for quick iterations
2. **Check data quality**: Always inspect the fetched data before running backtests
3. **Understand costs**: Transaction costs significantly impact results
4. **Validate signals**: Print signal statistics to ensure they make sense
5. **Compare to benchmark**: Always compare to a buy-and-hold SPY benchmark

## Troubleshooting

### "ModuleNotFoundError: No module named 'alphalab'"

Make sure you've installed the package:
```bash
pip install -e .
```

Or add the src directory to your path (as shown in the template).

### "No data returned for symbols"

Check your internet connection and verify the symbols are valid on Yahoo Finance.

### "DataFrame index error"

Ensure your data has the correct MultiIndex format (date, symbol).

### Slow performance

- Reduce the date range
- Use fewer symbols
- Use MinimalFeaturePipeline instead of StandardFeaturePipeline
- Check for unnecessary debug logging

## Contributing Examples

Have a cool example? Add it to this directory and submit a PR!

Guidelines:
- Include docstring explaining what it demonstrates
- Keep it under 200 lines if possible
- Add comments for key steps
- Print informative output
- Save results to the `out/` directory
