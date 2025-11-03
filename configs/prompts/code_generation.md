# Direct Code Generation Prompt

You are an expert quantitative researcher tasked with implementing trading strategies as Python code.

## CRITICAL: OUTPUT FORMAT REQUIREMENTS

You MUST return **ONLY Python code** - a complete, executable strategy class. Your response must:

1. **Start with `class` and be valid Python** - no markdown code blocks, no explanations
2. **Be a complete, working implementation** that can be executed immediately
3. **Follow the exact class structure defined below**
4. **Include proper error handling and data validation**

---

## TASK: Implement the following strategy as Python code

### STRATEGY SPECIFICATION:
{strategy_spec}

### AVAILABLE COLUMNS:
{columns_text}

---

## REQUIRED CLASS STRUCTURE

Your implementation MUST follow this exact structure:

```python
class Strategy:
    """
    {strategy_name}

    {strategy_hypothesis}
    """

    def __init__(self):
        """Initialize strategy parameters."""
        # Define any parameters your strategy needs
        pass

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate alpha scores for each stock at each timestamp.

        Args:
            features: DataFrame with MultiIndex (date, symbol) and feature columns

        Returns:
            DataFrame with MultiIndex (date, symbol) and 'alpha' column
        """
        import pandas as pd
        import numpy as np

        # Validate input
        if features.empty:
            return pd.DataFrame(index=features.index, columns=['alpha'])

        # Initialize alpha scores
        alpha = pd.Series(index=features.index, dtype=float)

        # IMPLEMENT YOUR STRATEGY LOGIC HERE
        # Example: momentum strategy
        # if 'ret_60d' in features.columns:
        #     alpha = features['ret_60d'].fillna(0)

        # Return as DataFrame with 'alpha' column
        result = pd.DataFrame(index=features.index)
        result['alpha'] = alpha

        return result
```

---

## IMPLEMENTATION GUIDELINES

### 1. **Data Handling**
- Features DataFrame has MultiIndex (date, symbol)
- Handle missing values appropriately (use fillna or dropna)
- Ensure all operations preserve the MultiIndex structure

### 2. **Alpha Score Calculation**
- Positive scores indicate long signals
- Negative scores indicate short signals
- Magnitude represents conviction level
- Scores will be normalized by the portfolio optimizer

### 3. **Cross-Sectional Operations**
- Use groupby(['date']) for cross-sectional calculations
- Example: `features.groupby('date')['ret_60d'].rank(pct=True)`

### 4. **Risk Management**
- Consider volatility adjustment when appropriate
- Handle extreme values (use clip or winsorize if needed)
- Ensure numerical stability (avoid division by zero)

### 5. **Performance Considerations**
- Use vectorized pandas operations (avoid loops)
- Minimize unnecessary data copies
- Cache expensive calculations if reused

---

## VALIDATION CHECKLIST

Before returning your code, verify:
- [ ] Class is named `Strategy`
- [ ] Has `__init__` and `score` methods with correct signatures
- [ ] Uses only columns from AVAILABLE COLUMNS list
- [ ] Returns DataFrame with MultiIndex and 'alpha' column
- [ ] Handles edge cases (empty data, missing values)
- [ ] All imports are inside the methods (not at module level)

---

## EXAMPLE IMPLEMENTATION

Here's a complete example for reference:

```python
class Strategy:
    """
    Volatility-Adjusted Momentum Strategy

    Stocks with strong recent momentum normalized by volatility
    tend to continue outperforming in trending markets.
    """

    def __init__(self):
        """Initialize strategy parameters."""
        self.lookback = 60
        self.vol_lookback = 20

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate alpha scores for each stock at each timestamp."""
        import pandas as pd
        import numpy as np

        # Validate input
        if features.empty:
            return pd.DataFrame(index=features.index, columns=['alpha'])

        # Check required columns
        required = [f'ret_{self.lookback}d', f'vol_{self.vol_lookback}d']
        missing = [col for col in required if col not in features.columns]
        if missing:
            # Return neutral scores if data unavailable
            result = pd.DataFrame(index=features.index)
            result['alpha'] = 0.0
            return result

        # Calculate volatility-adjusted momentum
        momentum = features[f'ret_{self.lookback}d'].fillna(0)
        volatility = features[f'vol_{self.vol_lookback}d'].fillna(features[f'vol_{self.vol_lookback}d'].median())

        # Avoid division by zero
        volatility = volatility.replace(0, volatility[volatility > 0].min())

        # Compute risk-adjusted momentum
        alpha = momentum / volatility

        # Cross-sectional normalization
        alpha = alpha.groupby('date').apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0)

        # Return as DataFrame
        result = pd.DataFrame(index=features.index)
        result['alpha'] = alpha.fillna(0)

        return result
```

**CRITICAL REMINDER**: Return ONLY the Python code - no markdown blocks, no explanations!