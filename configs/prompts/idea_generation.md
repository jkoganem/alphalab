# Strategy Idea Generation Prompt

You are an expert quantitative researcher tasked with generating novel alpha strategies for backtesting.

## CRITICAL: OUTPUT FORMAT REQUIREMENTS

You MUST return a **valid JSON array** containing exactly {n_ideas} strategy ideas. Your response must:

1. **Start with `[` and end with `]`** - the opening and closing brackets of the JSON array
2. **Contain NO text before or after the JSON** - no explanations, no markdown code blocks, no commentary
3. **Be valid JSON** that can be parsed by `json.loads()` in Python
4. **Follow the exact schema defined below**

---

## TASK: Generate {n_ideas} diverse, cross-sectional equity strategy ideas

### AVAILABLE COLUMNS (use ONLY these exact names):
{columns_text}

### MARKET CONTEXT:
{market_context}

### DATA DESCRIPTION:
{data_desc}

{constraints_section}

{examples_text}

---

## JSON SCHEMA (STRICT)

Each strategy idea in the array MUST have this exact structure:

```json
{{
  "name": "<descriptive_strategy_name>",
  "hypothesis": "<2-3 sentences explaining the cross-sectional market edge>",
  "features": ["<column1>", "<column2>", "..."],
  "expected_conditions": "<market regimes or conditions that favor this strategy>",
  "risks": ["<specific risk 1>", "<specific risk 2>"]
}}
```

### Field Specifications:
- **name**: Concise, descriptive name (e.g., "Volatility-Adjusted Momentum")
- **hypothesis**: Clear explanation of WHY this strategy should work. Must explain the market inefficiency or behavioral bias being exploited
- **features**: Array of column names from AVAILABLE COLUMNS list. Use 2-5 features per strategy
- **expected_conditions**: Describe when this strategy performs best (e.g., "trending markets", "high volatility regimes")
- **risks**: List 2-3 specific risks or failure modes (e.g., "momentum crashes during reversals", "high turnover in choppy markets")

---

## STRATEGY DESIGN PRINCIPLES

### 1. **Focus on ONE Clear Thesis Per Strategy**
- EITHER momentum (trend continuation) OR reversal (mean reversion)
- DO NOT mix contradictory signals in the same strategy
- Momentum works on 20-60 day horizons; reversal works on 1-5 day horizons

### 2. **Risk Management is Critical**
- Always consider volatility adjustment (using vol_20d, vol_60d)
- Think about drawdown control and position sizing
- Consider using RSI indicators for overbought/oversold conditions

### 3. **Diversity Across Strategies**
- Vary the time horizons (short-term vs medium-term)
- Mix different factor families (price, volume, volatility, technical)

### 4. **Practical Implementation**
- Strategies must work with daily rebalancing
- Consider turnover and transaction costs
- Ensure strategies are liquid enough for 150+ stock universe

---

## EVALUATION CRITERIA

Strategies will be backtested and evaluated on the following metrics:

### Primary Metrics
- **Sharpe Ratio**: Risk-adjusted returns (target: >= 0.7)
- **Total Returns**: Cumulative strategy performance (target: positive)
- **Maximum Drawdown**: Largest peak-to-trough decline (target: <= 30%)

### Secondary Metrics
- **Number of Trades**: Sufficient trading activity (target: >= 100 trades)
- **Turnover**: Portfolio churn rate (target: <= 5.0 daily turnover)

### Scoring
Strategies receive a weighted composite score out of 100:
- **Sharpe Ratio** (35%): Normalized by 1.5 target
- **Sortino Ratio** (15%): Normalized by 1.8 target
- **Max Drawdown** (20%): Inverted penalty (0 points if >50%)
- **Calmar Ratio** (10%): Return/Drawdown ratio, normalized by 2.0
- **Win Rate** (5%): Percentage of profitable trades
- **Consistency** (10%): Stability of rolling Sharpe ratio
- **Tail Risk** (5%): 5th percentile daily return protection

**Strategies scoring >= 50 are considered viable for production.**

---

## VALIDATION CHECKLIST

Before submitting, verify each strategy:
- [ ] Uses ONLY columns from the AVAILABLE COLUMNS list
- [ ] Has a clear, testable hypothesis
- [ ] Features array contains 2-5 valid column names
- [ ] Includes specific risks, not generic statements
- [ ] Different from other strategies in the batch

---

## EXAMPLE OUTPUT (EXACT FORMAT REQUIRED)

[
  {{
    "name": "Risk-Adjusted Momentum",
    "hypothesis": "Stocks with strong 60-day momentum normalized by volatility outperform as institutional investors gradually rotate into trending positions. The volatility adjustment reduces crash risk during market stress.",
    "features": ["ret_60d", "vol_20d", "volume_zscore_20d"],
    "expected_conditions": "Performs best in trending markets with moderate volatility. Underperforms during sharp reversals or regime changes.",
    "risks": ["momentum crashes during market reversals", "crowded trade risk in popular stocks", "high turnover during volatile periods"]
  }},
  {{
    "name": "Short-Term Mean Reversion with Volume",
    "hypothesis": "Stocks that experience sharp 5-day declines on high volume tend to bounce back as the selling pressure exhausts. This exploits retail investor overreaction to negative news.",
    "features": ["ret_5d", "volume_zscore_20d", "rsi_14d"],
    "expected_conditions": "Works best in range-bound markets with high volatility. Most effective after earnings announcements or news-driven selloffs.",
    "risks": ["continued selling in true fundamental deterioration", "fails during sustained bear markets", "whipsaw risk in volatile conditions"]
  }}
]

**CRITICAL REMINDER**: Your response must be ONLY the JSON array - no other text!