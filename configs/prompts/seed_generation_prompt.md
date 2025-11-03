# SEED STRATEGY GENERATION PROMPT

You are an expert quantitative researcher tasked with generating novel alpha strategies for backtesting.

## CRITICAL: OUTPUT FORMAT REQUIREMENTS

You MUST return a **valid JSON array** containing 5-10 strategy specifications. Your response must:

1. **Start with `[` and end with `]`** - the opening and closing brackets of the JSON array
2. **Contain NO text before or after the JSON** - no explanations, no markdown code blocks, no commentary
3. **Be valid JSON** that can be parsed by `json.loads()` in Python
4. **Follow the exact schema defined below**

---

## JSON SCHEMA (STRICT)

Each strategy in the array MUST have this exact structure:

```json
{
  "family": "<MUST be exactly 'cross_sectional_rank'>",
  "factors": [
    {
      "name": "<descriptive_factor_name>",
      "expr": "<base_feature_from_list_below>",
      "transforms": ["<transform_1>", "<transform_2>", "..."]
    }
  ],
  "combine": {
    "method": "<MUST be 'weighted_mean'>",
    "weights": [<array of floats summing to ~1.0>]
  },
  "post": ["<post_transform_1>", "<post_transform_2>", "..."],
  "rationale": "<1-2 sentence explanation of the strategy logic>"
}
```

---

## FIELD SPECIFICATIONS

### 1. `family` (string, REQUIRED)
- **MUST be exactly**: `"cross_sectional_rank"`
- **NO other values allowed**

### 2. `factors` (array, REQUIRED)
- **Array of 1-5 factor objects**
- Each factor object has:
  - `name` (string): Descriptive name (lowercase_with_underscores)
  - `expr` (string): **MUST be ONE of the base features listed below**
  - `transforms` (array of strings): Ordered list of transformations

### 3. `combine` (object, REQUIRED)
- `method` (string): **MUST be exactly** `"weighted_mean"`
- `weights` (array of floats): One weight per factor, should sum to approximately 1.0

### 4. `post` (array of strings, REQUIRED)
- **Post-processing transforms applied AFTER combining factors**
- **MUST include**: `"cs_rank_pct_centered"` and `"normalize:sum_abs"`
- Common pattern: `["cs_rank_pct_centered", "clip:-2,2", "normalize:sum_abs"]`

### 5. `rationale` (string, REQUIRED)
- 1-2 sentence explanation of the strategy's market hypothesis

---

## AVAILABLE BASE FEATURES (for `expr` field)

### Price Returns (Technical)
- `ret_1d`, `ret_5d`, `ret_10d`, `ret_20d`, `ret_21d`, `ret_60d`, `ret_126d`, `ret_252d`

### Volatility (Technical)
- `vol_10d`, `vol_20d`, `vol_60d`
- `parkinson_20d` (Parkinson volatility estimator using high-low range)
- `hl_vol_20d` (High-low volatility)

### Volume/Liquidity (Technical)
- `adv_20d` (Average dollar volume over 20 days)
- `volume_zscore_20d`, `volume_zscore_60d` (Volume z-scores)
- `turnover_20d` (Volume/shares outstanding ratio)

### Statistical Features
- `zscore_5d`, `zscore_20d` (Price z-scores)
- `skew_20d`, `skew_60d` (Skewness of returns)
- `kurt_20d`, `kurt_60d` (Kurtosis of returns)

### Technical Indicators
- `rsi_14d`, `rsi_28d` (Relative Strength Index, 0-100)

### Gap Features
- `gap` (Open/previous close - 1)
- `gap_1d` (1-day lagged gap)
- `gap_z20` (Gap z-score over 20 days)

### Seasonality
- `day_of_week` (0-6, Monday=0)
- `day_of_month` (1-31)
- `month` (1-12)


---

## AVAILABLE TRANSFORMS

### Arithmetic Operations
- `multiply:<number>` - Multiply by constant (e.g., `"multiply:2.0"`)
- `divide:<number>` - Divide by constant (e.g., `"divide:5.0"`)
- `add:<number>` - Add constant (e.g., `"add:1.0"`)
- `subtract:<number>` - Subtract constant (e.g., `"subtract:0.5"`)
- `multiply:-1` - Flip sign (for reversals)

### Cross-Sectional Normalization
- `cs_zscore` - Z-score within each time period
- `cs_robust_zscore` - Robust z-score (median/MAD)
- `cs_rank` - Rank (0 to N-1)
- `cs_rank_pct` - Rank percentile (0.0 to 1.0)
- `cs_rank_pct_centered` - Centered rank (-0.5 to +0.5)

### Outlier Handling
- `winsor:-3,3` - Winsorize at ±3 standard deviations
- `winsor:-2,2` - Winsorize at ±2 standard deviations
- `clip:-2,2` - Hard clip at ±2

### Time-Series Operations
- `ts_demean:20` - Remove 20-period rolling mean
- `ts_zscore:60` - Rolling 60-period z-score
- `ts_rank:20` - Rolling 20-period rank
- `ts_delta:5` - 5-period difference

### Final Normalization (for `post`)
- `normalize:sum_abs` - Scale so absolute weights sum to 1.0
- `normalize:l2` - L2 normalization

---

## STRATEGY DESIGN PRINCIPLES

### 1. **Signal Construction**
- Choose ONE clear thesis: pure momentum OR pure reversal
- Pure Momentum: Use `ret_60d` or `ret_126d` as primary factor
- Pure Reversal: Use `-ret_5d` or `-ret_1d` as primary factor (note the minus sign!)
- Optional supporting factors: volume confirmation, technical indicators

### 2. **Normalization Pipeline**
```json
"transforms": [
  "cs_robust_zscore",     ← Cross-sectional normalization (FIRST!)
  "winsor:-3,3"            ← Outlier handling
],
"post": [
  "cs_rank_pct_centered",  ← Convert to centered ranks
  "clip:-2,2",             ← Final clipping
  "normalize:sum_abs"      ← Portfolio constraint
]
```

**WARNING**: DO NOT use `divide:vol_20d` or other volatility division - this causes catastrophic failures during volatility spikes.

### 3. **Multi-Factor Combining**
```json
"factors": [
  {"name": "momentum", "expr": "ret_60d", "transforms": ["cs_robust_zscore", "winsor:-3,3"]},
  {"name": "volume_signal", "expr": "volume_zscore_20d", "transforms": ["cs_robust_zscore", "winsor:-3,3"]}
],
"combine": {
  "method": "weighted_mean",
  "weights": [0.7, 0.3]   ← 70% momentum, 30% volume confirmation
}
```

---

## VALIDATION CHECKLIST

Before submitting, verify:

- [ ] **Valid JSON**: Can be parsed by `json.loads()`
- [ ] **No wrapping**: No markdown code blocks, no text before/after JSON
- [ ] **family**: Exactly `"cross_sectional_rank"`
- [ ] **expr**: Uses ONLY features from the Available Base Features list
- [ ] **combine.method**: Exactly `"weighted_mean"`
- [ ] **combine.weights**: Array length matches `factors` array length
- [ ] **post**: Includes both `"cs_rank_pct_centered"` and `"normalize:sum_abs"`
- [ ] **rationale**: Present for each strategy (1-2 sentences)

---

## EXAMPLE OUTPUT (EXACT FORMAT REQUIRED)

```json
[
  {
    "family": "cross_sectional_rank",
    "factors": [
      {
        "name": "momentum",
        "expr": "ret_60d",
        "transforms": ["cs_robust_zscore", "winsor:-3,3"]
      }
    ],
    "combine": {"method": "weighted_mean", "weights": [1.0]},
    "post": ["cs_rank_pct_centered", "clip:-2,2", "normalize:sum_abs"],
    "rationale": "Pure momentum with cross-sectional normalization for relative strength ranking"
  },
  {
    "family": "cross_sectional_rank",
    "factors": [
      {
        "name": "short_reversal",
        "expr": "-ret_5d",
        "transforms": ["cs_robust_zscore", "winsor:-3,3"]
      }
    ],
    "combine": {"method": "weighted_mean", "weights": [1.0]},
    "post": ["cs_rank_pct_centered", "clip:-2,2", "normalize:sum_abs"],
    "rationale": "Short-term mean reversion betting on price reversals within cross-section"
  },
  {
    "family": "cross_sectional_rank",
    "factors": [
      {
        "name": "momentum",
        "expr": "ret_126d",
        "transforms": ["cs_robust_zscore", "winsor:-3,3"]
      },
      {
        "name": "volume_signal",
        "expr": "volume_zscore_20d",
        "transforms": ["cs_robust_zscore", "winsor:-3,3"]
      }
    ],
    "combine": {"method": "weighted_mean", "weights": [0.7, 0.3]},
    "post": ["cs_rank_pct_centered", "clip:-2,2", "normalize:sum_abs"],
    "rationale": "Medium-term momentum confirmed by volume signals for robust directional bias"
  }
]
```

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

## YOUR TASK

Generate 5-10 novel, creative alpha strategies using:
1. **Different lookback periods** (1d, 5d, 10d, 20d, 60d, 126d, 252d)
2. **Cross-sectional normalization** (robust z-scores, ranking)
3. **Multi-factor combinations** (momentum + volume, reversal + RSI)

**Focus on strategies that**:
- Choose ONE clear thesis (momentum OR reversal - don't mix!)
- Maximize Sharpe ratio while controlling drawdown
- Exploit behavioral biases or market inefficiencies
- Use cross-sectional normalization (NOT volatility division)
- Combine complementary signals intelligently
- Have clear, testable hypotheses
- Balance returns with risk management

**CRITICAL REMINDER**: Your response must be ONLY the JSON array - no other text!
