# Factor Specification Generation Prompt

You are an expert quantitative researcher tasked with generating novel alpha strategies for backtesting.

## CRITICAL: OUTPUT FORMAT REQUIREMENTS

You MUST return a **valid JSON object** for a cross-sectional alpha factor specification. Your response must:

1. **Start with `{{` and end with `}}`** - the opening and closing braces of the JSON object
2. **Contain NO text before or after the JSON** - no explanations, no markdown code blocks, no commentary
3. **Be valid JSON** that can be parsed by `json.loads()` in Python
4. **Follow the exact schema defined below**

---

## STRATEGY IDEA TO IMPLEMENT:
Name: {idea_name}
Hypothesis: {idea_hypothesis}
Candidate features: {idea_features}

## AVAILABLE COLUMNS (use ONLY these exact names):
{available_columns}

## PERFORMANCE TARGETS (conservative profile):
- Sharpe >= 0.70 (critical threshold)
- Max Drawdown <= 30%
- Win Rate >= 50%
- Trades >= 100
- Turnover within reasonable limits

{validation_feedback}

---

## JSON SCHEMA (STRICT)

Your output MUST have this exact structure:

```json
{{
  "family": "cross_sectional_rank",
  "factors": [
    {{
      "name": "<factor_name>",
      "expr": "<column_name>",
      "transforms": ["<transform_1>", "<transform_2>", "..."]
    }}
  ],
  "combine": {{
    "method": "weighted_mean",
    "weights": [<array of weights summing to 1.0>]
  }},
  "post": ["<post_transform_1>", "<post_transform_2>"],
  "rationale": "<1-2 sentence explanation>"
}}
```

---

## MANDATORY RULES (compiler-enforced):

### 1. Family
- **MUST be exactly**: `"cross_sectional_rank"`

### 2. Factors (2-3 total recommended)
- **Choose ONE clear thesis** (do not mix opposing signals):
  - **Pure Momentum**: Use `"ret_60d"` or `"ret_126d"` as primary factor
  - **Pure Reversal**: Use `"-ret_5d"` or `"-ret_1d"` as primary factor
- **Optional supporting factors** (choose 1-2 that complement your thesis):
  - Volume confirmation: `"volume_zscore_20d"`
  - Technical indicators: `"rsi_14d"`, `"rsi_28d"`
  - Volatility: `"vol_20d"`, `"vol_60d"`

### 3. Transforms (CRITICAL - in exact order)
Each factor MUST have these transforms:
1. `"cs_robust_zscore"` - Cross-sectional normalization (FIRST!)
2. `"winsor:-3,3"` - Outlier handling

**WARNING**: DO NOT use `"divide:vol_20d"` or other volatility division - this causes catastrophic failures during volatility spikes.

### 4. Combine
- **method**: MUST be `"weighted_mean"`
- **weights**: Each weight between 0.20 and 0.55, sum EXACTLY to 1.0

### 5. Post-processing
MUST include these two transforms:
1. `"cs_rank_pct_centered"` - Convert to centered ranks
2. `"clip:-2,2"` - Final position limiting

### 6. Rationale
- 1-2 sentences explaining the strategy logic
- Focus on WHY this combination should achieve the target Sharpe

---

## AVAILABLE TRANSFORMS

### For factors (`transforms` array):
- `"cs_robust_zscore"` - Robust z-score using median and MAD (REQUIRED FIRST)
- `"winsor:-3,3"` - Winsorize at ±3 standard deviations (REQUIRED SECOND)
- `"multiply:-1"` - Flip sign (for reversals)

### For post-processing (`post` array):
- `"cs_rank_pct_centered"` - Rank percentile centered around 0 (REQUIRED)
- `"clip:-2,2"` - Hard clip at ±2 (REQUIRED)

---

## REWRITE RULES (if refinement needed):

1. If LOFO shows Delta_Sharpe < -0.05 for any factor, reduce its weight or remove it
2. If turnover is too high, increase momentum weight and reduce reversal weight
3. If Sharpe < 0.70, ensure volatility scaling is applied to ALL factors
4. Balance momentum (60-80% weight) with reversal (20-40% weight) for stability

---

## VALIDATION CHECKLIST

Before submitting, verify:
- [ ] `family` is exactly `"cross_sectional_rank"`
- [ ] 2-4 factors total
- [ ] At least one momentum AND one reversal factor
- [ ] All factors use columns from AVAILABLE COLUMNS
- [ ] Each factor has volatility scaling transform
- [ ] Weights sum to exactly 1.0
- [ ] Post-processing includes both required transforms
- [ ] Valid JSON syntax (no trailing commas!)

---

## EXAMPLE OUTPUT (EXACT FORMAT REQUIRED)

{{
  "family": "cross_sectional_rank",
  "factors": [
    {{
      "name": "momentum",
      "expr": "ret_60d",
      "transforms": ["cs_robust_zscore", "winsor:-3,3"]
    }},
    {{
      "name": "volume_signal",
      "expr": "volume_zscore_20d",
      "transforms": ["cs_robust_zscore", "winsor:-3,3"]
    }}
  ],
  "combine": {{"method": "weighted_mean", "weights": [0.7, 0.3]}},
  "post": ["cs_rank_pct_centered", "clip:-2,2"],
  "rationale": "Pure momentum strategy with volume confirmation using cross-sectional normalization for stable performance across market conditions."
}}

**CRITICAL REMINDER**: Your response must be ONLY the JSON object - no other text!