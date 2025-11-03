# LLM Prompt Management System

This directory contains all prompts used for LLM-powered strategy generation in Alpha Backtest Lab.

## Purpose

Storing prompts in separate markdown files provides:
- **Easy Iteration**: Modify prompts without touching source code
- **Version Control**: Track prompt changes with clear git diffs
- **A/B Testing**: Swap prompts easily to compare performance
- **Documentation**: Prompts become self-documenting
- **Collaboration**: Non-technical stakeholders can review/modify prompts

## Directory Structure

```
configs/prompts/
├── README.md                        # This file
├── idea_generation.md               # Strategy idea brainstorming
├── factor_spec_generation.md        # JSON factor spec generation
├── batch_spec_generation.md         # Batch JSON spec generation
└── code_generation.md               # Direct Python code generation (legacy)
```

## Template Variables

Prompts support template variable substitution using Python's `str.format()`:

### Common Variables

- `{n_ideas}` - Number of ideas to generate
- `{market_context}` - Description of current market conditions
- `{data_desc}` - Available data description
- `{constraints_section}` - User preferences or default constraints
- `{columns_text}` - Available column names
- `{idea_name}` - Strategy name
- `{idea_hypothesis}` - Strategy hypothesis
- `{idea_features}` - Candidate features
- `{available_columns}` - JSON list of available columns
- `{validation_feedback}` - Validation errors from previous iteration

## Usage

### In Python Code

```python
from alphalab.ai.prompt_loader import PromptLoader

# Initialize loader
loader = PromptLoader()

# Load and render prompt
prompt = loader.load("idea_generation",
                     n_ideas=10,
                     market_context="Bull market with high volatility",
                     columns_text=str(available_columns))
```

### Prompt Format

Prompts are markdown files with template variables in `{curly_braces}`:

```markdown
# Example Prompt

Generate {n_ideas} trading strategies using these columns:

{columns_text}

Market context: {market_context}
```

## Prompt Descriptions

### `idea_generation.md`
**Purpose**: Generate high-level strategy ideas in plain English
**Input**: Market context, available data, user preferences
**Output**: JSON array of strategy ideas with hypotheses and features
**Used by**: `StrategyGenerator.generate_ideas()`

### `factor_spec_generation.md`
**Purpose**: Convert strategy idea to JSON factor specification
**Input**: Strategy idea, available columns, validation feedback
**Output**: JSON factor spec for cross-sectional ranking strategy
**Used by**: `StrategyGenerator.generate_spec()`

### `batch_spec_generation.md`
**Purpose**: Generate multiple factor specs in one LLM call
**Input**: List of strategy ideas, available columns
**Output**: JSON array of factor specifications
**Used by**: `StrategyGenerator.generate_batch_specs()`

### `code_generation.md` (Legacy)
**Purpose**: Generate Python code directly from idea
**Input**: Strategy idea, data schema
**Output**: Python class implementation
**Status**: Deprecated in favor of JSON spec + compiler approach

## Modifying Prompts

When modifying prompts, follow these guidelines:

### 1. Test Changes Locally First
```bash
# Run a small experiment with your modified prompt
python examples/personalized_strategy_discovery.py \
  --model gpt-4o-mini \
  --n-strategies 3 \
  --max-iterations 5
```

### 2. Document What Changed
Include comments in git commits explaining:
- What changed in the prompt
- Why the change was made
- Expected impact on strategy quality

### 3. A/B Test When Possible
- Keep the old prompt as `prompt_name_v1.md`
- Create new version as `prompt_name_v2.md`
- Compare results before switching

### 4. Monitor Key Metrics
After prompt changes, track:
- **Parse Success Rate**: % of LLM responses that parse correctly
- **Compilation Success Rate**: % of specs that compile without errors
- **Backtest Success Rate**: % of strategies that run to completion
- **Average Sharpe Ratio**: Strategy performance metric
- **Score Distribution**: Composite score percentiles

## Versioning

Prompt versions are tracked via git commits. For major changes:

1. Copy current prompt to `{name}_v{N}.md`
2. Modify the main prompt file
3. Commit with descriptive message
4. If revert needed, restore from version file

Example:
```bash
# Before major change
cp idea_generation.md idea_generation_v1.md
git add idea_generation_v1.md
git commit -m "Archive idea generation prompt v1 before major rewrite"

# Make changes to idea_generation.md
# ...

git add idea_generation.md
git commit -m "Refactor idea generation prompt: Add macro conditioning guidance"
```

## Troubleshooting

### Prompt Too Long
- LLMs have token limits (e.g., gpt-4o: 128k tokens)
- If prompts are rejected, reduce example count or verbosity

### Low Parse Success Rate
- Add more specific output format requirements
- Include more examples of correct JSON structure
- Add validation rules explicitly

### Strategies Don't Meet Quality Bar
- Adjust performance targets in prompt
- Add more constraints or guardrails
- Include negative examples (what NOT to do)

## Future Enhancements

Planned improvements:
- [ ] Few-shot examples stored separately and injected dynamically
- [ ] Prompt templates with inheritance (base + specialized)
- [ ] Automatic prompt optimization via LLM scoring
- [ ] Prompt analytics dashboard showing success rates by version

---

Last Updated: November 1, 2025
