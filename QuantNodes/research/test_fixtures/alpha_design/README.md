# alpha_design (M2 refactor: moved from quant_alpha.alpha{101,158}_design/)

Reference modules for Alpha 101 (Kakushadze 2015, arXiv:1601.00991) and
Alpha 158 (Qlib, Yang et al. 2020, arXiv:2009.11189) **design philosophy**.

These were previously vendored as `QuantNodes.research.quant_alpha.alpha101_design`
and `QuantNodes.research.quant_alpha.alpha158_design` but had **no production
consumer** — they were only imported by `tests/quant_alpha/test_designs.py`.

Moved to `test_fixtures/` per M2 of the refactor plan
(see `docs/refactor/REFACTOR_PLAN.md`). The few-shot examples and design
principles are preserved for future use as Alpha-GPT-style prompt
construction material.

## Subpackages

- `alpha101_design` — 8 design principles + 10 core operators + A-share
  compatibility + 10 few-shot examples.
- `alpha158_design` — 4 feature categories + 158 total features
  (9 KBAR + 20 Price + 5 Volume + 124 Rolling) + 360 template.

## Usage

```python
from QuantNodes.research.test_fixtures.alpha_design.alpha101_design import (
    DESIGN_PHILOSOPHY,
    ALPHA101_FEW_SHOT_EXAMPLES,
    get_few_shot_prompt,
)
```