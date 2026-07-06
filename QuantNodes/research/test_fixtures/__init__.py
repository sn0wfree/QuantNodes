# QuantNodes research test fixtures (data-only, optional)

Subpackages here provide **vendor-style test data + reference modules** that
are *not* part of the production runtime path. Use cases:

1. Smoke tests (`scripts/research/run_101_alphas_v2.py` can consume
   `alphas101/data/track_b_checkpoint.json` directly when vendored).
2. Few-shot examples + design philosophy references for Alpha-GPT-style
   workflows (`alpha_design/`).

These modules are excluded from `pip install quantnodes` default behavior
(no `__init__.py`-level re-export) but available to tests.