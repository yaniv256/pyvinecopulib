# Review: feature/incomplete-data-fitting vs main
Base branch: `chore/plotting-mypy-fixes`

- **New incomplete-data fitting helper (`src/pyvinecopulib/_python_helpers/incomplete_data.py`)**
  - Adds `fit_vine_incomplete` to fit R-vine copulas when observations are missing per edge, using adaptive truncation and independence fallbacks; keeps pseudo-observations tree-by-tree and stops early when all edges are independent.
  - Adds `get_complete_counts` plus `_parse_edge` helper to expose per-edge completeness counts for diagnostics.
  - Exports the helpers via `src/pyvinecopulib/__init__.py` so they are public API.
  - Needed to support modeling on incomplete datasets and to provide visibility into how many observations back each fitted edge.

- **Example notebook for incomplete data (`examples/08_incomplete_data.ipynb`)**
  - Demonstrates the new adaptive-fitting workflow on data with missingness.
  - Provides a usage guide for users working with incomplete observations.

- **Plotting/type safety fixes and mypy config** *(split to base branch `chore/plotting-mypy-fixes` — a maintenance branch)*
  - `pairs_copula_data` types inputs/outputs, tolerates `None`, and normalizes axes handling when `subplots` returns a scalar or mocked object.
  - `bicop_plot` uses an array for uniform-margin adjustment (`adj`) to keep broadcasting consistent.
  - `pyproject.toml` mypy target bumped to Python 3.11 and scope narrowed to `src` (tests/examples/docs excluded).
  - Rationale: keep lint/type stability changes isolated from the new incomplete-data feature for a focused review. This maintenance branch now underpins `feature/incomplete-data-fitting`.

## Notes / considerations
- The new API surface (`fit_vine_incomplete`, `get_complete_counts`) is now public; even as a minor update, add a brief patch-level changelog note (e.g., 0.7.6) calling out the helper and notebook.
- Adaptive truncation currently uses independence when counts are low and updates both variables’ pseudo-observations; if a different convention is desired, document it in the notebook and docstrings.
