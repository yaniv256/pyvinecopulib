# pyvinecopulib Working Notes

## Repository Scope
- Python bindings for the `vinecopulib` C++ library plus build scripts/stubs.
- Everything lives at `C:\Users\ybenami\Documents\pyvinecopulib`; dependent projects (e.g., portfolio-construction) point to this checkout via `pip install -e ..\pyvinecopulib`.

## Environment Setup
- README-backed guidance: stick to the conda-managed workflow (`make env-conda` ➜ `conda activate pyvinecopulib`). Standalone Python installs (even a pure 3.12) aren’t validated for builds/tests here.
1. Preferred Windows launch command (working today):
   ```powershell
   & "$env:WINDIR\System32\WindowsPowerShell\v1.0\powershell.exe" -ExecutionPolicy Bypass -NoExit `
     -Command "& 'c:\TradeTech\Miniconda3_20221221\shell\condabin\conda-hook.ps1'; conda activate pyvinecopulib"
   cd C:\Users\ybenami\Documents\pyvinecopulib
   python scripts\generate_requirements.py --format yml
   conda env create -f environment.yml   # or: make env-conda (once GNU Make is available)
   conda activate pyvinecopulib
   make dev-setup
   ```
2. On WSL or standalone Linux: install Miniconda (`bash Miniconda3-latest-Linux-x86_64.sh`), then run the same `make env-conda` + `make dev-setup` flow.
3. Always use the activated conda env before running builds/tests (`conda activate pyvinecopulib`).

## Common Commands
- `pip install -e .` — ensures editable imports for downstream repos.
- `make quick-check` — fast lint/type/test loop.
- `make check-all` — full pre-commit suite (lint, type-check, tests, examples).
- `make test` / `make test-fast` — full vs. quick unit tests.
- `make test-examples` — run the bundled notebooks.
- `make docs` / `make metadata` — rebuild docstrings, stubs, and documentation.

## Development Guidelines
- Keep the submodules (`vinecopulib`, `wdm`) synced via `git submodule update --init --recursive`.
- Run `make pre-commit` before pushing; hooks enforce formatting (ruff, clang-format, cmake-format).
- Document API changes in `CHANGELOG.md` and regenerate stubs/docstrings when touching C++ bindings.
- Coordinate version bumps with the C++ repo; Python wheels should track the same semantic version.

## Local Usage Notes
- Downstream projects should **not** vendor pyvinecopulib; point them to this repo via `pip install -e ..\pyvinecopulib` or through the conda env.
- When running Jupyter examples, start `jupyter notebook` from the activated env (`conda activate pyvinecopulib`).
- Large builds happen out-of-tree; clean artifacts with `make clean` if you need a fresh build.

## Support
- Report upstream issues in this repo; keep internal hacks documented in PR descriptions.
- For environment issues (compiler paths, Eigen/Boost detection), capture the exact `cmake` log before escalating.
