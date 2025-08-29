# Contributing to SGN‑Guard

Thanks for your interest in contributing! This project focuses on pragmatic, non‑novel stability heuristics for decentralized training.

## Development setup
- Python >= 3.9
- Create a virtualenv and install deps:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Tests and lint
```bash
pytest -q
mypy src
ruff check .
```

Notes:
- `mypy` is configured to ignore missing imports for third‑party packages.
- `ruff` uses default rules with a reasonable line length.

## Pull requests
- Keep changes small and focused; include tests when feasible.
- Document user‑visible changes in `README.md`.
- Ensure CI passes (tests + typecheck + lint).

## Code of conduct
Please follow our [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
