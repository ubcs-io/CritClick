# Suggested CI configuration (not committed)

> **Why this is a doc, not a workflow file**
> The repo's current Git Personal Access Token (PAT) lacks the `workflow`
> scope, so GitHub rejects any push that adds or modifies files under
> `.github/workflows/`. Until a token with that scope is available, CI is
> documented here as a suggested snippet for a project maintainer to add
> manually. See `.clinerules` for the full policy.

This is the CI matrix that `tester` is designed to pass. To enable it,
create `.github/workflows/ci.yml` by hand with the contents below.

## Suggested `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    name: Test (${{ matrix.os }} / Python ${{ matrix.python-version }})
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
        shell: bash

      - name: Install Linux headless deps
        if: runner.os == 'Linux'
        run: |
          sudo apt-get update
          sudo apt-get install -y xvfb python-xlib
          pip install ".[headless]"

      - name: Lint (ruff)
        run: ruff check src/ tests/
        shell: bash

      - name: Type check (mypy)
        run: mypy src/ || true
        shell: bash
        continue-on-error: true

      - name: Run tests (Linux, under Xvfb)
        if: runner.os == 'Linux'
        run: xvfb-run -a -s "-screen 0 1280x720x24" python -m pytest
        shell: bash

      - name: Run tests (macOS / Windows)
        if: runner.os != 'Linux'
        run: python -m pytest
        shell: bash

      - name: Self-test (capture/input sanity check)
        if: runner.os != 'Linux'
        run: tester --self-test || true
        shell: bash
        continue-on-error: true
```

## Notes

- **Linux** runs under `xvfb-run` (headless path) with `python-xlib` and the
  `headless` extra installed.
- **macOS / Windows** run in the runner's real desktop session and also
  exercise `tester --self-test` to confirm the capture/input backends work
  there.
- `mypy` and `--self-test` are kept non-blocking (`continue-on-error`) so
  they surface information without failing the build while the Windows/macOS
  capture paths are still being hardened.