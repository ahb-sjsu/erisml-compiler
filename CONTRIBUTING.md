# Contributing to ErisML Compiler

Thank you for your interest in contributing. Bug reports, feature
requests, and pull requests are all welcome.

## Getting help / asking questions

- **Bugs or unexpected behaviour:** open an issue at
  https://github.com/ahb-sjsu/erisml-compiler/issues. Include the
  compiler version (`eris-compile version`), Python version, OS, and
  a minimal reproducer if possible.
- **Feature requests / discussion:** open an issue and tag it
  `enhancement`. For substantive design proposals, prefer to attach a
  one-page design note rather than only a one-line title.
- **Security disclosures:** open a private security advisory via the
  repository's *Security* tab rather than a public issue.

## Setting up a development environment

```bash
git clone https://github.com/ahb-sjsu/erisml-compiler
cd erisml-compiler
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[test,calibration,monitor,dev]"
pytest
```

CI runs on Ubuntu against Python 3.10, 3.11, and 3.12. Please verify
locally on at least one of these before opening a PR.

## Submitting changes

1. Open an issue first if the change is substantive (more than a
   typo fix or a docstring tweak). This avoids parallel work and lets
   the design conversation happen before code lands.
2. Branch from `main`. Use a short descriptive branch name
   (`fix/...`, `feat/...`, `docs/...`).
3. Add or update tests for any code change. The full suite must pass
   (`pytest`). If you are adding a new public surface, add a test that
   would fail if that surface regressed.
4. Run `ruff check` and `black --check src tests` before opening the
   PR. The codebase aims for 100-character lines.
5. Open a PR against `main`. Describe both what changed and why; link
   the issue if there is one.

## Scope guidance

The compiler is a deliberately narrow piece of research software with
a specific philosophical stake (see `README.md` and the spec at
`ErisML-Compiler.md`). PRs that broaden the IR or add new extractor
tiers are welcome; PRs that *change the IR's semantic commitments*
need a design discussion in an issue first, because the IR is the
project's contract with downstream tooling (DEME, RLEF, the I-EIP
Monitor, the silicon emit). See `SCOPE.md` for what is in / stubbed /
deferred.

## Code of conduct

Be kind, be precise, and assume good faith. Disagreement is welcome;
disrespect is not. The maintainer reserves the right to close issues
or PRs that are abusive or off-topic.
