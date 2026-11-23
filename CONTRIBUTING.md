# Contributing to delbina

Thanks for helping. Small, focused pull requests are easiest to review.

## Setup

```bash
git clone https://github.com/delbina/delbina.git
cd delbina
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Guidelines

- Keep the core dependency-free apart from `tiktoken`. Integrations with other libraries go in docs or examples, or behind optional extras.
- Add tests for new behaviour. Tests must not need network or a model: pass `count_tokens=lambda s: len(s.split())` and a fake summarizer.
- Public functions get type hints and a docstring.
- Update `CHANGELOG.md` under the unreleased version.

## Good first contributions

- More `delbina eval` scenarios (multi-fact questions, code-editing tasks) and published results for popular models.
- New strategies (importance scoring, embedding-based retrieval of old turns).
- Summarizers for other local runtimes.
- Better token counting for non-OpenAI models.

## Releasing (maintainers)

1. Bump `__version__` in `src/delbina/__init__.py` and date the version in `CHANGELOG.md`.
2. Commit and push; wait for the `test` workflow to pass.
3. Publish a GitHub release tagged `vX.Y.Z` (no dot after the `v`). The `publish` workflow uploads to PyPI via trusted publishing after approval.
