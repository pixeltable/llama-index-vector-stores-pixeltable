# Contributing

## Development

```bash
git clone https://github.com/pixeltable/llama-index-vector-stores-pixeltable.git
cd llama-index-vector-stores-pixeltable
pip install -e ".[dev]"
pytest tests/ -v
ruff check . && ruff format --check .
```

## Releasing

Releases are published to PyPI automatically when a GitHub Release is created.

### One-time setup (trusted publishing)

1. Add a trusted publisher on [PyPI](https://pypi.org/manage/project/llama-index-vector-stores-pixeltable/settings/publishing/) (owner/repo/workflow/environment)
2. Create a GitHub environment called `pypi`

### Publishing a release

1. Bump version in `llama_index/vector_stores/pixeltable/__init__.py` and `pyproject.toml`
2. Commit and push to `main`
3. Create a GitHub Release with a `v*` tag (e.g. `v0.1.2`)
