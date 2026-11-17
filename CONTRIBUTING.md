# Contributing

One change per pull request. Run the unit tests and the linter before opening it:

```sh
uv sync --extra cpu --group dev
uv run pytest
uv run ruff check
```

The models are not in the repository. A change that needs a page to test with should name the
source and its licence in the pull request, and not commit the page.
