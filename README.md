# recall-kit

Lightweight memory integration for LLMs

## Installation

You can install the package using uv:

```bash
uv install recall-kit
```

Or using pip:

```bash
pip install recall-kit
```

## Usage

```python
from recall_kit import hello_world

# Default greeting
message = hello_world()
print(message)  # Output: Hello, World! Welcome to recall-kit.

# Custom greeting
message = hello_world("Python")
print(message)  # Output: Hello, Python! Welcome to recall-kit.
```

## Development

### Setup

1. Clone the repository
2. Install development dependencies with uv:
   ```bash
   uv install -e ".[dev]"
   ```

### Running Tests

```bash
uv run pytest
```

## Publishing to PyPI

This package is automatically published to PyPI when a new GitHub release is created.

### Automated Release Process

We provide a release script that automates version bumping and release creation:

```bash
# Install script dependencies
uv pip install -r scripts/requirements.txt

# Create a patch release (0.0.1 -> 0.1.1)
python scripts/release.py patch

# Or for minor/major releases
python scripts/release.py minor
python scripts/release.py major
```

The script will:
1. Update versions in all necessary files
2. Commit the changes
3. Create and push a git tag
4. Trigger the GitHub workflow to publish to PyPI

See `scripts/README.md` for more details.

### Manual Release Process

If you prefer to release manually:

1. Update the version in `pyproject.toml` and `recall_kit/__init__.py`
2. Create a new release on GitHub with a tag matching the version (e.g., `v0.0.1`)
3. The GitHub workflow will automatically build and publish the package to PyPI

## License

MIT
