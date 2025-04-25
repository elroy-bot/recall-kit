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

To publish a new version:

1. Update the version in `pyproject.toml`
2. Create a new release on GitHub with a tag matching the version (e.g., `v0.1.0`)
3. The GitHub workflow will automatically build and publish the package to PyPI using uv

## License

MIT
