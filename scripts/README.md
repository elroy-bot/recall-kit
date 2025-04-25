# Development Scripts

This directory contains utility scripts for development and release management.

## Release Script

The `release.py` script automates the process of bumping the version and creating a new release.

### Prerequisites

Install the required dependencies:

```bash
# Using uv
uv pip install -r scripts/requirements.txt

# Or using pip
pip install -r scripts/requirements.txt
```

### Usage

To bump the version and create a new release:

```bash
# For a patch release (0.0.1 -> 0.1.1)
python scripts/release.py patch

# For a minor release (0.0.1 -> 0.2.0)
python scripts/release.py minor

# For a major release (0.0.1 -> 1.0.0)
python scripts/release.py major
```

The script will:

1. Update the version in `pyproject.toml` and `recall_kit/__init__.py`
2. Commit these changes
3. Create a new git tag
4. Push the changes and tag to GitHub
5. Trigger the GitHub Actions workflow to publish to PyPI

### Notes

- The script will check for uncommitted changes and abort if there are any (except for the version files)
- You will be prompted to confirm before making any changes
- Make sure you have push access to the repository
