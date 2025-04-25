#!/usr/bin/env python
"""
Release script for recall-kit.

This script:
1. Bumps the version in pyproject.toml and __init__.py
2. Commits the changes
3. Creates and pushes a new tag to GitHub

Usage:
    python scripts/release.py [major|minor|patch]
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import toml


def get_current_version():
    """Get the current version from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        sys.exit("Error: pyproject.toml not found")

    data = toml.load(pyproject_path)
    return data["project"]["version"]


def bump_version(current_version, bump_type):
    """Bump the version according to semver rules."""
    major, minor, patch = map(int, current_version.split("."))

    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump_type == "minor":
        minor += 1
        patch = 0
    elif bump_type == "patch":
        patch += 1
    else:
        sys.exit(f"Error: Invalid bump type: {bump_type}")

    return f"{major}.{minor}.{patch}"


def update_pyproject_toml(new_version):
    """Update the version in pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    data = toml.load(pyproject_path)
    data["project"]["version"] = new_version

    with open(pyproject_path, "w") as f:
        toml.dump(data, f)

    print(f"Updated pyproject.toml version to {new_version}")


def update_init_py(new_version):
    """Update the version in __init__.py."""
    init_path = Path("recall_kit/__init__.py")
    if not init_path.exists():
        sys.exit("Error: recall_kit/__init__.py not found")

    with open(init_path, "r") as f:
        content = f.read()

    # Replace the version string
    new_content = re.sub(
        r'__version__\s*=\s*"[^"]+"',
        f'__version__ = "{new_version}"',
        content
    )

    with open(init_path, "w") as f:
        f.write(new_content)

    print(f"Updated __init__.py version to {new_version}")


def git_commit_and_tag(new_version):
    """Commit version changes and create a new tag."""
    # Check if git is available
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        sys.exit("Error: git command not found or not working")

    # Check if we're in a git repository
    try:
        subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                      check=True, capture_output=True)
    except subprocess.CalledProcessError:
        sys.exit("Error: Not in a git repository")

    # Check for uncommitted changes
    result = subprocess.run(["git", "status", "--porcelain"],
                           capture_output=True, text=True)
    if result.stdout.strip() and not all(line.startswith("M pyproject.toml") or
                                        line.startswith("M recall_kit/__init__.py")
                                        for line in result.stdout.strip().split("\n")):
        sys.exit("Error: There are uncommitted changes. Please commit or stash them first.")

    # Commit the version changes
    tag_name = f"v{new_version}"
    commit_message = f"Bump version to {new_version}"

    try:
        # Add the files
        subprocess.run(["git", "add", "pyproject.toml", "recall_kit/__init__.py"], check=True)

        # Commit
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        print(f"Committed changes: {commit_message}")

        # Create tag
        subprocess.run(["git", "tag", "-a", tag_name, "-m", f"Release {tag_name}"], check=True)
        print(f"Created tag: {tag_name}")

        # Push changes and tag
        subprocess.run(["git", "push", "origin", "main"], check=True)
        subprocess.run(["git", "push", "origin", tag_name], check=True)
        print(f"Pushed changes and tag to origin")

        print(f"\nSuccessfully released version {new_version}!")
        print(f"GitHub Actions workflow should now be triggered to publish to PyPI.")

    except subprocess.CalledProcessError as e:
        sys.exit(f"Error during git operations: {e}")


def main():
    parser = argparse.ArgumentParser(description="Bump version and create a release")
    parser.add_argument("bump_type", choices=["major", "minor", "patch"],
                        help="The type of version bump to perform")
    args = parser.parse_args()

    # Get current version
    current_version = get_current_version()
    print(f"Current version: {current_version}")

    # Bump version
    new_version = bump_version(current_version, args.bump_type)
    print(f"New version: {new_version}")

    # Confirm with user
    response = input(f"Bump version from {current_version} to {new_version}? [y/N] ")
    if response.lower() != "y":
        sys.exit("Aborted.")

    # Update files
    update_pyproject_toml(new_version)
    update_init_py(new_version)

    # Git operations
    response = input("Commit changes and create tag? [y/N] ")
    if response.lower() != "y":
        sys.exit("Version updated but not committed or tagged.")

    git_commit_and_tag(new_version)


if __name__ == "__main__":
    main()
