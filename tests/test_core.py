"""
Tests for the core module
"""
import pytest
from recall_kit.core import hello_world


def test_hello_world_default():
    """Test hello_world function with default parameter"""
    result = hello_world()
    assert result == "Hello, World! Welcome to recall-kit."


def test_hello_world_custom():
    """Test hello_world function with custom parameter"""
    result = hello_world("Python")
    assert result == "Hello, Python! Welcome to recall-kit."
