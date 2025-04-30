"""
Tests for the plugin system of Recall Kit.
"""

from typing import Any

import pytest

from recall_kit.core import StorageBackendProtocol
from recall_kit.plugins import (
    PluginRegistry,
    call_hooks,
    get_storage_backend,
    hookimpl,
    register_retrieve_fn,
    register_storage_backend,
)
from recall_kit.storage import SQLiteBackend
from tests.conftest import MockStorageBackend, mock_embed_text


def test_protocol_implementation():
    """Test that SQLiteBackend implements StorageBackendProtocol."""
    # Create an instance of SQLiteBackend
    backend = SQLiteBackend(":memory:")

    # Check if it implements the protocol
    assert isinstance(backend, StorageBackendProtocol)

    # This is possible because we added @runtime_checkable to the Protocol


@pytest.fixture
def plugin_registry():
    """Create a fresh registry for each test."""
    return PluginRegistry()


def test_register_embedding_fn(plugin_registry):
    """Test registering an embedding service."""
    # Create a mock embedding service class

    # Register the service
    plugin_registry.register_embedding_fn(mock_embed_text, "mock")

    # Get the service
    embed_fn = plugin_registry.get_embedding_fn("mock")
    assert embed_fn == mock_embed_text


def test_register_embedding_fn_with_aliases(plugin_registry):
    """Test registering an embedding service with aliases."""
    # Create a mock embedding service class

    # Register the service with aliases
    plugin_registry.register_embedding_fn(
        mock_embed_text, "mock", aliases=["mock1", "mock2"]
    )

    # Get the service by each alias
    embed_fn1 = plugin_registry.get_embedding_fn("mock1")
    embed_fn2 = plugin_registry.get_embedding_fn("mock2")

    # Check that we got the right class
    assert embed_fn1 == mock_embed_text
    assert embed_fn2 == mock_embed_text


def test_register_storage_backend(plugin_registry):
    """Test registering a storage backend."""

    # Register the backend
    plugin_registry.register_storage_backend(MockStorageBackend, "mock")

    # Get the backend
    backend_class = plugin_registry.get_storage_backend("mock")

    # Check that we got the right class
    assert backend_class == MockStorageBackend


def test_register_hook(plugin_registry):
    """Test registering a hook."""

    # Create a mock hook function
    def mock_hook(arg):
        return f"Hooked: {arg}"

    # Register the hook
    plugin_registry.register_hook("test_hook", mock_hook)

    # Get the hooks
    hooks = plugin_registry.get_hooks("test_hook")

    # Check that we got the right hook
    assert len(hooks) == 1
    assert hooks[0] == mock_hook
    assert hooks[0]("test") == "Hooked: test"


def test_call_hooks(plugin_registry):
    """Test calling hooks."""

    # Create mock hook functions
    def mock_hook1(arg):
        return f"Hook1: {arg}"

    def mock_hook2(arg):
        return f"Hook2: {arg}"

    # Register the hooks
    plugin_registry.register_hook("test_hook", mock_hook1)
    plugin_registry.register_hook("test_hook", mock_hook2)

    # Call the hooks
    results = plugin_registry.call_hooks("test_hook", "test")

    # Check the results
    assert len(results) == 2
    assert results[0] == "Hook1: test"
    assert results[1] == "Hook2: test"


def test_hookimpl_decorator():
    """Test the hookimpl decorator."""

    # Create a function with the decorator
    @hookimpl
    def test_hook(arg):
        return f"Decorated: {arg}"

    # Check that the function has the _is_hookimpl attribute
    assert hasattr(test_hook, "_is_hookimpl")
    assert test_hook._is_hookimpl

    # Check that the function still works
    assert test_hook("test") == "Decorated: test"


def test_register_and_get_storage_backend():
    """Test registering and getting a storage backend."""
    # Create a mock storage backend class

    # Register the backend
    register_storage_backend(MockStorageBackend, "mock_global")

    # Get the backend
    backend_class = get_storage_backend("mock_global")

    # Check that we got the right class
    assert backend_class == MockStorageBackend


def test_call_hooks_global():
    """Test calling hooks using the global function."""

    # Create a mock hook function
    def mock_hook(arg):
        return f"Global Hook: {arg}"

    # Register the hook
    from recall_kit.plugins import registry

    registry.register_hook("test_hook_global", mock_hook)

    # Call the hooks
    results = call_hooks("test_hook_global", "test")

    # Check the results
    assert len(results) == 1
    assert results[0] == "Global Hook: test"


def test_non_compliant_retrieve_function():
    """Test that a non-compliant retrieve function raises TypeError."""

    # Create a function that doesn't match the RetrieveFunction protocol
    # RetrieveFunction should take (storage, embedding_fn, request) and return List[Memory]
    def bad_retrieve_fn(wrong_param1: str, wrong_param2: int) -> str:
        """This function has the wrong signature for a RetrieveFunction."""
        return "This returns a string instead of List[Memory]"

    # Attempt to register the non-compliant function
    with pytest.raises(TypeError) as excinfo:
        register_retrieve_fn(bad_retrieve_fn, "bad_retrieve_fn")

    # Check that the error message mentions the specific issues
    error_msg = str(excinfo.value)
    assert "does not conform to RetrieveFunction protocol" in error_msg

    # Create another function with wrong return type
    def bad_return_type(storage: Any, embedding_fn: Any, request: Any) -> str:
        """This function has the right parameters but wrong return type."""
        return "This returns a string instead of List[Memory]"

    # Attempt to register the function with wrong return type
    with pytest.raises(TypeError) as excinfo:
        register_retrieve_fn(bad_return_type, "bad_return_type")

    # Check that the error message mentions return type
    error_msg = str(excinfo.value)
    assert "does not conform to RetrieveFunction protocol" in error_msg
