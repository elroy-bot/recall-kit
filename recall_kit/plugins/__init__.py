"""
Plugin system for Recall Kit.

This module provides a plugin system for extending Recall Kit's functionality,
including hooks for registering custom components.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Type

from ..protocols.base import (
    AugmentFunction,
    CompletionFunction,
    EmbeddingFunction,
    FilterFunction,
    RerankFunction,
    RetrieveFunction,
    StorageBackendProtocol,
)

# Import from discovery module
from .discovery import discover_plugins

# Import from registry module
from .registry import _check_protocol_conformance, registry

# Import from default module


# Re-export convenience functions that use the global registry


def register_embedding_fn(
    fn: EmbeddingFunction, name: str, aliases: Optional[List[str]] = None
) -> None:
    """
    Register an embedding service.

    Args:
        fn: The embedding service class to register
        name: The name of the embedding service
        aliases: Optional list of aliases for the service
    """
    registry.register_embedding_fn(fn, name, aliases)


def register_storage_backend(
    backend_class: Type[StorageBackendProtocol],
    name: str,
    aliases: Optional[List[str]] = None,
) -> None:
    """
    Register a storage backend.

    Args:
        backend_class: The storage backend class to register
        name: The name of the storage backend
        aliases: Optional list of aliases for the backend
    """
    registry.register_storage_backend(backend_class, name, aliases)


def get_embedding_fn(name: str) -> Optional[EmbeddingFunction]:
    """
    Get an embedding function by name.

    Args:
        name: The name of the embedding function

    Returns:
        The embedding function, or None if not found
    """
    return registry.get_embedding_fn(name)


def get_storage_backend(name: str) -> StorageBackendProtocol:
    """
    Get a storage backend by name.

    Args:
        name: The name of the storage backend

    Returns:
        The storage backend class, or None if not found
    """
    return registry.get_storage_backend(name)


def register_completion_fn(
    completion_fn: CompletionFunction, name: str, aliases: Optional[List[str]] = None
) -> None:
    """
    Register a completion function.

    Args:
        completion_fn: The completion function to register
        name: The name of the completion function
        aliases: Optional list of aliases for the function

    Raises:
        TypeError: If completion_fn does not conform to CompletionFunction protocol
    """
    # Check protocol conformance before registering
    _check_protocol_conformance(completion_fn, CompletionFunction)
    registry.register_completion_fn(completion_fn, name, aliases)


def get_completion_fn(name: str) -> Optional[CompletionFunction]:
    """
    Get a completion function by name.

    Args:
        name: The name of the completion function

    Returns:
        The completion function, or None if not found
    """
    return registry.get_completion_fn(name)


def register_retrieve_fn(
    retrieve_fn: RetrieveFunction, name: str, aliases: Optional[List[str]] = None
) -> None:
    """
    Register a retrieve function.

    Args:
        retrieve_fn: The retrieve function to register
        name: The name of the retrieve function
        aliases: Optional list of aliases for the function

    Raises:
        TypeError: If retrieve_fn does not conform to RetrieveFunction protocol
    """
    # Check protocol conformance before registering
    _check_protocol_conformance(retrieve_fn, RetrieveFunction)
    registry.register_retrieve_fn(retrieve_fn, name, aliases)


def get_filter_fn(name: str) -> FilterFunction:
    """
    Get a filter function by name.

    Args:
        name: The name of the filter function

    Returns:
        The filter function, or None if not found
    """
    return registry.get_filter_fn(name)


def register_rerank_fn(
    rerank_fn: RerankFunction, name: str, aliases: Optional[List[str]] = None
) -> None:
    """
    Register a rerank function.

    Args:
        rerank_fn: The rerank function to register
        name: The name of the rerank function
        aliases: Optional list of aliases for the function

    Raises:
        TypeError: If rerank_fn does not conform to RerankFunction protocol
    """
    # Check protocol conformance before registering
    _check_protocol_conformance(rerank_fn, RerankFunction)
    registry.register_rerank_fn(rerank_fn, name, aliases)


def get_rerank_fn(name: str) -> RerankFunction:
    """
    Get a rerank function by name.

    Args:
        name: The name of the rerank function

    Returns:
        The rerank function, or None if not found
    """
    return registry.get_rerank_fn(name)


def register_augment_fn(
    augment_fn: AugmentFunction, name: str, aliases: Optional[List[str]] = None
) -> None:
    """
    Register an augment function.

    Args:
        augment_fn: The augment function to register
        name: The name of the augment function
        aliases: Optional list of aliases for the function

    Raises:
        TypeError: If augment_fn does not conform to AugmentFunction protocol
    """
    # Check protocol conformance before registering
    _check_protocol_conformance(augment_fn, AugmentFunction)
    registry.register_augment_fn(augment_fn, name, aliases)


def get_augment_fn(name: str) -> AugmentFunction:
    """
    Get an augment function by name.

    Args:
        name: The name of the augment function

    Returns:
        The augment function, or None if not found
    """
    return registry.get_augment_fn(name)


def call_hooks(hook_name: str, *args, **kwargs) -> List[Any]:
    """
    Call all hook functions for a hook name.

    Args:
        hook_name: The name of the hook
        *args: Positional arguments to pass to the hook functions
        **kwargs: Keyword arguments to pass to the hook functions

    Returns:
        List of results from the hook functions
    """
    return registry.call_hooks(hook_name, *args, **kwargs)


# Initialize the plugin system
discover_plugins()
