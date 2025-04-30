"""
Plugin system for Recall Kit.

This module provides a plugin system for extending Recall Kit's functionality,
including hooks for registering custom components.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from types import ModuleType
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Protocol,
    Type,
    TypeVar,
    get_type_hints,
    runtime_checkable,
)

from .core import (
    AugmentFunction,
    CompletionFunction,
    EmbeddingFunction,
    FilterFunction,
    RerankFunction,
    RetrieveFunction,
    StorageBackendProtocol,
)


# Utility functions for signature checking
def check_signature_compatibility(
    fn: Callable, protocol_class: Type[Protocol]
) -> tuple[bool, str]:
    """
    Check if a function's signature is compatible with a protocol.

    Args:
        fn: The function to check
        protocol_class: The protocol class to check against

    Returns:
        A tuple of (is_compatible, error_message)
    """
    try:
        # Get normalized type hints for both the function and the protocol method
        protocol_hints = get_type_hints(protocol_class.__call__)
        function_hints = get_type_hints(fn)

        # Remove 'self' from protocol hints if present
        if "self" in protocol_hints:
            protocol_hints.pop("self")

        # Check if all required protocol parameters exist in the function
        for param_name, param_type in protocol_hints.items():
            if param_name == "return":
                continue
            if param_name not in function_hints:
                return False, f"Function missing required parameter: {param_name}"
            if function_hints[param_name] != param_type:
                return (
                    False,
                    f"Parameter '{param_name}' has type {function_hints[param_name]}, but protocol expects {param_type}",
                )

        # Check return type
        if "return" in protocol_hints and "return" in function_hints:
            if protocol_hints["return"] != function_hints["return"]:
                return (
                    False,
                    f"Return type {function_hints['return']} is not compatible with protocol return type {protocol_hints['return']}",
                )

        return True, ""
    except Exception as e:
        return False, f"Error checking signature compatibility: {str(e)}"


T = TypeVar("T")
# Define protocol classes for functions


@runtime_checkable
class MemoryProcessorProtocol(Protocol):
    def process(self, memory: Any) -> Any:
        ...


def _check_protocol_conformance(
    obj: Any, protocol_class: Type[Protocol], obj_name: str = None
) -> None:
    """
    Check if an object conforms to a protocol.

    Args:
        obj: The object to check
        protocol_class: The protocol class to check against
        obj_name: Optional name to use in error messages

    Raises:
        TypeError: If the object does not conform to the protocol
    """
    name = obj_name or getattr(obj, "__name__", str(obj))

    # Always perform a rigorous signature check
    is_compatible, error_msg = check_signature_compatibility(obj, protocol_class)
    if not is_compatible:
        raise TypeError(
            f"Function {name} does not conform to {protocol_class.__name__} protocol: {error_msg}"
        )

    # Also check with isinstance for runtime protocol conformance
    if not isinstance(obj, protocol_class):
        raise TypeError(
            f"Function {name} does not conform to {protocol_class.__name__} protocol at runtime"
        )


class PluginRegistry:
    """Registry for Recall Kit plugins."""

    def __init__(self):
        """Initialize a new plugin registry."""
        self._embedding_fns = {}
        self._storage_backends = {}
        self._embedding_fns = {}
        self._completion_fns = {}
        self._retrieve_fns = {}
        self._filter_fns = {}
        self._rerank_fns = {}
        self._augment_fns = {}
        self._hooks = {}

    def register_embedding_fn(
        self, fn: EmbeddingFunction, name: str, aliases: Optional[List[str]] = None
    ) -> None:
        """
        Register an embedding service.

        Args:
            fn: The embedding service class to register
            name: The name of the embedding service
            aliases: Optional list of aliases for the service

        Raises:
            TypeError: If fn does not conform to EmbeddingFunction protocol
        """
        _check_protocol_conformance(fn, EmbeddingFunction)

        self._embedding_fns[name] = fn
        if aliases:
            for alias in aliases:
                self._embedding_fns[alias] = fn

    def get_embedding_fn(self, name: str) -> Optional[EmbeddingFunction]:
        """
        Get an embedding service by name.

        Args:
            name: The name of the embedding service

        Returns:
            The embedding service class, or None if not found
        """
        return self._embedding_fns.get(name)

    def register_storage_backend(
        self,
        storage: Type[StorageBackendProtocol],
        name: str,
        aliases: Optional[List[str]] = None,
    ) -> None:
        """
        Register a storage backend.

        Args:
            backend_class: The storage backend class to register
            name: The name of the storage backend
            aliases: Optional list of aliases for the backend

        Raises:
            TypeError: If backend_class does not conform to StorageBackendProtocol
        """
        # Check if the class implements the StorageBackendProtocol
        if not isinstance(storage, StorageBackendProtocol):
            raise TypeError(
                f"{storage.__name__} does not conform to StorageBackendProtocol"
            )

        self._storage_backends[name] = storage
        if aliases:
            for alias in aliases:
                self._storage_backends[alias] = storage

    def get_storage_backend(self, name: str) -> Optional[Type[StorageBackendProtocol]]:
        """
        Get a storage backend by name.

        Args:
            name: The name of the storage backend

        Returns:
            The storage backend class, or None if not found
        """
        return self._storage_backends.get(name)

    def register_embedding_fn(
        self,
        embedding_fn: EmbeddingFunction,
        name: str,
        aliases: Optional[List[str]] = None,
    ) -> None:
        """
        Register an embedding function.

        Args:
            embedding_fn: The embedding function to register
            name: The name of the embedding function
            aliases: Optional list of aliases for the function

        Raises:
            TypeError: If embedding_fn does not conform to EmbeddingFunction protocol
        """
        _check_protocol_conformance(embedding_fn, EmbeddingFunction)

        self._embedding_fns[name] = embedding_fn
        if aliases:
            for alias in aliases:
                self._embedding_fns[alias] = embedding_fn

    def get_embedding_fn(self, name: str) -> Optional[EmbeddingFunction]:
        """
        Get an embedding function by name.

        Args:
            name: The name of the embedding function

        Returns:
            The embedding function, or None if not found
        """
        return self._embedding_fns.get(name)

    def register_completion_fn(
        self,
        completion_fn: CompletionFunction,
        name: str,
        aliases: Optional[List[str]] = None,
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
        _check_protocol_conformance(completion_fn, CompletionFunction)

        self._completion_fns[name] = completion_fn
        if aliases:
            for alias in aliases:
                self._completion_fns[alias] = completion_fn

    def get_completion_fn(self, name: str) -> Optional[CompletionFunction]:
        """
        Get a completion function by name.

        Args:
            name: The name of the completion function

        Returns:
            The completion function, or None if not found
        """
        return self._completion_fns.get(name)

    def register_retrieve_fn(
        self,
        retrieve_fn: Callable[[str, Any], List[Any]],
        name: str,
        aliases: Optional[List[str]] = None,
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
        # Check if the function conforms to the RetrieveFunction protocol
        from .core import RetrieveFunction

        _check_protocol_conformance(retrieve_fn, RetrieveFunction)

        self._retrieve_fns[name] = retrieve_fn
        if aliases:
            for alias in aliases:
                self._retrieve_fns[alias] = retrieve_fn

    def get_retrieve_fn(self, name: str) -> Optional[Callable[[str, Any], List[Any]]]:
        """
        Get a retrieve function by name.

        Args:
            name: The name of the retrieve function

        Returns:
            The retrieve function, or None if not found
        """
        return self._retrieve_fns.get(name)

    def register_filter_fn(
        self,
        filter_fn: Callable[[Any, Any], bool],
        name: str,
        aliases: Optional[List[str]] = None,
    ) -> None:
        """
        Register a filter function.

        Args:
            filter_fn: The filter function to register
            name: The name of the filter function
            aliases: Optional list of aliases for the function

        Raises:
            TypeError: If filter_fn does not conform to FilterFunction protocol
        """
        # Check if the function conforms to the FilterFunction protocol
        from .core import FilterFunction

        _check_protocol_conformance(filter_fn, FilterFunction)

        self._filter_fns[name] = filter_fn
        if aliases:
            for alias in aliases:
                self._filter_fns[alias] = filter_fn

    def get_filter_fn(self, name: str) -> Optional[Callable[[Any, Any], bool]]:
        """
        Get a filter function by name.

        Args:
            name: The name of the filter function

        Returns:
            The filter function, or None if not found
        """
        return self._filter_fns.get(name)

    def register_rerank_fn(
        self,
        rerank_fn: Callable[[List[Any], Any], List[Any]],
        name: str,
        aliases: Optional[List[str]] = None,
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
        # Check if the function conforms to the RerankFunction protocol
        from .core import RerankFunction

        _check_protocol_conformance(rerank_fn, RerankFunction)

        self._rerank_fns[name] = rerank_fn
        if aliases:
            for alias in aliases:
                self._rerank_fns[alias] = rerank_fn

    def get_rerank_fn(
        self, name: str
    ) -> Optional[Callable[[List[Any], Any], List[Any]]]:
        """
        Get a rerank function by name.

        Args:
            name: The name of the rerank function

        Returns:
            The rerank function, or None if not found
        """
        return self._rerank_fns.get(name)

    def register_augment_fn(
        self,
        augment_fn: Callable[[List[Any], Any], Any],
        name: str,
        aliases: Optional[List[str]] = None,
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
        # Check if the function conforms to the AugmentFunction protocol
        from .core import AugmentFunction

        _check_protocol_conformance(augment_fn, AugmentFunction)

        self._augment_fns[name] = augment_fn
        if aliases:
            for alias in aliases:
                self._augment_fns[alias] = augment_fn

    def get_augment_fn(self, name: str) -> Optional[Callable[[List[Any], Any], Any]]:
        """
        Get an augment function by name.

        Args:
            name: The name of the augment function

        Returns:
            The augment function, or None if not found
        """
        return self._augment_fns.get(name)

    def register_hook(self, hook_name: str, hook_func: Callable[..., Any]) -> None:
        """
        Register a hook function.

        Args:
            hook_name: The name of the hook
            hook_func: The hook function to register
        """
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        self._hooks[hook_name].append(hook_func)

    def get_hooks(self, hook_name: str) -> List[Callable[..., Any]]:
        """
        Get all hook functions for a hook name.

        Args:
            hook_name: The name of the hook

        Returns:
            List of hook functions
        """
        return self._hooks.get(hook_name, [])

    def call_hooks(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """
        Call all hook functions for a hook name.

        Args:
            hook_name: The name of the hook
            *args: Positional arguments to pass to the hook functions
            **kwargs: Keyword arguments to pass to the hook functions

        Returns:
            List of results from the hook functions
        """
        results = []
        for hook_func in self.get_hooks(hook_name):
            results.append(hook_func(*args, **kwargs))
        return results


# Global plugin registry
registry = PluginRegistry()


def hookimpl(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator for marking a function as a hook implementation.

    Args:
        func: The function to mark as a hook implementation

    Returns:
        The decorated function
    """
    func._is_hookimpl = True
    return func


def discover_plugins(namespace: str = "recall_kit_plugins") -> None:
    """
    Discover and load plugins from a namespace.

    Args:
        namespace: The namespace to search for plugins
    """
    try:
        # Find all modules in the namespace
        plugin_modules = {
            name: importlib.import_module(f"{namespace}.{name}")
            for finder, name, ispkg in pkgutil.iter_modules()
            if name.startswith(namespace)
        }

        # Register hooks from the modules
        for name, module in plugin_modules.items():
            register_module_hooks(module)
    except ImportError:
        # No plugins found
        pass


def register_module_hooks(module: ModuleType) -> None:
    """
    Register hooks from a module.

    Args:
        module: The module to register hooks from
    """
    for name, obj in inspect.getmembers(module):
        # Check if the object is a function and has the _is_hookimpl attribute
        if inspect.isfunction(obj) and getattr(obj, "_is_hookimpl", False):
            # Determine the hook name from the function name
            hook_name = name
            registry.register_hook(hook_name, obj)


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


def register_embedding_fn(
    embedding_fn: EmbeddingFunction, name: str, aliases: Optional[List[str]] = None
) -> None:
    """
    Register an embedding function.

    Args:
        embedding_fn: The embedding function to register
        name: The name of the embedding function
        aliases: Optional list of aliases for the function

    Raises:
        TypeError: If embedding_fn does not conform to EmbeddingFunction protocol
    """
    # Check protocol conformance before registering
    _check_protocol_conformance(embedding_fn, EmbeddingFunction)
    registry.register_embedding_fn(embedding_fn, name, aliases)


def get_embedding_fn(name: str) -> Optional[EmbeddingFunction]:
    """
    Get an embedding function by name.

    Args:
        name: The name of the embedding function

    Returns:
        The embedding function, or None if not found
    """
    return registry.get_embedding_fn(name)


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


def get_embedding_fn(name: str) -> Optional[EmbeddingFunction]:
    """
    Get an embedding service by name.

    Args:
        name: The name of the embedding service

    Returns:
        The embedding service class, or None if not found
    """
    return registry.get_embedding_fn(name)


def get_storage_backend(name: str) -> Optional[Type[StorageBackendProtocol]]:
    """
    Get a storage backend by name.

    Args:
        name: The name of the storage backend

    Returns:
        The storage backend class, or None if not found
    """
    return registry.get_storage_backend(name)


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


def get_retrieve_fn(name: str) -> Optional[Callable[[str, Any], List[Any]]]:
    """
    Get a retrieve function by name.

    Args:
        name: The name of the retrieve function

    Returns:
        The retrieve function, or None if not found
    """
    return registry.get_retrieve_fn(name)


def register_filter_fn(
    filter_fn: FilterFunction, name: str, aliases: Optional[List[str]] = None
) -> None:
    """
    Register a filter function.

    Args:
        filter_fn: The filter function to register
        name: The name of the filter function
        aliases: Optional list of aliases for the function

    Raises:
        TypeError: If filter_fn does not conform to FilterFunction protocol
    """
    # Check protocol conformance before registering
    _check_protocol_conformance(filter_fn, FilterFunction)
    registry.register_filter_fn(filter_fn, name, aliases)


def get_filter_fn(name: str) -> Optional[Callable[[Any, Any], bool]]:
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


def get_rerank_fn(name: str) -> Optional[Callable[[List[Any], Any], List[Any]]]:
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


def get_augment_fn(name: str) -> Optional[Callable[[List[Any], Any], Any]]:
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
