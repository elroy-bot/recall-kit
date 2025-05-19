"""
Plugin discovery mechanisms for Recall Kit.

This module provides utilities for discovering and loading plugins from
specified namespaces.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from types import ModuleType

from .registry import registry


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
