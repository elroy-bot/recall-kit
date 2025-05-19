"""
Recall Kit: Lightweight memory integrations for LLMs.

This package provides tools for creating, storing, retrieving, and consolidating
memories for Large Language Models.
"""

import warnings

import litellm

# Upstream warnings from litellm
warnings.filterwarnings(
    "ignore", message="Support for class-based `config` is deprecated"
)
warnings.filterwarnings(
    "ignore", message="There is no current event loop", category=DeprecationWarning
)

from recall_kit.core import RecallKit
from recall_kit.version import __version__

from .plugins import default

litellm.suppress_debug_info = True

__all__ = [
    "RecallKit",
    "__version__",
]
