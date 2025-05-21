"""
Memory-related data models for Recall Kit.

This module contains the Memory and MemorySource classes used throughout Recall Kit.
"""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


class MemorySource(BaseModel):
    """
    A source of memory, which can be converted into a Memory object.

    This represents the raw data from which a memory is created, such as
    a chat message, document, or other source.
    """

    text: str = Field(..., description="The raw text content of the memory source")
    title: str = Field(
        ..., description="A title or brief description of the memory source"
    )
    address: str = Field(
        ...,
        description="A unique identifier for the source (e.g., conversation:date:message:id)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the source"
    )
    user_id: int = Field(..., description="ID of the user who owns this source")
