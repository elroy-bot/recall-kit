"""
Data models for Recall Kit.

This module contains the core data models used throughout Recall Kit,
including Memory and MemorySource classes.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from .constants import ASSISTANT, SYSTEM, TOOL, USER


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

    def to_memory(self) -> Memory:
        """
        Convert this source to a Memory object.

        Args:
            user_id: ID of the user who owns this memory (defaults to 1)
        """
        return Memory(
            text=self.text,
            title=self.title,
            source_address=self.address,
            metadata=self.metadata,
            user_id=self.user_id,
        )


class Memory(BaseModel):
    """
    A memory that can be stored, retrieved, and used by an LLM.

    Memories are created from MemorySource objects and can be consolidated
    with other memories to form higher-level memories.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the memory",
    )
    text: str = Field(..., description="The text content of the memory")
    title: str = Field(..., description="A title or brief description of the memory")
    embedding: Optional[List[float]] = Field(
        None, description="Vector embedding of the memory text"
    )
    source_address: Optional[str] = Field(
        None, description="Address of the source that created this memory"
    )
    parent_ids: List[str] = Field(
        default_factory=list,
        description="IDs of parent memories if this is a consolidated memory",
    )
    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now, description="When the memory was created"
    )
    relevance: float = Field(
        default=0.0, description="Relevance score (set during retrieval)"
    )
    active: bool = Field(
        default=True, description="Whether this memory is active (available for search)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the memory"
    )
    user_id: int = Field(..., description="ID of the user who owns this memory")

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        """Validate that the embedding is a list of floats."""
        if v is None:
            return v
        if not isinstance(v, list):
            raise ValueError("Embedding must be a list of floats")
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Embedding must contain only numbers")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert the memory to a dictionary."""
        return self.model_dump(exclude={"embedding"})


class Message(BaseModel):
    """
    A message in a conversation.

    This represents a single message in a conversation, which can be from a user,
    assistant, or system.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the message",
    )
    role: str = Field(
        ...,
        description="The role of the message sender (system, user, assistant, tool)",
    )
    content: str = Field(..., description="The content of the message")
    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="When the message was created",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the message"
    )
    user_id: int = Field(..., description="ID of the user who owns this message")
    tool_call_id: Optional[str] = Field(
        None,
        description="ID of the tool call this message is responding to (for tool messages)",
    )
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        None, description="Tool calls made by this message (for assistant messages)"
    )

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Validate that the role is one of the allowed values."""
        allowed_roles = {SYSTEM, USER, ASSISTANT, TOOL}
        if v not in allowed_roles:
            raise ValueError(f"Role must be one of {allowed_roles}")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert the message to a dictionary."""
        result = self.model_dump()
        # Remove None values for tool_call_id and tool_calls
        if result.get("tool_call_id") is None:
            result.pop("tool_call_id", None)
        if result.get("tool_calls") is None:
            result.pop("tool_calls", None)
        return result


class MessageSet(BaseModel):
    """
    A set of messages in a conversation.

    This represents a collection of messages in a conversation, which can be
    used to track the history of a conversation.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the message set",
    )
    message_ids: List[str] = Field(
        default_factory=list, description="IDs of messages in this set"
    )
    active: bool = Field(default=True, description="Whether this message set is active")
    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="When the message set was created",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the message set"
    )
    user_id: int = Field(..., description="ID of the user who owns this message set")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the message set to a dictionary."""
        return self.model_dump()
