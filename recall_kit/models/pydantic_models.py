"""
Memory-related data models for Recall Kit.

"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SourceMetadata(BaseModel):
    source_type: str = Field(
        ...,
        description="The type of source (e.g., 'chat', 'document', etc.)",
    )
    source_id: int = Field(
        ...,
        description="A unique identifier for the source, such as a conversation ID or document ID",
    )
