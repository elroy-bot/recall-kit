"""
Protocol definitions for Recall Kit.

This module contains protocol definitions for various functions used in Recall Kit,
including retrieval, filtering, reranking, augmentation, embedding, completion,
and storage backend protocols.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable

from litellm import ModelResponse, Type
from pydantic import BaseModel


# Define type protocols for the callback functions
@runtime_checkable
class RetrieveFunction(Protocol):
    def __call__(
        self,
        storage: StorageBackendProtocol,
        embedding_fn: EmbeddingFunction,
        request: Any,
    ) -> List[Any]:
        ...


@runtime_checkable
class FilterFunction(Protocol):
    def __call__(self, memories: List[Any], request: Any) -> bool:
        ...


@runtime_checkable
class RerankFunction(Protocol):
    def __call__(self, memories: List[Any], request: Any) -> List[Any]:
        ...


@runtime_checkable
class AugmentFunction(Protocol):
    def __call__(self, memories: List[Any], request: Any) -> Any:
        ...


@runtime_checkable
class EmbeddingFunction(Protocol):
    def __call__(self, text: str) -> List[float]:
        ...


@runtime_checkable
class CompletionFunction(Protocol):
    def __call__(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        response_format: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None,
        additional_args: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        ...


@runtime_checkable
class StorageBackendProtocol(Protocol):
    def store_memory(self, memory: Any) -> None:
        ...

    def get_memory(self, memory_id: str) -> Optional[Any]:
        ...

    def get_all_memories(self) -> List[Any]:
        ...

    def search_memories(
        self, query_embedding: List[float], limit: int = 5
    ) -> List[Any]:
        ...

    def update_memory(self, memory: Any) -> None:
        ...

    def delete_memory(self, memory_id: str) -> bool:
        ...

    def store_message(self, message: Any) -> None:
        ...

    def get_message(self, message_id: str) -> Optional[Any]:
        ...

    def get_all_messages(self) -> List[Any]:
        ...

    def store_message_set(self, message_set: Any) -> None:
        ...

    def get_message_set(self, message_set_id: str) -> Optional[Any]:
        ...

    def get_active_message_set(self) -> Optional[Any]:
        ...

    def get_messages_in_set(self, message_set_id: str) -> List[Any]:
        ...

    def deactivate_all_message_sets(self) -> None:
        ...

    def get_all_message_sets(self) -> List[Any]:
        ...

    def create_user(self, token: str) -> int:
        ...

    def get_user_by_token(self, token: str) -> Optional[int]:
        ...

    def get_default_user_id(self) -> int:
        ...
