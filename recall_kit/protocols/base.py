"""
Protocol definitions for Recall Kit.

This module contains protocol definitions for various functions used in Recall Kit,
including retrieval, filtering, reranking, augmentation, embedding, completion,
and storage backend protocols.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Union, Unpack, runtime_checkable

from litellm import ChatCompletionRequest, ModelResponse, Type  # type: ignore
from pydantic import BaseModel

from ..models.memory import Memory
from ..models.message import Message, MessageSet


# Define type protocols for the callback functions
@runtime_checkable
class RetrieveFunction(Protocol):
    def __call__(
        self,
        storage: StorageBackendProtocol,
        embedding_fn: EmbeddingFunction,
        request: ChatCompletionRequest,
    ) -> List[Memory]:
        ...


@runtime_checkable
class FilterFunction(Protocol):
    def __call__(
        self,
        request: ChatCompletionRequest,
        memories: List[Memory],
    ) -> List[Memory]:
        ...


@runtime_checkable
class RerankFunction(Protocol):
    def __call__(
        self,
        request: ChatCompletionRequest,
        memories: List[Memory],
    ) -> List[Memory]:
        ...


@runtime_checkable
class AugmentFunction(Protocol):
    def __call__(
        self,
        request: ChatCompletionRequest,
        memories: List[Memory],
    ) -> ChatCompletionRequest:
        ...


@runtime_checkable
class EmbeddingFunction(Protocol):
    def __call__(self, text: str) -> List[float]:
        ...


@runtime_checkable
class CompletionFunction(Protocol):
    # Reflect the following params, supporting

    # model: Required[str]
    # messages: Required[List[AllMessageValues]]
    # frequency_penalty: float
    # logit_bias: dict
    # logprobs: bool
    # top_logprobs: int
    # max_tokens: int
    # n: int
    # presence_penalty: float
    # response_format: dict
    # seed: int
    # service_tier: str
    # stop: Union[str, List[str]]
    # stream_options: dict
    # temperature: float
    # top_p: float
    # tools: List[ChatCompletionToolParam]
    # tool_choice: ChatCompletionToolChoiceValues
    # parallel_tool_calls: bool
    # function_call: Union[str, dict]
    # functions: List
    # user: str
    # metadata: dict  # litellm specific param

    def __call__(
        self,
        **request: Unpack[ChatCompletionRequest],
    ) -> ModelResponse:
        ...


@runtime_checkable
class StorageBackendProtocol(Protocol):
    def store_memory(self, memory: Memory) -> None:
        ...

    def get_memory(self, memory_id: int) -> Optional[Memory]:
        ...

    def get_all_memories(self) -> List[Memory]:
        ...

    def search_memories(
        self, query_embedding: List[float], limit: int = 5
    ) -> List[Memory]:
        ...

    def update_memory(self, memory: Memory) -> None:
        ...

    def delete_memory(self, memory_id: int) -> bool:
        ...

    def store_message(self, message: Message) -> Message:
        ...

    def get_message(self, message_id: int) -> Optional[Message]:
        ...

    def get_all_messages(self) -> List[Message]:
        ...

    def store_message_set(self, message_set: MessageSet) -> None:
        ...

    def get_message_set(self, message_set_id: str) -> Optional[MessageSet]:
        ...

    def get_active_message_set(self) -> Optional[MessageSet]:
        ...

    def get_messages_in_set(self, message_set_id: str) -> List[Message]:
        ...

    def deactivate_all_message_sets(self) -> None:
        ...

    def get_all_message_sets(self) -> List[MessageSet]:
        ...

    def create_user(self, token: str) -> int:
        ...

    def get_user_by_token(self, token: str) -> Optional[int]:
        ...

    def get_default_user_id(self) -> int:
        ...
