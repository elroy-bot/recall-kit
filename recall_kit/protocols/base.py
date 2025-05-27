"""
Protocol definitions for Recall Kit.

This module contains protocol definitions for various functions used in Recall Kit,
including retrieval, filtering, reranking, augmentation, embedding, completion,
and storage backend protocols.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Unpack, runtime_checkable

from litellm import ModelResponse  # type: ignore
from litellm import AllMessageValues, ChatCompletionRequest  # type: ignore

from recall_kit.models import Embedding, Memory, MessageSet


# Define type protocols for the callback functions
@runtime_checkable
class RetrieveFunction(Protocol):
    def __call__(
        self,
        storage: StorageBackendProtocol,
        embedding_model: str,
        embedding_fn: EmbeddingFunction,
        request: ChatCompletionRequest,
    ) -> List[Memory]:
        ...


@runtime_checkable
class FilterFunction(Protocol):
    def __call__(
        self,
        request: Optional[ChatCompletionRequest],
        memories: List[Memory],
    ) -> List[Memory]:
        ...


@runtime_checkable
class RerankFunction(Protocol):
    def __call__(
        self,
        request: Optional[ChatCompletionRequest],
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
    def __call__(self, model: str, text: str) -> List[float]:
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

    def search_memories(
        self, query_embedding: List[float], limit: int = 5
    ) -> List[Memory]:
        ...

    def fetch_embedding(
        self,
        model: str,
        source_type: str,
        source_id: int,
    ) -> Optional[Embedding]:
        ...

    def get_active_memories(self) -> List[Memory]:
        ...

    def store_embedding(
        self, model: str, source_type: str, source_id: int, embedding: List[float]
    ) -> None:
        ...

    def update_memory(self, memory: Memory) -> None:
        ...

    def delete_memory(self, memory_id: int) -> bool:
        ...

    def store_message(self, message: AllMessageValues) -> int:
        ...

    def store_messages(self, messages: List[AllMessageValues]) -> List[int]:
        ...

    def get_message(self, message_id: int) -> Optional[AllMessageValues]:
        ...

    def store_message_set(self, message_set: MessageSet) -> int:
        ...

    def get_message_set(self, message_set_id: int) -> Optional[MessageSet]:
        ...

    def get_active_message_set(self) -> Optional[MessageSet]:
        ...

    def get_messages_in_set(self, message_set_id: int) -> List[AllMessageValues]:
        ...

    def deactivate_all_message_sets(self) -> None:
        ...

    def create_user(self, token: str) -> int:
        ...

    def get_user_by_token(self, token: Optional[str]) -> Optional[int]:
        ...

    def get_default_user_id(self) -> int:
        ...
