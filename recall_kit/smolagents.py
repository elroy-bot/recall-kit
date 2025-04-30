"""
Integration with smolagents for Recall Kit.

This module provides integration with the smolagents library, including
tools for creating and retrieving memories, and a custom agent memory
implementation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Callable, TypeVar

from recall_kit.plugins import get_embedding_fn

try:
    # Import the necessary classes from smolagents
    from smolagents import Tool, AgentMemory
    from smolagents.agents import MultiStepAgent, ToolCallingAgent

    # Type variable for the Agent class
    A = TypeVar('A', bound='Agent')

    # Define an Agent class that can be used with smolagents
    class Agent:
        """Compatibility wrapper for smolagents agent classes."""

        def __init__(
            self,
            memory: Optional[AgentMemory] = None,
            tools: Optional[List[Tool]] = None,
            llm: Optional[Any] = None,
            **kwargs: Any
        ):
            """Initialize an agent with the appropriate agent class."""
            # Store the memory for our own use
            self.memory = memory

            # Create the agent without passing memory
            self.agent = ToolCallingAgent(
                tools=tools or [],
                model=llm,
                **{k: v for k, v in kwargs.items() if k != 'memory'}
            )

        def run(self, input_text: str) -> str:
            """Run the agent on the input text."""
            # Get memory context if available
            context: str = ""
            if self.memory:
                context = self.memory.get_context(input_text)

            # Add context to the input if available
            if context:
                enhanced_input = f"{context}\n\nUser query: {input_text}"
            else:
                enhanced_input = input_text

            # Run the agent
            response = self.agent.run(enhanced_input)

            # Process the response to store in memory if needed
            if self.memory:
                self.memory.process_response(input_text, response)

            return response

except ImportError:
    raise ImportError(
        "smolagents is not installed. Install it with 'pip install smolagents' "
        "or 'pip install recall-kit[smolagents]'."
    )

from recall_kit import RecallKit, Memory


class RecallMemoryTool(Tool):
    """Tool for retrieving memories from Recall Kit."""

    name = "recall_memory"
    description = "Retrieve memories relevant to a query"

    def __init__(
        self,
        recall_kit: RecallKit,
        max_results: int = 3,
        relevance_threshold: float = 0.7,
        include_sources: bool = False,
    ):
        """
        Initialize the recall memory tool.

        Args:
            recall_kit: The RecallKit instance to use
            max_results: Maximum number of memories to return
            relevance_threshold: Minimum relevance score for memories
            include_sources: Whether to include source documents
        """
        self.recall_kit = recall_kit
        self.max_results = max_results
        self.relevance_threshold = relevance_threshold
        self.include_sources = include_sources

    def run(self, query: str) -> Dict[str, Any]:
        """
        Retrieve memories relevant to a query.

        Args:
            query: The query to search for

        Returns:
            Dictionary with the retrieved memories
        """
        # Search for memories
        memories = self.recall_kit.search(query, limit=self.max_results)

        # Filter by relevance threshold
        memories = [m for m in memories if m.relevance >= self.relevance_threshold]

        # Format the results
        results = []
        for memory in memories:
            memory_dict = {
                "id": memory.id,
                "text": memory.text,
                "title": memory.title,
                "relevance": memory.relevance,
                "created_at": memory.created_at.isoformat(),
            }

            if self.include_sources and memory.source_address:
                memory_dict["source"] = memory.source_address

            results.append(memory_dict)

        return {
            "memories": results,
            "count": len(results),
            "query": query,
        }


class CreateMemoryTool(Tool):
    """Tool for creating memories in Recall Kit."""

    name = "create_memory"
    description = "Create a new memory"

    def __init__(
        self,
        recall_kit: RecallKit,
        auto_consolidate: bool = False,
        importance_threshold: float = 0.0,
    ):
        """
        Initialize the create memory tool.

        Args:
            recall_kit: The RecallKit instance to use
            auto_consolidate: Whether to automatically consolidate memories
            importance_threshold: Minimum importance score for creating memories
        """
        self.recall_kit = recall_kit
        self.auto_consolidate = auto_consolidate
        self.importance_threshold = importance_threshold

    def run(self, text: str, title: Optional[str] = None, importance: float = 0.5) -> Dict[str, Any]:
        """
        Create a new memory.

        Args:
            text: The text content of the memory
            title: A title for the memory (auto-generated if not provided)
            importance: Importance score from 0-1

        Returns:
            Dictionary with the created memory information
        """
        # Check importance threshold
        if importance < self.importance_threshold:
            return {
                "created": False,
                "reason": f"Importance score {importance} is below threshold {self.importance_threshold}",
            }

        # Create the memory
        memory = self.recall_kit.create_memory(
            text=text,
            title=title,
            metadata={"importance": importance},
        )

        # Consolidate memories if enabled
        consolidated = []
        if self.auto_consolidate:
            consolidated = self.recall_kit.consolidate_memories()

        return {
            "created": True,
            "memory_id": memory.id,
            "title": memory.title,
            "consolidated_count": len(consolidated),
        }


class RecallKitAgentMemory(AgentMemory):
    """Custom agent memory implementation using Recall Kit."""

    def __init__(
        self,
        recall_kit: Optional[RecallKit] = None,
        system_prompt: Optional[str] = None,
        auto_consolidate: bool = False,
        memory_prefix: str = "Relevant memories from previous conversations:",
        max_memories: int = 5,
        relevance_threshold: float = 0.7,
        embedding_model: str = "text-embedding-3-small",
    ):
        """
        Initialize the RecallKit agent memory.

        Args:
            recall_kit: The RecallKit instance to use (creates a new one if None)
            system_prompt: The system prompt to use
            auto_consolidate: Whether to automatically consolidate memories
            memory_prefix: Prefix for memory context
            max_memories: Maximum number of memories to include
            relevance_threshold: Minimum relevance score for memories
            embedding_model: Embedding model to use if creating a new RecallKit
        """
        super().__init__(system_prompt=system_prompt or "")

        # Initialize RecallKit if not provided
        if recall_kit is not None:
            self.recall_kit = recall_kit
        else:
            # Get embedding function from registry
            embedding_fn = get_embedding_fn(embedding_model)

            # Create storage backend
            from recall_kit.storage import SQLiteBackend
            storage = SQLiteBackend()

            # Create RecallKit with storage and embedding function
            self.recall_kit = RecallKit(
                storage=storage,
                embedding_fn=embedding_fn
            )

        self.system_prompt = system_prompt
        self.auto_consolidate = auto_consolidate
        self.memory_prefix = memory_prefix
        self.max_memories = max_memories
        self.relevance_threshold = relevance_threshold

        # Configuration options
        self.auto_memory_capture = True
        self.consolidation_interval = 10
        self.memory_decay = 0.0
        self.include_sources = False

        # Internal state
        self._memory_counter = 0

    def add_memory(self, text: str, title: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a memory to the agent's memory.

        Args:
            text: The text content of the memory
            title: A title for the memory (auto-generated if not provided)
            metadata: Additional metadata for the memory

        Returns:
            The ID of the created memory
        """
        memory = self.recall_kit.create_memory(
            text=text,
            title=title,
            metadata=metadata or {},
        )

        # Increment memory counter
        self._memory_counter += 1

        # Check if we should consolidate
        if self.auto_consolidate and self._memory_counter >= self.consolidation_interval:
            self.recall_kit.consolidate_memories()
            self._memory_counter = 0

        return memory.id

    def get_relevant_memories(self, query: str) -> List[Memory]:
        """
        Get memories relevant to a query.

        Args:
            query: The query to search for

        Returns:
            List of relevant Memory objects
        """
        memories = self.recall_kit.search(query, limit=self.max_memories)

        # Apply relevance threshold
        memories = [m for m in memories if m.relevance >= self.relevance_threshold]

        # Apply memory decay if enabled
        if self.memory_decay > 0:
            import datetime

            now = datetime.datetime.now()
            for memory in memories:
                # Calculate age in days
                age_days = (now - memory.created_at).total_seconds() / (24 * 60 * 60)
                # Apply decay factor
                memory.relevance *= (1.0 - self.memory_decay * age_days)

            # Re-sort by decayed relevance
            memories.sort(key=lambda m: m.relevance, reverse=True)

        return memories

    def get_context(self, query: str) -> str:
        """
        Get memory context for a query.

        Args:
            query: The query to get context for

        Returns:
            Memory context as a string
        """
        memories = self.get_relevant_memories(query)

        if not memories:
            return ""

        # Format memories as context
        memory_texts = []
        for i, memory in enumerate(memories, 1):
            text = f"{i}. {memory.text}"
            if self.include_sources and memory.source_address:
                text += f" (Source: {memory.source_address})"
            memory_texts.append(text)

        context = f"{self.memory_prefix}\n" + "\n".join(memory_texts)
        return context

    def update_system_prompt(self, system_prompt: str) -> str:
        """
        Update the system prompt with memory context.

        Args:
            system_prompt: The original system prompt

        Returns:
            The updated system prompt with memory context
        """
        self.system_prompt = system_prompt
        return system_prompt

    def process_response(self, query: str, response: str) -> None:
        """
        Process a response to capture memories.

        Args:
            query: The user query
            response: The agent's response
        """
        if not self.auto_memory_capture:
            return

        # Simple approach: create a memory from the interaction
        memory_text = f"User: {query}\nAssistant: {response}"
        title = query[:50] + "..." if len(query) > 50 else query

        self.add_memory(
            text=memory_text,
            title=title,
            metadata={"type": "conversation"},
        )

    def configure(self, **kwargs: Any) -> None:
        """
        Configure the agent memory.

        Args:
            **kwargs: Configuration options
        """
        # Update configuration options
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
