# How recall-kit Works

This document provides a technical overview of recall-kit's architecture and internal workings.

## Core Architecture

recall-kit is built around a modular architecture with several key components:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Integration    │     │     Core        │     │  Memory Sources │
│  Interfaces     │◄────┤   Pipeline      │◄────┤  (via pluggy)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        ▲                       ▲                       ▲
        │                       │                       │
        └───────────────┬───────┴───────────────┬───────┘
                        │                       │
                ┌───────┴───────┐       ┌───────┴───────┐
                │  Processors   │       │   Utilities   │
                │ (Filter/Rank) │       │ & Helpers     │
                └───────────────┘       └───────────────┘
```

### Component Responsibilities

1. **Integration Interfaces**: Adapters that expose recall-kit functionality through different interfaces (Python API, CLI, MCP, etc.)
2. **Core Pipeline**: Orchestrates the memory retrieval and processing workflow
3. **Memory Sources**: Pluggable backends for storing and retrieving memories
4. **Processors**: Components that transform, filter, and rank memories
5. **Utilities**: Helper functions and shared code used across the library

## Plugin System

recall-kit uses [pluggy](https://pluggy.readthedocs.io/) to implement its extensible memory source system:

```python
# Inside recall-kit core
import pluggy

# Define the hookspec namespace
hookspec = pluggy.HookspecMarker("recall_kit")
hookimpl = pluggy.HookimplMarker("recall_kit")

class MemorySourceSpec:
    """Specifications for memory source plugins."""

    @hookspec
    def store(self, memory, metadata=None):
        """Store a memory with optional metadata."""

    @hookspec
    def retrieve(self, query, limit=10, **params):
        """Retrieve memories based on a query."""

    @hookspec
    def delete(self, memory_id):
        """Delete a memory by ID."""
```

Plugin implementations can then be registered:

```python
# In a plugin implementation
from recall_kit import hookimpl

class VectorStoreMemorySource:
    """Vector store implementation of memory source."""

    @hookimpl
    def store(self, memory, metadata=None):
        # Implementation details...

    @hookimpl
    def retrieve(self, query, limit=10, **params):
        # Implementation details...

    @hookimpl
    def delete(self, memory_id):
        # Implementation details...
```

## Memory Pipeline in Detail

The recall pipeline is the core of recall-kit's functionality. Here's how each stage works:

### 1. Retrieve

The retrieve stage fetches candidate memories from the configured memory source:

```python
def retrieve(query, source, params):
    """Fetch candidate memories from the source."""
    plugin_manager = get_plugin_manager()
    source_plugin = plugin_manager.get_plugin(source)

    # Call the plugin's retrieve method
    candidates = source_plugin.retrieve(
        query=query,
        **params
    )

    return candidates
```

### 2. Filter

The filter stage removes irrelevant or low-quality memories:

```python
def apply_filters(memories, filters):
    """Apply a series of filters to memories."""
    result = memories

    for filter_fn in filters:
        result = filter_fn(result)

    return result
```

Common filters include:
- Relevance threshold filtering
- Time-based decay
- Duplicate removal
- Content-based filtering

### 3. Rerank

The rerank stage orders memories by relevance or importance:

```python
def rerank(memories, ranker, query):
    """Reorder memories using the specified ranker."""
    if ranker is None:
        return memories

    scores = ranker.score(memories, query)

    # Combine memories with scores and sort
    ranked = sorted(
        zip(memories, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # Return just the memories in ranked order
    return [memory for memory, _ in ranked]
```

### 4. Reflect

The reflect stage generates meta-information or summaries about retrieved memories:

```python
def reflect(memories, reflector, query):
    """Generate reflections on the retrieved memories."""
    if reflector is None:
        return memories, None

    reflection = reflector.process(memories, query)

    return memories, reflection
```

### 5. Store

The store stage optionally stores new derived memories:

```python
def store_reflection(reflection, source, query):
    """Store the reflection as a new memory if enabled."""
    if reflection is None:
        return

    plugin_manager = get_plugin_manager()
    source_plugin = plugin_manager.get_plugin(source)

    # Create a new memory from the reflection
    new_memory = {
        "content": reflection,
        "metadata": {
            "type": "reflection",
            "query": query,
            "timestamp": time.time()
        }
    }

    # Store the new memory
    source_plugin.store(new_memory)
```

## Integration Methods

### Tool-based Integration

The tool-based integration provides a simple Python API:

```python
class MemoryToolkit:
    def __init__(self, config=None):
        self.config = config or default_config()
        self.pipeline = RecallPipeline(self.config)
        self.source = self.config.get("source", "in_memory")

    def store(self, memory, metadata=None):
        """Store a memory."""
        plugin_manager = get_plugin_manager()
        source_plugin = plugin_manager.get_plugin(self.source)
        return source_plugin.store(memory, metadata)

    def recall(self, query, **params):
        """Recall memories related to the query."""
        return self.pipeline.execute(query, **params)
```

### MCP Integration

The MCP integration exposes memory capabilities through the Model Context Protocol:

```python
class MCPMemoryServer:
    def __init__(self, config=None):
        self.config = config or default_config()
        self.toolkit = MemoryToolkit(self.config)

    def start(self, host="localhost", port=8000):
        """Start the MCP server."""
        server = MCPServer(host, port)

        # Register tools
        server.register_tool(
            "store_memory",
            self._store_memory,
            {"memory": str, "metadata": dict}
        )

        server.register_tool(
            "recall_memories",
            self._recall_memories,
            {"query": str, "params": dict}
        )

        # Start the server
        server.start()

    def _store_memory(self, memory, metadata=None):
        """MCP tool implementation for storing memories."""
        return self.toolkit.store(memory, metadata)

    def _recall_memories(self, query, params=None):
        """MCP tool implementation for recalling memories."""
        return self.toolkit.recall(query, **(params or {}))
```

### Model Wrapper Integration

The model wrapper provides an OpenAI-compatible interface:

```python
class MemoryEnabledModel:
    def __init__(self, base_model, memory_config=None):
        self.base_model = base_model
        self.memory_config = memory_config or {}
        self.toolkit = MemoryToolkit(self.memory_config)

    def generate(self, prompt, **params):
        """Generate a response with memory augmentation."""
        # Recall relevant memories
        memories = self.toolkit.recall(prompt)

        # Format memories as context
        context = self._format_memories_as_context(memories)

        # Augment the prompt with memory context
        augmented_prompt = f"{context}\n\n{prompt}"

        # Call the base model with the augmented prompt
        return self._call_base_model(augmented_prompt, **params)

    def _format_memories_as_context(self, memories):
        """Format memories as context for the model."""
        if not memories:
            return ""

        formatted = ["Here are some relevant memories:"]
        for i, memory in enumerate(memories, 1):
            formatted.append(f"{i}. {memory['content']}")

        return "\n".join(formatted)

    def _call_base_model(self, prompt, **params):
        """Call the base model with the given prompt."""
        # Implementation depends on the base model type
        # This is a simplified example
        return some_model_api.generate(prompt, **params)
```

### Command Line Interface

The CLI is implemented using a command-line parser like argparse or click:

```python
def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="recall-kit CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start a chat session")
    chat_parser.add_argument("--model", default="gpt-3.5-turbo", help="Model to use")

    # Config command
    config_parser = subparsers.add_parser("config", help="Configure settings")
    config_parser.add_argument("action", choices=["get", "set"], help="Action to perform")
    config_parser.add_argument("key", help="Configuration key")
    config_parser.add_argument("value", nargs="?", help="Configuration value (for set)")

    # Parse arguments
    args = parser.parse_args()

    # Handle commands
    if args.command == "chat":
        start_chat_session(args.model)
    elif args.command == "config":
        if args.action == "get":
            get_config(args.key)
        elif args.action == "set":
            set_config(args.key, args.value)
```

## Customization Points

recall-kit provides several key customization points:

### 1. Memory Sources

Create custom memory sources by implementing the memory source plugin interface:

```python
from recall_kit import register_memory_source

@register_memory_source("my_database")
class DatabaseMemorySource:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.db = connect_to_db(connection_string)

    def store(self, memory, metadata=None):
        # Store memory in database

    def retrieve(self, query, limit=10, **params):
        # Retrieve memories from database
```

### 2. Custom Filters

Create custom filters to control which memories are included:

```python
from recall_kit.filters import register_filter

@register_filter("sentiment")
class SentimentFilter:
    def __init__(self, min_sentiment=0.0):
        self.min_sentiment = min_sentiment
        self.analyzer = SentimentAnalyzer()

    def __call__(self, memories):
        """Filter memories based on sentiment score."""
        return [
            memory for memory in memories
            if self.analyzer.analyze(memory["content"]) >= self.min_sentiment
        ]
```

### 3. Custom Rankers

Create custom rankers to control memory ordering:

```python
from recall_kit.rankers import register_ranker

@register_ranker("personalization")
class PersonalizationRanker:
    def __init__(self, user_profile):
        self.user_profile = user_profile

    def score(self, memories, query):
        """Score memories based on user profile relevance."""
        scores = []
        for memory in memories:
            relevance = self._calculate_user_relevance(memory, self.user_profile)
            scores.append(relevance)

        return scores
```

### 4. Custom Reflectors

Create custom reflectors to generate meta-information:

```python
from recall_kit.reflectors import register_reflector

@register_reflector("key_points")
class KeyPointsReflector:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model

    def process(self, memories, query):
        """Extract key points from memories."""
        combined = "\n".join(memory["content"] for memory in memories)

        prompt = f"""
        Based on these memories:
        {combined}

        Extract 3-5 key points relevant to: {query}
        """

        # Use an LLM to generate the reflection
        response = some_llm_api.generate(prompt, model=self.model)

        return response
```

## Performance Considerations

recall-kit is designed with performance in mind:

1. **Lazy Loading**: Plugins and components are loaded only when needed
2. **Caching**: Results can be cached at various levels of the pipeline
3. **Batching**: Operations can be batched for efficiency
4. **Async Support**: Key operations support async/await for non-blocking I/O
5. **Streaming**: Large memory sets can be processed as streams

## Security Considerations

When using recall-kit, consider these security aspects:

1. **Memory Isolation**: Ensure memories from different users/contexts don't leak
2. **Data Privacy**: Be mindful of what is stored in memories
3. **Authentication**: Secure MCP and API endpoints
4. **Sanitization**: Validate and sanitize inputs to prevent injection attacks

## Debugging and Monitoring

recall-kit provides tools for debugging and monitoring:

1. **Logging**: Comprehensive logging throughout the pipeline
2. **Tracing**: Optional tracing of memory operations
3. **Metrics**: Performance and usage metrics
4. **Visualization**: Tools to visualize memory relationships and access patterns
