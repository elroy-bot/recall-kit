# recall-kit Examples

This document provides examples of different ways to integrate and use recall-kit.

## Basic Memory Operations

### Storing and Retrieving Memories

```python
from recall_kit import MemoryManager

# Initialize memory manager
memory = MemoryManager()

# Store memories
memory.store("User mentioned they're planning a trip to Japan in May")
memory.store("User prefers vegetarian food options")
memory.store("User is interested in historical landmarks and museums")

# Retrieve relevant memories
japan_memories = memory.recall("What do we know about the user's trip to Japan?")
food_memories = memory.recall("What are the user's food preferences?")

# Print results
print("Trip information:")
for mem in japan_memories:
    print(f"- {mem.content}")

print("\nFood preferences:")
for mem in food_memories:
    print(f"- {mem.content}")
```

## Integration Examples

### Tool-based Integration

The tool-based integration provides a simple Python API:

```python
from recall_kit import MemoryToolkit

# Initialize the toolkit
memory_tools = MemoryToolkit()

# Store memories
memory_tools.store("User mentioned they prefer dark mode")
memory_tools.store("User is working on a machine learning project")

# Recall memories
relevant_memories = memory_tools.recall("user preferences")
project_memories = memory_tools.recall("what project is the user working on")

# Use the memories in your application
print("User preferences:")
for memory in relevant_memories:
    print(f"- {memory.content}")
```

### MCP Integration

Expose memory capabilities through the Model Context Protocol:

```python
from recall_kit.integrations import MCPMemoryServer

# Create and start an MCP server with memory capabilities
memory_server = MCPMemoryServer()
memory_server.start(host="localhost", port=8000)

# The server now exposes these tools:
# - store_memory: Store a new memory
# - recall_memories: Retrieve relevant memories

# Example client usage (in a different application):
from mcp_client import MCPClient

client = MCPClient("localhost:8000")
client.call_tool("store_memory", {
    "memory": "User mentioned they're working on a presentation",
    "metadata": {"category": "work", "priority": "high"}
})

results = client.call_tool("recall_memories", {
    "query": "What is the user working on?",
    "params": {"limit": 5}
})
```

### Model Wrapper Integration

Wrap your existing LLM with memory capabilities through an OpenAI-compatible endpoint:

```python
from recall_kit.integrations import MemoryEnabledModel

# Create a memory-enabled wrapper around your model
memory_model = MemoryEnabledModel(
    base_model="your-existing-model",
    memory_config={
        "source": "vector_store",
        "recall_strategy": "semantic_search"
    }
)

# Use it like any OpenAI-compatible model
response = memory_model.generate("What were we talking about earlier?")
print(response)

# The model automatically:
# 1. Retrieves relevant memories based on the prompt
# 2. Formats them as context
# 3. Augments the prompt with this context
# 4. Calls the base model with the augmented prompt
```

### Command Line Interface

Use recall-kit directly from the command line:

```bash
# Start a chat session with memory enabled
recall-chat

# Configure memory settings
recall-config set memory.source vector_store
recall-config set memory.max_items 50

# Export memories to a file
recall-export --format json --output memories.json

# Import memories from a file
recall-import --file memories.json
```

### llm Package Integration

Integrate with [Simon Willison's llm package](https://github.com/simonw/llm):

```python
# Install the plugin
pip install recall-kit-llm

# Use in llm
import llm
from recall_kit_llm import RecallPlugin

# The plugin will automatically add memory capabilities to llm
response = llm.prompt("What did we discuss earlier?")

# You can also configure the plugin
llm.plugins.configure("recall", {
    "source": "vector_store",
    "max_memories": 10
})
```

### LiteLLM Integration

Add memory to [LiteLLM](https://github.com/BerriAI/litellm) for multi-model support:

```python
from litellm import completion
from recall_kit.integrations import setup_litellm_memory

# Configure memory for litellm
setup_litellm_memory(
    memory_source="vector_store",
    recall_strategy="semantic_search"
)

# Use litellm with memory capabilities
response = completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "What were we discussing earlier?"}]
)

# You can also use it with different models
response_claude = completion(
    model="anthropic.claude-3-opus",
    messages=[{"role": "user", "content": "Can you summarize our conversation?"}]
)
```

## Advanced Examples

### Custom Memory Source

Create a custom memory source:

```python
from recall_kit import register_memory_source

@register_memory_source("my_database")
class DatabaseMemorySource:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.db = connect_to_db(connection_string)

    def store(self, memory, metadata=None):
        """Store memory in database"""
        metadata = metadata or {}
        self.db.execute(
            "INSERT INTO memories (content, metadata) VALUES (?, ?)",
            (memory, json.dumps(metadata))
        )

    def retrieve(self, query, limit=10, **params):
        """Retrieve memories from database"""
        # This is a simplified example - in a real implementation,
        # you would use semantic search or other retrieval methods
        results = self.db.execute(
            "SELECT content, metadata FROM memories WHERE content LIKE ? LIMIT ?",
            (f"%{query}%", limit)
        ).fetchall()

        return [
            {
                "content": row[0],
                "metadata": json.loads(row[1]),
                "relevance": calculate_relevance(row[0], query)
            }
            for row in results
        ]

# Use the custom memory source
from recall_kit import MemoryManager

memory = MemoryManager(
    source="my_database",
    source_config={
        "connection_string": "sqlite:///memories.db"
    }
)
```

### Custom Recall Pipeline

Create a custom recall pipeline:

```python
from recall_kit import RecallPipeline
from recall_kit.filters import TimeDecayFilter
from recall_kit.rankers import RelevanceRanker

# Create custom components
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

# Create the pipeline
pipeline = RecallPipeline(
    retriever="semantic_search",
    filters=[
        TimeDecayFilter(half_life_days=7),
        SentimentFilter(min_sentiment=0.2)
    ],
    ranker=RelevanceRanker(),
    reflector="summarize_key_points",
    store_reflections=True
)

# Use the pipeline
memories = pipeline.execute("What does the user like?")
```

### Async Usage

Use recall-kit with async/await:

```python
import asyncio
from recall_kit import AsyncMemoryManager

async def main():
    # Initialize async memory manager
    memory = AsyncMemoryManager()

    # Store memories asynchronously
    await memory.store("User mentioned they're interested in AI research")

    # Retrieve memories asynchronously
    results = await memory.recall("What are the user's interests?")

    # Process results
    for result in results:
        print(f"Memory: {result.content}")
        print(f"Relevance: {result.relevance}")

# Run the async function
asyncio.run(main())
```

### Batch Operations

Process multiple memories efficiently:

```python
from recall_kit import MemoryManager

# Initialize memory manager
memory = MemoryManager()

# Batch store memories
memories = [
    "User mentioned they enjoy hiking on weekends",
    "User has a dog named Max",
    "User is learning to play the guitar",
    "User prefers coffee over tea"
]

memory.batch_store(memories)

# Batch recall with multiple queries
queries = [
    "What are the user's hobbies?",
    "What are the user's preferences?"
]

results = memory.batch_recall(queries)

# Process results
for query, memories in zip(queries, results):
    print(f"Query: {query}")
    for mem in memories:
        print(f"- {mem.content}")
    print()
