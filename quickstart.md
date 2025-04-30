# Quickstart Guide

This guide will help you get started with recall-kit quickly.

## Installation

```bash
# Using pip
pip install recall-kit

# Using uv
uv install recall-kit
```

## Basic Usage

### 1. Initialize a Memory Manager

```python
from recall_kit import MemoryManager

# Create a memory manager with default settings
memory = MemoryManager()
```

### 2. Store Memories

```python
# Store a simple text memory
memory.store("The user prefers dark mode for all applications")

# Store with metadata
memory.store(
    "User's favorite color is blue",
    metadata={
        "category": "preferences",
        "timestamp": "2025-04-30T10:00:00Z"
    }
)
```

### 3. Retrieve Memories

```python
# Simple retrieval
results = memory.recall("What are the user's preferences?")

# Print the retrieved memories
for result in results:
    print(f"Memory: {result.content}")
    print(f"Relevance: {result.relevance}")
    print(f"Metadata: {result.metadata}")
    print("---")
```

### 4. Configure Memory Settings

```python
# Create a memory manager with custom settings
memory = MemoryManager(
    source="vector_store",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    pipeline_config={
        "max_memories": 10,
        "filter_threshold": 0.7
    }
)
```

## Integrating with an LLM

### Using the Model Wrapper

```python
from recall_kit.integrations import MemoryEnabledModel

# Create a memory-enabled model
memory_model = MemoryEnabledModel(
    base_model="your-existing-model",
    memory_config={
        "source": "vector_store",
        "recall_strategy": "semantic_search"
    }
)

# Generate a response with memory context
response = memory_model.generate("What were we talking about earlier?")
print(response)
```

### Using the CLI

```bash
# Start a chat session with memory enabled
recall-chat

# Configure memory settings
recall-config set memory.source vector_store
recall-config set memory.max_items 50
```

## Next Steps

- Check out the [examples](examples.md) for more advanced usage patterns
- Learn about [how recall-kit works](how_it_works.md) under the hood
- Explore different [use cases](use_cases.md) for recall-kit
- Read the full [API documentation](https://recall-kit.readthedocs.io/) for detailed reference
