# recall-kit

A flexible toolbox for adding memory capabilities to Large Language Models (LLMs).

## Overview

recall-kit provides a comprehensive set of tools and interfaces for integrating memory capabilities into LLM applications. Unlike other memory solutions, recall-kit doesn't prescribe a single approach to memory management. Instead, it offers a flexible framework that allows developers to implement memory in ways that best suit their specific use cases.

## Key Features

- **Flexibility First**: Choose the memory integration approach that works best for your application
- **Multiple Integration Options**: Add memories via tools, Model Context Protocol (MCP), or an OpenAI-compatible model wrapper
- **Pluggable Memory Sources**: Use the built-in plugin system to specify any memory source you need
- **Customizable Recall Pipeline**: Define your own pipeline or use the defaults for memory retrieval and processing
- **Extensible Architecture**: Built with extensibility in mind, allowing for easy customization at every level

## Installation

```bash
# Using pip
pip install recall-kit

# Using uv
uv install recall-kit
```

## Integration Options

recall-kit offers multiple ways to integrate memory capabilities:

### Tool-based Integration

Add memory capabilities through function calls in your application:

```python
from recall_kit import MemoryToolkit

memory_tools = MemoryToolkit()
memory_tools.store("User mentioned they prefer dark mode")
relevant_memories = memory_tools.recall("user preferences")
```

### MCP Integration

Expose memory capabilities through the Model Context Protocol:

```python
from recall_kit.integrations import MCPMemoryServer

# Create and start an MCP server with memory capabilities
memory_server = MCPMemoryServer()
memory_server.start()

# The server can now be connected to any MCP-compatible client
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
```

### Command Line Interface

Use recall-kit directly from the command line:

```bash
# Start a chat session with memory enabled
recall-chat

# Configure memory settings
recall-config set memory.source vector_store
recall-config set memory.max_items 50
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
```

## Memory Sources

recall-kit uses [pluggy](https://pluggy.readthedocs.io/) to allow for flexible memory source specification:

```python
from recall_kit import register_memory_source

@register_memory_source("my_custom_source")
class CustomMemorySource:
    def store(self, memory):
        # Implementation for storing memories
        pass

    def retrieve(self, query, **params):
        # Implementation for retrieving memories
        pass
```

Built-in memory sources include:
- Vector stores (using various embedding models)
- Simple in-memory storage
- File-based persistence
- Database integrations

## Recall Pipeline

The recall pipeline in recall-kit consists of customizable stages:

1. **Retrieve**: Fetch candidate memories from the memory source
2. **Filter**: Remove irrelevant or low-quality memories
3. **Rerank**: Order memories by relevance or importance
4. **Reflect**: Generate meta-information or summaries about retrieved memories
5. **Store**: Optionally store new derived memories

Each stage can be customized:

```python
from recall_kit import RecallPipeline
from recall_kit.filters import TimeDecayFilter
from recall_kit.rankers import RelevanceRanker

pipeline = RecallPipeline(
    retriever="semantic_search",
    filters=[TimeDecayFilter(half_life_days=7)],
    ranker=RelevanceRanker(),
    reflector="summarize_key_points",
    store_reflections=True
)

memories = pipeline.execute("What does the user like?")
```

## Default Configuration

recall-kit comes with sensible defaults that work out of the box, while allowing for customization at any level:

```python
from recall_kit import MemoryManager

# Use with all defaults
memory = MemoryManager()

# Or customize as needed
memory = MemoryManager(
    source="vector_store",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    pipeline_config={
        "max_memories": 10,
        "filter_threshold": 0.7
    }
)
```

## Examples

Check out the `examples/` directory for complete usage examples:

- Basic memory storage and retrieval
- Custom memory sources
- Advanced recall pipeline configuration
- Integration with popular LLM frameworks

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to get started.

## License

MIT
