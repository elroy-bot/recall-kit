# How this repo implements plugins and extendability
This document describes how the LLM package implements its plugin system.

The goal is to provide insight to a package author who would like to implement his own plugin-supporting package.

Of particular interest is the way in which the package centers it's functionality around the plugin system, such that even core functionality is in fact implemented via plugins.

## Ways in which users can add plugins to this package

The `llm` package offers several ways for users to extend and customize its functionality through plugins:

### 1. Installing Pre-built Plugins

Users can install existing plugins from PyPI or other sources:

```bash
# Install a plugin from PyPI
llm install llm-gpt4all

# Install from a local directory (development mode)
llm install -e path/to/plugin-directory

# Install from a GitHub repository or gist
llm install https://github.com/username/llm-plugin-name
```

### 2. Creating Custom Plugins

Users can create their own plugins by implementing one or more of the following hook points:

#### a. Adding New Models

Users can add support for new LLM models by implementing the `register_models` hook:

```python
from llm import hookimpl, Model

@hookimpl
def register_models(register):
    register(MyCustomModel(), aliases=("custom",))

class MyCustomModel(Model):
    model_id = "my-custom-model"

    def execute(self, prompt, stream, response, conversation):
        # Implementation for the model
        yield "Generated response"
```

#### b. Adding New Embedding Models

Users can add support for new embedding models by implementing the `register_embedding_models` hook:

```python
from llm import hookimpl, EmbeddingModel

@hookimpl
def register_embedding_models(register):
    register(MyEmbeddingModel(), aliases=("my-embed",))

class MyEmbeddingModel(EmbeddingModel):
    model_id = "my-embedding-model"

    def embed_batch(self, texts):
        # Implementation for embedding
        return [...]
```

#### c. Adding New CLI Commands

Users can add new commands to the `llm` CLI tool:

```python
from llm import hookimpl
import click

@hookimpl
def register_commands(cli):
    @cli.command(name="my-command")
    def my_command():
        "Documentation for my command"
        click.echo("Command output")
```

#### d. Adding Template Loaders

Users can add new template loaders for use with `llm -t prefix:name`:

```python
from llm import hookimpl, Template

@hookimpl
def register_template_loaders(register):
    register("my-prefix", my_template_loader)

def my_template_loader(template_path):
    # Implementation to load templates
    return Template(...)
```

#### e. Adding Fragment Loaders

Users can add new fragment loaders for use with `llm -f prefix:argument`:

```python
from llm import hookimpl, Fragment

@hookimpl
def register_fragment_loaders(register):
    register("my-fragments", my_fragment_loader)

def my_fragment_loader(argument):
    # Implementation to load fragments
    return Fragment(...)
```

### 3. Controlling Plugin Loading

Users can control which plugins are loaded using the `LLM_LOAD_PLUGINS` environment variable:

```bash
# Disable all plugins
LLM_LOAD_PLUGINS='' llm ...

# Load only specific plugins
LLM_LOAD_PLUGINS='llm-gpt4all,llm-cluster' llm ...
```

### 4. Configuring Extra Models

Users can add additional OpenAI-compatible models by creating a YAML file at `~/.llm/extra-openai-models.yaml` with model configurations.

### 5. Viewing Installed Plugins

Users can view installed plugins and their capabilities:

```bash
llm plugins
```

This plugin architecture allows the `llm` package to be highly extensible while maintaining a clean core codebase. Even the default OpenAI models are implemented as a plugin (`llm.default_plugins.openai_models`), demonstrating how the package centers its functionality around the plugin system.
