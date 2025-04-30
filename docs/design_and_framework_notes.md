# Competitor / framework notes

These are design and implementation guidelines from the `recall-kit` author.

The motivation for this library is that I do not find any of the existing options to be flexible or liteweight enough.

Each toolset I've seen requires you to consider it a _framework_, and use _it_ as the predominant framework the user works with.

`recall-kit` is lightweight collection of functions, that can easily fit into many other frameworks.

`recall-kit` demonstrates how easy it is to integrate with other frameworks by demonstrating it, and adding implementations for a variety of other frameworks (ie even though I'm not a fan of `langchain`, I could demonstrate that you _can_ use it, without needing to install it.)

`recall-kit` is _not a low code solution_. There will not be a WYSIWIG agentic UI, and we do not target non-developer users.

`recall-kit` is not a freemium version of an enterprise project. Memory as a service is not viable, we do not aspire to starting a company with it as a primary offering.

`recall-kit` _shows its work_. It should be easy to discover what prompt is running where. Templating may be necessary but shoud be managed carefully.

## Design outputlook

In general, I think programming with LLM's benefits from the [Unix Philosophy](https://en.wikipedia.org/wiki/Unix_philosophy), in particular the way in which each program's output can be an input to another:

> Expect the output of every program to become the input to another, as yet unknown, program. Don't clutter output with extraneous information. Avoid stringently columnar or binary input formats. Don't insist on interactive input.

and:

> Write programs to handle text streams, because that is a universal interface.

This example from `semantic_kernel` illustrates how violating this becomes clunky:

```python
async def main():
    # Configure structured output format
    settings = OpenAIChatPromptExecutionSettings()
    settings.response_format = MenuItem

    # Create agent with plugin and settings
    agent = ChatCompletionAgent(
        service=AzureChatCompletion(),
        name="SK-Assistant",
        instructions="You are a helpful assistant.",
        plugins=[MenuPlugin()],
        arguments=KernelArguments(settings)
    )

    response = await agent.get_response(messages="What is the price of the soup special?")
    print(response.content)
```

Setting the output type is very clunky, and it's relatively tricky to chain together calls.

I generally prefer more functional programming, with very thin classes mostly serving just to encapsulate data.

An example of this not working is this Langchain example:

```python
class ChatParrotLink(BaseChatModel):
    """A custom chat model that echoes the first `parrot_buffer_length` characters
    of the input.

    When contributing an implementation to LangChain, carefully document
    the model including the initialization parameters, include
    an example of how to initialize the model and include any relevant
    links to the underlying models documentation or API.

    Example:

        .. code-block:: python

            model = ChatParrotLink(parrot_buffer_length=2, model="bird-brain-001")
            result = model.invoke([HumanMessage(content="hello")])
            result = model.batch([[HumanMessage(content="hello")],
                                 [HumanMessage(content="world")]])
    """

    model_name: str = Field(alias="model")
    """The name of the model"""
    parrot_buffer_length: int
    """The number of characters from the last message of the prompt to be echoed."""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    stop: Optional[List[str]] = None
    max_retries: int = 2

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Override the _generate method to implement the chat model logic.

        This can be a call to an API, a call to a local model, or any other
        implementation that generates a response to the input prompt.
```

There's too much magic here, the user needs to know how to overwrite the `generate` method.


My previous attempt at a library resulted in a very large Context class which involved many parameters:

```python
class ElroyContext:
    _db: Optional[DbSession] = None

    def __init__(
        self,
        *,
        # Basic Configuration
        config_path: Optional[str] = None,
        database_url: str,
        show_internal_thought: bool,
        system_message_color: str,
        assistant_color: str,
        user_input_color: str,
        warning_color: str,
        internal_thought_color: str,
        user_token: str,
        custom_tools_path: List[str] = [],
        # API Configuration
        openai_api_key: Optional[str] = None,
        openai_api_base: Optional[str] = None,
        openai_embedding_api_base: Optional[str] = None,
        # Model Configuration
        chat_model: Optional[str] = None,
        chat_model_api_key: Optional[str] = None,
        chat_model_api_base: Optional[str] = None,
        embedding_model: str,
        embedding_model_api_key: Optional[str] = None,
        embedding_model_api_base: Optional[str] = None,
        embedding_model_size: int,
        enable_caching: bool = True,
        inline_tool_calls: bool = False,
        # Context Management
        max_assistant_loops: int,
        max_tokens: int,
        max_context_age_minutes: float,
        min_convo_age_for_greeting_minutes: float,
        # Memory Management
        memory_cluster_similarity_threshold: float,
        max_memory_cluster_size: int,
        min_memory_cluster_size: int,
        memories_between_consolidation: int,
        messages_between_memory: int,
        l2_memory_relevance_distance_threshold: float,
        # Basic Configuration
        debug: bool,
        default_persona: Optional[str] = None,  # The generic persona to use if no persona is specified
        default_assistant_name: str,  # The generic assistant name to use if no assistant name is specified
        use_background_threads: bool,  # Whether to use background threads for certain operations
        max_ingested_doc_lines: int,  # The maximum number of lines to ingest from a document
        exclude_tools: List[str] = [],  # Tools to exclude from the tool registry
        reflect: bool,
    ):

        self.params = SimpleNamespace(**{k: v for k, v in locals().items() if k != "self"})

        self.reflect = reflect

        self.user_token = user_token
        self.show_internal_thought = show_internal_thought
        self.default_assistant_name = default_assistant_name
        self.default_persona = default_persona or PERSONA
        self.debug = debug
        self.max_tokens = max_tokens
        self.max_assistant_loops = max_assistant_loops
        self.l2_memory_relevance_distance_threshold = l2_memory_relevance_distance_threshold

        self.context_refresh_target_tokens = int(max_tokens / 3)
        self.memory_cluster_similarity_threshold = memory_cluster_similarity_threshold
        self.min_memory_cluster_size = min_memory_cluster_size
        self.max_memory_cluster_size = max_memory_cluster_size
        self.memories_between_consolidation = memories_between_consolidation
        self.messages_between_memory = messages_between_memory
        self.inline_tool_calls = inline_tool_calls
        self.use_background_threads = use_background_threads
        self.max_ingested_doc_lines = max_ingested_doc_lines
```


## Python specific notes

I do think dependency injection is useful, but it's difficult to do well in python. This example from pydantic ai is too magical:

```
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from bank_database import DatabaseConn


@dataclass
class SupportDependencies:
    customer_id: int
    db: DatabaseConn


class SupportOutput(BaseModel):
    support_advice: str = Field(description='Advice returned to the customer')
    block_card: bool = Field(description="Whether to block the customer's card")
    risk: int = Field(description='Risk level of query', ge=0, le=10)


support_agent = Agent(
    'openai:gpt-4o',
    deps_type=SupportDependencies,
    output_type=SupportOutput,
    system_prompt=(
        'You are a support agent in our bank, give the '
        'customer support and judge the risk level of their query.'
    ),
)
```


## Other frameworks

These are notes by the package author on other packages.

In general, I discount multi-agent systems, and the "agent" abstraction in general. The "agent" is a for loop with a pipeline.

Langchain, however, abstracts too much away. It is very difficult to discover what prompt ran where.

### toolz

Toolz is a very useful tool, especially in the way you can pipe one output to another. I make extensive use of toolz. Ideally, recall-kit will surface as the user facing abstraction a collection of functions.

### Pydantic

Pydantic is nice for building with LLM's because it converts naturally to text. We should use this for any data stuctures that recall-kit handles.

### semantic_kernel

Adding functions seems pretty easy. The ability to add them with (mostly) a vanilla function is nice:

```python
    @kernel_function(description="Provides the price of the requested menu item.")
    def get_item_price(
        self, menu_item: Annotated[str, "The name of the menu item."]
    ) -> Annotated[str, "Returns the price of the menu item."]:
        return "$9.99"
```

However the handling of getting the agent to output tool calls is weird and clunky:
```python
async def main():
    # Configure structured output format
    settings = OpenAIChatPromptExecutionSettings()
    settings.response_format = MenuItem

    # Create agent with plugin and settings
    agent = ChatCompletionAgent(
        service=AzureChatCompletion(),
        name="SK-Assistant",
        instructions="You are a helpful assistant.",
        plugins=[MenuPlugin()],
        arguments=KernelArguments(settings)
    )

    response = await agent.get_response(messages="What is the price of the soup special?")
    print(response.content)
```

### Letta

Letta's code quality is not good enough, and not performant. It's focus on Agentic UI is misplaced.




