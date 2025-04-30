# 2025-05-06

TODO:
- react to context overflow errors by splitting up the inputs, summarizing, and creating memories.
- accept user_id in cli

# 2025-05-02

Initial implementation is promising. A few tweaks are needed:

- The Consolidation process should be driven by the LLM. An LLM should consider the content, and rewrite it. Source metadata should be retained, both between a memory and it's immediate ancestor, and the root ancestor (source info). In other words, when a memory is recalled, a tool call should be able to retrieve the content of the source material for it.


For consolidation, see code in `elroy_prompts.py` and `memory_consolidation.py`. Note that for the prompts, they should be general and not necessarily oriented around a primary user.

- For the embedding_model, I'd rather it accept an embedding function. Similarly for a completion function. We should leverage `litellm` to handle support for many providers.

See `elroy_llm_client.py` for some inspiration.

- Embeddings should all be stored in one table, with the key pointing back to the source row. So, for example, if someone adds a new type of source data for memories (which we may also want to calculate embeddings on), it should be stored in the same table as those of memories (though, we will want enough metadata captured such that we can query the vector table for only one type of embedded data, e.g. only memories, only chat transcripts, only docs)




