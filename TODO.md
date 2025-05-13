TODO

- break out some of the very long functions in core.py into utility functions, e.g. the repairing missing tool_call_id. These should be in short concise utility functions. We shouldn't have a very long core.py

- be sure to NOT catch and rethrow generic exceptions. we should handle context window exceeded, but other errors should be raised

- need to figure out how to pass model name into recall_kit, when default embedding function is being used

- next milestone: LongBench runs successfully

- 

