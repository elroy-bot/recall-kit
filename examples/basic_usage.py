#!/usr/bin/env python
"""
Basic usage example for recall-kit
"""
from recall_kit import hello_world

# Default greeting
message = hello_world()
print(message)

# Custom greeting
message = hello_world("Python Developer")
print(message)
