def test_basic_message(recall_kit):
    request = {
        "model": "openai/gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ],
        "temperature": 0.7,
    }
    recall_kit.completion(request)
