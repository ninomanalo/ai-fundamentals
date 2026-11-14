messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant that explains technology simply."
    },
    {
        "role": "user",
        "content": "What is artificial intelligence?"
    },
    {
        "role": "assistant",
        "content": "Artificial intelligence allows computers to perform tasks that normally require human intelligence."
    },
    {
        "role": "user",
        "content": "Give me a real-world example."
    }
]

print("LLM Conversation")
print("================")

for message in messages:
    role = message["role"]
    content = message["content"]

    print(f"\nRole: {role}")
    print(f"Message: {content}")

print("\nConversation Summary")
print("====================")

print(f"Total messages: {len(messages)}")

user_messages = [
    message for message in messages
    if message["role"] == "user"
]

assistant_messages = [
    message for message in messages
    if message["role"] == "assistant"
]

print(f"User messages: {len(user_messages)}")
print(f"Assistant messages: {len(assistant_messages)}")