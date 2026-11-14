
import tiktoken

text = """
Artificial intelligence is transforming how businesses
operate. AI engineers build applications using large
language models, APIs, and automation tools.
"""

encoding = tiktoken.get_encoding("cl100k_base")

tokens = encoding.encode(text)

print("TOKEN ANALYZER")
print("==============")

print("\nOriginal Text:")
print(text)

print("\nToken IDs:")
print(tokens)

print("\nDecoded Tokens:")

for token in tokens:
    decoded_token = encoding.decode([token])
    print(f"Token ID: {token} | Text: {repr(decoded_token)}")

print("\nToken Statistics")
print("================")

print(f"Character count: {len(text)}")
print(f"Word count: {len(text.split())}")
print(f"Token count: {len(tokens)}")

