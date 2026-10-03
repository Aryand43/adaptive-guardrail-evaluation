Sanitized OpenRouter chat-completions response shapes for offline adapter tests. No real
prompts, outputs, keys or request IDs. Error messages are prefixed `SANITIZED:` so tests can
assert they never leak into exceptions or logs.
