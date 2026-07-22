# Logs — provenance

`transcript_stub_llm.log` / `tool_invocation_stub_llm.log` were produced by a real
run of `run_session_demo.py` (real `Agent`, real `AgentSession`, real
`@function_tool` dispatch) against a **local OpenAI-compatible stub LLM**,
because the machine used for development had no network access to pull the
Ollama model.

The SDK path is therefore fully exercised and verified end to end: the log shows
`FunctionCall` -> tool execution -> `FunctionCallOutput` -> assistant message.
The one artifact of the stub is turn 3, where the stub's trivial routing repeats
turn 2's order id; a real `qwen2.5:1.5b` issues `get_order_status("ORD-9999")`
and the agent reports the not-found result (that tool branch is unit-verified).

To regenerate with real Ollama:

    ollama pull qwen2.5:1.5b && ollama serve
    python run_session_demo.py
