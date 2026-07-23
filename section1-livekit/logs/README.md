# Logs — provenance

These files are produced by a **real** run of `run_session_demo.py`:

- `transcript.log` — the conversation (USER / AGENT turns).
- `tool_invocation.log` — every `FunctionCall` the LLM emitted and its
  `FunctionCallOutput`. Proof that the LLM actually invoked the tool.
- `session.log` — the full run log (SDK + app).

The pipeline is genuine end to end: real `Agent`, real `AgentSession`, real
Ollama LLM (`qwen2.5:1.5b`), and real `@function_tool` dispatch by the SDK. Only
STT/TTS are not exercised here, because the demo is text-driven (`AgentSession.run()`);
the audio legs run in the live `worker.py` voice path.

The log shows the required chain — `FunctionCall → tool execution →
FunctionCallOutput → assistant message` — including the `ORD-9999` not-found
branch, which returns the tool's message (not a hallucination).

To regenerate:

    ollama serve            # with qwen2.5:1.5b pulled
    python run_session_demo.py
