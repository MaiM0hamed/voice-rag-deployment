# Section 1 — LiveKit Voice Agent

A customer-support **voice agent** built on the real
[`livekit-agents`](https://github.com/livekit/agents) SDK (v1.6.6). The LLM
decides — on its own — to call a `@function_tool`, and the SDK dispatches it.

```
Microphone → STT (Deepgram) → LLM (Ollama) → @function_tool → TTS (Deepgram Aura) → Speaker
```

## Requirements

- **Python 3.10+** (developed on 3.13)
- **[Ollama](https://ollama.com)** running locally with a tool-capable model:
  ```bash
  ollama pull qwen2.5:1.5b
  ```
  > `qwen2.5:1.5b` reliably emits tool calls with the system prompt in `agent.py`.
  > Any larger `qwen2.5` / `llama3.1` model works too (set `OLLAMA_MODEL`).
- **A Deepgram API key** (free tier) — used for *both* STT and TTS. Only needed
  for the live voice worker, not the headless demo.
- **A LiveKit server** — LiveKit Cloud (free dev tier) or self-hosted — only for
  the live voice worker.

Python packages (see `requirements.txt`):

| Package | Purpose |
|---|---|
| `livekit-agents==1.6.6` | `Agent`, `AgentSession`, `@function_tool` |
| `livekit-plugins-openai==1.6.6` | `openai.LLM.with_ollama()` → local Ollama |
| `livekit-plugins-deepgram==1.6.6` | real streaming STT + TTS |
| `livekit-plugins-silero==1.6.6` | free local VAD (barge-in / turn detection) |
| `python-dotenv==1.0.1` | load `.env` |

## Installation

```bash
cd section1-livekit
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then fill in the values
```

## Environment variables

All configuration is read from `.env` — **nothing is hardcoded**.

| Variable | Used by | Notes |
|---|---|---|
| `OLLAMA_URL` | both | default `http://localhost:11434` |
| `OLLAMA_MODEL` | both | default `qwen2.5:1.5b` (tool-capable) |
| `LIVEKIT_URL` | worker | e.g. `wss://<you>.livekit.cloud` or `ws://localhost:7880` |
| `LIVEKIT_API_KEY` | worker | LiveKit room credential |
| `LIVEKIT_API_SECRET` | worker | LiveKit room credential |
| `DEEPGRAM_API_KEY` | worker | one key covers STT **and** TTS |
| `DEEPGRAM_MODEL` | worker | STT model, default `nova-3` |
| `DEEPGRAM_TTS_MODEL` | worker | TTS voice, default `aura-2-andromeda-en` |

> Never commit a real `.env`; it is git-ignored.

## How to run

### 1. Live voice pipeline (real mic → speaker)

Requires `.env` with the LiveKit + Deepgram values and Ollama running.

```bash
python worker.py dev
```

The worker registers with LiveKit and waits for a room. Join the room with any
LiveKit client (e.g. the [Agents Playground](https://agents-playground.livekit.io))
and speak. The worker logs the full lifecycle (see below).

### 2. Headless demo (no server, mic, or Deepgram key needed)

Proves the LLM → tool → output → LLM-response chain using the SDK's official
`AgentSession.run()`. Only Ollama is required.

```bash
python run_session_demo.py
```

It writes:
- `demo_transcript.md` — a readable transcript with tool calls
- `logs/transcript.log`, `logs/tool_invocation.log`, `logs/session.log`

## How to test

```bash
python run_session_demo.py
cat logs/tool_invocation.log     # must be NON-EMPTY
```

A passing run shows real tool invocations, e.g.:

```
LLM invoked tool: get_order_status(arguments={"order_id": "ORD-1001"})
Tool returned: get_order_status -> Order ORD-1001 is shipped. Carrier: Aramex. ETA: 2026-07-19. (is_error=False)
```

The `ORD-9999` turn returns a clean "No order found", proving the value came from
the **tool** (not the model's imagination). Verify tool-calling directly against
your model at any time:

```bash
python - <<'PY'
import json, urllib.request
p={"model":"qwen2.5:1.5b","stream":False,"messages":[
  {"role":"user","content":"Where is my order ORD-1001?"}],
  "tools":[{"type":"function","function":{"name":"get_order_status",
   "parameters":{"type":"object","properties":{"order_id":{"type":"string"}},
   "required":["order_id"]}}}]}
r=urllib.request.Request("http://localhost:11434/v1/chat/completions",
   json.dumps(p).encode(),{"Content-Type":"application/json"})
print(json.load(urllib.request.urlopen(r))["choices"][0]["message"].get("tool_calls"))
PY
```

## Example conversation

```
User:      What's the status of my order ORD-1001?
  → tool:  get_order_status({"order_id": "ORD-1001"})
  → out:   Order ORD-1001 is shipped. Carrier: Aramex. ETA: 2026-07-19.
Assistant: Your order was shipped with Aramex and should arrive by July 19th.
```

See `demo_transcript.md` for the full real transcript.

## Architecture

| File | Role |
|---|---|
| `agent.py` | `SupportAgent(Agent)` — persona + the `@function_tool` `get_order_status`. The tool's JSON schema is **derived** from type hints + docstring, not hand-written. |
| `worker.py` | Production entrypoint. Builds a real `AgentSession` (Deepgram STT + Aura TTS + Silero VAD, `allow_interruptions=True`) and runs as a LiveKit worker. |
| `run_session_demo.py` | Headless proof via `AgentSession.run()` — real Agent/LLM/tool dispatch without a server or audio hardware. |
| `logs/` | Real run artifacts (transcript, tool-invocation, session log). |

**Provider swap seam:** STT/TTS/LLM are passed to `AgentSession` as SDK objects, so
swapping (e.g. Deepgram → ElevenLabs TTS) is a one-line constructor change — nothing
in `agent.py` or the tool changes. See `NOTES.md`.

## Troubleshooting

- **`tool_invocation.log` is empty / the agent invents order details.**
  The model didn't emit a tool call. Root cause is almost always a *weak system
  prompt*, not the SDK — the schema is generated correctly. Small models need an
  explicit "you MUST call the tool" instruction (that's why `agent.py`'s prompt is
  forceful). Confirmed: `qwen2.5:1.5b` fails with a soft prompt and succeeds with
  the mandatory one. As a last resort use a larger model via `OLLAMA_MODEL`.
- **`ImportError: cannot import name 'deepgram' from 'livekit.plugins'`.**
  The Deepgram plugin isn't installed: `pip install livekit-plugins-deepgram==1.6.6`
  (or `pip install -r requirements.txt`).
- **`Missing required environment variable(s): ...`.** The worker fails fast when a
  `LIVEKIT_*` or `DEEPGRAM_API_KEY` value is absent. Fill them in `.env`.
- **Ollama connection refused.** Start it: `ollama serve`, and confirm the model is
  pulled: `ollama list`.
