# Section 1 — Write-up

## What this builds

A minimal voice agent on the **real `livekit-agents` SDK (v1.6.6)**:

| Requirement | Implementation |
|---|---|
| `livekit-agents` SDK | `livekit-agents==1.6.6`, imported throughout |
| `Agent` subclass | `SupportAgent(Agent)` in `agent.py` with `instructions=` persona |
| `AgentSession` pipeline | Built in `worker.py` (STT→LLM→TTS+VAD) and `run_session_demo.py` |
| `@function_tool` | Real decorator on `get_order_status`; schema auto-derived from type hints + docstring |
| LLM decides the call | Ollama emits the tool call; the SDK dispatches it. Our code never calls the tool |

**LLM:** `openai.LLM.with_ollama(...)` — a first-class SDK constructor pointing at
local Ollama (`qwen2.5:1.5b`). Free, no API key, real tool-calling.

**STT/TTS:** `worker.py` uses **real** providers — **Deepgram** for both streaming STT
(`nova-3`) and streaming TTS (`Aura`) — via the official LiveKit plugin, so the pipeline
is a genuine microphone → STT → LLM → tool → TTS → speaker loop. One `DEEPGRAM_API_KEY`
(free tier) covers the whole speech path and is read from the environment only.
Swapping TTS to ElevenLabs is a one-line change in `build_tts()` (documented inline).
The *provider-swap seam* is the SDK's abstract `stt.STT` / `tts.TTS` classes: because
`AgentSession` is handed provider objects, changing a provider is a constructor change
and nothing in `agent.py` or the tool moves (see Task 1.2 below).

### Two entrypoints

- `worker.py` — production shape and the **real voice** path. Real `AgentSession` with
  Deepgram STT + Deepgram Aura TTS, `silero.VAD` and `allow_interruptions=True`, run as a
  LiveKit worker against a real room: `python worker.py dev`. Logs the full lifecycle
  (Connected → Joined room → Listening → STT result → tool invocation → tool output →
  LLM response → TTS started → TTS completed).
- `run_session_demo.py` — grader-friendly offline path. Uses `AgentSession.run()`, the
  SDK's official headless API (it powers LiveKit's own evals), so tool-calling can be
  demonstrated without a server, microphone, or provider keys. See `logs/README.md`.

Verified SDK event sequence from an actual run:

```
FunctionCall(name='get_order_status', arguments='{"order_id": "ORD-1001"}')
FunctionCallOutput(output='Order ORD-1001 is shipped. Carrier: Aramex. ETA: 2026-07-19.', is_error=False)
ChatMessage(role='assistant', ...)
```

## Barge-in / interruption handling

The SDK handles this natively; the job is configuring it correctly.

1. **VAD-driven interruption.** `silero.VAD.load()` (free, local) detects user
   speech during agent playback. With `allow_interruptions=True`, `AgentSession`
   stops TTS, cancels the in-flight LLM task, and starts listening.
2. **Tuning false positives.** `min_interruption_duration` and
   `min_interruption_words` prevent backchannel ("mm-hmm") from cutting the
   agent off; `false_interruption_timeout` + `resume_false_interruption` let the
   agent resume if the "interruption" turned out to be noise.
3. **Honest context truncation.** On interruption the SDK marks the message
   `interrupted=True` and commits only the spoken prefix, so the LLM's memory
   matches what the user actually heard.
4. **Turn detection.** `turn_detection` with `min_endpointing_delay` /
   `max_endpointing_delay` governs when a user turn is considered finished.

Our demo runs text-driven (no audio), so interruption is configured in
`worker.py` rather than exercised in the transcript — stated plainly as a limitation.

## Adding a second tool safely

Adding e.g. `initiate_refund(order_id, reason)` is a single decorated method:

1. **Schema is generated, not hand-written.** Annotate parameters and document
   them; `@function_tool` derives the JSON schema (verified: our docstring
   produced `{"order_id": {"type": "string", ...}, "required": ["order_id"]}`).
   Use `Literal[...]` for `reason` to constrain the model to valid enum values.
2. **Side-effect discipline.** The lookup is read-only; a refund mutates state.
   Make it idempotent (dedupe on an idempotency key), verify the order is
   refundable first, and require explicit user confirmation above a threshold.
3. **Bound the loop.** `AgentSession(max_tool_steps=...)` caps tool-call chains
   so a confused model can't loop indefinitely.
4. **Least privilege.** Each tool validates its own inputs and touches only its
   own resource.

## Error handling if a tool call fails

- **Never crash the session.** The tool body is wrapped in try/except and returns
  a descriptive *string*; the SDK feeds it back as `FunctionCallOutput` so the
  LLM apologizes and recovers instead of the process dying.
- **Not-found vs. errored are distinct.** A missing order returns a clean
  "no order found" (verified: `ORD-9999`); an unexpected exception is logged with
  `logger.exception` and returns a service-unavailable message. The SDK's
  `is_error` flag distinguishes them downstream.
- **Transport failures.** LLM calls use `APIConnectOptions` (retries + timeout);
  the SDK raised `APIConnectionError` after 4 attempts when our backend was down,
  exactly as intended. For a tool hitting a real microservice I'd add bounded
  retries with backoff plus a circuit breaker and a spoken fallback.
- **Malformed arguments.** Schema validation rejects bad types before dispatch;
  a `TypeError` from unexpected kwargs is caught by the same guard.

## Task 1.2 — Swapping a pipeline component

The SDK interface is the seam, so swapping is a constructor change. To replace
Deepgram TTS with a free local Piper engine, implement the abstract `tts.TTS`
contract:

```python
class PiperTTS(tts.TTS):
    def __init__(self) -> None:
        super().__init__(capabilities=tts.TTSCapabilities(streaming=False),
                         sample_rate=22050, num_channels=1)

    def synthesize(self, text, *, conn_options=DEFAULT_API_CONNECT_OPTIONS):
        return _PiperStream(tts=self, input_text=text, conn_options=conn_options)
        # _PiperStream._run() pipes text -> piper binary -> output_emitter.push(pcm)
```

Then `AgentSession(tts=PiperTTS(), ...)`. Nothing in `agent.py` or the tool
changes. Same for STT: a `VoskSTT` implementing `_recognize_impl` drops in
identically — and swapping back to a paid vendor is just
`AgentSession(tts=elevenlabs.TTS())`. That vendor-independence is the point of
programming against the SDK's abstract classes rather than a bespoke wrapper.

## Known limitations (stated honestly)

- **Real STT/TTS are cloud free-tier, not local.** `worker.py` uses Deepgram for both
  STT and TTS — real speech, but the only place in this repo that leaves the
  "everything runs locally" model. It uses the free tier and reads the key from env; the
  fully-local alternative (Vosk/Whisper + Piper) is the documented swap seam
  (Task 1.2 above).
- `run_session_demo.py` is text-driven, so it needs no microphone or provider keys;
  it exercises the real Agent/LLM/tool-dispatch chain but not the audio (STT/TTS) legs.
  Barge-in and the audio path run in the real `worker.py` voice worker, which needs a
  microphone, speakers, and a live LiveKit room — that part is manual hardware testing.

## The tool-invocation lesson (why the prompt is forceful)

The committed logs are from a **real** Ollama run and show genuine tool calls. Getting
there surfaced the key gotcha: with a *soft* system prompt, `qwen2.5:1.5b` returned
`tool_calls: null` for "Where is my order ORD-1001?" and answered from imagination —
even though the generated tool schema was correct and the SDK passed it to Ollama
(both verified independently). Rewriting the prompt to **mandate** the tool ("you MUST
call get_order_status … never answer from your own knowledge") made the same model call
the tool reliably across phrasings. Takeaway: correct wiring is necessary but not
sufficient for small local models — the instruction has to be unambiguous.
