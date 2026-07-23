"""LiveKit worker: runs the real AgentSession STT -> LLM -> TTS pipeline.

This is the production-shaped entrypoint. It builds a genuine `AgentSession` with
REAL providers and hands it the `SupportAgent`. The SDK owns the pipeline: turn
detection, interruption, LLM streaming, and tool dispatch.

Pipeline (real, end to end):
    Microphone -> Deepgram STT -> Ollama LLM -> @function_tool -> Deepgram TTS -> Speaker

Run against a LiveKit server (self-hosted free, or LiveKit Cloud dev tier):
    python worker.py dev

Env (see .env.example):
    LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET   (LiveKit room)
    DEEPGRAM_API_KEY                                   (real STT + TTS, one key)
    DEEPGRAM_MODEL, DEEPGRAM_TTS_MODEL                 (STT / TTS model overrides)
    OLLAMA_URL, OLLAMA_MODEL                           (local LLM)

Every secret is read from the environment only; nothing is hardcoded.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from livekit.agents import AgentSession, JobContext, WorkerOptions, cli
from livekit.agents.llm import ChatMessage, FunctionCall, FunctionCallOutput
from livekit.plugins import deepgram, openai, silero

from agent import SupportAgent

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("worker")

# Secrets are read from the environment by the plugins / LiveKit CLI themselves;
# we only validate their presence so the worker fails fast with a clear message
# instead of a deep stack trace mid-call. Deepgram serves BOTH STT and TTS here,
# so a single key covers the whole speech path. The LIVEKIT_* trio is what the
# worker needs to reach the room.
_REQUIRED_KEYS = (
    "LIVEKIT_URL",
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
    "DEEPGRAM_API_KEY",
)


def _require_env() -> None:
    """Fail fast if a required secret is missing. Never logs the value."""
    missing = [name for name in _REQUIRED_KEYS if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + ". Add them to .env (LiveKit: https://cloud.livekit.io, "
            "Deepgram: https://console.deepgram.com). See .env.example."
        )


def build_llm() -> openai.LLM:
    """Real LiveKit LLM backed by local Ollama (free, OpenAI-compatible).

    Low temperature keeps tool-routing deterministic -- a support agent should
    reliably call get_order_status rather than occasionally answer from memory.
    """
    return openai.LLM.with_ollama(
        model=os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"),
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434") + "/v1",
        temperature=0.0,
    )


def build_stt() -> deepgram.STT:
    """Real streaming speech-to-text (Deepgram). Reads DEEPGRAM_API_KEY from env."""
    return deepgram.STT(model=os.getenv("DEEPGRAM_MODEL", "nova-3"), language="en")


def build_tts() -> deepgram.TTS:
    """Real streaming text-to-speech (Deepgram Aura). Reads DEEPGRAM_API_KEY from env.

    Deepgram covers STT and TTS with one key. To use ElevenLabs instead, install
    livekit-plugins-elevenlabs, set ELEVEN_API_KEY (with the text_to_speech
    permission enabled), and return `elevenlabs.TTS()` here.
    """
    return deepgram.TTS(model=os.getenv("DEEPGRAM_TTS_MODEL", "aura-2-andromeda-en"))


def _attach_logging(session: AgentSession) -> None:
    """Log the full voice pipeline lifecycle using the SDK's own events.

    No new abstraction: these are AgentSession's native events. Together they
    produce the required trace -- STT result, tool invocation, tool output, LLM
    response, and TTS start/stop.
    """

    @session.on("user_input_transcribed")
    def _on_transcript(ev) -> None:  # STT result
        final = getattr(ev, "is_final", True)
        logger.info("STT result (final=%s): %r", final, ev.transcript)

    @session.on("conversation_item_added")
    def _on_item(ev) -> None:  # tool invocation / output / LLM response
        item = ev.item
        if isinstance(item, FunctionCall):
            logger.info(
                "LLM invoked tool: %s(arguments=%s)", item.name, item.arguments
            )
        elif isinstance(item, FunctionCallOutput):
            logger.info(
                "Tool returned: %s -> %s (is_error=%s)",
                item.name,
                item.output,
                item.is_error,
            )
        elif isinstance(item, ChatMessage) and item.role == "assistant":
            logger.info("LLM response: %s", (item.text_content or "").strip())

    @session.on("agent_state_changed")
    def _on_state(ev) -> None:  # TTS start/stop + listening
        if ev.new_state == "speaking":
            logger.info("TTS started")
        elif ev.old_state == "speaking":
            logger.info("TTS completed")
        if ev.new_state == "listening":
            logger.info("Listening...")


async def entrypoint(ctx: JobContext) -> None:
    """Job entrypoint invoked by the LiveKit worker for each room."""
    _require_env()

    await ctx.connect()
    logger.info("Connected to LiveKit (%s)", os.getenv("LIVEKIT_URL", "?"))

    session = AgentSession(
        stt=build_stt(),
        llm=build_llm(),
        tts=build_tts(),
        vad=silero.VAD.load(),      # free, local VAD -> enables barge-in
        allow_interruptions=True,   # user can talk over the agent
    )
    _attach_logging(session)

    await session.start(agent=SupportAgent(), room=ctx.room)
    logger.info("Joined room %s", ctx.room.name)
    logger.info("Listening...")

    # Greet immediately so the TTS path is exercised on join (and the
    # "TTS started/completed" logs appear without the user speaking first).
    await session.generate_reply(
        instructions="Greet the user briefly and offer to help with their order."
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
