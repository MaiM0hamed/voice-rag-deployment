"""LiveKit worker: runs the real AgentSession STT -> LLM -> TTS pipeline.

This is the production-shaped entrypoint. It builds a genuine `AgentSession`
with our free providers and hands it the `SupportAgent`. The SDK owns the
pipeline: turn detection, interruption, LLM streaming, and tool dispatch.

Run against a LiveKit server (free self-hosted or LiveKit Cloud dev tier):
    python worker.py dev

Env (see .env.example): LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET,
OLLAMA_URL, OLLAMA_MODEL.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from livekit.agents import AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import openai, silero

from agent import SupportAgent
from mock_providers import MockSTT, MockTTS

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("worker")


def build_llm() -> openai.LLM:
    """Real LiveKit LLM backed by local Ollama (free, OpenAI-compatible)."""
    return openai.LLM.with_ollama(
        model=os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"),
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434") + "/v1",
    )


async def entrypoint(ctx: JobContext) -> None:
    """Job entrypoint invoked by the LiveKit worker for each room."""
    await ctx.connect()

    session = AgentSession(
        stt=MockSTT(),
        llm=build_llm(),
        tts=MockTTS(),
        vad=silero.VAD.load(),      # free, local VAD -> enables barge-in
        allow_interruptions=True,   # user can talk over the agent
    )

    await session.start(agent=SupportAgent(), room=ctx.room)
    logger.info("AgentSession started in room %s", ctx.room.name)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
