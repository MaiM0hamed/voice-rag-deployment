"""Headless demo of the REAL AgentSession pipeline (no LiveKit server needed).

Uses `AgentSession.run()` -- the SDK's official API for driving a session
outside a room (it powers LiveKit's own evals). The Agent, AgentSession, the
@function_tool dispatch and the LLM are all real SDK components; only STT/TTS
are free local stubs, which the brief permits.

This produces the transcript + tool-invocation log required by Task 1.1.

Run:
    ollama pull qwen2.5:1.5b && ollama serve
    python run_session_demo.py
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import AgentSession
from livekit.agents.llm import ChatMessage, FunctionCall, FunctionCallOutput
from livekit.plugins import openai

from agent import SupportAgent

load_dotenv()

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "session.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("demo")

USER_TURNS = [
    "Hi, are you a real person?",
    "What's the status of my order ORD-1001?",
    "Thanks! Can you check ORD-9999 too?",
]


async def main() -> None:
    """Drive scripted turns through a real AgentSession and log the results."""
    session = AgentSession(
        llm=openai.LLM.with_ollama(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"),
            base_url=os.getenv("OLLAMA_URL", "http://localhost:11434") + "/v1",
        )
    )

    # NOTE: capture_run=True makes start() await an initial agent-initiated
    # reply. Our agent waits for the user, so we start plainly and use run().
    await session.start(agent=SupportAgent())
    logger.info("AgentSession started (text-driven, no room)")

    transcript = LOG_DIR / "transcript.log"
    tool_log = LOG_DIR / "tool_invocation.log"

    with transcript.open("w", encoding="utf-8") as ts, tool_log.open(
        "w", encoding="utf-8"
    ) as tl:
        for turn in USER_TURNS:
            logger.info("USER: %s", turn)
            ts.write(f"USER: {turn}\n")

            result = await session.run(user_input=turn)

            for event in result.events:
                item = getattr(event, "item", None)

                if isinstance(item, FunctionCall):
                    line = f"LLM invoked tool: {item.name}(arguments={item.arguments})"
                    logger.info(line)
                    tl.write(line + "\n")

                elif isinstance(item, FunctionCallOutput):
                    line = (
                        f"Tool returned: {item.name} -> {item.output} "
                        f"(is_error={item.is_error})"
                    )
                    logger.info(line)
                    tl.write(line + "\n")

                elif isinstance(item, ChatMessage) and item.role == "assistant":
                    reply = (item.text_content or "").strip()
                    logger.info("AGENT: %s", reply)
                    ts.write(f"AGENT: {reply}\n\n")

    await session.aclose()
    logger.info("Wrote %s and %s", transcript, tool_log)


if __name__ == "__main__":
    asyncio.run(main())
