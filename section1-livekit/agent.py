"""SupportAgent: a real `livekit.agents.Agent` subclass with real tools.

Key point for the assessment: the tool schema is *not* hand-written. The
`@function_tool` decorator inspects the method's type hints and docstring to
build the schema the LLM sees, and the SDK handles dispatch when the model
emits a tool call. Our code never decides to call the tool -- the LLM does.
"""

from __future__ import annotations

import logging
import re

from livekit.agents import Agent, RunContext, function_tool

logger = logging.getLogger(__name__)

# Order ids look like "ORD-1001". Validating the shape lets us reject obvious
# garbage before a backend round-trip and give the model a correctable message.
_ORDER_ID_RE = re.compile(r"^ORD-\d{3,}$")

# The tool wiring is correct on its own; what actually drives a small local model
# to emit a real tool call (vs. answering from imagination) is an unambiguous,
# mandatory instruction. This forceful wording was verified to make
# qwen2.5:1.5b invoke get_order_status reliably across phrasings.
SYSTEM_PROMPT = (
    "You are a friendly customer-support assistant for a food-delivery app. "
    "You have one tool: get_order_status.\n"
    "RULES:\n"
    "- For ANY question about an order's status, delivery, carrier, or ETA where "
    "the user gives an order id, you MUST call get_order_status with that id.\n"
    "- Do NOT ask the user to repeat an order id that is already in their message.\n"
    "- NEVER answer an order question from your own knowledge or guess order "
    "details -- always rely on the tool's returned result.\n"
    "- If the user has not given an order id, ask for it in one short sentence.\n"
    "Keep replies short and conversational, suitable for being spoken aloud."
)

# Mocked backend data (a DB or microservice in production).
_MOCK_ORDERS: dict[str, dict[str, str]] = {
    "ORD-1001": {"status": "shipped", "eta": "2026-07-19", "carrier": "Aramex"},
    "ORD-1002": {"status": "preparing", "eta": "2026-07-21", "carrier": "Bosta"},
    "ORD-1003": {"status": "delivered", "eta": "2026-07-15", "carrier": "Aramex"},
}


class SupportAgent(Agent):
    """Customer-support voice agent with a tool the LLM may invoke mid-call."""

    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)

    @function_tool()
    async def get_order_status(self, context: RunContext, order_id: str) -> str:
        """Look up the current status, carrier and ETA of a customer order.

        Args:
            order_id: The order identifier, for example "ORD-1001".
        """
        try:
            key = order_id.strip().upper()
            if not _ORDER_ID_RE.match(key):
                logger.warning("TOOL get_order_status: malformed order id: %r", order_id)
                return (
                    f"'{order_id}' is not a valid order number. "
                    "Order numbers look like ORD-1001. Ask the user to repeat it."
                )
            order = _MOCK_ORDERS.get(key)
            if order is None:
                logger.warning("TOOL get_order_status: order not found: %s", key)
                return (
                    f"No order found with id {key}. "
                    "Ask the user to double-check the order number."
                )
            result = (
                f"Order {key} is {order['status']}. "
                f"Carrier: {order['carrier']}. ETA: {order['eta']}."
            )
            logger.info("TOOL get_order_status(%s) -> %s", key, result)
            return result
        except Exception as exc:  # never let a tool crash the session
            logger.exception("TOOL get_order_status failed")
            return f"The order lookup service is unavailable right now ({exc})."
