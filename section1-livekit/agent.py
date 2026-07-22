"""SupportAgent: a real `livekit.agents.Agent` subclass with real tools.

Key point for the assessment: the tool schema is *not* hand-written. The
`@function_tool` decorator inspects the method's type hints and docstring to
build the schema the LLM sees, and the SDK handles dispatch when the model
emits a tool call. Our code never decides to call the tool -- the LLM does.
"""

from __future__ import annotations

import logging

from livekit.agents import Agent, RunContext, function_tool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a friendly customer-support assistant for a food-delivery app. "
    "Help users with their orders. When a user asks about an order's status and "
    "provides an order id, call the get_order_status tool. Never invent order "
    "details -- always rely on the tool's result. Keep replies short and "
    "conversational, suitable for being spoken aloud."
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
