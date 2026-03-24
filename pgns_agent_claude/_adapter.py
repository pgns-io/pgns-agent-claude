# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""Claude Agent SDK adapter for pgns-agent."""

from __future__ import annotations

__all__ = ["ClaudeAdapter"]

from typing import TYPE_CHECKING, Any

from pgns_agent import Adapter

if TYPE_CHECKING:
    from claude_agent_sdk import Agent


class ClaudeAdapter(Adapter):
    """Thin adapter wrapping Anthropic's Claude Agent SDK into pgns-agent.

    Accepts a **pre-configured** :class:`claude_agent_sdk.Agent` instance and
    delegates to its ``run()`` method.  Framework authentication (Anthropic API
    key, model selection, tools) is the caller's responsibility — the adapter
    receives a ready-to-use agent.

    The task input dict is expected to contain a ``"prompt"`` key with the user
    message string.  Only ``"prompt"`` is consumed; additional keys are ignored.

    Example::

        from claude_agent_sdk import Agent
        from pgns_agent import AgentServer
        from pgns_agent_claude import ClaudeAdapter

        claude = Agent(model="claude-sonnet-4-20250514")
        server = AgentServer(name="my-claude-agent", description="Claude-powered agent")
        server.use(ClaudeAdapter(claude))
        server.listen(3000)

    The returned dict always includes an ``"output"`` key with the agent's
    response text and a ``"metadata"`` key with framework telemetry (model,
    token usage, stop reason).
    """

    def __init__(self, agent: Agent) -> None:
        """Initialise the adapter with a configured Claude Agent SDK agent.

        Args:
            agent: A :class:`claude_agent_sdk.Agent` instance.  Must be fully
                configured (model, tools, system prompt) before being passed
                to the adapter.
        """
        self._agent = agent

    async def handle(self, task_input: dict[str, Any]) -> dict[str, Any]:
        """Run the Claude agent with the task input and return the result.

        Extracts ``task_input["prompt"]`` and passes it to the agent's
        ``run()`` method.  Falls back to an empty string when ``"prompt"``
        is absent.

        Returns:
            A dict with ``"output"`` (agent response text) and ``"metadata"``
            (model, usage, stop_reason).
        """
        prompt = task_input.get("prompt") or ""
        result = await self._agent.run(prompt)

        # result is duck-typed — the SDK is not imported at runtime, so guard
        # each field with hasattr to tolerate minimal or future result shapes.
        metadata: dict[str, Any] = {}
        if hasattr(result, "model") and result.model is not None:
            metadata["model"] = result.model
        if hasattr(result, "stop_reason") and result.stop_reason is not None:
            metadata["stop_reason"] = result.stop_reason
        if hasattr(result, "usage") and result.usage is not None:
            usage: dict[str, int] = {}
            if hasattr(result.usage, "input_tokens"):
                usage["input_tokens"] = result.usage.input_tokens
            if hasattr(result.usage, "output_tokens"):
                usage["output_tokens"] = result.usage.output_tokens
            if usage:
                metadata["usage"] = usage

        response: dict[str, Any] = {"output": result.output}
        if metadata:
            response["metadata"] = metadata
        return response
