# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ClaudeAdapter — Claude Agent SDK integration with pgns-agent."""

from __future__ import annotations

import dataclasses
from typing import Any
from unittest.mock import AsyncMock

import pytest

from pgns_agent import Adapter, AgentServer
from pgns_agent_claude import ClaudeAdapter

# ---------------------------------------------------------------------------
# Mock Claude Agent SDK objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MockUsage:
    input_tokens: int = 100
    output_tokens: int = 50


@dataclasses.dataclass
class MockResult:
    output: str = "Hello from Claude!"
    model: str = "claude-sonnet-4-20250514"
    stop_reason: str = "end_turn"
    usage: MockUsage = dataclasses.field(default_factory=MockUsage)


class MockAgent:
    """Simulates a claude_agent_sdk.Agent for testing."""

    def __init__(self, result: MockResult | None = None) -> None:
        self._result = result or MockResult()
        self.run = AsyncMock(return_value=self._result)


class FailingAgent:
    """Agent whose run() raises an exception."""

    def __init__(self) -> None:
        self.run = AsyncMock(side_effect=RuntimeError("Claude API error"))


class MinimalResultAgent:
    """Agent returning a result with only output, no metadata attributes."""

    def __init__(self) -> None:
        result = type("Result", (), {"output": "minimal response"})()
        self.run = AsyncMock(return_value=result)


class NoneMetadataAgent:
    """Agent returning a result where metadata fields are None."""

    def __init__(self) -> None:
        result = type(
            "Result", (), {"output": "response", "model": None, "stop_reason": None, "usage": None}
        )()
        self.run = AsyncMock(return_value=result)


# ---------------------------------------------------------------------------
# Adapter base class contract
# ---------------------------------------------------------------------------


class TestClaudeAdapterIsAdapter:
    def test_is_adapter_subclass(self) -> None:
        adapter = ClaudeAdapter(MockAgent())  # type: ignore[arg-type]
        assert isinstance(adapter, Adapter)

    def test_handle_is_async(self) -> None:
        import inspect

        adapter = ClaudeAdapter(MockAgent())  # type: ignore[arg-type]
        assert inspect.iscoroutinefunction(adapter.handle)


# ---------------------------------------------------------------------------
# Adapter.handle() — direct invocation
# ---------------------------------------------------------------------------


class TestHandleDirect:
    @pytest.mark.asyncio
    async def test_returns_output(self) -> None:
        agent = MockAgent()
        adapter = ClaudeAdapter(agent)  # type: ignore[arg-type]

        result = await adapter.handle({"prompt": "What is 2+2?"})

        assert result["output"] == "Hello from Claude!"
        agent.run.assert_awaited_once_with("What is 2+2?")

    @pytest.mark.asyncio
    async def test_returns_metadata(self) -> None:
        agent = MockAgent()
        adapter = ClaudeAdapter(agent)  # type: ignore[arg-type]

        result = await adapter.handle({"prompt": "hi"})

        assert result["metadata"]["model"] == "claude-sonnet-4-20250514"
        assert result["metadata"]["stop_reason"] == "end_turn"
        assert result["metadata"]["usage"]["input_tokens"] == 100
        assert result["metadata"]["usage"]["output_tokens"] == 50

    @pytest.mark.asyncio
    async def test_missing_prompt_defaults_to_empty(self) -> None:
        agent = MockAgent()
        adapter = ClaudeAdapter(agent)  # type: ignore[arg-type]

        await adapter.handle({})

        agent.run.assert_awaited_once_with("")

    @pytest.mark.asyncio
    async def test_extra_keys_ignored(self) -> None:
        agent = MockAgent()
        adapter = ClaudeAdapter(agent)  # type: ignore[arg-type]

        result = await adapter.handle({"prompt": "test", "context": "extra", "user_id": 42})

        assert result["output"] == "Hello from Claude!"
        agent.run.assert_awaited_once_with("test")

    @pytest.mark.asyncio
    async def test_custom_result_values(self) -> None:
        custom = MockResult(
            output="custom reply",
            model="claude-opus-4-20250514",
            stop_reason="max_tokens",
            usage=MockUsage(input_tokens=500, output_tokens=200),
        )
        agent = MockAgent(result=custom)
        adapter = ClaudeAdapter(agent)  # type: ignore[arg-type]

        result = await adapter.handle({"prompt": "tell me everything"})

        assert result["output"] == "custom reply"
        assert result["metadata"]["model"] == "claude-opus-4-20250514"
        assert result["metadata"]["stop_reason"] == "max_tokens"
        assert result["metadata"]["usage"]["input_tokens"] == 500
        assert result["metadata"]["usage"]["output_tokens"] == 200

    @pytest.mark.asyncio
    async def test_minimal_result_no_metadata(self) -> None:
        agent = MinimalResultAgent()
        adapter = ClaudeAdapter(agent)  # type: ignore[arg-type]

        result = await adapter.handle({"prompt": "hi"})

        assert result["output"] == "minimal response"
        assert "metadata" not in result

    @pytest.mark.asyncio
    async def test_none_metadata_fields_excluded(self) -> None:
        agent = NoneMetadataAgent()
        adapter = ClaudeAdapter(agent)  # type: ignore[arg-type]

        result = await adapter.handle({"prompt": "hi"})

        assert result["output"] == "response"
        assert "metadata" not in result

    @pytest.mark.asyncio
    async def test_propagates_exceptions(self) -> None:
        agent = FailingAgent()
        adapter = ClaudeAdapter(agent)  # type: ignore[arg-type]

        with pytest.raises(RuntimeError, match="Claude API error"):
            await adapter.handle({"prompt": "hi"})


# ---------------------------------------------------------------------------
# AgentServer.use() wiring — registration
# ---------------------------------------------------------------------------


class TestUseRegistration:
    def test_registers_as_default_handler(self) -> None:
        server = AgentServer("claude-test", "test agent")
        server.use(ClaudeAdapter(MockAgent()))  # type: ignore[arg-type]

        assert "default" in server.handlers or len(server.handlers) == 1

    def test_registers_as_named_skill(self) -> None:
        server = AgentServer("claude-test", "test agent")
        server.use(ClaudeAdapter(MockAgent()), skill="chat")  # type: ignore[arg-type]

        assert "chat" in server.handlers


# ---------------------------------------------------------------------------
# AgentServer.use() — full HTTP round-trip via TestClient
# ---------------------------------------------------------------------------


class TestHandlePromptNone:
    @pytest.mark.asyncio
    async def test_none_prompt_normalized_to_empty(self) -> None:
        agent = MockAgent()
        adapter = ClaudeAdapter(agent)  # type: ignore[arg-type]

        await adapter.handle({"prompt": None})

        agent.run.assert_awaited_once_with("")


class TestHttpRoundTrip:
    def _make_server(self, agent: Any = None, *, skill: str | None = None) -> AgentServer:
        server = AgentServer("claude-test", "Claude adapter test agent")
        server.use(ClaudeAdapter(agent or MockAgent()), skill=skill)  # type: ignore[arg-type]
        return server

    def test_sync_task_completes(self) -> None:
        server = self._make_server()
        client = server.test_client()

        resp = client.send_task({"prompt": "Hello Claude"})

        assert resp.status == "completed"
        assert resp.result["output"] == "Hello from Claude!"
        assert resp.result["metadata"]["model"] == "claude-sonnet-4-20250514"

    def test_named_skill_dispatch(self) -> None:
        server = self._make_server(skill="chat")
        client = server.test_client()

        resp = client.send_task({"prompt": "Hi"}, skill="chat")

        assert resp.status == "completed"
        assert resp.result["output"] == "Hello from Claude!"

    def test_null_input_normalized(self) -> None:
        server = self._make_server()
        client = server.test_client()

        resp = client.send_task(None)

        assert resp.status == "completed"
        assert resp.result["output"] == "Hello from Claude!"

    def test_adapter_error_returns_failed(self) -> None:
        server = self._make_server(agent=FailingAgent())
        client = server.test_client(raise_server_exceptions=False)

        resp = client.send_task({"prompt": "hi"})

        assert resp.status == "failed"

    def test_async_mode_task_dispatched(self) -> None:
        from starlette.testclient import TestClient

        server = self._make_server()
        client = TestClient(server.app())
        resp = client.post(
            "/",
            json={"id": "t-async-claude", "input": {"prompt": "Hello"}},
            headers={"Prefer": "respond-async"},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["id"] == "t-async-claude"
        assert data["status"] == "submitted"

    def test_metadata_passthrough(self) -> None:
        custom = MockResult(
            output="response",
            model="claude-opus-4-20250514",
            stop_reason="end_turn",
            usage=MockUsage(input_tokens=10, output_tokens=20),
        )
        server = self._make_server(agent=MockAgent(result=custom))
        client = server.test_client()

        resp = client.send_task({"prompt": "test"})

        assert resp.result["metadata"]["model"] == "claude-opus-4-20250514"
        assert resp.result["metadata"]["usage"]["input_tokens"] == 10
        assert resp.result["metadata"]["usage"]["output_tokens"] == 20


# ---------------------------------------------------------------------------
# Agent Card integration
# ---------------------------------------------------------------------------


class TestAgentCard:
    def test_default_adapter_excluded_from_skills(self) -> None:
        server = AgentServer("claude-test", "test")
        server.use(ClaudeAdapter(MockAgent()))  # type: ignore[arg-type]

        client = server.test_client()
        card = client.build_agent_card()
        assert len(card.skills) == 0

    def test_named_adapter_appears_in_skills(self) -> None:
        server = AgentServer("claude-test", "test")
        server.use(ClaudeAdapter(MockAgent()), skill="chat")  # type: ignore[arg-type]

        client = server.test_client()
        card = client.build_agent_card()
        skill_ids = {s.id for s in card.skills}
        assert "chat" in skill_ids
