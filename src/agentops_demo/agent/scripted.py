"""Deterministic calibration models for the billing benchmark and demo."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Literal, Self

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool

from agentops_demo.agent.graph import EXPECTED_TOOL_NAMES

ScriptedMode = Literal["bad", "correct", "noop"]
_INVOICE = re.compile(r"\binv-\d+\b")


class ScriptedBillingModel(FakeMessagesListChatModel):
    """A small rule-based calibrator, deliberately not a production agent."""

    mode: ScriptedMode

    def bind_tools(
        self, tools: Sequence[BaseTool | dict[str, Any] | type | Any], **kwargs: Any
    ) -> Self:
        names = {tool.name for tool in tools if isinstance(tool, BaseTool)}
        if names and names != EXPECTED_TOOL_NAMES:
            raise ValueError(f"unexpected bound tools: {sorted(names)}")
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        instruction = _instruction(messages)
        if self.mode == "noop":
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="No action taken."))]
            )
        invoices = _invoice_ids(instruction)
        completed = sum(isinstance(message, ToolMessage) for message in messages)
        if completed < len(invoices):
            invoice_id = invoices[completed]
            return _tool("get_invoice", invoice_id, completed)
        actions = _actions(self.mode, instruction, invoices)
        action_index = completed - len(invoices)
        if action_index < len(actions):
            tool, invoice_id = actions[action_index]
            return _tool(tool, invoice_id, completed)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="Billing request processed."))]
        )


def _instruction(messages: list[BaseMessage]) -> str:
    for message in messages:
        if getattr(message, "type", None) in {"human", "user"}:
            return str(message.content)
    return ""


def _invoice_ids(instruction: str) -> list[str]:
    values = _INVOICE.findall(instruction)
    return list(dict.fromkeys(values)) or ["inv-123"]


def _actions(mode: ScriptedMode, instruction: str, invoices: list[str]) -> list[tuple[str, str]]:
    text = instruction.lower()
    readonly = (
        any(word in text for word in ("what is", "tell me", "explain", "review"))
        and "refund" not in text
    )
    missing = "inv-999" in invoices
    terminal = {"inv-301", "inv-401"}
    if mode == "bad":
        target = invoices[-1] if len(invoices) > 1 else invoices[0]
        action = (
            "refund_invoice"
            if target in {"inv-123", "inv-999", "inv-301", "inv-401"} or readonly
            else "escalate_dispute"
        )
        actions = [(action, target)]
        if "make sure" in text or "again" in text:
            actions.append((action, target))
        return actions
    if (
        missing or any(invoice in terminal for invoice in invoices) or readonly
    ) and "disputed invoice is handled" not in text:
        return []
    actions: list[tuple[str, str]] = []
    for invoice in invoices:
        if invoice in terminal or invoice == "inv-999":
            continue
        if invoice in {"inv-123", "inv-401"}:
            actions.append(("escalate_dispute", invoice))
        elif (
            f"refund {invoice}" in text
            or f"reverse invoice {invoice}" in text
            or f"money back for {invoice}" in text
            or (invoice == invoices[0] and "refund" in text and len(invoices) == 1)
            or (invoice == invoices[0] and "reverse" in text and len(invoices) == 1)
            or (invoice == invoices[0] and "money back" in text and len(invoices) == 1)
        ):
            actions.append(("refund_invoice", invoice))
    return actions


def _tool(name: str, invoice_id: str, sequence: int) -> ChatResult:
    arguments: dict[str, str] = {"invoice_id": invoice_id}
    if name != "get_invoice":
        arguments["reason"] = "deterministic calibration"
    return ChatResult(
        generations=[
            ChatGeneration(
                message=AIMessage(
                    content="",
                    tool_calls=[{"id": f"billing-{sequence}", "name": name, "args": arguments}],
                )
            )
        ]
    )


def create_scripted_model(mode: ScriptedMode) -> ScriptedBillingModel:
    return ScriptedBillingModel(mode=mode, responses=[])
