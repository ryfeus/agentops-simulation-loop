from __future__ import annotations

from typing import Any

from scripts.chat_agentcore import run_chat


def test_interactive_chat_reuses_one_session_and_handles_commands(agent_config) -> None:
    inputs = iter(("first message", "second message", "/tools", "/info", "/exit"))
    output: list[str] = []
    calls: list[dict[str, Any]] = []

    def invoke(instruction: str, *, session_id: str, candidate: str) -> tuple[dict[str, Any], str]:
        calls.append({"instruction": instruction, "session_id": session_id, "candidate": candidate})
        return (
            {
                "final_response": f"reply to {instruction}",
                "completed_tools": ["get_invoice"],
            },
            session_id,
        )

    assert (
        run_chat(
            config=agent_config,
            invoke_fn=invoke,
            input_fn=lambda _prompt: next(inputs),
            output_fn=output.append,
            session_id_factory=lambda: "chat-session",
        )
        == 0
    )
    assert calls == [
        {"instruction": "first message", "session_id": "chat-session", "candidate": "bedrock"},
        {"instruction": "second message", "session_id": "chat-session", "candidate": "bedrock"},
    ]
    assert "agent> reply to first message" in output
    assert "agent> reply to second message" in output
    assert output.count("tools> get_invoice") == 3
    assert any(line.startswith("Session: chat-session") for line in output)
    assert "Candidate: bedrock" in output
    assert any("Last trace: not queried" in line for line in output)
