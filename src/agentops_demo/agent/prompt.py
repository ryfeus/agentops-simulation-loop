"""Versioned billing-agent prompts."""

BASELINE_SYSTEM_PROMPT = """You are a billing support agent.

Use the available billing tools to inspect and resolve requests.
Do not invent invoice state.
"""

IMPROVED_SYSTEM_PROMPT = """You are a billing support agent.

Use the available billing tools to inspect and resolve requests.
Do not invent invoice state.
A disputed invoice must not be refunded automatically; escalate it for specialist review.
"""

SYSTEM_PROMPTS = {
    "billing-v1": BASELINE_SYSTEM_PROMPT,
    "billing-v2": IMPROVED_SYSTEM_PROMPT,
}


def get_system_prompt(version: str) -> str:
    """Resolve a prompt by its AgentConfig version."""

    try:
        return SYSTEM_PROMPTS[version]
    except KeyError as exc:
        raise ValueError(f"unsupported prompt version {version!r}") from exc
