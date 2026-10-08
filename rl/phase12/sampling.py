"""Explicit independent request seeds for TRL's colocated vLLM backend."""

from __future__ import annotations

import copy
from typing import Any

from rl.phase12.common import fingerprint

SEED_POLICY = "sha256-pass-seed-generation-call-request-index-v1"


def bind_request_seeds(engine: Any, seed: int) -> None:
    """TRL 1.13 seeds its colocated engine with rank zero, ignoring the pass seed.

    A single fixed SamplingParams.seed would make repeated prompts identical.
    Give each request an independent deterministic stream, while leaving native
    generation, weight synchronization, and tool handling untouched.
    """
    original = engine.generate
    generation_call = 0

    def generate(prompts: list[Any], *, sampling_params: Any, **kwargs: Any) -> Any:
        nonlocal generation_call
        if isinstance(sampling_params, list):
            raise ValueError("unexpected upstream per-request sampling; review seed adapter")
        parameters = []
        for index in range(len(prompts)):
            params = copy.deepcopy(sampling_params)
            params.seed = int(fingerprint([seed, generation_call, index])[:8], 16) % (2**31)
            parameters.append(params)
        generation_call += 1
        return original(prompts, sampling_params=parameters, **kwargs)

    engine.generate = generate
