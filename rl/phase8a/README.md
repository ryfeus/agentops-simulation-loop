# Phase 8A isolated runtime

This directory is intentionally an independent uv project. Run it only on the
provisioned G6 host through `make aws-rl-smoke-run`; root CI must never resolve
or install its CUDA dependencies.

Phase 8A requested vLLM 0.29.0, but TRL 1.13.0's published dependency metadata
limits its vLLM extra to 0.28.0. The project therefore pins vLLM 0.28.0; the
lockfile and collected `versions.json` record the resolved runtime.

Before a paid run, use `make rl-smoke-config-check` to construct the control
and colocated-vLLM TRL configurations from this exact lockfile. The vLLM extra
is deliberately selected only by the remote G6 bootstrap, so this local gate
checks the locked TRL/PEFT API without downloading model weights or installing
NVIDIA-only wheels.
