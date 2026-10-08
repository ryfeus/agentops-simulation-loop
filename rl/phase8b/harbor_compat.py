"""Compatibility primitives for TRL's direct Harbor environment lifecycle."""

from __future__ import annotations

from harbor.models.trial.config import ServiceVolumeConfig
from harbor.models.trial.paths import TrialPaths


def trial_log_mounts(paths: TrialPaths) -> list[ServiceVolumeConfig]:
    """Return the Harbor Trial log bind mounts required by Docker environments.

    TRL 1.13 creates ``HarborEnv`` directly instead of through a Harbor Trial.
    Harbor 0.22 Docker environments require this Trial mount list for
    ``/logs/verifier`` to be visible on the host. Phase 8B restores that
    contract until upstream integration handles Trial mounts and verifier
    topology itself.
    """

    return [
        {
            "type": "bind",
            "source": str(paths.verifier_dir.resolve()),
            "target": "/logs/verifier",
        },
        {
            "type": "bind",
            "source": str(paths.agent_dir.resolve()),
            "target": "/logs/agent",
        },
        {
            "type": "bind",
            "source": str(paths.artifacts_dir.resolve()),
            "target": "/logs/artifacts",
        },
    ]
