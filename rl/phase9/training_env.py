"""Private Phase 9 instrumentation over the proven Phase 8C Harbor lifecycle."""

from __future__ import annotations

from rl.phase8c.baseline_env import BaselineBillingHarborEnv


class TrainingBillingHarborEnv(BaselineBillingHarborEnv):
    """No additional policy-visible tools; Phase 9 only changes optimizer state."""
