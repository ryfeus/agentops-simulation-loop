from __future__ import annotations

import pytest

from scripts.trigger_online_failure import attempts_for_candidate


def test_scripted_bad_defaults_to_one_attempt() -> None:
    assert attempts_for_candidate("scripted-bad", None) == 1


def test_bedrock_defaults_to_bounded_retries() -> None:
    assert attempts_for_candidate("bedrock", None) == 5
    assert attempts_for_candidate("bedrock", 2) == 2


def test_attempt_count_must_be_positive() -> None:
    with pytest.raises(ValueError, match="between 1 and 5"):
        attempts_for_candidate("scripted-bad", 0)


def test_scripted_correct_is_rejected_from_failure_trigger() -> None:
    with pytest.raises(ValueError, match="cannot be used"):
        attempts_for_candidate("scripted-correct", None)
