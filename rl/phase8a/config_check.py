"""Exercise the pinned TRL configuration API without model, CUDA, or AWS work."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from train_smoke import build_training_config


def check() -> None:
    with TemporaryDirectory() as temporary:
        output = Path(temporary)
        control = build_training_config(
            mode="control", output=output / "control", vllm_gpu_memory_utilization=0.30
        )
        vllm = build_training_config(
            mode="vllm", output=output / "vllm", vllm_gpu_memory_utilization=0.30
        )
    assert control.use_vllm is False
    assert vllm.use_vllm is True
    assert vllm.vllm_mode == "colocate"
    assert vllm.vllm_tensor_parallel_size == 1
    for config in (control, vllm):
        assert config.max_completion_length == 32
        assert config.num_generations == 2
        assert config.model_init_kwargs["dtype"] == "bfloat16"


if __name__ == "__main__":
    check()
    print("Phase 8A locked TRL configuration: PASS")
