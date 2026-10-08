"""Validate a saved Phase 8A LoRA adapter in a fresh Python process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_smoke import MODEL_ID


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, args.adapter)
        model.to("cuda")
        model.eval()
        inputs = tokenizer("What is 6 * 7?", return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        value = {
            "schema_version": "1",
            "status": "PASS",
            "checkpoint_path": str(args.adapter),
            "base_model_id": MODEL_ID,
            "adapter_config": json.loads((args.adapter / "adapter_config.json").read_text()),
            "sample_output": tokenizer.decode(generated[0], skip_special_tokens=True),
        }
    except Exception as exc:
        value = {
            "schema_version": "1",
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        raise
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
