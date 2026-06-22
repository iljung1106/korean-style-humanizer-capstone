#!/usr/bin/env python3
"""Stage 8B generate-heavy SimPO-CPO from the Stage 8A SFT anchor."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve()
TRAINING_ROOT = SCRIPT.parents[2]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from pipeline_v2.train.stage03_simpo_curriculum import main as stage03_main


DEFAULTS = {
    "--dataset": [str(TRAINING_ROOT / "data" / "pipeline_v2" / "simpo_curriculum" / "stage08_generate_heavy.jsonl")],
    "--output": [str(TRAINING_ROOT / "outputs" / "pipeline_v2" / "stage08b_simpo_cpo_from_stage08a")],
    "--adapter-path": [str(TRAINING_ROOT / "outputs" / "pipeline_v2" / "stage08a_sft_anchor_from_stage07i" / "policy")],
    "--preference-loss": ["simpo"],
    "--learning-rate": ["7e-7"],
    "--beta": ["2.2"],
    "--simpo-gamma": ["1.0"],
    "--cpo-alpha": ["0.22"],
    "--max-grad-norm": ["0.3"],
    "--max-steps": ["-1"],
    "--num-train-epochs": ["2.0"],
    "--batch-size": ["1"],
    "--grad-accum": ["8"],
    "--max-seq-length": ["4096"],
    "--max-prompt-length": ["2048"],
    "--save-steps": ["50"],
    "--save-total-limit": ["2"],
    "--run-name": ["pipeline_v2_stage08b_simpo_cpo_from_stage08a"],
}


def has_arg(argv: list[str], name: str) -> bool:
    return any(item == name or item.startswith(name + "=") for item in argv)


def main() -> None:
    argv = list(sys.argv[1:])
    injected: list[str] = []
    for key, values in DEFAULTS.items():
        if has_arg(argv, key):
            continue
        for value in values:
            injected.extend([key, value])
    sys.argv = [sys.argv[0], *injected, *argv]
    stage03_main()


if __name__ == "__main__":
    main()
