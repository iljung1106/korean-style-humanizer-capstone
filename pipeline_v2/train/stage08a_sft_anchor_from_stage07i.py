#!/usr/bin/env python3
"""Stage 8A SFT anchor from Stage 7I, without unlikelihood or CPT rows."""

from __future__ import annotations

import sys
from pathlib import Path
import os


SCRIPT = Path(__file__).resolve()
TRAINING_ROOT = SCRIPT.parents[2]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from pipeline_v2.train.stage01_cpt_lite import main as stage01_main


DEFAULTS = {
    "--dataset": [str(TRAINING_ROOT / "data" / "pipeline_v2" / "cpt_mixed_probe.jsonl")],
    "--output": [str(TRAINING_ROOT / "outputs" / "pipeline_v2" / "stage08a_sft_anchor_from_stage07i")],
    "--adapter-path": [
        os.environ.get(
            "STAGE07I_ADAPTER_PATH",
            str(TRAINING_ROOT / "outputs" / "pipeline_v2" / "stage07i" / "policy"),
        )
    ],
    "--include-row-type": ["continuation_sft", "format_sft"],
    "--row-type-balance": ["continuation_sft=3,format_sft=2"],
    "--anti-slop-ul-weight": ["0.0"],
    "--max-steps": ["-1"],
    "--num-train-epochs": ["1.15"],
    "--learning-rate": ["6e-7"],
    "--max-grad-norm": ["0.3"],
    "--save-steps": ["50"],
    "--save-total-limit": ["2"],
    "--manifest-stage": ["stage08a_sft_anchor_from_stage07i"],
    "--run-name": ["pipeline_v2_stage08a_sft_anchor_from_stage07i"],
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
    stage01_main()


if __name__ == "__main__":
    main()
