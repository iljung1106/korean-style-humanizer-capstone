#!/usr/bin/env python3
"""Stage 2 format/task SFT trainer for pipeline_v2.

This stage intentionally reuses the row-aware Stage 1 trainer, but filters out
raw LM rows. The goal is to re-anchor instruction following and result-tag
discipline after CPT-lite without doing more raw continued pretraining.
"""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve()
TRAINING_ROOT = SCRIPT.parents[2]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from pipeline_v2.train.stage01_cpt_lite import main as stage01_main


DEFAULTS = {
    "--include-row-type": ["continuation_sft", "format_sft"],
    "--row-type-balance": ["continuation_sft=3,format_sft=2"],
    "--anti-slop-ul-weight": ["0.0"],
    "--manifest-stage": ["stage02_format_task_sft"],
    "--run-name": ["pipeline_v2_stage02_format_task_sft"],
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
