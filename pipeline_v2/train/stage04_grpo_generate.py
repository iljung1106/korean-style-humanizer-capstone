#!/usr/bin/env python3
"""Stage 4 generate-only GRPO trainer for pipeline_v2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve()
TRAINING_ROOT = SCRIPT.parents[2]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from pipeline_v2.lib.grpo_training import add_common_grpo_args, run_grpo_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pipeline_v2 Stage 4 generate-only GRPO.")
    add_common_grpo_args(parser, task="generate", default_output_name="stage04_grpo_generate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pause_path = Path(args.output).parent / "PAUSE_BEFORE_STAGE04"
    if pause_path.exists():
        print(f"[stage04/pause] refusing to start because {pause_path} exists", flush=True)
        raise SystemExit(75)
    run_grpo_stage(args, stage="stage04_grpo_generate", task="generate")


if __name__ == "__main__":
    main()
