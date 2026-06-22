#!/usr/bin/env python3
"""Stage 5 rewrite-only GRPO trainer for pipeline_v2."""

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
    parser = argparse.ArgumentParser(description="Train pipeline_v2 Stage 5 rewrite-only GRPO.")
    add_common_grpo_args(parser, task="rewrite", default_output_name="stage05_grpo_rewrite")
    return parser.parse_args()


def main() -> None:
    run_grpo_stage(parse_args(), stage="stage05_grpo_rewrite", task="rewrite")


if __name__ == "__main__":
    main()
