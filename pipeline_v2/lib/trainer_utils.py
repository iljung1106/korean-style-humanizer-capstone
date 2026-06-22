"""Training helpers that do not depend on legacy project modules."""

from __future__ import annotations

import json
import random
from typing import Any


def parse_report_to(value: str) -> list[str] | str:
    lowered = value.strip().lower()
    if lowered in {"", "none", "no", "false"}:
        return []
    if lowered == "all":
        return "all"
    return [item.strip() for item in value.split(",") if item.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def print_json(prefix: str, payload: dict[str, Any]) -> None:
    print(f"{prefix} {json.dumps(payload, ensure_ascii=False)}", flush=True)

