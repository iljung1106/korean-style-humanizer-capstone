"""Row-type balanced sampling for pipeline_v2."""

from __future__ import annotations

import random
from collections import OrderedDict
from typing import Any, Iterable


def parse_row_type_weights(spec: str) -> OrderedDict[str, int]:
    weights: OrderedDict[str, int] = OrderedDict()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid row-type balance item {item!r}; expected row_type=count.")
        name, value = item.split("=", 1)
        name = name.strip()
        count = int(value.strip())
        if not name:
            raise ValueError(f"Invalid row-type balance item {item!r}; empty row_type.")
        if count <= 0:
            raise ValueError(f"Invalid row-type balance item {item!r}; count must be positive.")
        weights[name] = count
    if not weights:
        raise ValueError("Row-type balance spec is empty.")
    return weights


def weighted_round_robin_pattern(weights: OrderedDict[str, int], present_types: Iterable[str]) -> list[str]:
    present = set(present_types)
    remaining = OrderedDict((name, count) for name, count in weights.items() if name in present)
    pattern: list[str] = []
    while any(count > 0 for count in remaining.values()):
        for name in list(remaining):
            if remaining[name] <= 0:
                continue
            pattern.append(name)
            remaining[name] -= 1
    if not pattern:
        raise ValueError("No row types from the balance spec are present in the dataset.")
    return pattern


class BalancedRowTypeSampler:
    """A deterministic replacement sampler that cycles through row-type quotas."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        weights: OrderedDict[str, int],
        *,
        seed: int,
        num_samples: int | None = None,
    ) -> None:
        self.rows = rows
        self.weights = weights
        self.seed = seed
        self.epoch = 0
        self.num_samples = num_samples or len(rows)
        self.groups: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            row_type = str(row.get("row_type") or "unknown")
            self.groups.setdefault(row_type, []).append(index)
        self.pattern = weighted_round_robin_pattern(weights, self.groups)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        pools: dict[str, list[int]] = {}
        offsets: dict[str, int] = {}
        for row_type, indices in self.groups.items():
            shuffled = list(indices)
            rng.shuffle(shuffled)
            pools[row_type] = shuffled
            offsets[row_type] = 0

        for position in range(self.num_samples):
            row_type = self.pattern[position % len(self.pattern)]
            pool = pools[row_type]
            offset = offsets[row_type]
            if offset >= len(pool):
                rng.shuffle(pool)
                offset = 0
            yield pool[offset]
            offsets[row_type] = offset + 1

    def metadata(self, preview_samples: int = 64) -> dict[str, Any]:
        preview_count = min(preview_samples, self.num_samples)
        preview_counts: dict[str, int] = {}
        for row_type in self.pattern:
            preview_counts[row_type] = 0
        for index in range(preview_count):
            row_type = self.pattern[index % len(self.pattern)]
            preview_counts[row_type] = preview_counts.get(row_type, 0) + 1
        return {
            "type": "balanced_row_type",
            "weights": dict(self.weights),
            "present_row_types": {name: len(indices) for name, indices in sorted(self.groups.items())},
            "pattern": self.pattern,
            "num_samples": self.num_samples,
            "preview_samples": preview_count,
            "preview_counts": preview_counts,
        }

