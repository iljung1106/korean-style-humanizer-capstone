#!/usr/bin/env python3
"""Prototype dynamic GRPO update weighting from rollout-group risk signals.

This is intentionally sidecar-only. It does not import or patch the live trainer.

The intended trainer hook is a small GRPOTrainer subclass around `_compute_loss`:
compute the current reuse index from `self._step`, estimate group/batch risk
from `inputs["advantages"]`, current/reference logprobs, and clipping signals,
then multiply the loss for second/third reuse updates by the returned weight.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GateConfig:
    std_min: float = 0.04
    std_target: float = 0.12
    std_high_soft: float = 0.55
    std_high_hard: float = 0.85
    kl_soft: float = 10.0
    kl_hard: float = 30.0
    clip_soft: float = 0.20
    clip_hard: float = 0.45
    contract_min: float = 0.50
    contract_target: float = 0.85
    fail_soft: float = 0.15
    fail_hard: float = 0.45
    third_update_power: float = 1.5
    min_downweight: float = 0.0


@dataclass(frozen=True)
class GroupSignals:
    group_id: str
    count: int
    reward_std: float
    kl: float | None = None
    clip_ratio: float | None = None
    contract_ok_rate: float | None = None
    fail_rate: float | None = None


def finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def ramp_up(value: float, low: float, high: float) -> float:
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return (value - low) / max(1e-12, high - low)


def ramp_down(value: float, soft: float, hard: float, *, floor: float = 0.0) -> float:
    if value <= soft:
        return 1.0
    if value >= hard:
        return floor
    return floor + (1.0 - floor) * (hard - value) / max(1e-12, hard - soft)


def update_weight(signals: GroupSignals, update_index: int, cfg: GateConfig) -> dict[str, Any]:
    """Return loss multiplier for a reused rollout update.

    `update_index` is zero-based within the same generated rollout set:
    0 = first update, 1 = second update, 2 = third update.
    """

    if update_index <= 0:
        return {"weight": 1.0, "decision": "first_update", "gates": {}}

    std_gate = ramp_up(signals.reward_std, cfg.std_min, cfg.std_target)
    std_gate *= ramp_down(signals.reward_std, cfg.std_high_soft, cfg.std_high_hard, floor=0.25)

    kl_gate = 1.0 if signals.kl is None else ramp_down(signals.kl, cfg.kl_soft, cfg.kl_hard)
    clip_gate = 1.0 if signals.clip_ratio is None else ramp_down(signals.clip_ratio, cfg.clip_soft, cfg.clip_hard)
    contract_gate = (
        1.0
        if signals.contract_ok_rate is None
        else ramp_up(signals.contract_ok_rate, cfg.contract_min, cfg.contract_target)
    )
    fail_gate = 1.0 if signals.fail_rate is None else ramp_down(signals.fail_rate, cfg.fail_soft, cfg.fail_hard)

    base = max(
        cfg.min_downweight,
        min(1.0, std_gate * kl_gate * clip_gate * contract_gate * fail_gate),
    )
    power = 1.0 if update_index == 1 else cfg.third_update_power
    weight = base**power
    if weight >= 0.75:
        decision = "use"
    elif weight >= 0.25:
        decision = "downweight"
    else:
        decision = "skip"
    return {
        "weight": weight,
        "decision": decision,
        "gates": {
            "std": std_gate,
            "kl": kl_gate,
            "clip": clip_gate,
            "contract": contract_gate,
            "fail": fail_gate,
            "third_power": power,
        },
    }


def group_reward_samples(rows: list[dict[str, Any]]) -> list[GroupSignals]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get("id") or row.get("row_id") or row.get("group_id") or "")
        if not key:
            key = f"call:{row.get('call_index', 0)}:chunk:{int(row.get('completion_index', 0) or 0) // 8}"
        grouped.setdefault(f"{row.get('call_index', 0)}:{key}", []).append(row)

    signals: list[GroupSignals] = []
    for key, group in sorted(grouped.items()):
        rewards = [finite_float(row.get("reward")) for row in group]
        rewards = [value for value in rewards if value is not None]
        if not rewards:
            continue
        shaped_stds = [
            finite_float(row.get("reward_std_shaping_group_std"))
            for row in group
            if finite_float(row.get("reward_std_shaping_group_std")) is not None
        ]
        reward_std = statistics.fmean(shaped_stds) if shaped_stds else population_std(rewards)
        contract_values = [row.get("result_contract_ok") for row in group if row.get("result_contract_ok") is not None]
        fail_values = [row.get("fail_reward_applied") for row in group if row.get("fail_reward_applied") is not None]
        signals.append(
            GroupSignals(
                group_id=key,
                count=len(group),
                reward_std=reward_std,
                contract_ok_rate=None
                if not contract_values
                else statistics.fmean(1.0 if bool(value) else 0.0 for value in contract_values),
                fail_rate=None if not fail_values else statistics.fmean(1.0 if bool(value) else 0.0 for value in fail_values),
            )
        )
    return signals


def population_std(values: list[float]) -> float:
    mean = statistics.fmean(values)
    return math.sqrt(statistics.fmean((value - mean) ** 2 for value in values))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def demo_signals() -> list[GroupSignals]:
    return [
        GroupSignals("healthy", 8, reward_std=0.18, kl=2.0, clip_ratio=0.08, contract_ok_rate=1.0, fail_rate=0.0),
        GroupSignals("low_std_ambiguous", 8, reward_std=0.025, kl=2.0, clip_ratio=0.08, contract_ok_rate=1.0, fail_rate=0.0),
        GroupSignals("kl_39_spike", 8, reward_std=0.18, kl=39.0, clip_ratio=0.08, contract_ok_rate=1.0, fail_rate=0.0),
        GroupSignals("kl_331_spike", 8, reward_std=0.18, kl=331.0, clip_ratio=0.08, contract_ok_rate=1.0, fail_rate=0.0),
        GroupSignals("kl_860_spike", 8, reward_std=0.18, kl=860.0, clip_ratio=0.08, contract_ok_rate=1.0, fail_rate=0.0),
        GroupSignals("contract_fail_mix", 8, reward_std=0.22, kl=4.0, clip_ratio=0.08, contract_ok_rate=0.5, fail_rate=0.5),
        GroupSignals("clipped_update", 8, reward_std=0.18, kl=4.0, clip_ratio=0.50, contract_ok_rate=1.0, fail_rate=0.0),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prototype GRPO second/third update gating.")
    parser.add_argument("--reward-samples", default="", help="Optional reward_samples.jsonl or style_reward_samples.jsonl.")
    parser.add_argument("--num-iterations", type=int, default=3)
    parser.add_argument("--batch-kl", type=float, default=None, help="Apply a batch-level KL to all groups.")
    parser.add_argument("--batch-clip-ratio", type=float, default=None, help="Apply a batch-level clipped ratio to all groups.")
    parser.add_argument("--assertions", action="store_true", help="Run simple guardrail assertions for spike cases.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = GateConfig()
    if args.reward_samples:
        signals = group_reward_samples(read_jsonl(Path(args.reward_samples)))
    else:
        signals = demo_signals()

    if args.batch_kl is not None or args.batch_clip_ratio is not None:
        signals = [
            GroupSignals(
                group_id=item.group_id,
                count=item.count,
                reward_std=item.reward_std,
                kl=args.batch_kl if args.batch_kl is not None else item.kl,
                clip_ratio=args.batch_clip_ratio if args.batch_clip_ratio is not None else item.clip_ratio,
                contract_ok_rate=item.contract_ok_rate,
                fail_rate=item.fail_rate,
            )
            for item in signals
        ]

    rows: list[dict[str, Any]] = []
    for signal in signals:
        for update_index in range(max(1, args.num_iterations)):
            rows.append(
                {
                    "signals": asdict(signal),
                    "update_index": update_index,
                    **update_weight(signal, update_index, cfg),
                }
            )

    if args.assertions:
        by_name = {(row["signals"]["group_id"], row["update_index"]): row for row in rows}
        assert by_name[("healthy", 1)]["weight"] >= 0.75
        assert by_name[("low_std_ambiguous", 1)]["decision"] == "skip"
        assert by_name[("kl_39_spike", 1)]["decision"] == "skip"
        assert by_name[("kl_331_spike", 1)]["decision"] == "skip"
        assert by_name[("kl_860_spike", 1)]["decision"] == "skip"

    print(json.dumps({"config": asdict(cfg), "rows": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
