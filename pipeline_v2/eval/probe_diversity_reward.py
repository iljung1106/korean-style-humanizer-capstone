#!/usr/bin/env python3
"""Probe pipeline_v2 group diversity reward shaping.

This is intentionally dependency-light. If kiwipiepy is installed, the imported
reward code will include POS 4gram similarity; otherwise the probe still checks
character and word ngram behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_v2.lib.grpo_training import apply_generate_diversity_bonus, pairwise_text_similarity_components


TEXT_A = (
    "비는 오래된 처마 끝에서 가느다랗게 흘러내렸다. "
    "주인공은 젖은 골목을 지나며 아직 끝나지 않은 약속을 떠올렸다. "
    "등불은 흔들렸고, 사람들의 낮은 목소리는 안개처럼 번졌다. "
) * 2
TEXT_B = (
    "비는 오래된 처마 끝에서 가느다랗게 흘러내렸다. "
    "주인공은 젖은 골목을 지나며 아직 끝나지 않은 약속을 떠올렸다. "
    "등불은 흔들렸고, 사람들의 낮은 목소리는 안개처럼 번졌다. "
) * 2
TEXT_C = (
    "새벽의 시장은 문을 여는 상인들의 손놀림으로 먼저 깨어났다. "
    "그녀는 아직 식지 않은 국밥 냄새 사이로 편지 봉투를 감추고 걸었다. "
    "멀리서 종소리가 울리자 멈춰 있던 하루가 천천히 움직였다. "
) * 2
TEXT_D = (
    "검은 성벽 위로 모래바람이 치솟았다. "
    "소년은 낡은 창을 쥔 채 이름 없는 별자리를 올려다보았다. "
    "오늘 밤만 지나면 아무도 믿지 않던 예언이 피로 증명될 터였다. "
) * 2


def kiwi_status() -> dict[str, Any]:
    try:
        from pipeline_v2.lib.gui_style_scoring import kiwi_available

        return {"kiwi_available": bool(kiwi_available())}
    except Exception as exc:
        return {"kiwi_available": False, "error": f"{type(exc).__name__}: {exc}"}


def base_entries() -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "row_id": "same_prompt",
            "task": "generate",
            "failed": False,
            "result_text": text,
            "reward": 0.0,
        }
        for index, text in enumerate([TEXT_A, TEXT_B, TEXT_C, TEXT_D])
    ]


def run_mode(mode: str, max_bonus: float, target_distance: float) -> dict[str, Any]:
    args = argparse.Namespace(
        group_diversity_bonus_max=max_bonus,
        group_diversity_mode=mode,
        group_diversity_target_distance=target_distance,
        group_diversity_density_smoothing=0.10,
        group_diversity_min_chars=20,
        num_generations=4,
    )
    entries = deepcopy(base_entries())
    apply_generate_diversity_bonus(entries, args)
    rows = []
    for entry in entries:
        rows.append(
            {
                "index": entry["index"],
                "reward": entry["reward"],
                "diversity_bonus": entry.get("diversity_bonus", 0.0),
                "diversity_distance": entry.get("diversity_distance"),
                "diversity_density": entry.get("diversity_density"),
                "diversity_rarity": entry.get("diversity_rarity"),
                "diversity_loo_contribution": entry.get("diversity_loo_contribution"),
                "diversity_similarity_char": entry.get("diversity_similarity_char"),
                "diversity_similarity_word": entry.get("diversity_similarity_word"),
                "diversity_similarity_content": entry.get("diversity_similarity_content"),
                "diversity_similarity_pos": entry.get("diversity_similarity_pos"),
                "diversity_mmr_quality": entry.get("diversity_mmr_quality"),
                "diversity_mmr_redundancy": entry.get("diversity_mmr_redundancy"),
                "diversity_mmr_contribution": entry.get("diversity_mmr_contribution"),
            }
        )
    bonuses = [float(row["diversity_bonus"]) for row in rows]
    finite_bonuses = [value for value in bonuses if math.isfinite(value)]
    return {
        "mode": mode,
        "rows": rows,
        "sum_bonus": sum(finite_bonuses),
        "max_bonus_abs": max(abs(value) for value in finite_bonuses) if finite_bonuses else 0.0,
        "duplicate_pair_similarity": pairwise_text_similarity_components(TEXT_A, TEXT_B),
        "different_pair_similarity": pairwise_text_similarity_components(TEXT_A, TEXT_D),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["all", "density_adjusted", "dra_density", "leave_one_out", "sgrpo_leave_one_out", "mmr_reweighted", "distance_threshold"],
        default="all",
    )
    parser.add_argument("--max-bonus", type=float, default=0.03)
    parser.add_argument("--target-distance", type=float, default=0.55)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    modes = ["density_adjusted", "leave_one_out", "mmr_reweighted", "distance_threshold"] if args.mode == "all" else [args.mode]
    result = {
        "kiwi": kiwi_status(),
        "settings": {
            "max_bonus": args.max_bonus,
            "target_distance": args.target_distance,
        },
        "modes": [run_mode(mode, args.max_bonus, args.target_distance) for mode in modes],
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    for mode_result in result["modes"]:
        print(f"mode={mode_result['mode']} sum_bonus={mode_result['sum_bonus']:.6f}")
        for row in mode_result["rows"]:
            print(
                f"  idx={row['index']} reward={row['reward']:+.5f} "
                f"bonus={row['diversity_bonus']:+.5f} distance={row['diversity_distance']}"
            )


if __name__ == "__main__":
    main()
