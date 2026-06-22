#!/usr/bin/env python3
"""Prepare phase-end rewrite/generate evaluation sets.

The generated files are stable for a given seed and do not depend on legacy GUI
modules. Source/control chunks are built from raw text using paragraph-aware
chunking from `metrics_gui_compatible`.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
PHASE_EVAL_ROOT = SCRIPT.parent
PIPELINE_ROOT = SCRIPT.parents[2]
TRAINING_ROOT = SCRIPT.parents[3]
REPO_ROOT = SCRIPT.parents[4]
if str(PHASE_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE_EVAL_ROOT))
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from metrics_gui_compatible import hangul_ratio, paragraph_chunks, read_text
from pipeline_v2.lib.preference_data import (
    add_generate_length_instruction,
    add_generate_style_guidance_instruction,
    add_result_contract_instruction,
    add_rewrite_style_guidance_instruction,
    apply_system_prompt_variant,
    as_chat_messages,
)


DEFAULT_AI_DIR = REPO_ROOT / "data" / "raw" / "ai_novels"
DEFAULT_HUMAN_DIR = REPO_ROOT / "data" / "raw" / "human_novels"
DEFAULT_GENERATE_PROMPTS = TRAINING_ROOT / "data" / "processed" / "grpo_generate_prompts.jsonl"
DEFAULT_OUTPUT_DIR = TRAINING_ROOT / "data" / "phase_eval_v2"

ARTIFACT_RE = re.compile(
    r"(작품\s*소개|저자\s*소개|장편소설|텍본|스캔본|다운로드|원본\s*링크|https?://|"
    r"무단\s*전재|무단\s*복제|ISBN|[ⓒ©]|"
    r"(?:^|\n)\s*(?:판권|지은이|발행처|출판사|전자책|정가|펴낸곳|펴낸이|등록번호)\s*[:：])",
    re.I,
)

REWRITE_SYSTEM = "당신은 한국 웹소설 문체 교정 작가입니다. AI가 쓴 듯한 문장을 자연스러운 한국 웹소설 본문으로 고칩니다."
REWRITE_USER = "아래 원문을 자연스러운 한국 웹소설 본문으로 다시 쓰세요.\n\n원문:\n{source_text}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def text_files(root: Path) -> list[Path]:
    by_stem: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
            continue
        if path.stem.lower() in {"all_stories", "all_stories.txt"}:
            continue
        key = str(path.with_suffix(""))
        previous = by_stem.get(key)
        if previous is None or (previous.suffix.lower() != ".txt" and path.suffix.lower() == ".txt"):
            by_stem[key] = path
    return sorted(by_stem.values())


def build_chunk_candidates(
    root: Path,
    *,
    group: str,
    target_chars: int,
    min_chars: int,
    max_chars: int,
    min_hangul_ratio: float,
    max_chunks_per_file: int,
    limit_files: int = 0,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    files = text_files(root)
    if limit_files > 0:
        files = files[:limit_files]
    for path in files:
        try:
            text = read_text(path)
        except Exception:
            continue
        pieces = paragraph_chunks(text, target_chars=target_chars, min_chars=min_chars, max_chars=max_chars)
        kept = 0
        for index, piece in enumerate(pieces):
            if kept >= max_chunks_per_file:
                break
            if hangul_ratio(piece) < min_hangul_ratio:
                continue
            if ARTIFACT_RE.search(piece):
                continue
            candidates.append(
                {
                    "id": f"{group}-{len(candidates):06d}",
                    "group": group,
                    "source_file": str(path),
                    "chunk_id": f"{path.name}:{index}",
                    "text": piece,
                    "char_len": len(piece),
                    "hangul_ratio": hangul_ratio(piece),
                }
            )
            kept += 1
    return candidates


def sample_rows(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    if len(rows) <= count:
        copied = list(rows)
    else:
        copied = rng.sample(rows, count)
    copied.sort(key=lambda row: str(row.get("id", "")))
    return copied


def make_rewrite_rows(ai_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(ai_chunks):
        source_text = str(row["text"])
        variant_seed = f"phase-rewrite-{index:04d}:{row.get('id', '')}"
        prompt = [
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": REWRITE_USER.format(source_text=source_text)},
        ]
        prompt = apply_system_prompt_variant(prompt, task="rewrite", variant_seed=variant_seed)
        prompt = add_result_contract_instruction(prompt)
        prompt = add_rewrite_style_guidance_instruction(prompt, variant_seed=variant_seed, shuffle_bullets=True)
        rows.append(
            {
                "id": f"phase-rewrite-{index:04d}",
                "task": "rewrite",
                "source_text": source_text,
                "reference_text": "",
                "source_file": row.get("source_file", ""),
                "source_chunk_id": row.get("chunk_id", ""),
                "prompt": prompt,
            }
        )
    return rows


def make_control_rows(rows: list[dict[str, Any]], *, group: str) -> list[dict[str, Any]]:
    return [
        {
            "id": f"{group}-{index:04d}",
            "group": group,
            "source_file": row.get("source_file", ""),
            "chunk_id": row.get("chunk_id", ""),
            "text": row.get("text", ""),
            "char_len": row.get("char_len", 0),
            "hangul_ratio": row.get("hangul_ratio", 0.0),
        }
        for index, row in enumerate(rows)
    ]


def make_generate_rows(rows: list[dict[str, Any]], count: int, rng: random.Random, prompt_kind: str) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row.get("task") == "generate" and (not prompt_kind or row.get("prompt_kind") == prompt_kind)
    ]
    selected = sample_rows(candidates, count, rng)
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        task = "continuation" if row.get("prompt_kind") == "continuation" else "generate"
        variant_seed = f"phase-generate-{index:04d}:{row.get('id', '')}"
        prompt = as_chat_messages(row.get("prompt"))
        prompt = apply_system_prompt_variant(prompt, task=task, variant_seed=variant_seed)
        prompt = add_result_contract_instruction(prompt)
        prompt = add_generate_length_instruction(
            prompt,
            int(row.get("max_output_chars") or 0),
            3000 if task == "generate" else 0,
        )
        prompt = add_generate_style_guidance_instruction(prompt, variant_seed=variant_seed, shuffle_bullets=True)
        out = {
            "id": f"phase-generate-{index:04d}-{row.get('id', '')}",
            "task": "generate",
            "prompt_kind": row.get("prompt_kind", ""),
            "prompt": prompt,
            "source_text": row.get("source_text", ""),
            "reference_text": row.get("reference_text", ""),
            "source": row.get("source", ""),
        }
        if "max_output_chars" in row:
            out["max_output_chars"] = row["max_output_chars"]
        normalized.append(out)
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare phase-end style eval sets.")
    parser.add_argument("--ai-novel-dir", default=str(DEFAULT_AI_DIR))
    parser.add_argument(
        "--ai-control-dir",
        default="",
        help="Optional separate AI control corpus. Defaults to --ai-novel-dir.",
    )
    parser.add_argument("--human-novel-dir", default=str(DEFAULT_HUMAN_DIR))
    parser.add_argument("--generate-prompts", default=str(DEFAULT_GENERATE_PROMPTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=260601)
    parser.add_argument("--rewrite-count", type=int, default=30)
    parser.add_argument("--generate-count", type=int, default=30)
    parser.add_argument("--control-count", type=int, default=30)
    parser.add_argument("--generate-prompt-kind", default="new_writing")
    parser.add_argument("--target-chars", type=int, default=5000)
    parser.add_argument("--min-chars", type=int, default=2500)
    parser.add_argument("--max-chars", type=int, default=7000)
    parser.add_argument("--min-hangul-ratio", type=float, default=0.45)
    parser.add_argument("--max-chunks-per-file", type=int, default=2)
    parser.add_argument("--limit-ai-files", type=int, default=0, help="Smoke-test limit; 0 means all files.")
    parser.add_argument("--limit-human-files", type=int, default=0, help="Smoke-test limit; 0 means all files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    ai_control_dir = Path(args.ai_control_dir) if args.ai_control_dir else Path(args.ai_novel_dir)

    ai_candidates = build_chunk_candidates(
        Path(args.ai_novel_dir),
        group="ai_source",
        target_chars=args.target_chars,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        min_hangul_ratio=args.min_hangul_ratio,
        max_chunks_per_file=args.max_chunks_per_file,
        limit_files=args.limit_ai_files,
    )
    ai_control_candidates = (
        ai_candidates
        if ai_control_dir == Path(args.ai_novel_dir)
        else build_chunk_candidates(
            ai_control_dir,
            group="ai_source",
            target_chars=args.target_chars,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            min_hangul_ratio=args.min_hangul_ratio,
            max_chunks_per_file=args.max_chunks_per_file,
            limit_files=args.limit_ai_files,
        )
    )
    human_candidates = build_chunk_candidates(
        Path(args.human_novel_dir),
        group="human_control",
        target_chars=args.target_chars,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        min_hangul_ratio=args.min_hangul_ratio,
        max_chunks_per_file=args.max_chunks_per_file,
        limit_files=args.limit_human_files,
    )
    generate_source_rows = read_jsonl(Path(args.generate_prompts))

    selected_ai = sample_rows(ai_candidates, args.rewrite_count, rng)
    selected_ai_control = sample_rows(ai_control_candidates, args.control_count, rng)
    selected_human = sample_rows(human_candidates, args.control_count, rng)
    rewrite_rows = make_rewrite_rows(selected_ai)
    generate_rows = make_generate_rows(generate_source_rows, args.generate_count, rng, args.generate_prompt_kind)
    ai_control_rows = make_control_rows(selected_ai_control, group="ai_source")
    human_control_rows = make_control_rows(selected_human, group="human_control")

    outputs = {
        "rewrite_prompts": rewrite_rows,
        "generate_prompts": generate_rows,
        "ai_source_controls": ai_control_rows,
        "human_controls": human_control_rows,
    }
    for name, rows in outputs.items():
        write_jsonl(output_dir / f"{name}.jsonl", rows)

    manifest = {
        "time": time.time(),
        "output_dir": str(output_dir),
        "inputs": {
            "ai_novel_dir": str(Path(args.ai_novel_dir)),
            "ai_control_dir": str(ai_control_dir),
            "human_novel_dir": str(Path(args.human_novel_dir)),
            "generate_prompts": str(Path(args.generate_prompts)),
        },
        "args": vars(args),
        "candidate_counts": {
            "ai": len(ai_candidates),
            "ai_control": len(ai_control_candidates),
            "human": len(human_candidates),
            "generate": len(generate_source_rows),
        },
        "rows": {name: {"path": str(output_dir / f"{name}.jsonl"), "count": len(rows)} for name, rows in outputs.items()},
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
