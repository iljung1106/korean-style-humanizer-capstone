#!/usr/bin/env python3
"""Build a mixed raw-LM / continuation-SFT / format-SFT dataset.

Standalone by design: no imports from existing project scripts.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
TRAINING_ROOT = SCRIPT.parents[2]

DEFAULT_RAW = TRAINING_ROOT / "data" / "pipeline_v2" / "cpt_raw_chunks.jsonl"
DEFAULT_SFT = TRAINING_ROOT / "data" / "processed" / "sft_train.jsonl"
DEFAULT_OUTPUT = TRAINING_ROOT / "data" / "pipeline_v2" / "cpt_mixed_train.jsonl"
DEFAULT_MANIFEST = TRAINING_ROOT / "data" / "pipeline_v2" / "cpt_mixed_train.manifest.json"
CHAPTER_INDEX_RE = re.compile(r"^(?:제\s*)?\d{1,4}\s*[장화](?:[.\s].*)?$")
FORMAT_ARTIFACT_RE = re.compile(
    r"(작품\s*소개|저자\s*소개|장편소설|텍본|스캔본|다운로드|원본\s*링크|https?://|"
    r"무단\s*전재|무단\s*복제|ISBN|[ⓒ©]|"
    r"(?:^|\n)\s*(?:판권|지은이|발행처|출판사|전자책|정가|펴낸곳|펴낸이|등록번호)\s*[:：])",
    re.I,
)

SYSTEM_CONTINUATION = "당신은 한국 웹소설 작가입니다. 자연스럽고 흡입력 있는 한국어 장르소설 본문을 씁니다."
USER_CONTINUATION = (
    "아래 이전 본문을 자연스럽게 이어 쓰세요.\n\n"
    "요구사항:\n"
    "- 앞 문맥의 시점, 인물 관계, 말투, 장르 분위기를 유지하세요.\n"
    "- 요약하거나 해설하지 말고, 바로 다음 장면을 이어 쓰세요.\n"
    "- 사고 과정, 분석, 메모, 시스템 문구는 출력하지 마세요.\n"
    "- 최종 결과물만 <result>와 </result> 사이에 작성하세요.\n"
    "- </result> 이후에는 아무것도 출력하지 마세요.\n\n"
    "문체 지침:\n"
    "- AI가 쓴 글처럼 보이는 과잉 수식, 균일한 문장 리듬, 반복적인 감탄과 강조를 피하세요.\n"
    "- 짧은 문장과 긴 문장을 섞고, 같은 길이와 같은 종결이 반복되지 않게 하세요. 같은 문장 구조를 반복하기보단 다양성을 추구하세요.\n"
    "- 같은 어미가 반복되지 않도록, '-ㅆ다' 말고도 다양한 종결 어미를 사용하세요.\n"
    "- '그 순간', '마치', '압도적인', '미묘한', '운명처럼', '그것은 단순한' 같은 상투적 강조 표현을 반복하지 마세요.\n"
    "- 번역투처럼 보이는 연결 표현이나 대명사의 사용을 줄이세요.\n"
    "- '마치 ~ 같았다'나 '~한 것처럼', '~에 가까웠다' 같은 비유 표현을 줄이세요.\n"
    "- '심장이 쿵 떨어졌다'나 '손끝이 미세하게 떨렸다' 같은 의미가 불명확하고 무의미하게 강조되는 표현을 자제하세요.\n"
    "- 내면 독백은 짧고 날카롭게 쓰고, 같은 판단을 여러 번 풀어 설명하지 마세요.\n"
    "- 추상적인 요약, 작가 메모 등을 포함하지 말고, 실제 웹소설의 일부분처럼 읽힐 수 있도록 쓰세요.\n\n"
    "이전 본문:\n{context}"
)


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


def assistant_text(messages: list[dict[str, Any]]) -> str:
    parts = [str(item.get("content") or "") for item in messages if item.get("role") == "assistant"]
    return "\n\n".join(part for part in parts if part.strip())


def wrap_result_tags(text: str) -> str:
    stripped = text.strip()
    if "<result>" in stripped and "</result>" in stripped:
        return stripped
    return f"<result>\n{stripped}\n</result>"


def result_wrapped_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wrapped: list[dict[str, Any]] = []
    for item in messages:
        copied = dict(item)
        if copied.get("role") == "assistant":
            copied["content"] = wrap_result_tags(str(copied.get("content") or ""))
        wrapped.append(copied)
    return wrapped


def format_artifact_reason(text: str) -> str | None:
    head = text[:4000]
    lines = [line.strip() for line in head.splitlines() if line.strip()]
    chapter_index_lines = sum(1 for line in lines if CHAPTER_INDEX_RE.fullmatch(line))
    if "목차" in head and chapter_index_lines >= 2:
        return "table_of_contents"
    if FORMAT_ARTIFACT_RE.search(head):
        return "front_matter_or_metadata"
    return None


def make_continuation_row(row: dict[str, Any], *, context_chars: int, completion_chars: int) -> dict[str, Any] | None:
    text = str(row.get("text") or "").strip()
    if len(text) < context_chars + 600:
        return None
    context = text[:context_chars].strip()
    completion = text[context_chars : context_chars + completion_chars].strip()
    if len(context) < 400 or len(completion) < 500:
        return None
    if format_artifact_reason(completion) is not None:
        return None
    return {
        "row_type": "continuation_sft",
        "id": f"cont-{row.get('id')}",
        "messages": [
            {"role": "system", "content": SYSTEM_CONTINUATION},
            {"role": "user", "content": USER_CONTINUATION.format(context=context)},
            {"role": "assistant", "content": f"<result>\n{completion}\n</result>"},
        ],
        "source_row_id": row.get("id"),
        "source_file": row.get("source_file"),
        "context_chars": len(context),
        "completion_chars": len(completion),
    }


def make_format_sft_rows(
    sft_rows: list[dict[str, Any]],
    max_rows: int,
    *,
    wrap_result: bool,
    filter_artifacts: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    dropped: dict[str, int] = {}
    for index, row in enumerate(sft_rows):
        messages = row.get("messages")
        if not isinstance(messages, list) or not assistant_text(messages).strip():
            continue
        reason = format_artifact_reason(assistant_text(messages)) if filter_artifacts else None
        if reason is not None:
            dropped[reason] = dropped.get(reason, 0) + 1
            continue
        rows.append(
            {
                "row_type": "format_sft",
                "id": f"format-{row.get('id') or index}",
                "messages": result_wrapped_messages(messages) if wrap_result else messages,
                "source": row.get("source", ""),
                "source_row_id": row.get("id"),
            }
        )
        if max_rows > 0 and len(rows) >= max_rows:
            break
    return rows, dropped


def choose_rows(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    if len(rows) >= count:
        return rng.sample(rows, count)
    return list(rows) + [rng.choice(rows) for _ in range(count - len(rows))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pipeline_v2 mixed CPT/SFT JSONL.")
    parser.add_argument("--raw-chunks", default=str(DEFAULT_RAW))
    parser.add_argument("--format-sft", default=str(DEFAULT_SFT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--total-rows", type=int, default=0, help="If >0, sample rows to match ratios.")
    parser.add_argument("--raw-lm-ratio", type=float, default=0.50)
    parser.add_argument("--continuation-sft-ratio", type=float, default=0.35)
    parser.add_argument("--format-sft-ratio", type=float, default=0.15)
    parser.add_argument("--context-chars", type=int, default=1600)
    parser.add_argument("--completion-chars", type=int, default=2600)
    parser.add_argument("--max-format-sft-rows", type=int, default=0)
    parser.add_argument("--wrap-format-result-tags", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filter-format-artifacts", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    raw_rows = read_jsonl(Path(args.raw_chunks))
    sft_rows = read_jsonl(Path(args.format_sft))

    raw_lm_rows = [
        {
            "row_type": "raw_lm",
            "id": row.get("id"),
            "text": row.get("text"),
            "source_file": row.get("source_file"),
            "source_row_id": row.get("id"),
        }
        for row in raw_rows
        if str(row.get("text") or "").strip()
    ]
    continuation_rows = [
        item
        for item in (
            make_continuation_row(row, context_chars=args.context_chars, completion_chars=args.completion_chars)
            for row in raw_rows
        )
        if item is not None
    ]
    format_rows, format_dropped = make_format_sft_rows(
        sft_rows,
        args.max_format_sft_rows,
        wrap_result=args.wrap_format_result_tags,
        filter_artifacts=args.filter_format_artifacts,
    )

    if args.total_rows > 0:
        raw_count = round(args.total_rows * args.raw_lm_ratio)
        cont_count = round(args.total_rows * args.continuation_sft_ratio)
        format_count = max(0, args.total_rows - raw_count - cont_count)
        rows = (
            choose_rows(raw_lm_rows, raw_count, rng)
            + choose_rows(continuation_rows, cont_count, rng)
            + choose_rows(format_rows, format_count, rng)
        )
    else:
        rows = raw_lm_rows + continuation_rows + format_rows
    rng.shuffle(rows)

    output = Path(args.output)
    manifest = Path(args.manifest)
    write_jsonl(output, rows)
    stats = {
        "time": time.time(),
        "output": str(output),
        "rows": len(rows),
        "available": {
            "raw_lm": len(raw_lm_rows),
            "continuation_sft": len(continuation_rows),
            "format_sft": len(format_rows),
        },
        "dropped": {
            "format_sft": format_dropped,
        },
        "selected": {
            "raw_lm": sum(1 for row in rows if row.get("row_type") == "raw_lm"),
            "continuation_sft": sum(1 for row in rows if row.get("row_type") == "continuation_sft"),
            "format_sft": sum(1 for row in rows if row.get("row_type") == "format_sft"),
        },
        "args": vars(args),
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
