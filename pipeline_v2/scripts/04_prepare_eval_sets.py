#!/usr/bin/env python3
"""Prepare fixed evaluation prompt sets for pipeline_v2.

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

DEFAULT_GRPO = TRAINING_ROOT / "data" / "processed" / "grpo_mixed_prompts.jsonl"
DEFAULT_RAW = TRAINING_ROOT / "data" / "pipeline_v2" / "cpt_raw_chunks.jsonl"
DEFAULT_SFT = TRAINING_ROOT / "data" / "processed" / "sft_train.jsonl"
DEFAULT_OUTPUT_DIR = TRAINING_ROOT / "data" / "eval_v2"
EVAL_ARTIFACT_RE = re.compile(
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
    "- 요약하거나 해설하지 말고 바로 다음 장면을 이어 쓰세요.\n"
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

FORMAT_STOP_SYSTEM = "당신은 한국 웹소설 작가입니다. 지시된 출력 형식을 엄격히 지킵니다."
FORMAT_STOP_USER = (
    "짧은 한국 웹소설 장면을 작성하세요.\n\n"
    "조건:\n"
    "- 900자에서 1400자 사이\n"
    "- 소설 본문만 작성\n"
    "- 반드시 <result>로 시작해 </result>로 닫기\n"
    "- </result> 이후에는 어떤 글자도 출력하지 않기\n\n"
    "장면 조건: {seed}"
)

FORMAT_STOP_SEEDS = (
    "비 내리는 헌터 아카데미 복도에서 조교가 금지된 소환진을 발견한다.",
    "무림맹 후원 깊은 곳에서 늦게 도착한 서신 한 장 때문에 회의가 멈춘다.",
    "폐허가 된 지하 연구소에서 주인공이 동료의 이름표를 주워 든다.",
    "회귀한 재벌가 막내가 첫 주주총회 직전에 예상 밖의 변수를 만난다.",
    "던전 입구의 편의점에서 야간 아르바이트생이 이상한 손님을 맞는다.",
    "마법학 교수로 위장한 엘프가 한국식 인스턴트 커피를 처음 마신다.",
    "낡은 표국의 연무장에서 새벽 수련을 지켜보던 검사가 결심을 바꾼다.",
    "우주 정거장의 무인 공장에서 관리 AI가 오래된 명령을 다시 읽는다.",
    "도시 외곽의 임시 대피소에서 전직 아이돌 매니저가 생존자 명단을 확인한다.",
    "사라진 황녀의 방 앞에서 호위기사가 문 너머의 발소리를 듣는다.",
    "탑의 37층 휴게실에서 초보 성좌가 잘못 보낸 후원 메시지를 회수하려 한다.",
    "황궁 서고의 봉인된 사다리 아래에서 어린 서기관이 금빛 비늘을 발견한다.",
    "멸망한 서버의 마지막 랭커가 튜토리얼 마을의 잡화점 주인으로 깨어난다.",
    "전쟁 직전의 항구 도시에서 무명 용병이 의뢰인의 진짜 신분을 눈치챈다.",
    "달빛 아래 기루 지붕 위에서 정보상이 비밀 장부를 넘기기 직전 망설인다.",
    "마왕군 보급창에서 말단 병사가 인간 왕국의 편지를 몰래 읽는다.",
    "눈 덮인 북부 성벽에서 후계자 후보 둘이 같은 암살자를 기다린다.",
    "낡은 PC방 심야석에서 주인공이 게임 속 길드 채팅을 현실에서 듣는다.",
    "폐교가 된 마법학교 강의실에서 오래된 칠판 문장이 스스로 바뀐다.",
    "도깨비 시장의 새벽 경매장에서 기억을 담은 유리병이 너무 낮은 값에 팔린다.",
    "성좌 방송 대기실에서 담당 PD가 주인공에게 마지막 경고를 보낸다.",
    "귀환한 검성이 평범한 분식집 앞에서 오래전 제자의 흔적을 찾는다.",
    "지하철 막차 안에서 회귀자가 이번 생에는 놓치지 않겠다고 다짐한다.",
    "사막 던전의 모래폭풍 속에서 길잡이가 일부러 잘못된 방향을 가리킨다.",
    "황태자의 약혼식 직전, 하녀로 위장한 마법사가 초대장을 태워 버린다.",
    "재난 이후의 방송국 옥상에서 생존자 대표가 거짓 구조 신호를 듣는다.",
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


def sample_rows(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    if len(rows) <= count:
        return list(rows)
    return rng.sample(rows, count)


def normalize_prompt_row(row: dict[str, Any], *, prefix: str, index: int) -> dict[str, Any]:
    copied = {
        "id": f"{prefix}-{index:04d}-{row.get('id')}",
        "task": row.get("task"),
        "prompt": row.get("prompt"),
    }
    for key in ("source_text", "reference_text", "max_output_chars"):
        if key in row:
            copied[key] = row[key]
    return copied


def make_continuation_eval_rows(
    raw_rows: list[dict[str, Any]],
    count: int,
    rng: random.Random,
    *,
    context_chars: int,
    reference_chars: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in raw_rows:
        text = str(row.get("text") or "").strip()
        if len(text) < context_chars + reference_chars + 200:
            continue
        context = text[:context_chars].strip()
        reference = text[context_chars : context_chars + reference_chars].strip()
        if len(context) < 400 or len(reference) < 400:
            continue
        if EVAL_ARTIFACT_RE.search(context) or EVAL_ARTIFACT_RE.search(reference):
            continue
        candidates.append(
            {
                "id": f"eval-cont-{row.get('id')}",
                "task": "continuation",
                "prompt": [
                    {"role": "system", "content": SYSTEM_CONTINUATION},
                    {"role": "user", "content": USER_CONTINUATION.format(context=context)},
                ],
                "source_text": context,
                "reference_text": reference,
                "source_row_id": row.get("id"),
                "source_file": row.get("source_file"),
            }
        )
    return sample_rows(candidates, count, rng)


def assistant_text(messages: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        str(item.get("content") or "").strip()
        for item in messages
        if item.get("role") == "assistant" and str(item.get("content") or "").strip()
    )


def make_format_stop_rows(sft_rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seeds = list(FORMAT_STOP_SEEDS)
    rng.shuffle(seeds)
    for index, seed in enumerate(seeds[:count]):
        rows.append(
            {
                "id": f"eval-format-seed-{index:03d}",
                "task": "format_stop",
                "prompt": [
                    {"role": "system", "content": FORMAT_STOP_SYSTEM},
                    {"role": "user", "content": FORMAT_STOP_USER.format(seed=seed)},
                ],
            }
        )
    if len(rows) >= count:
        return rows

    shuffled = list(sft_rows)
    rng.shuffle(shuffled)
    seen_prompts = {json.dumps(row.get("prompt", ""), ensure_ascii=False, sort_keys=True) for row in rows}
    for row in shuffled:
        messages = row.get("messages")
        if not isinstance(messages, list) or not assistant_text(messages):
            continue
        prompt = [dict(item) for item in messages if item.get("role") != "assistant"]
        if not prompt:
            continue
        prompt.append(
            {
                "role": "user",
                "content": "위 요청에 답하되 반드시 <result>와 </result>로 감싸고, </result> 뒤에는 아무것도 쓰지 마세요.",
            }
        )
        key = json.dumps(prompt, ensure_ascii=False, sort_keys=True)
        if key in seen_prompts:
            continue
        seen_prompts.add(key)
        rows.append(
            {
                "id": f"eval-format-sft-{row.get('id')}",
                "task": "format_stop",
                "prompt": prompt,
                "source_row_id": row.get("id"),
            }
        )
        if len(rows) >= count:
            break
    return rows


def prompt_has_korean(row: dict[str, Any]) -> bool:
    text = json.dumps(row.get("prompt", ""), ensure_ascii=False)
    return bool(re.search(r"[\uac00-\ud7a3]", text))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare fixed pipeline_v2 eval prompt JSONL files.")
    parser.add_argument("--grpo-prompts", default=str(DEFAULT_GRPO))
    parser.add_argument("--raw-chunks", default=str(DEFAULT_RAW))
    parser.add_argument("--sft", default=str(DEFAULT_SFT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=240531)
    parser.add_argument("--generate-count", type=int, default=48)
    parser.add_argument("--rewrite-count", type=int, default=48)
    parser.add_argument("--continuation-count", type=int, default=48)
    parser.add_argument("--format-stop-count", type=int, default=24)
    parser.add_argument("--continuation-context-chars", type=int, default=1600)
    parser.add_argument("--continuation-reference-chars", type=int, default=1800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)

    grpo_rows = [row for row in read_jsonl(Path(args.grpo_prompts)) if prompt_has_korean(row)]
    raw_rows = read_jsonl(Path(args.raw_chunks))
    sft_rows = read_jsonl(Path(args.sft))

    generate_rows = [
        normalize_prompt_row(row, prefix="eval-gen", index=index)
        for index, row in enumerate(
            sample_rows([row for row in grpo_rows if row.get("task") == "generate"], args.generate_count, rng)
        )
    ]
    rewrite_rows = [
        normalize_prompt_row(row, prefix="eval-rewrite", index=index)
        for index, row in enumerate(
            sample_rows([row for row in grpo_rows if row.get("task") == "rewrite"], args.rewrite_count, rng)
        )
    ]
    continuation_rows = make_continuation_eval_rows(
        raw_rows,
        args.continuation_count,
        rng,
        context_chars=args.continuation_context_chars,
        reference_chars=args.continuation_reference_chars,
    )
    format_stop_rows = make_format_stop_rows(sft_rows, args.format_stop_count, rng)

    outputs = {
        "fixed_generate_prompts": generate_rows,
        "fixed_rewrite_prompts": rewrite_rows,
        "fixed_continuation_prompts": continuation_rows,
        "fixed_format_stop_prompts": format_stop_rows,
    }
    manifest = {
        "time": time.time(),
        "output_dir": str(output_dir),
        "input": {
            "grpo_prompts": str(args.grpo_prompts),
            "raw_chunks": str(args.raw_chunks),
            "sft": str(args.sft),
        },
        "rows": {},
        "args": vars(args),
    }
    for name, rows in outputs.items():
        path = output_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        manifest["rows"][name] = {"path": str(path), "count": len(rows)}

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
