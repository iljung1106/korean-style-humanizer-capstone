#!/usr/bin/env python3
"""Prepare Stage06 generate evaluation sets.

Creates two generate-only eval dirs:
- new_writing: write from scratch prompts
- continuation: continue previous novel text prompts

Both dirs use the phase_eval schema expected by run_generation_vllm.py:
generate_prompts.jsonl, ai_source_controls.jsonl, human_controls.jsonl,
rewrite_prompts.jsonl, manifest.json.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
PHASE_EVAL_ROOT = ROOT / "pipeline_v2" / "eval" / "phase_eval"
if str(PHASE_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE_EVAL_ROOT))

from prepare_eval_sets import ARTIFACT_RE, read_jsonl, write_jsonl  # noqa: E402


DEFAULT_OUT = ROOT / "outputs" / "local_reports" / "stage06b_generate_eval"
DEFAULT_GENERATE_PROMPTS = ROOT / "data" / "processed" / "grpo_generate_prompts.jsonl"
DEFAULT_HUMAN_DIR = ROOT.parents[0] / "data" / "raw" / "human_novels"
DEFAULT_CONTROL_DIR = ROOT / "outputs" / "local_reports" / "stage06b_100_eval" / "eval_v2"

NEW_WRITING_SYSTEM = "당신은 한국 웹소설 작가입니다. 사용자가 제시한 조건으로 바로 읽히는 장르소설 본문을 씁니다."
CONTINUATION_SYSTEM = "당신은 한국 웹소설 작가입니다. 자연스럽고 흡입력 있는 한국어 장르소설 본문을 씁니다."

EXTRA_NEW_WRITING_SPECS = [
    ("현대 판타지", "회귀/방송국", "1인칭 남성 주인공", "생방송 직전", "폐지 직전 예능 PD로 회귀한 전직 스타 연출가", "새벽의 방송국 편집실", "첫 방송을 살리기 위해 망한 코너를 뒤집는 중", "윗선의 방해와 출연자의 돌발 행동을 동시에 처리해야 함", "빠른 판단과 건조한 긴장감", ["편집 타임라인", "대본 뭉치", "꺼진 모니터"]),
    ("무협", "문파재건/복수", "3인칭 남성 주인공 중심", "폐허 진입", "몰락한 검문 문주의 마지막 제자", "비 내리는 산문 앞", "십 년 만에 돌아와 무너진 현판을 다시 세우는 순간", "옛 사형의 배신 흔적을 발견함", "절제된 분노와 무거운 귀환", ["부러진 현판", "녹슨 검", "젖은 도포"]),
    ("로맨스 판타지", "정략혼/궁정암투", "1인칭 여성 주인공", "무도회장 대치", "황태자와 파혼하려는 공작 영애", "샹들리에가 흔들리는 왕궁 연회장", "파혼 선언 직전에 황제가 직접 말을 걸어옴", "도망칠 명분과 가문의 안전을 동시에 지켜야 함", "우아하지만 살얼음 같은 긴장", ["흰 장갑", "와인 잔", "가문의 문장"]),
    ("헌터물", "던전경영/각성", "1인칭 남성 주인공", "상태창 확인", "폐급 던전을 상속받은 무명 헌터", "균열이 새는 낡은 지하 사무실", "던전 코어가 말을 걸기 시작함", "던전을 팔아 빚을 갚을지 직접 키울지 선택해야 함", "막막함 속의 기묘한 기대감", ["금 간 코어", "압류 딱지", "낡은 의자"]),
    ("SF", "우주개척/군상극", "3인칭 여성 주인공 중심", "착륙 사고 직후", "식민선의 임시 선장이 된 항법사", "모래폭풍이 몰아치는 외계 행성", "통신이 끊긴 상태에서 생존자를 점검함", "구조 신호를 보낼 전력을 생명유지 장치에서 빼야 함", "차갑고 실무적인 절박함", ["깨진 헬멧", "산소 게이지", "붉은 모래"]),
    ("아카데미물", "천재/오해", "1인칭 남성 주인공", "시험장 입장", "마법 이론만 외운 줄 알았던 실전 꼴찌", "왕립 아카데미 지하 실습장", "실수로 금지된 고대식을 발동함", "재능을 숨길지 시험을 뒤집을지 결정해야 함", "당황과 계산이 섞인 분위기", ["분필 가루", "마력 측정석", "감독관의 호각"]),
    ("게임 판타지", "NPC빙의/공략", "1인칭 남성 주인공", "퀘스트 수락", "튜토리얼 마을의 잡화상 NPC로 빙의한 고인물", "비 오는 초보자 마을 상점", "첫 플레이어가 엉뚱한 선택지를 고름", "세계 멸망 루트를 막기 위해 규칙 안에서 힌트를 줘야 함", "유쾌하지만 불길한 예감", ["나무 계산대", "초보자 검", "젖은 망토"]),
    ("현대물", "법정/복수", "3인칭 남성 주인공 중심", "증인석 선서", "누명을 쓰고 검사직을 잃은 변호사", "지방법원 3호 법정", "과거 자신을 몰락시킨 증인이 다시 거짓말을 시작함", "증거를 지금 공개하면 더 큰 배후를 놓칠 수 있음", "차분하고 날카로운 압박", ["녹취록", "물컵", "증인 선서문"]),
    ("로맨스", "재회/오피스", "1인칭 여성 주인공", "엘리베이터 안", "전 연인 회사에 파견 온 위기관리 컨설턴트", "퇴근 직전의 고층 빌딩 엘리베이터", "정전으로 둘만 갇힌 상황", "사적인 감정과 회사의 비밀을 분리해야 함", "건조한 대화 속에 남은 감정", ["비상등", "사원증", "멈춘 층수 표시"]),
    ("대체역사", "전쟁/외교", "3인칭 남성 주인공 중심", "밀서 수령", "역사를 알고 있는 젊은 통역관", "개항기 밤의 외교관저", "전쟁을 앞당길 수 있는 오역을 발견함", "진실을 말하면 자신과 가족이 위험해짐", "조용한 공포와 정치적 긴장", ["밀랍 봉인", "양초", "젖은 외투"]),
    ("판타지", "용병/마검", "1인칭 여성 주인공", "계약 협상", "저주받은 마검을 든 전직 성기사", "용병 길드 뒤편 작은 회의실", "터무니없는 호위 의뢰를 받음", "의뢰인이 과거 적국의 왕자임을 알아챔", "비꼼과 경계가 섞인 대화", ["계약서", "검은 장갑", "은화 주머니"]),
    ("현대 판타지", "퇴마/도시괴담", "3인칭 여성 주인공 중심", "현장 조사", "소문난 가짜 무당으로 살던 진짜 퇴마사", "새벽의 폐업한 찜질방", "장난 의뢰인 줄 알았던 사건에서 진짜 흔적을 발견함", "경찰보다 먼저 원혼의 이름을 알아내야 함", "눅눅하고 불쾌한 긴장", ["젖은 수건", "깨진 거울", "녹슨 열쇠"]),
    ("무협", "의원/독공", "1인칭 남성 주인공", "진맥 장면", "독에 내성이 있는 떠돌이 의원", "전염병이 도는 강가 마을", "죽은 줄 알았던 환자가 손목을 붙잡음", "치료하면 추적자가 몰려오지만 외면할 수 없음", "담담한 책임감과 불안", ["약탕기", "검은 핏줄", "대나무 침상"]),
    ("헌터물", "레이드/정치", "3인칭 여성 주인공 중심", "브리핑 도중", "협회에서 밀려난 S급 전략가", "거대 게이트 앞 임시 지휘소", "현장 지휘권을 빼앗긴 채 재난을 예측함", "틀린 명령을 방치할지 모두 앞에서 뒤집을지 선택해야 함", "날카로운 판단과 현장감", ["작전 지도", "무전기", "갈라진 방벽"]),
    ("로맨스 판타지", "회귀/육아물", "1인칭 여성 주인공", "아침 식탁", "폭군이 될 조카를 키우게 된 회귀자 이모", "북부 저택의 긴 식당", "아이가 처음으로 거짓말을 함", "혼내야 할지 숨은 두려움을 먼저 알아봐야 할지 고민함", "따뜻하지만 긴장된 가족극", ["식은 수프", "작은 손", "찢어진 그림"]),
    ("스포츠", "야구/재기", "1인칭 남성 주인공", "불펜 투구", "어깨 부상 후 독립리그로 내려간 전직 유망주", "한여름 지방구장 불펜", "스카우터가 보는 앞에서 마지막 공을 던짐", "구속을 포기하고 제구로 승부해야 함", "땀 냄새 나는 현실감과 절박함", ["낡은 글러브", "로진백", "전광판"]),
    ("현대물", "요리/가족", "3인칭 여성 주인공 중심", "가게 오픈 전", "망한 국밥집을 물려받은 호텔 셰프", "새벽 시장 골목의 작은 가게", "첫 손님이 돌아가신 아버지의 단골임을 알게 됨", "레시피를 바꿀지 원래 맛을 지킬지 결정해야 함", "담백하고 생활감 있는 분위기", ["육수 냄비", "낡은 메뉴판", "비닐 앞치마"]),
    ("판타지", "마탑/정치극", "3인칭 남성 주인공 중심", "청문회 시작", "마탑의 최연소 회계감사관", "대마법사들이 모인 원형 회의장", "장부 속 사라진 마석의 행방을 추궁함", "진범을 알지만 지금 말하면 증거가 사라짐", "지적이고 차가운 압박", ["두꺼운 장부", "푸른 마석", "봉인된 서랍"]),
    ("아포칼립스", "생존/가족", "1인칭 여성 주인공", "차량 고장", "동생을 데리고 피난 중인 정비공", "버려진 고속도로 휴게소", "연료를 찾다가 다른 생존자의 흔적을 발견함", "도움을 요청할지 숨어서 지나갈지 선택해야 함", "건조하고 서늘한 생존감", ["빈 기름통", "유리 조각", "낡은 담요"]),
    ("무협", "암살/잠입", "3인칭 여성 주인공 중심", "지붕 위 대기", "황궁에 숨어든 백발의 암살자", "비 내리는 황도 밤거리", "목표가 예상보다 어린 소년임을 확인함", "명령을 따를지 진실을 확인할지 흔들림", "차갑고 절제된 암살극", ["검은 기와", "젖은 비녀", "짧은 비수"]),
    ("현대 판타지", "재벌/상속전", "1인칭 여성 주인공", "이사회 입장", "사생아로 숨겨져 살던 천재 투자자", "초고층 본사 회의실", "죽은 회장의 유언장이 공개되는 자리", "지분은 얻었지만 모두가 적인 상황", "차분한 독기와 권력 싸움", ["유언장", "녹음기", "차가운 회의탁자"]),
    ("로맨스", "계약연애/연예계", "3인칭 남성 주인공 중심", "촬영장 대기", "스캔들로 추락한 국민 배우", "눈 내리는 야외 세트장", "가짜 연애 상대가 실제 첫사랑임을 알게 됨", "연기를 해야 하는데 감정이 먼저 흔들림", "잔잔하지만 불편한 설렘", ["핫팩", "대본", "흰 숨"]),
    ("판타지", "던전/퍼즐", "1인칭 남성 주인공", "문양 해석", "기억을 잃은 함정 해체사", "고대 유적의 회전하는 석실", "벽화가 자신의 과거를 가리키는 것을 발견함", "출구를 열면 동료가 위험해질 수 있음", "미스터리와 압박감", ["석판", "모래시계", "피 묻은 붕대"]),
    ("무협", "상단/정보전", "1인칭 여성 주인공", "장부 확인", "상단주의 서녀로 태어난 정보상", "밤의 객잔 2층 방", "거래 장부 속에서 반란군의 암호를 발견함", "팔면 큰돈이 되지만 나라가 흔들릴 수 있음", "영리하고 냉정한 계산", ["먹물", "주판", "잠긴 창문"]),
    ("SF", "인공지능/수사", "3인칭 남성 주인공 중심", "취조실 입장", "인간 감정을 학습한 수사 보조 AI의 관리자", "하얀 조명이 강한 미래 경찰청 취조실", "AI가 용의자의 거짓말보다 수사관의 거짓말을 먼저 지적함", "도구로 남길지 증인으로 인정할지 판단해야 함", "차갑고 불편한 윤리적 긴장", ["투명 패널", "음성 기록", "흰 장갑"]),
    ("현대 판타지", "학원/괴담", "1인칭 남성 주인공", "야간 순찰", "귀신을 못 보는 척하는 학생회장", "불 꺼진 고등학교 복도", "방송실에서 자신의 목소리가 흘러나옴", "따라가면 규칙을 어기지만 그냥 두면 누군가 사라짐", "청춘물과 공포의 경계", ["학생증", "삐걱이는 문", "교내 방송"]),
    ("판타지", "왕위계승/전쟁", "3인칭 여성 주인공 중심", "전령 도착", "왕위를 포기한 국경의 공주", "눈 덮인 성벽 위", "수도 함락 소식을 듣고 군사들이 자신을 바라봄", "도망칠 자유와 책임 사이에서 선택해야 함", "웅장하지만 절제된 결단", ["피 묻은 깃발", "얼어붙은 성문", "낡은 왕관"]),
    ("현대물", "의학/응급실", "1인칭 여성 주인공", "응급 콜", "징계 직전의 외상외과 전문의", "새벽 세 시 대학병원 응급실", "자신을 고소한 보호자의 가족이 실려 옴", "개인 감정을 접고 수술실로 들어가야 함", "빠르고 건조한 현장감", ["수술 장갑", "심전도 소리", "핏자국"]),
    ("게임 판타지", "랭커/은퇴", "3인칭 남성 주인공 중심", "로그인 직후", "정체를 숨긴 전 서버 1위 랭커", "폐허가 된 초보자 성문 앞", "계정 삭제 전 마지막 접속에서 과거 길드원이 위기에 처함", "도와주면 정체가 드러나지만 외면할 수 없음", "쓸쓸한 재회와 액션", ["녹슨 창", "길드 문장", "시스템 알림"]),
]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prompt_from_spec(spec: dict[str, Any], index: int) -> dict[str, Any]:
    genre = str(spec["genre"])
    subgenre = str(spec["subgenre"])
    pov = str(spec["point_of_view"])
    opening = str(spec["opening_type"])
    protagonist = str(spec["protagonist"])
    setting = str(spec["setting"])
    situation = str(spec["situation"])
    conflict = str(spec["conflict"])
    mood = str(spec["mood"])
    style_notes = str(spec["style_notes"])
    must_include = spec.get("must_include") or []
    avoid = spec.get("avoid") or ["과도한 설정 설명", "요약식 전개", "메타 발화"]
    target = int(spec.get("target_length_chars") or 2200)
    content = (
        "아래 조건을 바탕으로 한국어 웹소설 본문을 새로 작성하세요.\n\n"
        f"장르: {genre} / {subgenre}\n"
        f"시점: {pov}\n"
        f"도입 방식: {opening}\n"
        f"주인공: {protagonist}\n"
        f"배경: {setting}\n"
        f"상황: {situation}\n"
        f"핵심 갈등: {conflict}\n"
        f"분위기: {mood}\n"
        f"문체 참고: {style_notes}\n"
        f"반드시 포함할 요소: {', '.join(map(str, must_include))}\n"
        f"피할 요소: {', '.join(map(str, avoid))}\n\n"
        "요구사항:\n"
        f"- {target:,}자 안팎으로 쓰되, 최소 1,500자 이상 작성하세요.\n"
        "- 요약, 설정 설명문, 작법 조언, 사과, 메타 발화 없이 소설 본문만 출력하세요.\n"
        "- 한국어 웹소설 독자가 읽는 본문처럼 작성하세요."
    )
    return {
        "id": f"stage06b-new-{index:04d}",
        "task": "generate",
        "prompt_kind": "new_writing",
        "source": "generate_prompt_specs_extended",
        "source_text": "",
        "reference_text": "",
        "prompt": [
            {"role": "system", "content": NEW_WRITING_SYSTEM},
            {"role": "user", "content": content},
        ],
    }


def continuation_prompt(source_text: str, index: int, source_file: str, chunk_id: str) -> dict[str, Any]:
    content = (
        "아래 이전 본문을 자연스럽게 이어 쓰세요.\n\n"
        "요구사항:\n"
        "- 앞 문맥의 시점, 인물 관계, 말투, 장르 분위기를 유지하세요.\n"
        "- 앞 내용을 요약하거나 해설하지 말고, 바로 다음 장면을 이어 쓰세요.\n"
        "- 사과, 설명, 분석, 제목, 목차, 메타 발화를 쓰지 마세요.\n"
        "- 한국어 소설 본문만 출력하세요.\n"
        "- 최소 1,500자 이상 작성하세요.\n\n"
        f"이전 본문:\n{source_text}"
    )
    return {
        "id": f"stage06b-cont-{index:04d}",
        "task": "generate",
        "prompt_kind": "continuation",
        "source": "human_raw_chunk",
        "source_text": source_text,
        "reference_text": "",
        "source_file": source_file,
        "source_chunk_id": chunk_id,
        "prompt": [
            {"role": "system", "content": CONTINUATION_SYSTEM},
            {"role": "user", "content": content},
        ],
    }


def build_new_writing_prompts(path: Path, count: int, seed: int) -> list[dict[str, Any]]:
    source = [row for row in read_jsonl(path) if row.get("task") == "generate" and row.get("prompt_kind") == "new_writing"]
    rows: list[dict[str, Any]] = []
    for row in source:
        out = dict(row)
        out["id"] = f"stage06b-new-{len(rows):04d}-{row.get('id', '')}"
        rows.append(out)
    for spec_tuple in EXTRA_NEW_WRITING_SPECS:
        if len(rows) >= count:
            break
        genre, subgenre, pov, opening, protagonist, setting, situation, conflict, mood, must_include = spec_tuple
        spec = {
            "genre": genre,
            "subgenre": subgenre,
            "point_of_view": pov,
            "opening_type": opening,
            "protagonist": protagonist,
            "setting": setting,
            "situation": situation,
            "conflict": conflict,
            "mood": mood,
            "style_notes": "장면이 실제로 진행되도록 대사와 행동을 섞고, 요약식 설명을 피하세요.",
            "must_include": must_include,
            "avoid": ["과도한 설정 설명", "요약식 전개", "메타 발화"],
            "target_length_chars": 2200,
        }
        rows.append(prompt_from_spec(spec, len(rows)))
    random.Random(seed).shuffle(rows)
    return rows[:count]


def tail_context(text: str, max_chars: int = 3200) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    sliced = text[-max_chars:]
    paragraph_break = sliced.find("\n\n")
    if 0 <= paragraph_break < 600:
        sliced = sliced[paragraph_break + 2 :]
    return sliced.strip()


def build_continuation_prompts_from_controls(human_controls: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    selected = human_controls[:]
    random.Random(seed).shuffle(selected)
    rows: list[dict[str, Any]] = []
    for row in selected:
        if len(rows) >= count:
            break
        text = tail_context(str(row.get("text") or ""))
        if len(text) < 1200 or ARTIFACT_RE.search(text):
            continue
        rows.append(
            continuation_prompt(
                text,
                len(rows),
                str(row.get("source_file") or ""),
                str(row.get("chunk_id") or ""),
            )
        )
    if len(rows) < count:
        raise RuntimeError(f"not enough continuation rows: {len(rows)} < {count}")
    return rows


def load_controls(control_dir: Path, count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reuse existing unpaired control distributions.

    Generate evaluation is not paired. We only need stable human/AI baseline
    distributions, so avoid rescanning raw AI novels here.
    """
    ai = read_jsonl(control_dir / "ai_source_controls.jsonl")[:count]
    human = read_jsonl(control_dir / "human_controls.jsonl")[:count]
    if len(ai) < count or len(human) < count:
        raise RuntimeError(f"not enough controls in {control_dir}: ai={len(ai)} human={len(human)} count={count}")
    return ai, human


def write_eval_dir(out_dir: Path, prompts: list[dict[str, Any]], ai_controls: list[dict[str, Any]], human_controls: list[dict[str, Any]], kind: str, args: argparse.Namespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "generate_prompts.jsonl", prompts)
    write_jsonl(out_dir / "rewrite_prompts.jsonl", [])
    write_jsonl(out_dir / "ai_source_controls.jsonl", ai_controls)
    write_jsonl(out_dir / "human_controls.jsonl", human_controls)
    write_json(
        out_dir / "manifest.json",
        {
            "time": time.time(),
            "kind": kind,
            "args": vars(args),
            "rows": {
                "generate_prompts": len(prompts),
                "rewrite_prompts": 0,
                "ai_source_controls": len(ai_controls),
                "human_controls": len(human_controls),
            },
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--generate-prompts", default=str(DEFAULT_GENERATE_PROMPTS))
    parser.add_argument("--human-novel-dir", default=str(DEFAULT_HUMAN_DIR))
    parser.add_argument("--control-dir", default=str(DEFAULT_CONTROL_DIR))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=260605)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    new_prompts = build_new_writing_prompts(Path(args.generate_prompts), args.count, args.seed)
    ai_controls, human_controls = load_controls(Path(args.control_dir), args.count)
    cont_prompts = build_continuation_prompts_from_controls(human_controls, args.count, args.seed + 1)
    write_eval_dir(output_root / "new_writing_eval_v2", new_prompts, ai_controls, human_controls, "new_writing", args)
    write_eval_dir(output_root / "continuation_eval_v2", cont_prompts, ai_controls, human_controls, "continuation", args)
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "new_writing": len(new_prompts),
                "continuation": len(cont_prompts),
                "ai_controls": len(ai_controls),
                "human_controls": len(human_controls),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
