from __future__ import annotations

import csv
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


CAPSTONE_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = CAPSTONE_ROOT.parent
TRAINING_ROOT = WORKSPACE_ROOT / "gemma4_style_rl_training"

if str(CAPSTONE_ROOT) not in sys.path:
    sys.path.insert(0, str(CAPSTONE_ROOT))

from pipeline_v2.lib import gui_style_scoring as style_scoring  # noqa: E402


DEFAULT_VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "https://lower-sector-renaissance-workstation.trycloudflare.com")
DEFAULT_MODEL = os.getenv("VLLM_MODEL", "gemma4-webnovel-stage08b")
REFERENCE_PATH = Path(
    os.getenv(
        "STYLE_REFERENCE_PATH",
        str(TRAINING_ROOT / "data" / "processed" / "gui_style_reward_reference_stage06.json"),
    )
)
ANTI_SLOP_LEXICON_PATH = Path(
    os.getenv(
        "ANTI_SLOP_LEXICON_PATH",
        str(TRAINING_ROOT / "data" / "processed" / "anti_slop_lexicon.json"),
    )
)
METRIC_CSV_PATH = CAPSTONE_ROOT / "data" / "metrics" / "stage06b_rewrite_filtered_metrics_with_explanations.csv"

RESULT_OPEN_TAG = "<result>"
RESULT_CLOSE_TAG = "</result>"
REWRITE_SYSTEM_PROMPT = (
    "당신은 한국 웹소설 문체 교정 작가입니다. AI가 쓴 듯한 문장을 자연스러운 한국 웹소설 본문으로 고칩니다."
)
COMMON_STYLE_GUIDANCE_BULLETS = (
    "같은 구조의 문장이 세 개 이상 연속되거나, 감정 어휘가 한 문단에 밀집되거나, 형용사·부사가 중첩 수식되는 패턴을 피하세요.",
    "동일한 서술어, 감정어 및 내용어가 너무 잦게 나오지 않도록 다양한 표현을 섞어 사용하세요. 꼭 필요한 경우가 아니라면 의미가 달라지지 않는 선에서 표현을 조정하세요.",
    "유사한 길이의 문장을 연속하지 마세요. 긴 호흡과 짧은 호흡의 문장을 섞어 사용하세요. 글 내에 충분히 다양한 길이의 문장이 있어야 합니다.",
    "같은 종결 어미가 반복되지 않도록 '-ㅆ다' 외에도 '-까?', '-지', '-나', 명사형 등 다양한 종결 방식을 문맥에 맞게 사용하세요.",
    "'그 순간', '마치', '압도적인', '미묘한', '운명처럼', '그것은 단순한' 같은 상투적 강조 표현을 반복하지 마세요.",
    "'그것', '그', '그녀', '그들' 같은 대명사가 반복되지 않도록, 구체적인 대상 이름이나 자연스러운 생략을 사용하세요.",
    "'마치 ~같았다', '~한 것처럼', '~에 가까웠다' 같은 비유 표현을 줄이세요.",
    "의미가 불명확한 신체 반응 묘사나 과장된 감정 강조를 자제하세요.",
    "요약, 작가 메모 등을 포함하지 말고, 실제 연재분 발췌와 같은 형태로 본문만 작성하세요.",
)


class ScoreRequest(BaseModel):
    text: str = Field(..., min_length=1)


class HumanizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    temperature: float = Field(0.65, ge=0.0, le=1.5)
    top_p: float = Field(0.9, ge=0.05, le=1.0)
    max_tokens: int = Field(1800, ge=128, le=8192)
    model: str | None = None
    base_url: str | None = None


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def pct(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return round(max(0.0, min(100.0, value)), 1)


def load_metric_rows() -> list[dict[str, Any]]:
    if not METRIC_CSV_PATH.exists():
        return []
    with METRIC_CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


METRIC_ROWS = load_metric_rows()
SCORER = style_scoring.GuiStyleScorer(
    reference_path=REFERENCE_PATH,
    anti_slop_lexicon_path=ANTI_SLOP_LEXICON_PATH,
    translationese_model_path=Path(
        os.getenv(
            "TRANSLATIONESE_MODEL_PATH",
            str(TRAINING_ROOT / "models" / "translationese_svm" / "svm_detector.joblib"),
        )
    ),
)


def parenthesis_pair_per_1k_chars(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0
    left_count = sum(stripped.count(ch) for ch in "([{（［｛")
    right_count = sum(stripped.count(ch) for ch in ")]}）］｝")
    return 1000.0 * min(left_count, right_count) / max(1, len(stripped))


def ai_likeness_for_metric(value: float, stats: dict[str, Any]) -> float:
    anti = style_scoring.anti_ai_score(value, stats)
    if not math.isfinite(anti):
        return float("nan")
    return pct((1.0 - anti) * 50.0)


def human_band_label(value: float, stats: dict[str, Any]) -> str:
    q25 = finite_float(stats.get("human_q25"))
    q75 = finite_float(stats.get("human_q75"))
    q10 = finite_float(stats.get("human_q10"))
    q90 = finite_float(stats.get("human_q90"))
    if math.isfinite(q25) and math.isfinite(q75) and q25 <= value <= q75:
        return "human_iqr"
    if math.isfinite(q10) and math.isfinite(q90) and q10 <= value <= q90:
        return "human_band"
    return "outside"


def metric_weight(row: dict[str, Any], stats: dict[str, Any] | None) -> float:
    from_row = abs(finite_float(row.get("human_ai_cliffs_delta"), 0.0))
    if from_row > 0:
        return from_row
    if stats:
        return abs(finite_float(stats.get("weight"), 0.2))
    return 0.05


def enrich_score(raw_score: dict[str, Any]) -> dict[str, Any]:
    result_text = str(raw_score.get("result_text") or "")
    metrics = dict(raw_score.get("metrics") or {})
    metrics["parenthesis_pair_per_1k_chars"] = parenthesis_pair_per_1k_chars(result_text)
    details: list[dict[str, Any]] = []
    weighted_ai_parts: list[tuple[float, float]] = []

    for row in METRIC_ROWS:
        name = str(row.get("metric") or "").strip()
        if not name:
            continue
        value = finite_float(metrics.get(name))
        stats = SCORER.metrics_reference.get(name)
        ai_likeness = ai_likeness_for_metric(value, stats) if stats and math.isfinite(value) else float("nan")
        if math.isfinite(ai_likeness):
            weighted_ai_parts.append((ai_likeness, metric_weight(row, stats)))
        details.append(
            {
                "metric": name,
                "metric_ko": row.get("metric_ko") or name,
                "presentation_note": row.get("presentation_note") or "",
                "description": row.get("description") or "",
                "value": round(value, 6) if math.isfinite(value) else None,
                "ai_likeness": ai_likeness if math.isfinite(ai_likeness) else None,
                "status": human_band_label(value, stats) if stats and math.isfinite(value) else "reference_missing",
                "improved_in_eval": str(row.get("improved")).lower() == "true",
                "gap_closure": finite_float(row.get("gap_closure")),
                "human_mean": finite_float(stats.get("human_mean")) if stats else None,
                "ai_mean": finite_float(stats.get("ai_mean")) if stats else None,
                "human_q25": finite_float(stats.get("human_q25")) if stats else None,
                "human_q75": finite_float(stats.get("human_q75")) if stats else None,
            }
        )

    if weighted_ai_parts:
        overall_ai = sum(value * weight for value, weight in weighted_ai_parts) / sum(weight for _value, weight in weighted_ai_parts)
    else:
        raw_gui_score = finite_float(raw_score.get("score"), 0.0)
        overall_ai = (1.0 - raw_gui_score) * 50.0

    family_scores = {
        key: pct((1.0 - finite_float(value, 0.0)) * 50.0)
        for key, value in (raw_score.get("family_scores") or {}).items()
    }
    return {
        "ai_likeness": pct(overall_ai),
        "human_likeness": pct(100.0 - overall_ai),
        "raw_gui_score": round(finite_float(raw_score.get("score")), 6),
        "result_text": result_text,
        "metrics": metrics,
        "metric_details": details,
        "family_ai_likeness": family_scores,
        "scorer_status": raw_score.get("scorer_status") or {},
    }


def score_text(text: str, *, require_result_tags: bool = False) -> dict[str, Any]:
    raw = SCORER.score_text(text, require_result_tags=require_result_tags)
    return enrich_score(raw)


def latest_rewrite_user_prompt(source: str) -> str:
    contract = (
        "출력 형식:\n"
        f"- 최종 결과물만 {RESULT_OPEN_TAG}와 {RESULT_CLOSE_TAG} 사이에 작성하세요.\n"
        f"- {RESULT_CLOSE_TAG} 이후에는 아무것도 출력하지 마세요.\n"
        f"- {RESULT_OPEN_TAG} 안에는 최종 본문만 넣고, 분석, 메모, 시스템 문구는 넣지 마세요."
    )
    guidance = (
        "재작성 목표:\n"
        "- 사건 순서, 인물 관계, 설정 정보, 장면의 핵심 의미는 유지하세요.\n"
        "- 단순 교정이나 일부 단어 치환에 그치지 말고, 문장 구조와 서술 호흡을 충분히 고치세요. 목표는 주어진 본문과 비교했을 때 더 몰입이 되는 글을 만드는 것입니다.\n"
        "- 원문을 짧게 요약하거나 과도하게 늘리지 말고, 원문 분량의 85~115% 수준을 유지하세요. 원문이 부적절한 경우에는 범위 내에서 조정해도 좋습니다.\n\n"
        "문체 지침:\n"
        + "\n".join(f"- {bullet}" for bullet in COMMON_STYLE_GUIDANCE_BULLETS)
    )
    return (
        "아래 원문을 자연스러운 한국 웹소설 본문으로 다시 쓰세요.\n\n"
        f"{contract}\n\n"
        f"{guidance}\n\n"
        f"원문:\n{source}"
    ).strip()


def strip_result_tags(text: str) -> str:
    match = re.search(r"<result>(.*?)</result>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


async def call_vllm(request: HumanizeRequest) -> dict[str, Any]:
    base_url = (request.base_url or DEFAULT_VLLM_BASE_URL).rstrip("/")
    model = request.model or DEFAULT_MODEL
    messages = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": latest_rewrite_user_prompt(request.text)},
    ]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens,
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=20.0)) as client:
            response = await client.post(f"{base_url}/v1/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"vLLM 호출 실패: {exc}") from exc

    try:
        content = data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"vLLM 응답 형식 오류: {data}") from exc
    return {"raw_output": str(content), "messages": messages, "model": model, "base_url": base_url}


app = FastAPI(title="Korean Webnovel Style Humanizer UI", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model": DEFAULT_MODEL,
        "vllm_base_url": DEFAULT_VLLM_BASE_URL,
        "reference_exists": REFERENCE_PATH.exists(),
        "anti_slop_lexicon_exists": ANTI_SLOP_LEXICON_PATH.exists(),
        "metric_count": len(METRIC_ROWS),
    }


@app.get("/api/config")
def config() -> dict[str, Any]:
    return {
        "model": DEFAULT_MODEL,
        "vllm_base_url": DEFAULT_VLLM_BASE_URL,
        "system_prompt": REWRITE_SYSTEM_PROMPT,
        "prompt_preview": latest_rewrite_user_prompt("{source}"),
        "metric_rows": METRIC_ROWS,
    }


@app.post("/api/score")
def score_endpoint(request: ScoreRequest) -> dict[str, Any]:
    return score_text(request.text, require_result_tags=False)


@app.post("/api/humanize")
async def humanize_endpoint(request: HumanizeRequest) -> dict[str, Any]:
    source_score = score_text(request.text, require_result_tags=False)
    generated = await call_vllm(request)
    raw_output = generated["raw_output"]
    output_score = score_text(raw_output, require_result_tags=True)
    clean_output = output_score["result_text"] or strip_result_tags(raw_output)

    if not output_score["result_text"] and clean_output:
        output_score = score_text(clean_output, require_result_tags=False)

    improvement = round(source_score["ai_likeness"] - output_score["ai_likeness"], 1)
    metric_delta = []
    output_by_name = {item["metric"]: item for item in output_score["metric_details"]}
    for source_item in source_score["metric_details"]:
        out_item = output_by_name.get(source_item["metric"])
        if not out_item:
            continue
        source_ai = source_item.get("ai_likeness")
        output_ai = out_item.get("ai_likeness")
        metric_delta.append(
            {
                "metric": source_item["metric"],
                "metric_ko": source_item["metric_ko"],
                "source_ai_likeness": source_ai,
                "output_ai_likeness": output_ai,
                "delta": round(float(source_ai) - float(output_ai), 1)
                if source_ai is not None and output_ai is not None
                else None,
                "description": source_item.get("description") or "",
            }
        )

    return {
        "source_text": request.text,
        "output_text": clean_output,
        "raw_output": raw_output,
        "source_score": source_score,
        "output_score": output_score,
        "improvement": improvement,
        "metric_delta": metric_delta,
        "prompt": generated["messages"],
        "model": generated["model"],
        "base_url": generated["base_url"],
    }
