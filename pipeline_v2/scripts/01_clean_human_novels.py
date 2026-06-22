#!/usr/bin/env python3
"""Clean raw Korean webnovel text files into standalone CPT chunks.

This script is intentionally dependency-free and does not import project-local
modules. It reads raw `.txt` files, removes obvious metadata/noise, chunks on
paragraph boundaries, filters low-quality chunks, and writes JSONL rows:

    {"row_type": "raw_lm", "id": "...", "text": "...", ...}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Iterable


SCRIPT = Path(__file__).resolve()
TRAINING_ROOT = SCRIPT.parents[2]
WORKSPACE_ROOT = SCRIPT.parents[3]

DEFAULT_INPUT_DIR = WORKSPACE_ROOT / "data" / "raw" / "human_novels"
DEFAULT_OUTPUT = TRAINING_ROOT / "data" / "pipeline_v2" / "cpt_raw_chunks.jsonl"
DEFAULT_MANIFEST = TRAINING_ROOT / "data" / "pipeline_v2" / "cpt_raw_chunks.manifest.json"

TEXT_EXTENSIONS = {".txt"}
ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")
LONG_RULE_RE = re.compile(r"^\s*[-_=*#]{5,}\s*$")
BRACKET_FILE_RE = re.compile(r"^\s*[\[\(]?\s*\d{1,5}[_\-.].{0,80}\.(?:txt|jpg|jpeg|png|gif)\s*[\]\)]?\s*$", re.I)
CHAPTER_FILE_RE = re.compile(r"^\s*-{3,}\s*\[\s*\d{1,5}[_\-.].*?\]\s*-{3,}\s*$")
URL_RE = re.compile(r"https?://\S+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTI_NEWLINE_RE = re.compile(r"\n{4,}")
SPACE_RE = re.compile(r"[ \t]{2,}")
CHAPTER_INDEX_RE = re.compile(r"^(?:제\s*)?\d{1,4}\s*[장화](?:[.\s].*)?$")
CHUNK_ARTIFACT_RE = re.compile(
    r"(작품\s*소개|저자\s*소개|장편소설|텍본|스캔본|다운로드|원본\s*링크|https?://|"
    r"무단\s*전재|무단\s*복제|ISBN|[ⓒ©]|"
    r"(?:^|\n)\s*(?:판권|지은이|발행처|출판사|전자책|정가|펴낸곳|펴낸이|등록번호)\s*[:：])",
    re.I,
)


def read_text(path: Path) -> str:
    data = path.read_bytes()
    for bom, encoding in (
        (b"\xff\xfe\x00\x00", "utf-32"),
        (b"\x00\x00\xfe\xff", "utf-32"),
        (b"\xff\xfe", "utf-16"),
        (b"\xfe\xff", "utf-16"),
        (b"\xef\xbb\xbf", "utf-8-sig"),
    ):
        if data.startswith(bom):
            return data.decode(encoding, errors="replace")
    if data.count(0) / max(1, len(data)) >= 0.03:
        for encoding in ("utf-16", "utf-16-le", "utf-16-be", "utf-32"):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                pass
    for encoding in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass
    return data.decode("utf-8", errors="replace")


def stable_id(*parts: object, length: int = 16) -> str:
    payload = "\n".join(str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def hangul_ratio(text: str) -> float:
    chars = [char for char in text if not char.isspace()]
    if not chars:
        return 0.0
    hangul = sum(1 for char in chars if "\uac00" <= char <= "\ud7a3")
    return hangul / len(chars)


def replacement_ratio(text: str) -> float:
    chars = [char for char in text if not char.isspace()]
    if not chars:
        return 0.0
    bad = sum(1 for char in chars if char == "\ufffd")
    return bad / len(chars)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    text = ZERO_WIDTH_RE.sub("", text)
    text = CONTROL_RE.sub("", text)
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            lines.append("")
            continue
        if LONG_RULE_RE.match(line):
            continue
        if CHAPTER_FILE_RE.match(line) or BRACKET_FILE_RE.match(line):
            continue
        if line.startswith(("원본 링크:", "작가:", "출처:", "다운로드:", "텍본", "스캔본")):
            continue
        if URL_RE.search(line):
            continue
        line = SPACE_RE.sub(" ", line)
        lines.append(line)
    text = "\n".join(lines)
    text = MULTI_NEWLINE_RE.sub("\n\n\n", text)
    return text.strip()


def paragraph_chunks(text: str, *, target_chars: int, max_chars: int, min_chars: int) -> Iterable[str]:
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        plen = len(paragraph)
        if plen > max_chars:
            if current and current_len >= min_chars:
                yield "\n\n".join(current).strip()
            current, current_len = [], 0
            start = 0
            while start < plen:
                end = min(plen, start + target_chars)
                if end < plen:
                    boundary = max(
                        paragraph.rfind("다.", start, end),
                        paragraph.rfind("요.", start, end),
                        paragraph.rfind("\n", start, end),
                        paragraph.rfind(". ", start, end),
                    )
                    if boundary > start + min_chars:
                        end = boundary + 2
                piece = paragraph[start:end].strip()
                if len(piece) >= min_chars:
                    yield piece
                start = end
            continue
        if current and current_len + plen + 2 > max_chars:
            if current_len >= min_chars:
                yield "\n\n".join(current).strip()
            current, current_len = [], 0
        current.append(paragraph)
        current_len += plen + 2
        if current_len >= target_chars:
            yield "\n\n".join(current).strip()
            current, current_len = [], 0
    if current and current_len >= min_chars:
        yield "\n\n".join(current).strip()


def chunk_artifact_reason(text: str) -> str | None:
    head = text[:3000]
    lines = [line.strip() for line in head.splitlines() if line.strip()]
    chapter_index_lines = sum(1 for line in lines if CHAPTER_INDEX_RE.fullmatch(line))
    if "목차" in head and chapter_index_lines >= 2:
        return "table_of_contents"
    if CHUNK_ARTIFACT_RE.search(head):
        return "front_matter_or_metadata"
    return None


def compact_fingerprint(text: str, window: int) -> str:
    compact = re.sub(r"\s+", "", text)
    return stable_id(compact[:window])


def iter_text_files(input_dir: Path, limit_files: int = 0) -> Iterable[Path]:
    count = 0
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        yield path
        count += 1
        if limit_files > 0 and count >= limit_files:
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean raw human webnovels into CPT JSONL chunks.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--min-chars", type=int, default=1200)
    parser.add_argument("--target-chars", type=int, default=4200)
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument("--min-hangul-ratio", type=float, default=0.45)
    parser.add_argument("--max-replacement-ratio", type=float, default=0.01)
    parser.add_argument("--dedupe-window", type=int, default=1800)
    parser.add_argument("--limit-files", type=int, default=0, help="Process only the first N txt files for smoke tests.")
    parser.add_argument("--limit-chunks", type=int, default=0, help="Stop after writing N chunks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output = Path(args.output)
    manifest = Path(args.manifest)
    if not input_dir.exists():
        raise FileNotFoundError(f"input dir not found: {input_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "time": time.time(),
        "input_dir": str(input_dir),
        "output": str(output),
        "files_seen": 0,
        "files_read": 0,
        "chunks_written": 0,
            "dropped": {
            "too_short": 0,
            "low_hangul": 0,
            "replacement_chars": 0,
            "artifact": 0,
            "duplicate": 0,
            "read_error": 0,
        },
        "args": vars(args),
    }
    seen: set[str] = set()

    with output.open("w", encoding="utf-8") as handle:
        for path in iter_text_files(input_dir, args.limit_files):
            stats["files_seen"] += 1
            try:
                text = normalize_text(read_text(path))
            except Exception:
                stats["dropped"]["read_error"] += 1
                continue
            stats["files_read"] += 1
            for chunk_index, chunk in enumerate(
                paragraph_chunks(
                    text,
                    target_chars=args.target_chars,
                    max_chars=args.max_chars,
                    min_chars=args.min_chars,
                )
            ):
                if len(chunk) < args.min_chars:
                    stats["dropped"]["too_short"] += 1
                    continue
                hratio = hangul_ratio(chunk)
                if hratio < args.min_hangul_ratio:
                    stats["dropped"]["low_hangul"] += 1
                    continue
                rratio = replacement_ratio(chunk)
                if rratio > 0 or rratio > args.max_replacement_ratio:
                    stats["dropped"]["replacement_chars"] += 1
                    continue
                if chunk_artifact_reason(chunk) is not None:
                    stats["dropped"]["artifact"] += 1
                    continue
                fp = compact_fingerprint(chunk, args.dedupe_window)
                if fp in seen:
                    stats["dropped"]["duplicate"] += 1
                    continue
                seen.add(fp)
                row_id = "raw-" + stable_id(path.as_posix(), chunk_index, chunk[:200], length=20)
                row = {
                    "row_type": "raw_lm",
                    "id": row_id,
                    "text": chunk,
                    "source_file": str(path),
                    "chunk_index": chunk_index,
                    "chars": len(chunk),
                    "hangul_ratio": round(hratio, 6),
                    "sha1": stable_id(chunk, length=40),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["chunks_written"] += 1
                if args.limit_chunks > 0 and stats["chunks_written"] >= args.limit_chunks:
                    manifest.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(json.dumps(stats, ensure_ascii=False))
                    return

    manifest.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
