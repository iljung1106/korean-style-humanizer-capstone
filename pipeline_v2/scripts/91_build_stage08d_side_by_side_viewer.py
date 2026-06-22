#!/usr/bin/env python3
"""Build a static side-by-side viewer for Stage06B vs Stage08D samples."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "outputs" / "local_reports"

INPUTS = {
    "rewrite": {
        "label": "Rewrite 100",
        "stage06b": REPORT_ROOT / "stage06b_100_eval" / "generations" / "rewrite_generations.jsonl",
        "stage08d": (
            REPORT_ROOT
            / "stage08d_vllm_eval"
            / "outputs"
            / "pipeline_v2"
            / "phase_eval"
            / "stage08d_rewrite100_eval"
            / "generations"
            / "rewrite_generations.jsonl"
        ),
    },
    "continuation": {
        "label": "Generate: Continuation 100",
        "stage06b": REPORT_ROOT
        / "stage06b_generate_eval"
        / "continuation_generations"
        / "generate_generations.jsonl",
        "stage08d": (
            REPORT_ROOT
            / "stage08d_vllm_eval"
            / "outputs"
            / "pipeline_v2"
            / "phase_eval"
            / "stage08d_generate_continuation100_eval"
            / "generations"
            / "generate_generations.jsonl"
        ),
    },
    "new": {
        "label": "Generate: New Writing 100",
        "stage06b": REPORT_ROOT / "stage06b_generate_eval" / "new_writing_generations" / "generate_generations.jsonl",
        "stage08d": (
            REPORT_ROOT
            / "stage08d_vllm_eval"
            / "outputs"
            / "pipeline_v2"
            / "phase_eval"
            / "stage08d_generate_new100_eval"
            / "generations"
            / "generate_generations.jsonl"
        ),
    },
}

OUT_DIR = REPORT_ROOT / "stage08d_vllm_eval_compare" / "side_by_side_viewer"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def strip_result(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"^\s*<result>\s*", "", text)
    text = re.sub(r"\s*</result>.*$", "", text, flags=re.S)
    return text.strip()


def prompt_text(prompt: Any) -> str:
    if not isinstance(prompt, list):
        return str(prompt or "")
    parts: list[str] = []
    for item in prompt:
        if not isinstance(item, dict):
            continue
        role = item.get("role", "")
        content = str(item.get("content") or "")
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


def row_key(row: dict[str, Any], index: int) -> str:
    source_chunk = str(row.get("source_chunk_id") or "")
    if source_chunk:
        return source_chunk
    row_id = str(row.get("id") or "")
    if row_id:
        return row_id
    return f"index:{index:04d}"


def compact_path(value: str) -> str:
    value = str(value or "")
    if not value:
        return ""
    return value.replace(str(ROOT.parent), "").lstrip("/")


def model_payload(row: dict[str, Any]) -> dict[str, Any]:
    generated = str(row.get("generated_text") or row.get("raw_generated_text") or "")
    return {
        "model_label": row.get("model_label", ""),
        "generated_text": strip_result(generated),
        "raw_generated_text": generated,
        "hit_result_close": bool(row.get("hit_result_close")),
        "stop_reason": row.get("stop_reason") or row.get("stop_reason_label") or "",
        "generated_tokens": row.get("generated_tokens"),
        "decoded_tokens": row.get("decoded_tokens"),
        "elapsed_sec_batch": row.get("elapsed_sec_batch"),
        "post_result_chars": row.get("post_result_chars"),
    }


def build_cases() -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for task, cfg in INPUTS.items():
        left_rows = read_jsonl(cfg["stage06b"])
        right_rows = read_jsonl(cfg["stage08d"])
        right_by_key = {row_key(row, index): row for index, row in enumerate(right_rows)}
        paired: list[dict[str, Any]] = []
        missing = 0
        for index, left in enumerate(left_rows):
            key = row_key(left, index)
            right = right_by_key.get(key)
            if right is None and index < len(right_rows):
                right = right_rows[index]
            if right is None:
                missing += 1
                continue
            source_text = str(left.get("source_text") or right.get("source_text") or "")
            prompt = prompt_text(left.get("prompt") or right.get("prompt"))
            paired.append(
                {
                    "index": index,
                    "id": left.get("id") or right.get("id") or f"{task}-{index:04d}",
                    "key": key,
                    "task": task,
                    "source_file": compact_path(left.get("source_file") or right.get("source_file") or ""),
                    "source_chunk_id": left.get("source_chunk_id") or right.get("source_chunk_id") or "",
                    "source_text": source_text,
                    "prompt_text": prompt,
                    "stage06b": model_payload(left),
                    "stage08d": model_payload(right),
                }
            )
        cases[task] = {
            "label": cfg["label"],
            "count": len(paired),
            "missing": missing,
            "items": paired,
        }
    return cases


HTML = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stage06B vs Stage08D Side-by-Side</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d8dde6;
      --text: #161a22;
      --muted: #596273;
      --accent: #1957d2;
      --accent-soft: #eaf0ff;
      --bad: #b42318;
      --good: #0f766e;
      --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      --sans: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: var(--sans);
      line-height: 1.55;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(246, 247, 249, 0.96);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(8px);
    }
    .wrap { max-width: 1680px; margin: 0 auto; padding: 16px 20px; }
    h1 { margin: 0 0 12px; font-size: 20px; font-weight: 760; letter-spacing: 0; }
    .toolbar {
      display: grid;
      grid-template-columns: auto minmax(260px, 1fr) auto auto auto;
      gap: 10px;
      align-items: center;
    }
    .tabs { display: flex; gap: 6px; flex-wrap: wrap; }
    button, select, input {
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
      border-radius: 6px;
      padding: 8px 10px;
      font: inherit;
      min-height: 38px;
    }
    button { cursor: pointer; }
    button.active {
      background: var(--accent);
      color: white;
      border-color: var(--accent);
    }
    input { width: 100%; }
    label.toggle {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: var(--muted);
      white-space: nowrap;
      font-size: 13px;
    }
    label.toggle input { width: auto; min-height: 0; }
    main.wrap { padding-top: 14px; }
    .meta {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      margin-bottom: 12px;
      align-items: start;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .source {
      padding: 12px 14px;
      max-height: 220px;
      overflow: auto;
      white-space: pre-wrap;
      font-size: 14px;
    }
    .details {
      padding: 10px 12px;
      color: var(--muted);
      font-size: 13px;
      min-width: 280px;
    }
    .details strong { color: var(--text); }
    .columns {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      align-items: start;
    }
    .column-header {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      align-items: center;
    }
    .column-header h2 { margin: 0; font-size: 16px; }
    .bad { color: var(--bad); }
    .good { color: var(--good); }
    .stats {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      justify-content: flex-end;
      font-size: 12px;
      color: var(--muted);
    }
    .pill {
      background: #f1f3f6;
      border: 1px solid #e1e5ec;
      border-radius: 999px;
      padding: 2px 8px;
      white-space: nowrap;
    }
    .text {
      padding: 16px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 15px;
      line-height: 1.72;
      max-height: calc(100vh - 270px);
      overflow: auto;
    }
    .hidden { display: none !important; }
    mark {
      background: #fff1a8;
      color: inherit;
      padding: 0 1px;
      border-radius: 2px;
    }
    .prompt {
      margin-top: 12px;
      padding: 12px 14px;
      white-space: pre-wrap;
      color: var(--muted);
      font-size: 13px;
      max-height: 180px;
      overflow: auto;
    }
    @media (max-width: 980px) {
      .toolbar { grid-template-columns: 1fr; }
      .meta, .columns { grid-template-columns: 1fr; }
      .text { max-height: none; }
    }
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <h1>Stage06B vs Stage08D Side-by-Side</h1>
      <div class="toolbar">
        <div class="tabs" id="tabs"></div>
        <input id="search" type="search" placeholder="id, 파일명, 본문 검색">
        <select id="caseSelect"></select>
        <button id="prevBtn" type="button">Prev</button>
        <button id="nextBtn" type="button">Next</button>
      </div>
      <div class="toolbar" style="grid-template-columns: repeat(4, auto) 1fr; margin-top: 8px;">
        <label class="toggle"><input id="showSource" type="checkbox" checked> 원문/조건</label>
        <label class="toggle"><input id="showPrompt" type="checkbox"> 프롬프트</label>
        <label class="toggle"><input id="rawText" type="checkbox"> raw result 태그 포함</label>
        <label class="toggle"><input id="onlyIssues" type="checkbox"> 미종료/length stop만</label>
        <span id="status" style="color: var(--muted); font-size: 13px;"></span>
      </div>
    </div>
  </header>
  <main class="wrap">
    <section class="meta">
      <div id="sourceCard" class="card source"></div>
      <div class="card details" id="details"></div>
    </section>
    <section id="promptCard" class="card prompt hidden"></section>
    <section class="columns">
      <article class="card">
        <div class="column-header">
          <h2>Stage06B</h2>
          <div class="stats" id="stats06"></div>
        </div>
        <div class="text" id="text06"></div>
      </article>
      <article class="card">
        <div class="column-header">
          <h2>Stage08D</h2>
          <div class="stats" id="stats08"></div>
        </div>
        <div class="text" id="text08"></div>
      </article>
    </section>
  </main>
  <script src="viewer_data.js"></script>
  <script>
    const state = { task: 'rewrite', index: 0, filtered: [] };
    const tabs = document.getElementById('tabs');
    const caseSelect = document.getElementById('caseSelect');
    const search = document.getElementById('search');
    const status = document.getElementById('status');
    const showSource = document.getElementById('showSource');
    const showPrompt = document.getElementById('showPrompt');
    const rawText = document.getElementById('rawText');
    const onlyIssues = document.getElementById('onlyIssues');

    function esc(value) {
      return String(value ?? '').replace(/[&<>"']/g, ch => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }[ch]));
    }
    function highlight(value, query) {
      const text = esc(value);
      if (!query) return text;
      const safe = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      return text.replace(new RegExp(safe, 'gi'), match => `<mark>${match}</mark>`);
    }
    function activeItems() {
      const all = VIEWER_DATA.tasks[state.task].items;
      const q = search.value.trim().toLowerCase();
      return all.filter(item => {
        const issue = !item.stage06b.hit_result_close || !item.stage08d.hit_result_close ||
          item.stage06b.stop_reason === 'length' || item.stage08d.stop_reason === 'length';
        if (onlyIssues.checked && !issue) return false;
        if (!q) return true;
        const hay = [
          item.id, item.key, item.source_file, item.source_chunk_id,
          item.source_text, item.stage06b.generated_text, item.stage08d.generated_text
        ].join('\n').toLowerCase();
        return hay.includes(q);
      });
    }
    function setTask(task) {
      state.task = task;
      state.index = 0;
      renderTabs();
      rebuildSelect();
      render();
    }
    function renderTabs() {
      tabs.innerHTML = '';
      Object.entries(VIEWER_DATA.tasks).forEach(([key, task]) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = key === state.task ? 'active' : '';
        btn.textContent = `${task.label} (${task.count})`;
        btn.onclick = () => setTask(key);
        tabs.appendChild(btn);
      });
    }
    function rebuildSelect() {
      state.filtered = activeItems();
      caseSelect.innerHTML = '';
      state.filtered.forEach((item, idx) => {
        const opt = document.createElement('option');
        opt.value = String(idx);
        opt.textContent = `${idx + 1}. ${item.id}`;
        caseSelect.appendChild(opt);
      });
      if (state.index >= state.filtered.length) state.index = Math.max(0, state.filtered.length - 1);
      caseSelect.value = String(state.index);
      status.textContent = `${state.filtered.length} / ${VIEWER_DATA.tasks[state.task].items.length} samples`;
    }
    function statPill(label, value, klass = '') {
      return `<span class="pill ${klass}">${esc(label)}: ${esc(value)}</span>`;
    }
    function stats(model) {
      const closeClass = model.hit_result_close ? 'good' : 'bad';
      return [
        statPill('close', model.hit_result_close ? 'yes' : 'no', closeClass),
        statPill('stop', model.stop_reason || ''),
        statPill('tokens', model.generated_tokens ?? ''),
        statPill('post', model.post_result_chars ?? 0),
      ].join('');
    }
    function render() {
      rebuildSelect();
      const item = state.filtered[state.index];
      if (!item) {
        document.getElementById('sourceCard').textContent = 'No samples';
        document.getElementById('text06').textContent = '';
        document.getElementById('text08').textContent = '';
        return;
      }
      caseSelect.value = String(state.index);
      const q = search.value.trim();
      document.getElementById('sourceCard').classList.toggle('hidden', !showSource.checked);
      document.getElementById('sourceCard').innerHTML = highlight(item.source_text || '(조건 생성: source_text 없음)', q);
      const promptCard = document.getElementById('promptCard');
      promptCard.classList.toggle('hidden', !showPrompt.checked);
      promptCard.innerHTML = highlight(item.prompt_text, q);
      document.getElementById('details').innerHTML = [
        `<strong>${esc(item.id)}</strong>`,
        `task: ${esc(item.task)}`,
        `chunk: ${esc(item.source_chunk_id || item.key)}`,
        `file: ${esc(item.source_file || '-')}`,
      ].join('<br>');
      const key = rawText.checked ? 'raw_generated_text' : 'generated_text';
      document.getElementById('text06').innerHTML = highlight(item.stage06b[key], q);
      document.getElementById('text08').innerHTML = highlight(item.stage08d[key], q);
      document.getElementById('stats06').innerHTML = stats(item.stage06b);
      document.getElementById('stats08').innerHTML = stats(item.stage08d);
    }
    function move(delta) {
      if (!state.filtered.length) return;
      state.index = (state.index + delta + state.filtered.length) % state.filtered.length;
      render();
    }
    document.getElementById('prevBtn').onclick = () => move(-1);
    document.getElementById('nextBtn').onclick = () => move(1);
    caseSelect.onchange = () => { state.index = Number(caseSelect.value || 0); render(); };
    [search, showSource, showPrompt, rawText, onlyIssues].forEach(el => {
      el.addEventListener('input', () => { state.index = 0; render(); });
      el.addEventListener('change', () => { state.index = 0; render(); });
    });
    document.addEventListener('keydown', ev => {
      if (ev.target && ['INPUT', 'SELECT'].includes(ev.target.tagName)) return;
      if (ev.key === 'ArrowLeft') move(-1);
      if (ev.key === 'ArrowRight') move(1);
    });
    renderTabs();
    rebuildSelect();
    render();
  </script>
</body>
</html>
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_by": Path(__file__).name,
        "tasks": build_cases(),
    }
    (OUT_DIR / "viewer_data.js").write_text(
        "window.VIEWER_DATA = " + json.dumps(payload, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )
    (OUT_DIR / "index.html").write_text(HTML, encoding="utf-8")
    summary = {
        task: {
            "label": data["label"],
            "count": data["count"],
            "missing": data["missing"],
            "stage06b_closed": sum(1 for item in data["items"] if item["stage06b"]["hit_result_close"]),
            "stage08d_closed": sum(1 for item in data["items"] if item["stage08d"]["hit_result_close"]),
        }
        for task, data in payload["tasks"].items()
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT_DIR / "index.html")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
