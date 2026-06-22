import { useEffect, useMemo, useState } from "react";
import {
  BarChart3,
  Clipboard,
  Eraser,
  FileText,
  Loader2,
  RefreshCw,
  Sparkles,
  Wand2,
} from "lucide-react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

const sampleText =
  "그 순간, 세상은 마치 얼어붙은 것처럼 조용해졌다. 유나는 압도적인 기운에 숨을 삼켰다. 그것은 단순한 두려움이 아니었다. 운명처럼 다가온 미묘한 예감이 그녀의 심장을 세차게 흔들었다.";

function clampScore(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return 0;
  return Math.max(0, Math.min(100, value));
}

function scoreTone(score) {
  if (score >= 70) return "bad";
  if (score >= 42) return "mid";
  return "good";
}

function formatValue(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  const n = Number(value);
  if (Math.abs(n) >= 100) return n.toFixed(1);
  if (Math.abs(n) >= 10) return n.toFixed(2);
  return n.toFixed(4);
}

function formatDelta(delta) {
  if (delta === null || delta === undefined || Number.isNaN(Number(delta))) return "-";
  const n = Number(delta);
  if (Math.abs(n) < 0.05) return "0.0p";
  return `${n > 0 ? "-" : "+"}${Math.abs(n).toFixed(1)}p`;
}

async function postJson(path, payload) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const data = await response.json();
      message = data.detail || message;
    } catch {
      // keep status text
    }
    throw new Error(message);
  }
  return response.json();
}

function ScorePill({ label, score, helper }) {
  const safeScore = clampScore(score);
  return (
    <div className={`score-pill ${scoreTone(safeScore)}`}>
      <span>{label}</span>
      <strong>{safeScore.toFixed(1)}</strong>
      {helper ? <small>{helper}</small> : null}
    </div>
  );
}

function Meter({ value, tone = "source" }) {
  const safeValue = clampScore(value);
  return (
    <div className="meter" aria-label={`${safeValue.toFixed(1)}점`}>
      <div className={`meter-fill ${tone}`} style={{ width: `${safeValue}%` }} />
    </div>
  );
}

function MetricBars({ rows }) {
  return (
    <section className="analysis-band">
      <div className="section-title">
        <BarChart3 size={18} />
        <h2>지표별 AI스러움</h2>
      </div>
      <div className="metric-bars">
        {rows.map((row) => {
          const source = row.source_ai_likeness ?? 0;
          const output = row.output_ai_likeness ?? 0;
          const delta = row.delta;
          return (
            <div className="metric-row" key={row.metric}>
              <div className="metric-label">
                <strong>{row.metric_ko}</strong>
                <span>{formatDelta(delta)}</span>
              </div>
              <div className="paired-bars">
                <div className="bar-line">
                  <span>원문</span>
                  <Meter value={source} tone="source" />
                  <b>{formatValue(source)}</b>
                </div>
                <div className="bar-line">
                  <span>변환</span>
                  <Meter value={output} tone="output" />
                  <b>{formatValue(output)}</b>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function DetailTable({ rows }) {
  return (
    <section className="detail-band">
      <div className="section-title">
        <FileText size={18} />
        <h2>세부 지표</h2>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>지표</th>
              <th>원문 AI</th>
              <th>변환 AI</th>
              <th>원문 값</th>
              <th>변환 값</th>
              <th>평가</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.metric}>
                <td>
                  <strong>{row.metric_ko}</strong>
                  <span>{row.description}</span>
                </td>
                <td>{formatValue(row.source_ai_likeness)}</td>
                <td>{formatValue(row.output_ai_likeness)}</td>
                <td>{formatValue(row.source_value)}</td>
                <td>{formatValue(row.output_value)}</td>
                <td className={row.delta >= 0 ? "good-text" : "warn-text"}>
                  {row.delta === null || row.delta === undefined
                    ? "참조 없음"
                    : row.delta >= 0
                      ? "인간 기준에 가까워짐"
                      : "AI 기준에 가까워짐"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function TextPanel({ title, score, value, onChange, readOnly, placeholder }) {
  return (
    <section className="text-panel">
      <div className="panel-head">
        <h2>{title}</h2>
        {score ? (
          <ScorePill label="AI스러움" score={score.ai_likeness} helper={`인간스러움 ${score.human_likeness?.toFixed?.(1) ?? "-"}점`} />
        ) : null}
      </div>
      <textarea
        value={value}
        onChange={(event) => onChange?.(event.target.value)}
        readOnly={readOnly}
        placeholder={placeholder}
      />
    </section>
  );
}

function App() {
  const [sourceText, setSourceText] = useState(sampleText);
  const [outputText, setOutputText] = useState("");
  const [sourceScore, setSourceScore] = useState(null);
  const [outputScore, setOutputScore] = useState(null);
  const [metricDelta, setMetricDelta] = useState([]);
  const [health, setHealth] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [temperature, setTemperature] = useState(0.65);

  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then((response) => response.json())
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  const detailedRows = useMemo(() => {
    if (!sourceScore || !outputScore) return [];
    const outputByMetric = new Map(outputScore.metric_details.map((item) => [item.metric, item]));
    return sourceScore.metric_details.map((item) => {
      const output = outputByMetric.get(item.metric) || {};
      const sourceAi = item.ai_likeness;
      const outputAi = output.ai_likeness;
      return {
        metric: item.metric,
        metric_ko: item.metric_ko,
        description: item.description,
        source_ai_likeness: sourceAi,
        output_ai_likeness: outputAi,
        source_value: item.value,
        output_value: output.value,
        delta:
          sourceAi !== null && sourceAi !== undefined && outputAi !== null && outputAi !== undefined
            ? Number(sourceAi) - Number(outputAi)
            : null,
      };
    });
  }, [sourceScore, outputScore]);

  async function scoreOnly() {
    setLoading(true);
    setError("");
    try {
      const source = await postJson("/api/score", { text: sourceText });
      setSourceScore(source);
      if (outputText.trim()) {
        const output = await postJson("/api/score", { text: outputText });
        setOutputScore(output);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function humanize() {
    setLoading(true);
    setError("");
    try {
      const data = await postJson("/api/humanize", {
        text: sourceText,
        temperature,
      });
      setOutputText(data.output_text || "");
      setSourceScore(data.source_score);
      setOutputScore(data.output_score);
      setMetricDelta(data.metric_delta || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function clearAll() {
    setSourceText("");
    setOutputText("");
    setSourceScore(null);
    setOutputScore(null);
    setMetricDelta([]);
    setError("");
  }

  const improvement =
    sourceScore && outputScore ? Number(sourceScore.ai_likeness) - Number(outputScore.ai_likeness) : null;
  const chartRows = metricDelta.length ? metricDelta : detailedRows;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Gemma4 WebNovel Style Humanizer</p>
          <h1>AI 텍스트 인간화 비교 UI</h1>
        </div>
        <div className="status-strip">
          <span>{health?.model || "gemma4-webnovel-stage08b"}</span>
          <span className={health?.reference_exists ? "ok" : "warn"}>style ref</span>
          <span className={health?.anti_slop_lexicon_exists ? "ok" : "warn"}>anti-slop</span>
        </div>
      </header>

      <section className="score-band">
        <ScorePill label="원문 AI스러움" score={sourceScore?.ai_likeness ?? 0} helper="낮을수록 인간 기준에 가까움" />
        <ScorePill label="개선폭" score={improvement === null ? 0 : Math.max(0, improvement)} helper={improvement === null ? "대기 중" : `${improvement.toFixed(1)}p 감소`} />
        <ScorePill label="변환 AI스러움" score={outputScore?.ai_likeness ?? 0} helper="모델 출력 기준" />
      </section>

      <section className="controls">
        <button type="button" onClick={humanize} disabled={loading || !sourceText.trim()} className="primary">
          {loading ? <Loader2 className="spin" size={18} /> : <Wand2 size={18} />}
          변환 및 평가
        </button>
        <button type="button" onClick={scoreOnly} disabled={loading || !sourceText.trim()}>
          <BarChart3 size={18} />
          점수만 계산
        </button>
        <button type="button" onClick={() => setSourceText(sampleText)} disabled={loading}>
          <Clipboard size={18} />
          예시
        </button>
        <button type="button" onClick={clearAll} disabled={loading}>
          <Eraser size={18} />
          지우기
        </button>
        <label className="slider">
          <RefreshCw size={16} />
          <span>temperature</span>
          <input
            type="range"
            min="0"
            max="1.2"
            step="0.05"
            value={temperature}
            onChange={(event) => setTemperature(Number(event.target.value))}
          />
          <b>{temperature.toFixed(2)}</b>
        </label>
      </section>

      {error ? (
        <div className="error-box">
          <Sparkles size={18} />
          <span>{error}</span>
        </div>
      ) : null}

      <section className="workspace">
        <TextPanel
          title="원문"
          score={sourceScore}
          value={sourceText}
          onChange={setSourceText}
          placeholder="AI가 작성한 한국어 웹소설 문단을 붙여넣으세요."
        />
        <TextPanel
          title="인간화 결과"
          score={outputScore}
          value={outputText}
          onChange={setOutputText}
          placeholder="변환 결과가 여기에 표시됩니다. 직접 붙여넣고 점수만 계산할 수도 있습니다."
        />
      </section>

      {chartRows.length ? <MetricBars rows={chartRows.slice(0, 12)} /> : null}
      {detailedRows.length ? <DetailTable rows={detailedRows} /> : null}
    </main>
  );
}

export default App;
