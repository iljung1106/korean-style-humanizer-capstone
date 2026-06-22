# Style Humanizer UI

Vite + React + FastAPI 기반의 한국어 웹소설 문체 인간화 비교 UI입니다.

- 좌측 원문과 우측 인간화 결과를 비교합니다.
- 기존 `pipeline_v2`의 stylometry scorer와 stage06 기준값을 사용합니다.
- vLLM OpenAI 호환 엔드포인트로 `gemma4-webnovel-stage08b` 모델을 호출합니다.
- 주요 문체 지표별 AI스러움 막대그래프와 세부 지표 테이블을 표시합니다.

## 실행

```bash
cd korean-style-humanizer-capstone/apps/style-humanizer-ui/backend
python3 -m uvicorn app:app --host 127.0.0.1 --port 8050
```

```bash
cd korean-style-humanizer-capstone/apps/style-humanizer-ui/frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

기본 vLLM 주소와 모델명은 환경변수로 바꿀 수 있습니다.

```bash
VLLM_BASE_URL=https://your-tunnel.example.com VLLM_MODEL=gemma4-webnovel-stage08b python3 -m uvicorn app:app --host 127.0.0.1 --port 8050
```
