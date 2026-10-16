# Asculto Web App — Inference-as-a-Service & Educational Interactive

A FastAPI app that serves the trained HLS-CMDS models as an explained inference
service and a ground-truth-backed listening game. See `../docs/WebApp.md` for the
full specification.

## Features

- **Analyze (IaaS):** upload one file for a full report, or many for a batch
  table. Parametrized by prediction heads, explanation depth and figures.
- **Report:** source routing, normal/abnormal screening, fine-grained typing,
  per-head post-softmax probability figures, signal-quality flags, and
  plain-language health-awareness narrative — always with a not-a-diagnosis
  disclaimer.
- **Learn (game):** scored listening rounds using dataset ground truth, with an
  explanation after every answer.
- **JSON API:** `POST /api/analyze`, `GET /api/game/round`,
  `POST /api/game/answer` for programmatic / IaaS use.
- **Personalized AI narrative (LLM):** `POST /api/llm/report` and the
  `patient_context` / `ai_narrative` report options produce a personalized
  report with medical / diet / environment recommendations and a follow-up chat
  (offline synthesizer by default; OpenAI-compatible LLM when configured).
  See `../docs/LLM_Integration.md`.

## Run

```bash
# from the repo root, using the project venv
.venv/bin/python -m uvicorn webapp.app.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000.

Environment overrides:

| var | default | purpose |
|---|---|---|
| `ASCULTO_DEVICE` | `cpu` | `cuda` for GPU inference |
| `ASCULTO_DATA_DIR` | `data/HLS-CMDS` | dataset root (game + demo) |
| `ASCULTO_BINARY_OUT` | `ml/runs` | binary campaign checkpoints |
| `ASCULTO_MC_OUT` | `ml/runs` (same as binary) | multiclass checkpoints |
| `ASCULTO_LLM_API_KEY` | *(unset)* | enables the OpenAI-compatible LLM narrative |
| `ASCULTO_LLM_BASE_URL` | `https://api.openai.com/v1` | any compatible endpooint (e.g. local Ollama) |
| `ASCULTO_LLM_MODEL` | `gpt-4o-mini` | LLM model name |

## Model heads

Heads are 5-fold checkpoint ensembles averaged at the probability level; they
warm-load on startup (see `/health`). Missing checkpoints are reported, not
fatal.

## Notes / limitations

- Models are trained on a **manikin** dataset — this is an educational/research
  prototype, **not a medical device**.
- Uploads are processed in a temp file and deleted; nothing is persisted.
- PoC runs inference in-process on CPU by default. The production path
  (ONNX/Triton GPU workers behind a queue, TypeScript/Hono gateway, object
  storage) is described in `../docs/TechStack.md` and `../docs/WebApp.md`.
