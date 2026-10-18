# Asculto Web App — Inference-as-a-Service & Educational Interactive

> Status: **implementation plan + specification (Stage 1 PoC)**
> Companion to `BusinessPlan.md`, `ProductComputeArchitecture.md`, `TechStack.md`.

## 1. Purpose and added value

The ML benchmark campaign produced strong, reproducible models but only as
offline artifacts. This app turns them into two user-facing products that
demonstrate the platform's value beyond research:

1. **Inference-as-a-Service (IaaS) — in-depth reporting.**
   A user uploads one or many auscultation recordings and receives a
   *parametrized, explainable report*: source routing (heart / lung / mixed),
   normal-vs-abnormal screening, fine-grained typing, calibrated per-head
   probability distributions, and health-awareness guidance — all rendered with
   figures and plain-language explanation.

2. **Educational Interactive (game mode).**
   A scored quiz built on the dataset's ground-truth labels: the app plays a
   real clip and asks the learner to identify the sound, then explains the
   answer. This is the "learning by listening" surface for students, community
   health workers and curious users, and a funnel into the platform.

Together they convert a research benchmark into (a) a service other products
can call, (b) an education product, and (c) a data/engagement flywheel.

## 2. Why this matters (strategic framing)

- **Demonstrates end-to-end value** — from raw audio to a human-readable,
  decision-support report, not just a leaderboard metric.
- **Two-sided growth** — clinicians/patients get awareness; students get
  training. Both generate engagement and (with consent) data.
- **Reusable service boundary** — the same inference API will back the future
  mobile app, the connected stethoscope tiers, and OEM integrations.
- **Educational infrastructure** aligns with the open-source / humanitarian
  mission already stated in the business plan.

## 3. Architecture

PoC stack (chosen for speed and to reuse the existing Python ML stack):

```
Browser (vanilla HTML/CSS/JS, server-rendered report pages)
        │  multipart upload / fetch JSON
        ▼
FastAPI app  ───────────────────────────────────────────────┐
  • routes: analyze (single/batch), report, game, health     │
  • report builder (figures + narrative + disclaimers)       │
  • model registry (cached, warm-loaded on startup)          │
        │                                                    │
        ▼                                                    │
Torch inference worker (in-process for PoC)                 │
  • cardia preprocessing (band-pass + log-mel)              │
  • 5-fold checkpoint ensembles per head                    │
  • post-softmax probabilities → figures + text             │
        ▲                                                    │
        │                                                    │
Dataset (HLS-CMDS) for game ground truth + demo clips ──────┘
```

Production target (documented, not built in this PoC): the same FastAPI
service behind a queue (Redis/SQS) with GPU workers, models exported to
ONNX/TensorRT, object storage for audio/reports, and a TypeScript/Hono API
gateway as recommended in `TechStack.md`. The PoC keeps a single Python
process so it is runnable and testable today.

## 4. Model registry (served heads)

| head | task | scheme | model | classes | source |
|---|---|---|---|---|---|
| `source` | sound | binary | ResNet-18 | heart / lung / mixed | `ml/runs` |
| `heart_binary` | heart | binary | ResNet-18 | normal / abnormal | `ml/runs` |
| `heart_types` | heart | 10class | Transformer | 10 heart types | `ml/runs_mc` |
| `lung_binary` | lung | binary | ResNet-18 | normal / abnormal | `ml/runs` |
| `lung_types` | lung | 6class | Transformer | 6 lung types | `ml/runs_mc` |

Each head is served as a **5-fold ensemble**: the post-softmax probabilities of
all fold checkpoints are averaged, which stabilises the report and gives a
natural spread. Checkpoints are warm-loaded once and cached in memory.

The classical champions (MFCC + RF/SVM) currently lead the gate but are not
persisted as artifacts; persisting them (joblib pipeline) is a listed next step
so the app can serve the champion model per task.

## 5. The parametrized report

Request parameters:

- `heads` — subset of `source, heart_binary, heart_types, lung_binary, lung_types`.
- `explanation` — `brief | standard | detailed` (controls narrative depth).
- `figures` — `none | key | all` (waveform, band-pass overlay, log-mel,
  per-head probability bars, optional mean-distribution context).
- `include_disclaimer` — always forced on for safety, toggle only controls
  prominence.
- `audio_playback` — embed the uploaded clip (raw + filtered) for review.

Report contents (per file):

1. **Summary card** — routed source, screening verdict, top typing, overall
   confidence, and a clear "not a diagnosis" banner.
2. **Signal quality** — duration, RMS, clipping, SNR estimate, and a warning if
   the clip is too short/clipped.
3. **Figures** — waveform, band-pass overlay, log-mel spectrogram, and a
   post-softmax probability bar chart for every requested head (actual class
   unknown for uploads; the head's arg-max is highlighted).
4. **Narrative** — plain-language interpretation of each head's output plus
   health-awareness context for the predicted condition (e.g. what a systolic
   murmur or wheeze is, when to seek care).
5. **Combined heart+lung view** — when both typing heads are requested, the
   16-class block view contextualises both detectors.
6. **Disclaimers** — PoC, manikin-trained, not a medical device, seek
   professional advice.

Batch mode: a table of all files (routed source, screening, top type,
confidence, quality flags) with drill-down links to each full report and a
downloadable CSV.

## 5B. Personalized AI narrative (LLM integration)

Report generation ends with an optional **AI narrative**: the structured head
probabilities are turned into a plain-language report personalized to the
user's stated health context and conversation, with **medical / diet /
environment** recommendations. Implementation is provider-agnostic
(`webapp/app/llm.py`):

- **LLMClient** — OpenAI-compatible `chat/completions`, enabled by
  `ASCULTO_LLM_*` env vars (also works with local Ollama).
- **LLMSynthesizer** — deterministic offline fallback with the same output
  shape, so the app works and stays testable without any API key.

The output JSON (`summary`, `findings`, `recommendations{medical,diet,environment}`,
`next_steps`, `provider`, `model`, `disclaimer`) is rendered in the report and
powers a follow-up chat (`POST /api/llm/report`). Guardrails: educational
output, never a diagnosis, PHI only when the user opts in. Research directions
— a token/embedding audio-encoder architecture and whole-breathing-cycle
modeling — are documented in `docs/LLM_Integration.md`.

## 5A. Temporal and multi-scale analysis

A whole recording spans several breathing/cardiac cycles, so a single
full-clip decision hides how the assessment evolves over time. The app supports
two modes:

- **Single window** — sweep the clip with one window (default 3 s, hop 0.5 s).
- **Multi-scale (default)** — sweep at several scales (default **1 s micro /
  3 s meso / 15 s macro**) and fuse them length-weighted, so short windows
  localise micro events while the longer windows / whole clip provide macro
  context. A 1–3 s window alone is ambiguous; the whole-clip pass is the most
  reliable and always runs.

For each head the app returns a time series of post-softmax probabilities and
renders:

- **Paired time-aligned figure** — the log-mel spectrogram (top) and the
  sliding-window head probabilities (bottom) share a time axis in seconds, with
  a translucent ribbon marking the arg-max class per window.
- **Multi-scale figure** — one probability panel per scale plus the whole-clip
  (macro) distribution as dashed reference lines, all time-aligned with the
  log-mel.
- **Live view** — the report also returns the raw series and renders a canvas
  chart whose playhead is synced to the audio element.
- **Fused prediction** — length-weighted combination of the scales, reported
  alongside the macro prediction.

Parameters: `temporal`, `temporal_mode` (`single|multiscale`), `scales`,
`window_s`, `hop_s` — exposed in the web form and the JSON API
(`POST /api/analyze`).

**Length handling** — CNN heads run natively on short-window spectrograms; the
Transformer heads pad a short window to the training length because their
learned positional embedding is fixed-length. Making the Transformer fully
variable-length (interpolated positions) is the follow-up that would let
multi-scale typing run natively. Compute implications are quantified in
`MobileCompute.md`.

## 6. Game mode (Educational Interactive)

- **Round generation** — sample a random clip from the dataset for a chosen
  domain (heart types / lung types / normal-vs-abnormal), return its audio URL,
  the question, and 4 multiple-choice options (correct + distractors).
- **Answering** — the client submits the choice; the server returns correctness,
  the ground-truth label, and an educational explanation of the sound and its
  clinical significance.
- **Scoring** — client-side streak/score (stateless server keeps the PoC
  simple); optional server-side sessions later.
- **Ground truth** — taken directly from `HS.csv` / `LS.csv` / `Mix.csv`
  (normalized, ID-verified), so the game is a genuine labelled listening drill.

## 7. API surface (PoC)

| method | path | purpose |
|---|---|---|
| GET | `/` | landing page (upload + tabs) |
| GET | `/game` | game page |
| POST | `/analyze` | single/batch upload → report page (HTML) |
| POST | `/api/analyze` | JSON inference (programmatic / IaaS); includes `viz` (log-mel/waveform), `quant` (quantified metrics), `panels`, per-head `probs_std` |
| GET | `/api/game/round` | next game round (JSON) |
| POST | `/api/game/answer` | grade an answer (JSON) |
| GET | `/game/audio/{sample_id}` | stream a dataset clip for the game |
| GET | `/matrix` | sound matrix: one educational tile per heart/lung type |
| GET | `/api/matrix/mel/{sample_id}` | log-mel + waveform payload for a dataset clip (JSON) |
| GET | `/health` | service + loaded-model status |
| GET | `/static/*` | css/js |

## 8. Safety, privacy and regulatory posture

- Explicit, prominent "**educational / research prototype — not a medical
  device, not a diagnosis**" messaging on every report and the landing page.
- Models are trained on a **manikin** dataset; the app must not imply clinical
  validity. This is stated in the report and the docs.
- Uploaded audio is processed in-memory/temp and not persisted in the PoC;
  batch reports are generated on demand. A production build would add explicit
  consent, retention policy and encryption.
- No PHI is required; the PoC is anonymous by design.

## 9. Implementation plan (this iteration)

1. **Docs** — this file plus updates to `BusinessPlan.md`,
   `ProductComputeArchitecture.md`, `TechStack.md`.
2. **Dependencies** — `fastapi`, `uvicorn[standard]`, `python-multipart`,
   `jinja2` (hardlinked via uv into the project venv).
3. **Inference core** — `webapp/app/inference.py`: registry, checkpoint
   ensembles, waveform preprocessing, prediction, and figure rendering.
4. **Report builder** — `webapp/app/report.py`: parametrized report assembly
   with narrative + health-awareness knowledge base + disclaimers.
5. **Game** — `webapp/app/game.py`: dataset-backed rounds and grading.
6. **API + UI** — `webapp/app/main.py`, `templates/`, `static/`.
7. **Tests** — FastAPI `TestClient` covering health, single analyze, batch
   analyze, game round/answer, using real dataset audio.

## 10. Future work

- Persist + serve the classical champions; add temperature scaling for
  calibrated probabilities in the report.
- Export heads to ONNX/TensorRT and move inference to GPU workers behind a
  queue; add the TypeScript/Hono gateway per `TechStack.md`.
- Accounts, saved history, shareable report links, PDF export (reuse
  `cardia.report`), multilingual narratives.
- Active-learning loop: flag low-confidence/ambiguous uploads for review.
- Multi-tenant IaaS with API keys, quotas and SLAs.
