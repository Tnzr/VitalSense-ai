# Asculto — Auscultation Benchmark, Inference-as-a-Service & Educational Game

Stage-1 AI proof of concept for automated heart/lung sound analysis on the
**[HLS-CMDS](https://doi.org/10.1109/IEEEDATA.2025.3566012)** dataset (Torabi,
Shirani & Reilly, IEEE Data Descriptions 2025): 535 digital-stethoscope recordings
from a clinical manikin, spanning heart-only, lung-only and mixed clips.

The project delivers three things:

1. **A reproducible ML benchmark** — 3 tasks × 4 model tiers × 5-fold
   leakage-safe grouped cross-validation, tracked in Weights & Biases, with
   auditable adopt/skip gates.
2. **Inference-as-a-Service** — a FastAPI web app that turns uploads into
   explained, parametrized reports with figures and health-awareness guidance.
3. **An educational game** — a ground-truth listening drill for training.

> ⚠️ **Educational / research prototype.** Models are trained on a *manikin*
> dataset. This is **not a medical device** and does **not** provide a diagnosis.

---

## Results at a glance

Best test **balanced accuracy** per stage (5 folds; gate = adopt only if
balanced-accuracy *or* macro-F1 improves ≥ 1 point over the best prior stage):

| task | scheme | stage 1 classical | stage 2 CNN | stage 3 temporal | stage 4 transformer | gate |
|---|---|---|---|---|---|---|
| sound (heart/lung/mixed) | 3-class | **0.934** | 0.938 | 0.851 | 0.860 | adopt s1 |
| heart normal/abnormal | binary | **0.909** | 0.653 | 0.668 | 0.465 | adopt s1 |
| lung normal/abnormal | binary | **0.969** | 0.728 | 0.640 | 0.858 | adopt s1 |
| heart sound typing | 10-class | **0.845** | 0.324 | 0.335 | 0.773 | adopt s1 |
| lung sound typing | 6-class | **0.977** | 0.808 | 0.456 | 0.828 | adopt s1 |

**Headline:** a cheap MFCC + classical baseline is the champion on every task
under the gate rule — a strong production candidate — while the Transformer is
the best deep tier and the most mobile-friendly (0.34 GFLOPs, 2 MB int8).

![Benchmark summary](docs/assets/report/benchmark.png)

Training dynamics (loss + validation balanced-accuracy / macro-F1 per epoch):

| sound ResNet-18 | heart 10-class Transformer | lung 6-class ResNet-18 |
|---|---|---|
| ![](docs/assets/report/curves_stage2-sound-resnet18-allfolds-s0.png) | ![](docs/assets/report/curves_stage4-heart-transformer-10class-allfolds-s0.png) | ![](docs/assets/report/curves_stage2-lung-resnet18-6class-allfolds-s0.png) |

## Confusion matrices (normalized relative to the actual class)

Binary screening, source routing, fine-grained typing, and the combined view:

| heart normal/abnormal | lung normal/abnormal | heart/lung/mixed |
|---|---|---|
| ![](docs/assets/report/cm_heart_binary.png) | ![](docs/assets/report/cm_lung_binary.png) | ![](docs/assets/report/cm_sound_binary.png) |

| heart 10-class | lung 6-class | combined 16×16 |
|---|---|---|
| ![](docs/assets/report/cm_heart_10class.png) | ![](docs/assets/report/cm_lung_6class.png) | ![](docs/assets/report/cm_combined.png) |

## Inference distributions (post-softmax head output)

Mean post-softmax probability per actual class — the model's pre-decision
belief, not just the arg-max:

| sound ResNet-18 | heart 10-class Transformer | lung 6-class Transformer |
|---|---|---|
| ![](docs/assets/report/meanprob_sound.png) | ![](docs/assets/report/meanprob_heart.png) | ![](docs/assets/report/meanprob_lung.png) |

## How the models improve — training evolution

One clip pushed through the model at epochs 1 / 5 / 20 / 60: **input waveform →
processing (band-pass + log-mel) → output probabilities**, showing the decision
flip from wrong to correct as training proceeds.

| heart 10-class | lung 6-class | sound ResNet-18 |
|---|---|---|
| ![](docs/assets/report/evolution_heart_10class.png) | ![](docs/assets/report/evolution_lung_6class.png) | ![](docs/assets/report/evolution_sound_binary.png) |

## Synchronized inference timeline (the app's key output)

Input spectrogram + stacked-area head probabilities + confidence line, all
time-aligned — so the assessment is visible across breathing/cardiac cycles
rather than as a single full-clip label:

![Synchronized inference timeline](docs/assets/report/timeline_source.png)

Multi-scale (1 s / 3 s / 15 s) view with the whole-clip distribution as dashed
reference lines:

![Multi-scale](docs/assets/report/multiscale_source.png)

---

## Listen — sample clips

Dataset recordings served directly from this repository. Clips are re-encoded
to 22.05 kHz with light compression so they play in browsers and native
players (the original 4 kHz WAVs decode as silence in HTML5 `<audio>` and many
system players); raw versions are in `docs/assets/audio/` too.

| clip | playback |
|---|---|
| **Heart — normal** | <audio controls preload="none" src="https://raw.githubusercontent.com/Tnzr/VitalSense-ai/main/docs/assets/audio/heart_normal_filtered.wav"></audio> |
| **Heart — abnormal** (atrial fibrillation) | <audio controls preload="none" src="https://raw.githubusercontent.com/Tnzr/VitalSense-ai/main/docs/assets/audio/heart_abnormal_filtered.wav"></audio> |
| **Lung — normal** | <audio controls preload="none" src="https://raw.githubusercontent.com/Tnzr/VitalSense-ai/main/docs/assets/audio/lung_normal_filtered.wav"></audio> |
| **Lung — wheeze** | <audio controls preload="none" src="https://raw.githubusercontent.com/Tnzr/VitalSense-ai/main/docs/assets/audio/lung_wheeze_filtered.wav"></audio> |

---

## Web app — Inference-as-a-Service + Educational Game

Upload (or record with a phone/computer mic) one file for a full report, or many for
a batch table. The report leads with **condition analysis** — 10 heart types / 6 lung
types plus normal/abnormal screening — with the heart/lung/mixed router only deciding
which condition panels lead. It adds an interactive signal explorer and a quantified
audio report.

| Landing / upload | Synchronized report | Learn (game) |
|---|---|---|
| ![](docs/walkthrough/01_landing.png) | ![](docs/walkthrough/04_report_timeline.png) | ![](docs/walkthrough/06_game.png) |

**Full step-by-step tour:** [`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md).

### Log-mel gallery — what each sound type looks like

Regenerate with `python scripts/make_mel_gallery.py` (first 6 s of one example per class).

![Heart sound types — log-mel](docs/assets/mel_gallery_heart.png)
![Lung sound types — log-mel](docs/assets/mel_gallery_lung.png)

App highlights:

- **Condition panels** — ranked class probabilities with ± fold-model disagreement
  whiskers; click a class for its description and to highlight it over time.
- **Signal explorer** — browser-drawn waveform, log-mel spectrogram and probability
  over time on one axis, synced to audio: click to seek, drag to zoom, hover for
  time / Hz / dB.
- **Quantified audio report** — heart/breathing rate estimates, rhythm regularity,
  band energy split, spectral centroid/roll-off/flatness, SNR, clipping and model
  certainty (fold spread, margin, entropy, temporal agreement); JSON/CSV export + print.
- **Sound matrix** (`/matrix`) — one tile per heart/lung type with audio, a live
  log-mel plot, listen-for / look-for cues, compare (up to 3) and a hide-labels quiz.
- **Phone-mic capture** — in-browser 15 s recording → WAV → the same report
  (flagged as out-of-domain: models were trained on 4 kHz stethoscope clips).
- **Parametrized report** — choose heads, explanation depth (brief/standard/detailed),
  figure level, and temporal mode (single window or multi-scale 1/3/15 s).
- **Signal quality** — duration, RMS, clipping and SNR flags.
- **Health-awareness narrative** — plain-language context for each finding, with a
  forced not-a-diagnosis disclaimer.
- **Batch analysis** — table of source, screening, top type, confidence, verdict, quality.
- **Game mode** — scored ground-truth listening rounds with explanations.
- **JSON API** — `POST /api/analyze`, `GET /api/game/round`, `POST /api/game/answer`.

## Personalized AI narrative (LLM)

`POST /api/llm/report` + the `patient_context` / `ai_narrative` report options
generate a personalized narrative with **medical / diet / environment**
recommendations grounded in the findings, plus a follow-up chat. Works out of
the box with an offline synthesizer; enable an OpenAI-compatible LLM (or local
Ollama) with `ASCULTO_LLM_*` env vars. See
[`docs/LLM_Integration.md`](docs/LLM_Integration.md).

**Research directions** (documented in `docs/LLM_Integration.md`):

- **Token/embedding architecture** — the audio Transformer's encoder already
  organises patch tokens and class embeddings; `encode()` + an embedding
  extractor make them the interface to an LLM (see the
  [class-embedding similarity](docs/assets/report/embeddings_heart.png)).
- **Whole-breathing-cycle modeling** — short snippets are ambiguous; the app's
  macro pass already uses the full 15 s clip, and the roadmap adds cycle
  segmentation, long-context encoders and cycle-aware training for proper
  whole-cycle diagnosis.

---

## Quick start

```bash
git clone git@github.com:Tnzr/VitalSense-ai.git
cd VitalSense-ai
./setup.sh          # venv + deps + downloads HLS-CMDS (37.8 MB) from Zenodo if absent
```

`setup.sh` creates the venv, installs dependencies, downloads the dataset from
its Zenodo source if needed, writes `.env` (`ASCULTO_DATA_DIR`), and
**verifies the manifests** (`sound=535 heart=195 lung=195`).

```bash
# Tests (manifests, leakage-safe splits, DSP, web API)
.venv/bin/python -m pytest -q tests

# Web app
.venv/bin/python -m uvicorn webapp.app.main:app --port 8010     # http://localhost:8010

# Train one head
.venv/bin/python -m cardia.cli --task heart --stage 2 --model resnet18 --smoke

# Full smoke campaign (all tasks × 4 stages, with gates)
cd ml && SMOKE=1 ./scripts/run_sequential.sh

# Mobile readiness: export + INT8 quantize + measure
.venv/bin/python -m cardia.export.mobile --out-dir runs/export

# Regenerate the report + walkthrough screenshots
.venv/bin/python -m cardia.report.report --out-dir report
.venv/bin/python scripts/screenshot_walkthrough.py
```

## Repository layout

| path | content |
|---|---|
| `ml/cardia/` | training/eval pipeline: `data/`, `models/`, `train/`, `report/`, `export/`, `wandb_utils/` |
| `ml/scripts/` | sequential campaign orchestrator + gates (`run_sequential.sh`) |
| `webapp/` | FastAPI app: inference registry, report engine, game, templates, static |
| `scripts/` | `make_assets.py`, `screenshot_walkthrough.py` |
| `deploy/` | container images (`api`, `inference`), worker service + Modal batch worker |
| `gateway/` | Cloudflare Worker (Hono): auth, quotas, jobs, Stripe webhooks |
| `frontend/` | static SPA (Vite) for Cloudflare Pages |
| `desktop/` | Electron + ONNX offline app |
| `infra/terraform/` | Cloudflare / Neon / Modal / GitHub modules (R2 remote state) |
| `finance/` | `pricing.yaml` + unit-economics model (emits `docs/Finance.md`) |
| `billing/` | Stripe product catalogue + setup script |
| `legal/` | ToS / Privacy / DPA / Cookie templates |
| `docs/` | strategy, architecture, tech stack, app spec, mobile-compute analysis, walkthrough |

## Notebooks

End-to-end notebooks (Jupyter, kernel `asculto`) covering the entire process:

| notebook | covers |
|---|---|
| [`notebooks/01_dataset_exploration.ipynb`](notebooks/01_dataset_exploration.ipynb) | dataset, counts, class balance, embedded audio, spectrograms, signal quality |
| [`notebooks/02_preprocessing_splits.ipynb`](notebooks/02_preprocessing_splits.ipynb) | band-pass DSP, augmentation, leakage-safe grouped splits |
| [`notebooks/03_benchmark_training.ipynb`](notebooks/03_benchmark_training.ipynb) | classical baseline, deep smoke training, campaign/gate tables |
| [`notebooks/04_report_embeddings_webapp.ipynb`](notebooks/04_report_embeddings_webapp.ipynb) | encoder embeddings, app-style report, README audio/screenshots |
| [`notebooks/05_llm_integration.ipynb`](notebooks/05_llm_integration.ipynb) | personalized AI narrative, recommendations, optional live LLM |

```bash
.venv/bin/python -m ipykernel install --user --name cardiasense   # one-time (setup.sh does this)
.venv/bin/python -m jupyter notebook notebooks/                   # or: jupyter lab
```

The executed notebooks are also exported to a single **solutions document**:
[`docs/Notebook_Solutions.pdf`](docs/Notebook_Solutions.pdf) (18 pages).
Regenerate it with `scripts/export_solutions_pdf.sh` (needs `xelatex` +
`pdfunite`).

## Documentation

- [`docs/BusinessPlan.md`](docs/BusinessPlan.md) — strategy, markets, revenue
- [`docs/ProductComputeArchitecture.md`](docs/ProductComputeArchitecture.md) — tiers, edge/cloud, app layer
- [`docs/TechStack.md`](docs/TechStack.md) — stack choices + migration path
- [`docs/WebApp.md`](docs/WebApp.md) — app spec (IaaS + game, multi-scale)
- [`docs/LLM_Integration.md`](docs/LLM_Integration.md) — personalized AI reports + research directions
- [`docs/MobileCompute.md`](docs/MobileCompute.md) — measured model profile, phone latency, quantization
- [`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md) — instructional usage walkthrough
- [`docs/RnD_PrototypingPlan.md`](docs/RnD_PrototypingPlan.md) — R&D plan
- [`docs/DatasetLicenses.md`](docs/DatasetLicenses.md) — dataset license registry + attribution
- [`docs/Deployment.md`](docs/Deployment.md) — deployment runbook (containers, Terraform, Workers, Pages)
- [`docs/Finance.md`](docs/Finance.md) — unit economics, break-even, sensitivity, investor metrics

## Data

HLS-CMDS — Torabi, Y., Shirani, S., & Reilly, J. P. (2025). *HLS-CMDS: Heart and
Lung Sounds Dataset Recorded from a Clinical Manikin using Digital Stethoscope.*
IEEE Data Descriptions. DOI [10.1109/IEEEDATA.2025.3566012](https://doi.org/10.1109/IEEEDATA.2025.3566012)
(Zenodo record [15376628](https://zenodo.org/records/15376628)). Downloaded
automatically by `setup.sh`; not committed to this repository.

## Disclaimer

Educational and research prototype only. Not a medical device. Models are
trained on manikin recordings and must not be used for clinical decision-making.
