// Asculto static frontend entry. Consumes the gateway/worker JSON contract and
// renders the report client-side (base64 figures come from the report engine).

declare const __API_BASE__: string;
const API_BASE = typeof __API_BASE__ === "string" ? __API_BASE__ : "/v1";

interface Head {
  key: string;
  label: string;
  pred: string;
  confidence: number;
  n_models: number;
  kind: string;
}

interface Report {
  filename: string;
  verdict?: string;
  quality?: { duration_s: number; flags: string[]; ok: boolean };
  heads: Head[];
  narrative: string[];
  figures: Record<string, string>;
  ai_narrative?: {
    summary: string;
    provider: string;
    model: string;
    recommendations: { medical: string[]; diet: string[]; environment: string[] };
    next_steps: string[];
    disclaimer: string;
  };
  error?: string;
}

const token = localStorage.getItem("asculto_api_key") || "";

const VERDICT_PILL: Record<string, string> = { "abnormal-flag": "pill-bad", "reference-flag": "pill-warn" };
const VERDICT_LABEL: Record<string, string> = {
  "abnormal-flag": "abnormality flagged",
  "reference-flag": "reference-only screen flagged",
};

function authHeaders(): HeadersInit {
  return token ? { authorization: `Bearer ${token}` } : {};
}

async function analyze(file: File, opts: Record<string, string | boolean>): Promise<Report> {
  const form = new FormData();
  form.append("file", file);
  form.append("heads", String(opts.heads || ""));
  form.append("explanation", String(opts.explanation || "standard"));
  form.append("figures", "all");
  form.append("temporal", String(!!opts.temporal));
  form.append("temporal_mode", "multiscale");
  form.append("scales", "1,3,15");
  form.append("ai_narrative", "true");
  form.append("patient_context", String(opts.patient_context || ""));

  const res = await fetch(`${API_BASE}/analyze`, { method: "POST", headers: authHeaders(), body: form });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  return (await res.json()) as Report;
}

function esc(s: string): string {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c] as string));
}

function render(rep: Report): void {
  const el = document.getElementById("results")!;
  if (rep.error) {
    el.innerHTML = `<div class="card warn"><h2>Could not analyze</h2><pre>${esc(rep.error)}</pre></div>`;
    return;
  }
  const heads = rep.heads
    .map((h) => `<tr><td>${esc(h.label)}</td><td><strong>${esc(h.pred)}</strong></td>
      <td>${(h.confidence * 100).toFixed(1)}%</td><td>${h.n_models} folds</td></tr>`)
    .join("");
  const narrative = rep.narrative.map((n) => `<li>${esc(n)}</li>`).join("");
  const figs = Object.entries(rep.figures || {})
    .map(([name, uri]) => `<figure><img src="${uri}" alt="${esc(name)}" /></figure>`)
    .join("");
  const ai = rep.ai_narrative
    ? `<div class="card"><h2>Personalized AI narrative
         <span class="muted">${esc(rep.ai_narrative.provider)} · ${esc(rep.ai_narrative.model)}</span></h2>
         <p>${esc(rep.ai_narrative.summary)}</p>
         <h3>Medical</h3><ul>${rep.ai_narrative.recommendations.medical.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>
         <h3>Diet</h3><ul>${rep.ai_narrative.recommendations.diet.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>
         <h3>Environment</h3><ul>${rep.ai_narrative.recommendations.environment.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>
         <h3>Next steps</h3><ul>${rep.ai_narrative.next_steps.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>
         <p class="muted">${esc(rep.ai_narrative.disclaimer)}</p></div>`
    : "";

  el.innerHTML = `
    <section class="report-head">
      <h1>Report — ${esc(rep.filename)}</h1>
      <span class="pill ${VERDICT_PILL[rep.verdict ?? ""] ?? "pill-ok"}">
        ${VERDICT_LABEL[rep.verdict ?? ""] ?? "no abnormality flagged"}</span>
    </section>
    ${rep.quality?.flags?.length ? `<div class="card warn"><strong>Quality:</strong> ${esc(rep.quality.flags.join("; "))}</div>` : ""}
    <div class="card"><h2>Summary</h2><ul class="narrative">${narrative}</ul></div>
    ${ai}
    <div class="card"><h2>Predictions</h2>
      <table><thead><tr><th>Head</th><th>Prediction</th><th>Confidence</th><th>Ensemble</th></tr></thead>
      <tbody>${heads}</tbody></table></div>
    ${figs ? `<div class="card"><h2>Figures</h2><div class="figures">${figs}</div></div>` : ""}
  `;
}

document.getElementById("analyze-form")!.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const status = document.getElementById("status")!;
  const file = (document.getElementById("file") as HTMLInputElement).files?.[0];
  if (!file) return;
  status.textContent = "Analyzing…";
  try {
    const rep = await analyze(file, {
      explanation: (document.getElementById("explanation") as HTMLSelectElement).value,
      temporal: (document.getElementById("temporal") as HTMLInputElement).checked,
      patient_context: (document.getElementById("patient_context") as HTMLInputElement).value,
    });
    render(rep);
    status.textContent = "";
  } catch (err) {
    status.textContent = `Error: ${(err as Error).message}`;
  }
});

// PWA installability: register the service worker on http(s) only (skips
// Capacitor's capacitor:// / file:// schemes where SWs are unsupported).
if ("serviceWorker" in navigator && location.protocol.startsWith("http")) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js").catch(() => {});
  });
}
