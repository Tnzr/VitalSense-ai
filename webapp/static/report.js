/* Report page: condition panels, signal explorer wiring, JSON/CSV export. */
(function () {
  "use strict";
  const el = document.getElementById("report-data");
  if (!el) return;
  const data = JSON.parse(el.textContent);
  const V = window.AscultoViz;
  const heads = Object.fromEntries((data.heads || []).map((h) => [h.key, h]));
  const pct = (x) => (x * 100).toFixed(1) + "%";
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  let explorer = null;
  const headSel = document.getElementById("trk-head");
  const legend = document.getElementById("trk-legend");

  function renderLegend(key) {
    if (!legend) return;
    const s = data.series && data.series[key];
    legend.innerHTML = s ? s.classes.map((c, k) =>
      `<button class="legend-item" data-cls="${esc(c)}"><span style="background:${V.PALETTE[k % V.PALETTE.length]}"></span>${esc(c)}</button>`
    ).join("") : "";
    legend.querySelectorAll(".legend-item").forEach((b) => b.addEventListener("click", () => {
      const on = b.classList.toggle("on");
      legend.querySelectorAll(".legend-item").forEach((o) => { if (o !== b) o.classList.remove("on"); });
      explorer && explorer.setHighlight(on ? b.dataset.cls : null);
    }));
  }

  function focusHead(key, cls) {
    if (!explorer || !data.series || !data.series[key]) return;
    if (headSel) headSel.value = key;
    explorer.setHead(key);
    renderLegend(key);
    if (cls) {
      explorer.setHighlight(cls);
      legend.querySelectorAll(".legend-item").forEach((o) => o.classList.toggle("on", o.dataset.cls === cls));
    }
  }

  // ---- condition panels
  const wrap = document.getElementById("panels");
  if (wrap) {
    data.panels.forEach((p) => {
      const h = heads[p.key];
      if (!h) return;
      const std = h.probs_std || h.probs.map(() => 0);
      const order = h.probs.map((v, i) => i).sort((a, b) => h.probs[b] - h.probs[a]);
      const top = order[0];
      const div = document.createElement("div");
      div.className = "panel" + (p.routed ? "" : " dim");
      div.innerHTML = `
        <div class="panel-head"><h3>${esc(h.label)}</h3>
          <span class="pill ${p.routed ? "pill-route" : ""}">${p.routed ? "routed" : "reference only"}</span></div>
        <div class="panel-top"><span class="top-pred">${esc(h.pred)}</span>
          <span class="top-conf">${pct(h.probs[top])} ± ${pct(std[top])}</span>
          <span class="muted small">${h.n_models} fold models</span></div>
        <ol class="bars">${order.map((i) => {
          const lo = Math.max(h.probs[i] - std[i], 0), hi = Math.min(h.probs[i] + std[i], 1);
          const k = data.knowledge[h.key + ":" + h.classes[i]];
          return `<li data-cls="${esc(h.classes[i])}" title="${esc(k ? k[0] : "")}" class="${i === top ? "top" : ""}">
            <span class="bar-label">${esc(h.classes[i])}</span>
            <span class="bar-track"><span class="bar-fill" style="width:${h.probs[i] * 100}%"></span>
              <span class="bar-err" style="left:${lo * 100}%;width:${(hi - lo) * 100}%"></span></span>
            <span class="bar-val">${pct(h.probs[i])}</span></li>`;
        }).join("")}</ol>
        <p class="panel-detail muted">${p.routed ? "Click a class for a plain-language description." :
          "The source router did not route this clip here; shown for comparison."}</p>`;
      const detail = div.querySelector(".panel-detail");
      div.querySelectorAll(".bars li").forEach((li) => li.addEventListener("click", () => {
        div.querySelectorAll(".bars li").forEach((o) => o.classList.toggle("sel", o === li));
        const c = li.dataset.cls, k = data.knowledge[h.key + ":" + c];
        const typing = h.kind === "multiclass";
        const slug = (p.key + "-" + c).toLowerCase().replace(/[^a-z0-9]+/g, "-");
        detail.innerHTML = `<b>${esc(c)}</b> — ${k ? esc(k[1] || k[0]) :
          esc(typing ? "" : "Screening head: probability that the clip is " + c + ".")}` +
          (typing ? ` <a href="/matrix#${slug}">Hear examples in the sound matrix →</a>` : "");
        focusHead(p.key, c);
      }));
      wrap.appendChild(div);
    });
  }

  // ---- signal explorer
  if (data.viz && document.getElementById("trk-mel")) {
    const keys = data.series ? Object.keys(data.series) : [];
    const lead = (data.panels.find((p) => keys.includes(p.key)) || {}).key || keys[0] || null;
    if (data.viz.source_duration > data.viz.duration + 0.5) {
      const note = document.getElementById("trk-trunc");
      note.hidden = false;
      note.textContent = `The explorer and quantified metrics cover the first ${data.viz.duration.toFixed(0)} s of this ` +
        `${data.viz.source_duration.toFixed(0)} s recording; playback and probability-over-time use the full clip.`;
    }
    explorer = V.SignalExplorer({
      viz: data.viz, audio: document.getElementById("report-audio"),
      wave: document.getElementById("trk-wave"), mel: document.getElementById("trk-mel"),
      prob: document.getElementById("trk-prob"), tip: document.getElementById("trk-tip"),
      range: document.getElementById("trk-range"), series: data.series, head: lead,
    });
    if (lead) focusHead(lead);
    if (headSel) headSel.addEventListener("change", () => focusHead(headSel.value));
  }

  // ---- exports
  const base = (data.filename || "report").replace(/\.[^.]+$/, "");
  function download(name, text, type) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([text], { type }));
    a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
  }
  const exportable = () => { const d = Object.assign({}, data); delete d.viz; return d; };
  const q = (v) => { const s = String(v ?? ""); return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; };
  const btnJ = document.getElementById("dl-json"), btnC = document.getElementById("dl-csv");
  if (btnJ) btnJ.addEventListener("click", () =>
    download(base + ".vitalsense.json", JSON.stringify(exportable(), null, 2), "application/json"));
  if (btnC) btnC.addEventListener("click", () => {
    const rows = [["section", "metric", "value", "unit", "note"]];
    (data.heads || []).forEach((h) => h.classes.forEach((c, i) => rows.push(
      ["Prediction: " + h.label, c, h.probs[i].toFixed(4), "probability",
        h.probs_std ? "fold std " + h.probs_std[i].toFixed(4) : ""])));
    ((data.quant && data.quant.sections) || []).forEach((s) => s.rows.forEach((r) =>
      rows.push([s.title, r.metric, r.value, r.unit, r.note])));
    download(base + ".vitalsense.csv", rows.map((r) => r.map(q).join(",")).join("\n"), "text/csv");
  });
})();
