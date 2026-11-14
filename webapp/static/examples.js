// Landing-page "Hear the difference": full-clip log-mel under each sample player.
(function () {
  const V = window.AscultoViz;
  if (!V) return;
  document.querySelectorAll("canvas.example-mel").forEach(async (c) => {
    const d = c.dataset;
    const q = new URLSearchParams({ task: d.task, scheme: d.scheme });
    const r = await fetch(`/api/matrix/mel/${encodeURIComponent(d.sample)}?${q}`);
    if (!r.ok) { c.hidden = true; return; }
    const viz = await r.json();
    const audio = c.closest(".example").querySelector("audio");
    V.MiniMel(c, viz, audio, viz.duration);
  });
})();
