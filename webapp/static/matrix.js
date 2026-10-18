/* Sound matrix: lazy mini log-mels, example switching, compare tray, quiz mode. */
(function () {
  "use strict";
  const V = window.AscultoViz;
  const cache = {};

  function melUrl(t, sid) {
    return `/api/matrix/mel/${encodeURIComponent(sid)}?task=${t.dataset.task}&scheme=${t.dataset.scheme}`;
  }
  function audioUrl(t, sid) {
    return `/game/audio/${encodeURIComponent(sid)}?task=${t.dataset.task}&scheme=${t.dataset.scheme}`;
  }
  async function getMel(t, sid) {
    const url = melUrl(t, sid);
    if (!cache[url]) cache[url] = fetch(url).then((r) => (r.ok ? r.json() : null));
    return cache[url];
  }

  async function show(tile, idx) {
    const ids = JSON.parse(tile.dataset.samples || "[]");
    if (!ids.length) return;
    const sid = ids[idx];
    tile.dataset.current = sid;
    const oldA = tile.querySelector("audio"), a = oldA.cloneNode();
    a.src = audioUrl(tile, sid); oldA.replaceWith(a);
    const oldC = tile.querySelector("canvas"), c = oldC.cloneNode();
    oldC.replaceWith(c);
    tile.querySelectorAll(".tile-samples .btn").forEach((b) => b.classList.toggle("on", +b.dataset.idx === idx));
    const viz = await getMel(tile, sid);
    if (viz) V.MiniMel(c, viz, a, 5);
    renderCompare();
  }

  const tiles = Array.from(document.querySelectorAll(".tile"));
  const io = new IntersectionObserver((entries) => entries.forEach((e) => {
    if (e.isIntersecting) { io.unobserve(e.target); show(e.target, 0); }
  }), { rootMargin: "200px" });
  tiles.forEach((t) => {
    io.observe(t);
    t.querySelectorAll(".tile-samples .btn").forEach((b) => b.addEventListener("click", () => show(t, +b.dataset.idx)));
    const cmp = t.querySelector(".cmp");
    if (cmp) cmp.addEventListener("change", () => {
      const on = tiles.filter((x) => x.querySelector(".cmp:checked"));
      if (on.length > 3) cmp.checked = false;
      renderCompare();
    });
    t.querySelector(".tile-label").addEventListener("click", () => t.classList.toggle("revealed"));
  });

  async function renderCompare() {
    const box = document.getElementById("compare"), row = document.getElementById("compare-row");
    const sel = tiles.filter((x) => x.querySelector(".cmp:checked"));
    box.hidden = !sel.length;
    row.innerHTML = "";
    for (const t of sel) {
      const ids = JSON.parse(t.dataset.samples), sid = t.dataset.current || ids[0];
      const col = document.createElement("div");
      col.className = "compare-col";
      col.innerHTML = `<h3 class="tile-label">${t.dataset.label}</h3><canvas class="mini-mel big"></canvas>
        <audio controls src="${audioUrl(t, sid)}"></audio>`;
      if (t.classList.contains("revealed")) col.classList.add("revealed");
      row.appendChild(col);
      const viz = await getMel(t, sid);
      if (viz) V.MiniMel(col.querySelector("canvas"), viz, col.querySelector("audio"), 5);
    }
  }

  document.getElementById("quiz-mode").addEventListener("change", (e) => {
    document.body.classList.toggle("quiz", e.target.checked);
    tiles.forEach((t) => t.classList.remove("revealed"));
  });
  document.addEventListener("play", (e) => {
    document.querySelectorAll("audio").forEach((a) => { if (a !== e.target) a.pause(); });
  }, true);
  if (location.hash) {
    const t = document.querySelector(location.hash);
    if (t) { t.classList.add("flash"); t.scrollIntoView({ block: "center" }); }
  }
})();
