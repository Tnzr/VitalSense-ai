/* Interactive signal visualizations: log-mel spectrogram, waveform and
 * probability-over-time tracks sharing one time axis, synced to <audio>. */
(function (global) {
  "use strict";
  const STOPS = [[0, 0, 4], [31, 12, 72], [85, 15, 109], [136, 34, 106], [186, 54, 85],
    [227, 89, 51], [249, 140, 10], [249, 201, 50], [252, 255, 164]];
  const LUT = (() => {
    const lut = new Uint8ClampedArray(256 * 3);
    for (let i = 0; i < 256; i++) {
      const x = (i / 255) * (STOPS.length - 1);
      const k = Math.min(Math.floor(x), STOPS.length - 2);
      const f = x - k;
      for (let c = 0; c < 3; c++) lut[i * 3 + c] = STOPS[k][c] + (STOPS[k + 1][c] - STOPS[k][c]) * f;
    }
    return lut;
  })();
  const PALETTE = ["#4cc2c4", "#f28e2b", "#e15759", "#59a14f", "#edc948", "#b07aa1",
    "#ff9da7", "#9c755f", "#76b7b2", "#bab0ac"];
  const hz2mel = (f) => 2595 * Math.log10(1 + f / 700);

  function decode(b64) {
    const bin = atob(b64);
    const a = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) a[i] = bin.charCodeAt(i);
    return a;
  }

  function melImage(mel) {
    const q = decode(mel.data);
    const W = mel.frames, H = mel.n_mels;
    const c = document.createElement("canvas");
    c.width = W; c.height = H;
    const ctx = c.getContext("2d");
    const img = ctx.createImageData(W, H);
    for (let m = 0; m < H; m++) {
      for (let t = 0; t < W; t++) {
        const v = q[m * W + t], o = ((H - 1 - m) * W + t) * 4;
        img.data[o] = LUT[v * 3]; img.data[o + 1] = LUT[v * 3 + 1]; img.data[o + 2] = LUT[v * 3 + 2];
        img.data[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    return { canvas: c, q };
  }

  function fit(canvas) {
    const dpr = global.devicePixelRatio || 1;
    const w = canvas.clientWidth || 600, h = canvas.clientHeight || 120;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }

  function melFreqY(f, h, mel) {
    const top = mel.hz.length ? hz2mel(mel.hz[mel.hz.length - 1] * 1.02) : hz2mel(2000);
    return h - (hz2mel(f) / top) * h;
  }

  function drawMel(ctx, w, h, img, mel, t0, t1, axes) {
    ctx.imageSmoothingEnabled = true;
    const sx = t0 / mel.hop_s, sw = Math.max((t1 - t0) / mel.hop_s, 1);
    ctx.drawImage(img.canvas, sx, 0, sw, mel.n_mels, 0, 0, w, h);
    if (!axes) return;
    ctx.font = "10px system-ui, sans-serif";
    ctx.fillStyle = "rgba(255,255,255,0.8)";
    ctx.strokeStyle = "rgba(255,255,255,0.15)";
    [100, 250, 500, 1000, 1500].forEach((f) => {
      const y = melFreqY(f, h, mel);
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      ctx.fillText(f >= 1000 ? f / 1000 + " kHz" : f + " Hz", 4, y - 2);
    });
  }

  function niceStep(span) {
    const steps = [0.1, 0.2, 0.5, 1, 2, 5, 10];
    return steps.find((s) => span / s <= 10) || 10;
  }

  /** Multi-track explorer. tracks: {wave, mel, prob} canvases (any may be null). */
  function SignalExplorer(o) {
    const viz = o.viz, dur = viz.duration;
    const img = melImage(viz.mel);
    const st = { t0: 0, t1: dur, hover: null, drag: null, head: o.head || null, highlight: null };
    const audio = o.audio || null;
    const canvases = [o.wave, o.mel, o.prob].filter(Boolean);

    const tOf = (canvas, x) => st.t0 + (x / canvas.clientWidth) * (st.t1 - st.t0);
    const xOf = (t, w) => ((t - st.t0) / (st.t1 - st.t0)) * w;
    const now = () => (audio ? audio.currentTime : st.cursor || 0);

    function overlay(ctx, w, h) {
      if (st.drag && st.drag.x1 !== undefined) {
        ctx.fillStyle = "rgba(76,194,196,0.18)";
        const a = Math.min(st.drag.t0, st.drag.t1), b = Math.max(st.drag.t0, st.drag.t1);
        ctx.fillRect(xOf(a, w), 0, xOf(b, w) - xOf(a, w), h);
      }
      if (st.hover !== null) {
        ctx.strokeStyle = "rgba(255,255,255,0.35)"; ctx.setLineDash([3, 3]);
        const x = xOf(st.hover, w);
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); ctx.setLineDash([]);
      }
      const px = xOf(now(), w);
      ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, h); ctx.stroke(); ctx.lineWidth = 1;
    }

    function drawWave() {
      const { ctx, w, h } = fit(o.wave);
      ctx.fillStyle = "#0b121a"; ctx.fillRect(0, 0, w, h);
      const n = viz.wave.min.length;
      ctx.strokeStyle = "#4cc2c4";
      ctx.beginPath();
      for (let x = 0; x < w; x++) {
        const ta = tOf(o.wave, x), tb = tOf(o.wave, x + 1);
        let i0 = Math.floor((ta / dur) * n), i1 = Math.max(Math.ceil((tb / dur) * n), i0 + 1);
        let lo = 0, hi = 0;
        for (let i = Math.max(i0, 0); i < Math.min(i1, n); i++) {
          lo = Math.min(lo, viz.wave.min[i]); hi = Math.max(hi, viz.wave.max[i]);
        }
        ctx.moveTo(x + 0.5, h / 2 - hi * h * 0.46); ctx.lineTo(x + 0.5, h / 2 - lo * h * 0.46);
      }
      ctx.stroke();
      overlay(ctx, w, h);
    }

    function drawMelTrack() {
      const { ctx, w, h } = fit(o.mel);
      drawMel(ctx, w, h, img, viz.mel, st.t0, st.t1, true);
      overlay(ctx, w, h);
    }

    function drawProb() {
      const { ctx, w, h } = fit(o.prob);
      ctx.fillStyle = "#0b121a"; ctx.fillRect(0, 0, w, h);
      const pad = 16, ph = h - pad;
      ctx.strokeStyle = "rgba(255,255,255,0.08)";
      ctx.fillStyle = "#93a4b5"; ctx.font = "10px system-ui";
      [0.25, 0.5, 0.75].forEach((p) => {
        ctx.beginPath(); ctx.moveTo(0, ph - p * ph); ctx.lineTo(w, ph - p * ph); ctx.stroke();
        ctx.fillText(p * 100 + "%", w - 26, ph - p * ph - 2);
      });
      const s = o.series && st.head ? o.series[st.head] : null;
      if (s && s.times.length) {
        const means = s.classes.map((_, k) => s.probs.reduce((a, r) => a + r[k], 0) / s.probs.length);
        const order = means.map((m, k) => [m, k]).sort((a, b) => b[0] - a[0]).map((x) => x[1]);
        const strong = new Set(order.slice(0, 4));
        order.slice().reverse().forEach((k) => {
          const hl = st.highlight === null ? strong.has(k) : st.highlight === s.classes[k];
          ctx.strokeStyle = PALETTE[k % PALETTE.length];
          ctx.globalAlpha = hl ? 1 : 0.18; ctx.lineWidth = hl ? 2 : 1;
          ctx.beginPath();
          s.times.forEach((t, i) => {
            const x = xOf(t, w), y = ph - s.probs[i][k] * ph;
            i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
          });
          ctx.stroke();
        });
        ctx.globalAlpha = 1; ctx.lineWidth = 1;
      } else {
        ctx.fillStyle = "#93a4b5"; ctx.font = "12px system-ui";
        ctx.fillText("Enable the temporal sweep to see probability over time.", 10, 20);
      }
      ctx.fillStyle = "#93a4b5"; ctx.font = "10px system-ui";
      const step = niceStep(st.t1 - st.t0);
      for (let t = Math.ceil(st.t0 / step) * step; t <= st.t1; t += step) {
        const x = xOf(t, w);
        ctx.fillText(t.toFixed(step < 1 ? 1 : 0) + "s", x + 2, h - 3);
        ctx.fillRect(x, ph, 1, 4);
      }
      overlay(ctx, w, ph);
    }

    function draw() {
      if (o.wave) drawWave();
      if (o.mel) drawMelTrack();
      if (o.prob) drawProb();
      if (o.range) o.range.textContent = `${st.t0.toFixed(1)}–${st.t1.toFixed(1)} s` +
        (st.t1 - st.t0 < dur - 1e-6 ? " (zoomed · double-click to reset)" : "");
    }

    function tip(canvas, e, t) {
      if (!o.tip) return;
      let html = `<b>${t.toFixed(2)} s</b>`;
      if (canvas === o.mel) {
        const mel = viz.mel, h = canvas.clientHeight;
        const top = hz2mel(mel.hz[mel.hz.length - 1] * 1.02);
        const f = 700 * (Math.pow(10, ((1 - e.offsetY / h) * top) / 2595) - 1);
        let m = 0;
        mel.hz.forEach((c, i) => { if (Math.abs(c - f) < Math.abs(mel.hz[m] - f)) m = i; });
        const fr = Math.min(Math.max(Math.round(t / mel.hop_s), 0), mel.frames - 1);
        const db = mel.db_min + (img.q[m * mel.frames + fr] / 255) * (mel.db_max - mel.db_min);
        html += ` · ${Math.round(f)} Hz · ${db.toFixed(1)} dB`;
      } else if (canvas === o.prob && o.series && st.head && o.series[st.head]) {
        const s = o.series[st.head];
        let i = 0;
        s.times.forEach((tt, j) => { if (Math.abs(tt - t) < Math.abs(s.times[i] - t)) i = j; });
        const top = s.classes.map((c, k) => [s.probs[i][k], c, k]).sort((a, b) => b[0] - a[0]).slice(0, 3);
        html += top.map(([p, c, k]) =>
          `<br><span style="color:${PALETTE[k % PALETTE.length]}">■</span> ${c} ${(p * 100).toFixed(0)}%`).join("");
      } else if (canvas === o.wave) {
        const n = viz.wave.max.length, i = Math.min(Math.floor((t / dur) * n), n - 1);
        html += ` · peak ${Math.max(Math.abs(viz.wave.min[i]), viz.wave.max[i]).toFixed(2)}`;
      }
      o.tip.innerHTML = html;
      o.tip.style.display = "block";
      const box = o.tip.parentElement.getBoundingClientRect();
      const r = canvas.getBoundingClientRect();
      o.tip.style.left = Math.min(r.left - box.left + e.offsetX + 12, box.width - 190) + "px";
      o.tip.style.top = (r.top - box.top + e.offsetY + 12) + "px";
    }

    canvases.forEach((c) => {
      c.style.touchAction = "none";
      c.addEventListener("pointermove", (e) => {
        const t = tOf(c, e.offsetX);
        st.hover = t;
        if (st.drag) { st.drag.t1 = t; st.drag.x1 = e.offsetX; }
        tip(c, e, t); draw();
      });
      c.addEventListener("pointerleave", () => {
        st.hover = null; if (o.tip) o.tip.style.display = "none"; draw();
      });
      c.addEventListener("pointerdown", (e) => {
        c.setPointerCapture(e.pointerId);
        st.drag = { t0: tOf(c, e.offsetX), x0: e.offsetX };
      });
      c.addEventListener("pointerup", (e) => {
        const d = st.drag; st.drag = null;
        if (!d) return;
        if (Math.abs(e.offsetX - d.x0) < 5) {
          const t = Math.min(Math.max(tOf(c, e.offsetX), 0), dur);
          if (audio) { audio.currentTime = t; } else { st.cursor = t; }
        } else {
          const a = Math.max(Math.min(d.t0, d.t1), 0), b = Math.min(Math.max(d.t0, d.t1), dur);
          if (b - a > 0.2) { st.t0 = a; st.t1 = b; }
        }
        draw();
      });
      c.addEventListener("dblclick", () => { st.t0 = 0; st.t1 = dur; draw(); });
    });

    if (audio) {
      let raf = null;
      const loop = () => {
        const t = audio.currentTime;
        if (!audio.paused && (t > st.t1 || t < st.t0)) {
          const span = st.t1 - st.t0;
          st.t0 = Math.min(Math.max(t - span * 0.1, 0), dur - span); st.t1 = st.t0 + span;
        }
        draw();
        raf = audio.paused ? null : requestAnimationFrame(loop);
      };
      ["play", "seeked", "timeupdate"].forEach((ev) => audio.addEventListener(ev, () => { if (!raf) loop(); }));
    }
    global.addEventListener("resize", draw);
    draw();
    return {
      setHead(k) { st.head = k; st.highlight = null; draw(); },
      setHighlight(c) { st.highlight = c; draw(); },
      palette: PALETTE,
      redraw: draw,
    };
  }

  /** Compact spectrogram with playhead + click-to-seek (matrix tiles). Shows a
   *  `span`-second page that follows the playhead so individual beats stay visible. */
  function MiniMel(canvas, viz, audio, span) {
    const img = melImage(viz.mel);
    const win = Math.min(span || viz.duration, viz.duration);
    const page = () => {
      const t = audio ? audio.currentTime : 0;
      const t0 = Math.min(Math.floor(t / win) * win, Math.max(viz.duration - win, 0));
      return [t0, t0 + win];
    };
    const draw = () => {
      const { ctx, w, h } = fit(canvas);
      const [t0, t1] = page();
      drawMel(ctx, w, h, img, viz.mel, t0, t1, canvas.clientHeight > 120);
      ctx.fillStyle = "rgba(255,255,255,0.75)"; ctx.font = "10px system-ui";
      ctx.fillText(`${t0.toFixed(0)}–${t1.toFixed(0)} s`, w - 44, 12);
      if (audio && audio.currentTime > 0) {
        const x = ((audio.currentTime - t0) / win) * w;
        ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
      }
    };
    let raf = null;
    const loop = () => { draw(); raf = audio.paused ? null : requestAnimationFrame(loop); };
    const kick = () => { if (!raf) loop(); };
    const seek = (e) => {
      const [t0] = page();
      audio.currentTime = t0 + (e.offsetX / canvas.clientWidth) * win;
      if (audio.paused) audio.play();
    };
    const evs = ["play", "seeked", "timeupdate"];
    if (audio) {
      evs.forEach((ev) => audio.addEventListener(ev, kick));
      canvas.addEventListener("click", seek);
    }
    global.addEventListener("resize", draw);
    draw();
    return {
      redraw: draw,
      dispose() {
        global.removeEventListener("resize", draw);
        if (raf) cancelAnimationFrame(raf);
        if (audio) { evs.forEach((ev) => audio.removeEventListener(ev, kick)); canvas.removeEventListener("click", seek); }
      },
    };
  }

  global.AscultoViz = { SignalExplorer, MiniMel, PALETTE };
})(window);
