/* In-browser microphone capture (phone or computer) -> 16-bit WAV -> /analyze. */
(function () {
  "use strict";
  const btn = document.getElementById("rec-btn");
  if (!btn) return;
  const status = document.getElementById("rec-status");
  const meter = document.getElementById("rec-meter");
  const preview = document.getElementById("rec-preview");
  const use = document.getElementById("rec-use");
  const input = document.getElementById("files-input");
  const form = document.getElementById("analyze-form");
  const MAX_S = 15;
  let ctx, stream, src, node, chunks, started, timer, blob, level = 0;

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.isSecureContext) {
    status.textContent = "Microphone capture needs HTTPS (or localhost) and a browser with getUserMedia.";
    btn.disabled = true;
    return;
  }

  function drawMeter() {
    const g = meter.getContext("2d"), w = meter.width, h = meter.height;
    g.fillStyle = "#0b121a"; g.fillRect(0, 0, w, h);
    const el = started ? (performance.now() - started) / 1000 : 0;
    g.fillStyle = "#4c72b0"; g.fillRect(0, 0, (Math.min(el, MAX_S) / MAX_S) * w, h / 3);
    g.fillStyle = level > 0.9 ? "#c44e52" : "#4cc2c4";
    g.fillRect(0, h / 2, Math.min(level * 4, 1) * w, h / 2);
  }

  async function start() {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    src = ctx.createMediaStreamSource(stream);
    node = ctx.createScriptProcessor(4096, 1, 1);
    chunks = [];
    node.onaudioprocess = (e) => {
      const d = e.inputBuffer.getChannelData(0);
      chunks.push(new Float32Array(d));
      let s = 0; for (let i = 0; i < d.length; i++) s += d[i] * d[i];
      level = Math.sqrt(s / d.length);
    };
    src.connect(node); node.connect(ctx.destination);
    started = performance.now();
    timer = setInterval(() => {
      drawMeter();
      const el = (performance.now() - started) / 1000;
      status.textContent = `Recording… ${el.toFixed(1)} / ${MAX_S} s at ${ctx.sampleRate} Hz`;
      if (el >= MAX_S) stop();
    }, 100);
    btn.textContent = "Stop";
    btn.classList.add("recording");
  }

  function stop() {
    clearInterval(timer);
    node.disconnect(); src.disconnect();
    stream.getTracks().forEach((t) => t.stop());
    const sr = ctx.sampleRate;
    ctx.close();
    const n = chunks.reduce((a, c) => a + c.length, 0);
    const pcm = new Float32Array(n);
    let o = 0; chunks.forEach((c) => { pcm.set(c, o); o += c.length; });
    blob = encodeWav(pcm, sr);
    preview.src = URL.createObjectURL(blob);
    preview.hidden = false;
    use.disabled = false;
    started = null; level = 0; drawMeter();
    status.textContent = `Captured ${(n / sr).toFixed(1)} s at ${sr} Hz. Listen back, then analyze.`;
    btn.textContent = "Record again";
    btn.classList.remove("recording");
  }

  function encodeWav(pcm, sr) {
    const buf = new ArrayBuffer(44 + pcm.length * 2), v = new DataView(buf);
    const str = (off, s) => { for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)); };
    str(0, "RIFF"); v.setUint32(4, 36 + pcm.length * 2, true); str(8, "WAVE");
    str(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    str(36, "data"); v.setUint32(40, pcm.length * 2, true);
    for (let i = 0; i < pcm.length; i++) {
      const s = Math.max(-1, Math.min(1, pcm[i]));
      v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Blob([buf], { type: "audio/wav" });
  }

  btn.addEventListener("click", () => {
    if (started) { stop(); return; }
    start().catch((e) => { status.textContent = "Microphone unavailable: " + e.message; });
  });

  use.addEventListener("click", () => {
    if (!blob) return;
    const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    const file = new File([blob], `mic-${stamp}.wav`, { type: "audio/wav" });
    try {
      const dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
      form.submit();
    } catch (e) {
      const fd = new FormData(form);
      fd.delete("files"); fd.append("files", file);
      status.textContent = "Analyzing…";
      fetch(form.action, { method: "POST", body: fd }).then((r) => r.text()).then((html) => {
        document.open(); document.write(html); document.close();
      });
    }
  });
  drawMeter();
})();
