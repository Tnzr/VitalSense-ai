// Over-time charts for saved exams: daily mean, min–max whisker, 7-day average.
(function () {
  var el = document.getElementById("exam-data");
  if (!el) return;
  var trends = JSON.parse(el.textContent || "{}");
  var tip = document.getElementById("chart-tip");
  var COLORS = { heart_binary: "#c44e52", lung_binary: "#4c72b0", heart_rate: "#c44e52", breathing_rate: "#4c72b0" };
  var DAY = 86400000;

  function day(d) { return Date.parse(d + "T00:00:00Z"); }  // UTC, matching toISOString() labels

  function draw(canvasId, legendId, keys, fixedRange, fmt) {
    var cv = document.getElementById(canvasId);
    if (!cv) return;
    var series = keys.filter(function (k) { return trends[k]; }).map(function (k) {
      return { key: k, t: trends[k] };
    });
    var legend = document.getElementById(legendId);
    legend.innerHTML = series.map(function (s) {
      return '<span><i style="background:' + COLORS[s.key] + '"></i>' + s.t.label +
        (s.t.unit === "probability" ? "" : " (" + s.t.unit + ")") + "</span>";
    }).join("") || '<span class="hint">No data for this chart yet.</span>';
    var ctx = cv.getContext("2d");
    var dpr = window.devicePixelRatio || 1;
    var W = cv.clientWidth, H = cv.clientHeight;
    cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    if (!series.length) return;
    var xs = [], ys = [];
    series.forEach(function (s) {
      s.t.daily.forEach(function (p) { xs.push(day(p.date)); ys.push(p.min, p.max); });
    });
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
    if (x1 === x0) { x0 -= DAY; x1 += DAY; }
    var y0 = fixedRange ? fixedRange[0] : Math.min.apply(null, ys), y1 = fixedRange ? fixedRange[1] : Math.max.apply(null, ys);
    if (!fixedRange) { var pad = (y1 - y0) * 0.15 || 5; y0 = Math.max(0, y0 - pad); y1 += pad; }
    var L = 48, R = 12, T = 10, B = 26;
    function X(t) { return L + (t - x0) / (x1 - x0) * (W - L - R); }
    function Y(v) { return T + (1 - (v - y0) / (y1 - y0)) * (H - T - B); }
    ctx.font = "11px system-ui, sans-serif"; ctx.fillStyle = "#93a4b5"; ctx.strokeStyle = "#22303d"; ctx.lineWidth = 1;
    for (var i = 0; i <= 4; i++) {
      var v = y0 + (y1 - y0) * i / 4, y = Y(v);
      ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(W - R, y); ctx.stroke();
      ctx.fillText(fmt(v), 4, y + 4);
    }
    var nT = Math.max(2, Math.min(6, Math.round((x1 - x0) / DAY) + 1, Math.floor((W - L - R) / 56)));
    for (i = 0; i < nT; i++) {
      var t = x0 + (x1 - x0) * i / Math.max(nT - 1, 1);
      ctx.fillText(new Date(t).toISOString().slice(5, 10), X(t) - 14, H - 8);
    }
    var marks = [];
    series.forEach(function (s) {
      var c = COLORS[s.key];
      ctx.strokeStyle = c; ctx.fillStyle = c;
      s.t.daily.forEach(function (p) {
        var x = X(day(p.date));
        ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x, Y(p.min)); ctx.lineTo(x, Y(p.max)); ctx.stroke();
        ctx.beginPath(); ctx.arc(x, Y(p.mean), 4, 0, 2 * Math.PI); ctx.fill();
        marks.push({ x: x, y: Y(p.mean), p: p, s: s });
      });
      ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]); ctx.beginPath();
      s.t.daily.forEach(function (p, j) {
        var x = X(day(p.date)), y = Y(p.rolling7);
        if (j) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      });
      ctx.stroke(); ctx.setLineDash([]);
    });
    cv.onmousemove = function (e) {
      var r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
      var best = null, bd = 400;
      marks.forEach(function (m) { var d = (m.x - mx) * (m.x - mx) + (m.y - my) * (m.y - my); if (d < bd) { bd = d; best = m; } });
      if (!best) { tip.hidden = true; return; }
      var p = best.p;
      tip.innerHTML = "<strong>" + best.s.t.label + "</strong> · " + p.date + "<br>mean " + fmt(p.mean) +
        " (range " + fmt(p.min) + "–" + fmt(p.max) + ", " + p.n + " exam" + (p.n === 1 ? "" : "s") +
        ")<br>7-day average " + fmt(p.rolling7);
      tip.hidden = false;
      tip.style.left = (e.pageX + 12) + "px"; tip.style.top = (e.pageY + 12) + "px";
    };
    cv.onmouseleave = function () { tip.hidden = true; };
  }

  function render() {
    draw("chart-screens", "legend-screens", ["heart_binary", "lung_binary"], [0, 1], function (v) { return v.toFixed(2); });
    draw("chart-rates", "legend-rates", ["heart_rate", "breathing_rate"], null, function (v) { return v.toFixed(0); });
  }
  render();
  window.addEventListener("resize", render);
})();
