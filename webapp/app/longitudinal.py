"""Exam summaries plus same-day (intraday) and over-time consistency analytics.

Everything here describes the behaviour of the model outputs across repeated
recordings. It is not a health assessment.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

SITES = [
    ("", "Not specified"),
    ("aortic", "Aortic (2nd right intercostal space)"),
    ("pulmonic", "Pulmonic (2nd left intercostal space)"),
    ("tricuspid", "Tricuspid (4th left intercostal space)"),
    ("mitral", "Mitral (apex)"),
    ("lung_ant_r", "Lung, anterior right"),
    ("lung_ant_l", "Lung, anterior left"),
    ("lung_post_r", "Lung, posterior right"),
    ("lung_post_l", "Lung, posterior left"),
]
SITE_KEYS = {k for k, _ in SITES}
SCREENS = [("heart_binary", "Heart screen"), ("lung_binary", "Lung screen")]
TYPES = [("heart_types", "Heart type"), ("lung_types", "Lung type")]
RATES = [("heart_rate", "Heart rate", "bpm"), ("breathing_rate", "Breathing rate", "breaths/min")]
_METRIC_KEYS = {
    "Heart rate estimate": "heart_rate",
    "Breathing rate estimate": "breathing_rate",
    "Beat interval variability (CV)": "beat_cv",
    "Cardiac periodicity": "cardiac_periodicity",
    "Respiratory periodicity": "respiratory_periodicity",
    "Dynamic SNR estimate": "snr_db",
}
BASELINE_N = 3
SD_SCALE = 0.25  # within-day SD of a probability at which the spread part of the score reaches 0


def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def exam_from_report(rep: Dict) -> Dict:
    """Compact, audio-free record of one analysed recording."""
    series = rep.get("series") or {}
    heads = []
    for h in rep.get("heads", []):
        classes, probs = list(h["classes"]), [float(p) for p in h["probs"]]
        top = classes.index(h["pred"]) if h["pred"] in classes else int(np.argmax(probs))
        agree = None
        ser = series.get(h["key"])
        if ser and ser.get("probs"):
            agree = round(float(np.mean(np.asarray(ser["probs"]).argmax(axis=1) == top)), 3)
        std = h.get("probs_std")
        heads.append({
            "key": h["key"], "label": h["label"], "kind": h["kind"], "pred": h["pred"],
            "confidence": round(float(h["confidence"]), 4), "classes": classes,
            "probs": [round(p, 4) for p in probs],
            "fold_std": round(float(std[top]), 4) if std else None,
            "window_agreement": agree,
        })
    metrics = {}
    for sec in (rep.get("quant") or {}).get("sections", []):
        for row in sec.get("rows", []):
            key = _METRIC_KEYS.get(row.get("metric"))
            val = _num(row.get("value"))
            if key and val is not None and not (key.endswith("_rate") and val <= 0):  # 0 = no rhythm found
                metrics[key] = round(val, 3)
    q = rep.get("quality") or {}
    return {
        "route": rep.get("route"), "verdict": rep.get("verdict"),
        "duration_s": q.get("duration_s"), "quality_ok": bool(q.get("ok", True)),
        "quality_flags": list(q.get("flags", [])), "heads": heads, "metrics": metrics,
    }


def _head(exam: Dict, key: str) -> Optional[Dict]:
    return next((h for h in exam.get("heads", []) if h["key"] == key), None)


def p_abnormal(exam: Dict, key: str) -> Optional[float]:
    h = _head(exam, key)
    if not h or "abnormal" not in h["classes"]:
        return None
    return float(h["probs"][h["classes"].index("abnormal")])


def _when(exam: Dict) -> datetime:
    try:
        return datetime.fromisoformat(exam["recorded_at"].replace("Z", "+00:00")).replace(tzinfo=None)
    except (KeyError, ValueError):
        return datetime.min


def flat_row(exam: Dict) -> Dict:
    t = _when(exam)
    row = {"id": exam.get("id"), "recorded_at": exam.get("recorded_at"), "date": t.date().isoformat(),
           "time": t.strftime("%H:%M"), "site": exam.get("site", ""), "note": exam.get("note", ""),
           "filename": exam.get("filename", ""), "verdict": exam.get("verdict"),
           "quality_ok": exam.get("quality_ok"), "duration_s": exam.get("duration_s")}
    for key, _ in SCREENS:
        p = p_abnormal(exam, key)
        row[f"{key}_p_abnormal"] = None if p is None else round(p, 4)
        h = _head(exam, key)
        row[f"{key}_pred"] = h["pred"] if h else None
    for key, _ in TYPES:
        h = _head(exam, key)
        row[f"{key}_pred"] = h["pred"] if h else None
        row[f"{key}_confidence"] = h["confidence"] if h else None
    for key, _, _ in RATES:
        row[key] = exam.get("metrics", {}).get(key)
    return row


def _stats(vals: List[float]) -> Optional[Dict]:
    if not vals:
        return None
    a = np.asarray(vals, dtype=float)
    return {"n": int(a.size), "mean": round(float(a.mean()), 4), "sd": round(float(a.std(ddof=1)), 4) if a.size > 1 else 0.0,
            "min": round(float(a.min()), 4), "max": round(float(a.max()), 4)}


def _agreement(preds: List[str]) -> Optional[Dict]:
    preds = [p for p in preds if p]
    if not preds:
        return None
    mode, count = Counter(preds).most_common(1)[0]
    return {"mode": mode, "share": round(count / len(preds), 3), "n": len(preds)}


def _trend(points: List[tuple]) -> Optional[Dict]:
    """OLS slope per week over (day_offset, value) points, with a 95% interval."""
    if len(points) < 2:
        return None
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    sxx = float(((x - x.mean()) ** 2).sum())
    if sxx == 0:
        return None
    slope = float(((x - x.mean()) * (y - y.mean())).sum() / sxx)
    out = {"slope_per_week": round(7 * slope, 4), "n_days": len(points), "ci95": None, "direction": "not enough days"}
    if len(points) >= 3:
        resid = y - (y.mean() + slope * (x - x.mean()))
        se = float(np.sqrt((resid ** 2).sum() / (len(points) - 2) / sxx))
        lo, hi = 7 * (slope - 1.96 * se), 7 * (slope + 1.96 * se)
        out["ci95"] = [round(lo, 4), round(hi, 4)]
        out["direction"] = "rising" if lo > 0 else "falling" if hi < 0 else "no clear trend"
    return out


def summarize(exams: List[Dict]) -> Dict:
    exams = sorted(exams, key=_when)
    rows = [flat_row(e) for e in exams]
    by_day: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_day[r["date"]].append(r)
    dates = sorted(by_day)

    days = []
    for d in dates:
        rs = by_day[d]
        day = {"date": d, "n": len(rs), "screens": {}, "types": {}, "rates": {}}
        for key, _ in SCREENS:
            st = _stats([r[f"{key}_p_abnormal"] for r in rs if r[f"{key}_p_abnormal"] is not None])
            if st:
                st["agreement"] = _agreement([r[f"{key}_pred"] for r in rs])
                day["screens"][key] = st
        for key, _ in TYPES:
            ag = _agreement([r[f"{key}_pred"] for r in rs])
            if ag:
                day["types"][key] = ag
        for key, _, _ in RATES:
            st = _stats([r[key] for r in rs if r[key] is not None])
            if st:
                day["rates"][key] = st
        day["quality_share"] = round(sum(1 for r in rs if r["quality_ok"]) / len(rs), 3)
        days.append(day)

    t0 = datetime.fromisoformat(dates[0]) if dates else None
    trends = {}
    for key, label, *unit in [(k, l) for k, l in SCREENS] + list(RATES):
        group = "screens" if key in dict(SCREENS) else "rates"
        pts, daily = [], []
        for day in days:
            st = day[group].get(key)
            if st:
                pts.append(((datetime.fromisoformat(day["date"]) - t0).days, st["mean"]))
                daily.append({"date": day["date"], "mean": st["mean"], "min": st["min"], "max": st["max"], "n": st["n"]})
        for i, p in enumerate(daily):
            lo = datetime.fromisoformat(p["date"]).toordinal() - 6
            win = [q["mean"] for q in daily[: i + 1] if datetime.fromisoformat(q["date"]).toordinal() >= lo]
            p["rolling7"] = round(float(np.mean(win)), 4)
        if not daily:
            continue
        field = f"{key}_p_abnormal" if group == "screens" else key
        vals = [r[field] for r in rows if r[field] is not None]
        base = vals[:BASELINE_N]
        latest = daily[-1]["mean"]
        change = None
        if len(vals) > BASELINE_N:
            bsd = float(np.std(base, ddof=1)) if len(base) > 1 else 0.0
            change = {"baseline_mean": round(float(np.mean(base)), 4), "latest_day_mean": latest,
                      "delta": round(latest - float(np.mean(base)), 4),
                      "z": round((latest - float(np.mean(base))) / bsd, 2) if bsd > 0 else None}
        trends[key] = {"label": label, "unit": unit[0] if unit else "probability", "daily": daily,
                       "trend": _trend(pts), "vs_baseline": change}

    parts, agreements, sds = [], [], []
    for day in days:
        if day["n"] < 2:
            continue
        for key, _ in SCREENS:
            st = day["screens"].get(key)
            if st and st["n"] >= 2:
                agree = st["agreement"]["share"] if st.get("agreement") else 1.0
                agreements.append(agree)
                sds.append(st["sd"])
                parts.append(0.5 * agree + 0.5 * max(0.0, 1 - st["sd"] / SD_SCALE))
    consistency = None
    if parts:
        consistency = {"score": round(100 * float(np.mean(parts))), "agreement": round(float(np.mean(agreements)), 3),
                       "mean_within_day_sd": round(float(np.mean(sds)), 4), "repeat_days": sum(1 for d in days if d["n"] >= 2)}

    summary = {
        "n_exams": len(rows), "n_days": len(dates),
        "span_days": (datetime.fromisoformat(dates[-1]) - t0).days + 1 if dates else 0,
        "quality_share": round(sum(1 for r in rows if r["quality_ok"]) / len(rows), 3) if rows else None,
        "rows": rows, "days": days, "trends": trends, "consistency": consistency,
    }
    summary["narrative"] = narrative(summary)
    return summary


def narrative(s: Dict) -> List[str]:
    if not s["n_exams"]:
        return ["No saved exams yet. Analyse a recording while signed in and tick \"Save to my exam history\"."]
    out = [f"{s['n_exams']} exam{'s' if s['n_exams'] != 1 else ''} over {s['span_days']} day"
           f"{'s' if s['span_days'] != 1 else ''} ({s['n_days']} recording day{'s' if s['n_days'] != 1 else ''})."]
    c = s["consistency"]
    if c:
        out.append(f"Same-day repeats gave the same screening result {100 * c['agreement']:.0f}% of the time, and the "
                   f"probability of an abnormal result varied by ±{c['mean_within_day_sd']:.2f} within a day on average. "
                   f"Consistency score: {c['score']}/100.")
    else:
        out.append("Record 2 or more exams on the same day to measure how consistent the model's results are.")
    for t in s["trends"].values():
        tr, unit = t["trend"], t["unit"]
        what = f"{t['label']} probability of an abnormal result" if unit == "probability" else f"{t['label']} estimate"

        def fmt(v, unit=unit):
            return f"{v:+.3f}" if unit == "probability" else f"{v:+.1f} {unit}"

        if tr and tr["ci95"]:
            out.append(f"{what}: {tr['direction']}, {fmt(tr['slope_per_week'])} per week "
                       f"(95% range {fmt(tr['ci95'][0])} to {fmt(tr['ci95'][1])}, {tr['n_days']} days).")
        ch = t["vs_baseline"]
        if ch:
            z = f", {ch['z']:+.1f} baseline SDs" if ch["z"] is not None else ""
            out.append(f"{what} on the latest day vs. the first {BASELINE_N} exams: {fmt(ch['delta'])}{z}.")
    if s["quality_share"] is not None and s["quality_share"] < 1:
        out.append(f"{100 * s['quality_share']:.0f}% of recordings passed the quality checks; low-quality recordings "
                   "make trends noisier.")
    out.append("These are trends in model outputs from models trained on manikin recordings. They are not a health "
               "assessment and do not measure recovery.")
    return out
