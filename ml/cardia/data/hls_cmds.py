"""HLS-CMDS manifest builder.

The dataset ships three metadata CSVs (``HS.csv``, ``LS.csv``, ``Mix.csv``)
and WAVs under ``HS/``, ``LS/`` and ``Mix/`` (optionally nested one level,
e.g. ``HS/HS/``). Each task manifest has one row per recording with a
``group`` column that ties the three files of a mixed recording together
(``H0001``/``L0001``/``M0001`` share group ``"1"``) so CV splits never leak a
mixed recording's sources across folds.

Tasks / schemes:
  sound  binary  -> heart (HS + Mix heart sources) / lung (LS + Mix lung sources) / mixed
  heart  binary  -> normal / abnormal;  10class -> HEART_TYPES
  lung   binary  -> normal / abnormal;  6class  -> LUNG_TYPES
"""
from __future__ import annotations

import glob
import os
from functools import lru_cache
from typing import Dict, List, Optional

import pandas as pd

from cardia.config import HEART_TYPES, LUNG_TYPES, SCHEMES_BY_TASK

SOUND_CLASSES = ["heart", "lung", "mixed"]
BINARY_CLASSES = ["normal", "abnormal"]


def class_names(task: str, scheme: str = "binary") -> List[str]:
    if task == "sound":
        return list(SOUND_CLASSES)
    if scheme == "binary":
        return list(BINARY_CLASSES)
    if task == "heart" and scheme == "10class":
        return list(HEART_TYPES)
    if task == "lung" and scheme == "6class":
        return list(LUNG_TYPES)
    raise ValueError(f"unsupported task/scheme: {task}/{scheme}")


@lru_cache(maxsize=16)
def _wav_index(data_dir: str) -> Dict[str, str]:
    """Map recording id (file stem) -> absolute path, for every WAV under data_dir."""
    index: Dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)):
        if "__MACOSX" in path:
            continue
        index.setdefault(os.path.splitext(os.path.basename(path))[0], os.path.abspath(path))
    return index


# Canonical file-name codes (README); a few LS.csv ids use stale codes
# (``M_C_LUA`` / ``M_G_LUA``) while the WAVs are named ``M_FC_LUA`` / ``M_CC_LUA``.
TYPE_CODES = {"Fine Crackles": "FC", "Coarse Crackles": "CC"}


def _resolve(data_dir: str, sample_id: str, row: Optional[dict] = None) -> str:
    index = _wav_index(data_dir)
    path = index.get(sample_id)
    if path is None and row is not None:
        sound_type = row.get("lung_type") if row.get("source") == "LS" else row.get("heart_type")
        code = TYPE_CODES.get(str(sound_type))
        if code and sample_id.count("_") == 2:
            gender, _, location = sample_id.split("_")
            path = index.get(f"{gender}_{code}_{location}")
            if path is not None:
                row["sample_id"] = os.path.splitext(os.path.basename(path))[0]
                if row.get("group") == sample_id:
                    row["group"] = row["sample_id"]
    if path is None:
        raise FileNotFoundError(f"HLS-CMDS recording {sample_id}.wav not found under {data_dir}")
    return path


def _read_csv(data_dir: str, name: str) -> pd.DataFrame:
    path = os.path.join(data_dir, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found; run ./setup.sh or set ASCULTO_DATA_DIR to the HLS-CMDS root")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df.apply(lambda s: s.str.strip() if pd.api.types.is_string_dtype(s) else s)


def _mix_group(mixed_id: str) -> str:
    return str(int("".join(ch for ch in mixed_id if ch.isdigit())))


def _heart_rows(data_dir: str) -> List[dict]:
    rows = []
    for r in _read_csv(data_dir, "HS.csv").to_dict("records"):
        sid = r["Heart Sound ID"]
        rows.append({"sample_id": sid, "group": sid, "source": "HS", "heart_type": r["Heart Sound Type"],
                     "gender": r["Gender"], "location": r["Location"]})
    for r in _read_csv(data_dir, "Mix.csv").to_dict("records"):
        rows.append({"sample_id": r["Heart Sound ID"], "group": _mix_group(r["Mixed Sound ID"]), "source": "MixH",
                     "heart_type": r["Heart Sound Type"], "gender": r["Gender"], "location": r["Location"]})
    return rows


def _lung_rows(data_dir: str) -> List[dict]:
    rows = []
    for r in _read_csv(data_dir, "LS.csv").to_dict("records"):
        sid = r["Lung Sound ID"]
        rows.append({"sample_id": sid, "group": sid, "source": "LS", "lung_type": r["Lung Sound Type"],
                     "gender": r["Gender"], "location": r["Location"]})
    for r in _read_csv(data_dir, "Mix.csv").to_dict("records"):
        rows.append({"sample_id": r["Lung Sound ID"], "group": _mix_group(r["Mixed Sound ID"]), "source": "MixL",
                     "lung_type": r["Lung Sound Type"], "gender": r["Gender"], "location": r["Location"]})
    return rows


def _mixed_rows(data_dir: str) -> List[dict]:
    return [{"sample_id": r["Mixed Sound ID"], "group": _mix_group(r["Mixed Sound ID"]), "source": "Mix",
             "heart_type": r["Heart Sound Type"], "lung_type": r["Lung Sound Type"],
             "gender": r["Gender"], "location": r["Location"]}
            for r in _read_csv(data_dir, "Mix.csv").to_dict("records")]


def build_task_manifest(data_dir: str, task: str, scheme: str = "binary",
                        out_dir: Optional[str] = None) -> pd.DataFrame:
    """Build (and optionally persist to ``out_dir``) the manifest for a task/scheme.

    Columns: sample_id, file_path, group, source, [heart_type|lung_type], gender,
    location, [is_clean], target, task_label.
    """
    if task not in SCHEMES_BY_TASK or scheme not in SCHEMES_BY_TASK[task]:
        raise ValueError(f"unsupported task/scheme: {task}/{scheme}")
    names = class_names(task, scheme)

    if task == "sound":
        rows = ([{**r, "task_label": "heart"} for r in _heart_rows(data_dir)]
                + [{**r, "task_label": "lung"} for r in _lung_rows(data_dir)]
                + [{**r, "task_label": "mixed"} for r in _mixed_rows(data_dir)])
        cols = ["sample_id", "file_path", "group", "source", "heart_type", "lung_type", "gender", "location"]
    else:
        type_col = f"{task}_type"
        base = _heart_rows(data_dir) if task == "heart" else _lung_rows(data_dir)
        rows = []
        for r in base:
            label = r[type_col] if scheme != "binary" else ("normal" if r[type_col] == "Normal" else "abnormal")
            rows.append({**r, "task_label": label})
        cols = ["sample_id", "file_path", "group", "source", type_col, "gender", "location"]
        if task == "heart":
            for r in rows:
                r["is_clean"] = True
            cols.append("is_clean")

    unknown = sorted({r["task_label"] for r in rows} - set(names))
    if unknown:
        raise ValueError(f"labels {unknown} not in class list for {task}/{scheme}")
    for r in rows:
        r["file_path"] = _resolve(data_dir, r["sample_id"], r)
        r["target"] = names.index(r["task_label"])
    df = pd.DataFrame(rows)[cols + ["target", "task_label"]]
    df["group"] = df["group"].astype(str)
    if df["sample_id"].duplicated().any():
        raise ValueError(f"duplicate sample ids in {task}/{scheme} manifest")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, f"{task}_{scheme}_manifest.csv"), index=False)
    return df
