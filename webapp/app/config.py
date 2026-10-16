"""Web app configuration: paths, served model registry, health-awareness knowledge base."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ML = os.path.join(REPO, "ml")

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(REPO, ".env"))
except Exception:  # pragma: no cover - dotenv is optional
    pass

DATA_DIR = os.environ.get("ASCULTO_DATA_DIR", os.path.join(REPO, "data", "HLS-CMDS"))
BINARY_OUT = os.environ.get("ASCULTO_BINARY_OUT", os.path.join(ML, "runs"))
MC_OUT = os.environ.get("ASCULTO_MC_OUT", os.path.join(REPO, "ml", "runs", "mc"))
DEVICE = os.environ.get("ASCULTO_DEVICE", "cpu")  # PoC default CPU; set "cuda" for GPU
SEED = int(os.environ.get("ASCULTO_SEED", "0"))


@dataclass
class Head:
    key: str
    label: str
    task: str
    scheme: str
    stage: int
    model: str
    classes: List[str]
    out_dir: str
    kind: str  # "source" | "binary" | "multiclass"


def _heart_types() -> List[str]:
    return [
        "Normal", "Late Diastolic Murmur", "Mid Systolic Murmur", "Late Systolic Murmur",
        "Atrial Fibrillation", "S3", "S4", "Early Systolic Murmur", "Tachycardia", "AV Block",
    ]


def _lung_types() -> List[str]:
    return ["Normal", "Wheezing", "Fine Crackles", "Coarse Crackles", "Rhonchi", "Pleural Rub"]


def model_registry() -> List[Head]:
    return [
        Head("source", "Source routing (heart / lung / mixed)", "sound", "binary", 2, "resnet18",
             ["heart", "lung", "mixed"], BINARY_OUT, "source"),
        Head("heart_binary", "Heart screening (normal / abnormal)", "heart", "binary", 2, "resnet18",
             ["normal", "abnormal"], BINARY_OUT, "binary"),
        Head("heart_types", "Heart sound typing (10 classes)", "heart", "10class", 4, "transformer",
             _heart_types(), MC_OUT, "multiclass"),
        Head("lung_binary", "Lung screening (normal / abnormal)", "lung", "binary", 2, "resnet18",
             ["normal", "abnormal"], BINARY_OUT, "binary"),
        Head("lung_types", "Lung sound typing (6 classes)", "lung", "6class", 4, "transformer",
             _lung_types(), MC_OUT, "multiclass"),
    ]


# ---------------------------------------------------------------- knowledge base
# Plain-language health-awareness context. NOT diagnostic advice.
KNOWLEDGE = {
    # heart
    "Normal": (
        "No abnormal heart-sound pattern was detected by the model.",
        "Normal heart sounds (S1 'lub' and S2 'dub') reflect valve closure. A normal result is reassuring, "
        "but a single recording cannot rule out disease.",
    ),
    "Late Diastolic Murmur": (
        "A murmur was detected in late diastole.",
        "Late-diastolic murmurs can occur with mitral stenosis or other valve conditions. They are best "
        "assessed by a clinician with a full examination.",
    ),
    "Mid Systolic Murmur": (
        "A murmur was detected during mid-systole.",
        "Mid-systolic (ejection) murmurs are common and often innocent, but can accompany aortic or "
        "pulmonary valve disease. A clinician can judge whether further testing is needed.",
    ),
    "Late Systolic Murmur": (
        "A murmur was detected in late systole.",
        "Late-systolic murmurs are often associated with mitral valve prolapse. They warrant clinical review.",
    ),
    "Atrial Fibrillation": (
        "An irregular rhythm consistent with atrial fibrillation was detected.",
        "Atrial fibrillation is an irregular heart rhythm that raises stroke risk. It is important to "
        "confirm with an ECG and discuss treatment with a clinician.",
    ),
    "S3": (
        "A third heart sound (S3) was detected.",
        "An S3 gallop can be normal in young people but often signals reduced heart pumping function in "
        "adults. Clinical correlation is advised.",
    ),
    "S4": (
        "A fourth heart sound (S4) was detected.",
        "An S4 gallop is often associated with a stiffened ventricle (e.g. from hypertension). A clinician "
        "should interpret it in context.",
    ),
    "Early Systolic Murmur": (
        "A murmur was detected in early systole.",
        "Early-systolic murmurs can be benign or reflect valve disease; clinical assessment is recommended.",
    ),
    "Tachycardia": (
        "A fast heart rate pattern was detected.",
        "Tachycardia may be a normal response to exercise or stress, or may indicate an arrhythmia. If it is "
        "persistent or accompanied by symptoms, seek medical advice.",
    ),
    "AV Block": (
        "A pattern consistent with atrioventricular (AV) block was detected.",
        "AV block is a conduction disturbance that can range from harmless to serious. It should be reviewed "
        "by a clinician, often with an ECG.",
    ),
    # lung
    "Wheezing": (
        "A wheeze was detected.",
        "Wheezing is a high-pitched whistling sound from narrowed airways, common in asthma and COPD. "
        "Persistent or severe wheeze needs medical assessment.",
    ),
    "Fine Crackles": (
        "Fine crackles (rales) were detected.",
        "Fine crackles are short, high-pitched sounds often heard in the lung bases; they can occur with "
        "pulmonary fibrosis or early fluid. Clinical review is advised.",
    ),
    "Coarse Crackles": (
        "Coarse crackles were detected.",
        "Coarse crackles are lower-pitched bubbling sounds that can indicate secretions or fluid in the "
        "airways. A clinician should assess the cause.",
    ),
    "Rhonchi": (
        "Rhonchi were detected.",
        "Rhonchi are low-pitched rattling sounds from larger airways, often due to mucus. They may ease with "
        "coughing but persistent cases warrant review.",
    ),
    "Pleural Rub": (
        "A pleural rub was detected.",
        "A pleural rub is a creaking sound from inflamed pleural surfaces, which can occur in pleurisy or "
        "infection. Medical evaluation is recommended.",
    ),
    "abnormal": (
        "The model flagged this recording as abnormal.",
        "An abnormal screening result is not a diagnosis. It means the pattern differed from normal; a "
        "clinician should interpret it with your history and a physical exam.",
    ),
    "normal": (
        "The model did not flag an abnormality in this recording.",
        "A normal screening result is reassuring but cannot exclude disease, especially from a single "
        "short recording.",
    ),
    "heart": (
        "The recording was routed as a heart-sound clip.",
        "It was analysed with the heart models (screening and typing).",
    ),
    "lung": (
        "The recording was routed as a lung-sound clip.",
        "It was analysed with the lung models (screening and typing).",
    ),
    "mixed": (
        "The recording was routed as a mixed heart+lung clip.",
        "Both heart and lung models were applied.",
    ),
}

DISCLAIMER = (
    "This is an educational / research prototype trained on a manikin dataset. It is NOT a medical device "
    "and does NOT provide a diagnosis. Always consult a qualified healthcare professional about any concern."
)

# Dataset attribution (required by the training data's CC BY 4.0 license).
DATASET_CREDIT = (
    "Models trained on HLS-CMDS — Torabi, Y., Shirani, S., & Reilly, J. P. (2025), "
    "IEEE Data Descriptions, DOI 10.1109/IEEEDATA.2025.3566012 (Zenodo 15376628). "
    "Dataset licensed CC BY 4.0. See docs/DatasetLicenses.md."
)
