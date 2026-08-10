from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
ENCODER_DIR = BASE_DIR / "encoders"
PLOTS_DIR = BASE_DIR / "plots"
REPORTS_DIR = BASE_DIR / "reports"


def ensure_dirs() -> None:
    for d in (DATA_DIR, MODEL_DIR, ENCODER_DIR, PLOTS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def resolve_data_file(*names: str) -> Path:
    for name in names:
        p = DATA_DIR / name
        if p.exists():
            return p
    tried = ", ".join(str(DATA_DIR / n) for n in names)
    raise FileNotFoundError(
        f"Missing data file. Tried: {tried}. Place the file in {DATA_DIR}."
    )
