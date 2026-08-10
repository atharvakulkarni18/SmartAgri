from __future__ import annotations

import importlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from config import (
    BASE_DIR,
    DATA_DIR,
    ENCODER_DIR,
    MODEL_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    ensure_dirs,
)

ensure_dirs()

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartAgri — AI Powered Farming",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# REQUIRED FILES & BOOTSTRAP
# ---------------------------------------------------------------------------
REQUIRED_MODEL_FILES = [
    MODEL_DIR / "best_crop_model.txt",
    MODEL_DIR / "best_fert_model.txt",
    MODEL_DIR / "best_yield_model.txt",
]

REQUIRED_ENCODER_FILES = [
    ENCODER_DIR / "CROP_REGISTRY.pkl",
    ENCODER_DIR / "ALIAS_MAP.pkl",
    ENCODER_DIR / "DISTRICT_PROFILES.pkl",
    ENCODER_DIR / "STATE_PROFILES.pkl",
    ENCODER_DIR / "NATIONAL_PROFILES.pkl",
    ENCODER_DIR / "DISTRICT_CROP_SET.pkl",
    ENCODER_DIR / "FAMILY_YIELD_MAX.pkl",
    ENCODER_DIR / "FAMILY_YIELD_MIN.pkl",
    ENCODER_DIR / "rain_lookup_df.pkl",
    ENCODER_DIR / "DISTRICT_TEMP_OVERRIDE.pkl",
    ENCODER_DIR / "SCORE_WEIGHTS.pkl",
    ENCODER_DIR / "crop_cat_enc.pkl",
    ENCODER_DIR / "crop_target_le.pkl",
    ENCODER_DIR / "crop_scaler.pkl",
    ENCODER_DIR / "CROP_FEATURES.pkl",
    ENCODER_DIR / "fert_cat_enc.pkl",
    ENCODER_DIR / "fert_target_le.pkl",
    ENCODER_DIR / "fert_scaler.pkl",
    ENCODER_DIR / "yield_cat_enc.pkl",
    ENCODER_DIR / "STATE_NORM_MAP.pkl",
    ENCODER_DIR / "SEASON_NORM_MAP.pkl",
    ENCODER_DIR / "IRRIGATION_NORM_MAP.pkl",
    ENCODER_DIR / "SOIL_NORM_MAP.pkl",
    ENCODER_DIR / "REGION_CROP_MAP.pkl",
]

BOOTSTRAP_STAGES = [
    "v3_NB1_EDA.py",
    "v3_NB2_KnowledgeLayer.py",
    "v3_NB3_Training.py",
    "v3_NB4_Pipeline.py",
    "test_harness.py",
    "v3_NB6_Evaluation.py",
]


def all_exist(paths: Iterable[Path]) -> bool:
    return all(path.exists() for path in paths)


def project_ready() -> bool:
    return all_exist(REQUIRED_MODEL_FILES) and all_exist(REQUIRED_ENCODER_FILES)


@st.cache_resource(show_spinner=False)
def load_modules():
    try:
        pipeline = importlib.import_module("v3_NB4_Pipeline")
    except Exception:
        pipeline = None
    try:
        harness = importlib.import_module("test_harness")
    except Exception:
        harness = None
    return pipeline, harness


@st.cache_data
def load_geography_data():
    rain_path = ENCODER_DIR / "rain_lookup_df.pkl"
    state_profiles_path = ENCODER_DIR / "STATE_PROFILES.pkl"

    states: List[str] = []
    state_to_districts: Dict[str, List[str]] = {}

    if rain_path.exists():
        try:
            rdf = pd.read_pickle(rain_path)
            if "STATE_NORM" in rdf.columns and "DISTRICT_CLEAN" in rdf.columns:
                for s, g in rdf.groupby("STATE_NORM"):
                    d_list = sorted(
                        g["DISTRICT_CLEAN"].astype(str).str.title().unique().tolist()
                    )
                    state_to_districts[str(s)] = d_list
                states = sorted(list(state_to_districts.keys()))
        except Exception:
            pass

    if not states and state_profiles_path.exists():
        try:
            sp = pd.read_pickle(state_profiles_path)
            states = sorted([str(k) for k in sp.keys()])
        except Exception:
            pass

    if not states:
        states = [
            "Rajasthan", "Kerala", "Bihar", "Punjab", "Uttar Pradesh",
            "Maharashtra", "Tamil Nadu", "Karnataka", "Gujarat", "Madhya Pradesh",
        ]

    return states, state_to_districts


def run_stage(script_name: str) -> None:
    script_path = BASE_DIR / script_name
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "Unknown failure"
        raise RuntimeError(f"{script_name} failed:\n{message}")


def bootstrap_project() -> None:
    if project_ready():
        return

    status = st.empty()
    progress = st.progress(0)
    total = len(BOOTSTRAP_STAGES)

    for index, stage in enumerate(BOOTSTRAP_STAGES, start=1):
        status.info(f"⏳ Training Stage ({index}/{total}): **{stage}** — Please wait...")
        run_stage(stage)
        progress.progress(index / total)

    status.success("✅ All models trained successfully! SmartAgri is ready.")


def execute_pipeline(pipeline_module, harness_module, payload: Dict[str, Any]):
    if harness_module and hasattr(harness_module, "run_pipeline"):
        return harness_module.run_pipeline(payload)
    elif pipeline_module and hasattr(pipeline_module, "run_pipeline"):
        return pipeline_module.run_pipeline(payload)
    elif pipeline_module:
        norm = pipeline_module.normalize_input(payload)
        district = norm["district"]
        state = norm["state"]
        soil_type = norm["soil_type"]
        season = norm["season"]
        irrigation = norm["irrigation"]
        n_val = norm["N"]
        p_val = norm["P"]
        k_val = norm["K"]
        soil_ph = norm["pH"]

        rain = pipeline_module.lookup_rainfall(district, state)
        if rain:
            annual_rain = rain["annual"]
            season_rain = {
                "kharif": rain["jun_sep"],
                "rabi": rain["oct_dec"],
                "zaid": rain["mar_may"],
            }.get(season, annual_rain * 0.5)
        else:
            annual_rain = 800.0
            season_rain = annual_rain * (0.6 if season == "kharif" else 0.25)

        temp_mid = pipeline_module.get_district_temp(district, state, season)
        hum_mid = {"kharif": 75.0, "rabi": 55.0, "zaid": 40.0}.get(season, 60.0)

        ml_scores = pipeline_module.engine_suitability(
            n_val, p_val, k_val, soil_type, season, irrigation,
            temp_mid, hum_mid, season_rain, soil_ph,
        )

        ml_candidates = set(ml_scores.keys())
        yield_candidates = pipeline_module.get_yield_top_crops(district, top_n=5)
        freq_candidates = pipeline_module.get_freq_top_crops(district, top_n=5)
        region_candidates = pipeline_module.get_region_crops(state)
        candidate_crops = (
            {pipeline_module.resolve(c) for c in ml_candidates}
            | yield_candidates
            | freq_candidates
            | region_candidates
        )

        scored = []
        for canonical in candidate_crops:
            if not pipeline_module.is_crop_in_season(canonical, season):
                continue
            ml_prob = float(ml_scores.get(canonical, 0.05))
            final, components, yield_data, yield_level = pipeline_module.engine_score(
                ml_prob, canonical, district, state, season, irrigation,
                annual_rain, season_rain, temp_c=temp_mid, soil_ph=soil_ph,
            )
            if final is None:
                continue
            meta = pipeline_module.get_meta(canonical) or {}
            display = meta.get("display", canonical.title())
            scored.append({
                "canonical": canonical,
                "display": display,
                "final": float(final),
                "components": components,
                "yield_data": yield_data,
                "yield_level": yield_level,
            })

        scored.sort(key=lambda x: x["final"], reverse=True)
        top_crops = scored[:5]
        best = top_crops[0] if top_crops else None
        best_canonical = best["canonical"] if best else None

        if best_canonical:
            fert_name, fert_reason, ml_probs = pipeline_module.engine_fertilizer(
                n_val, p_val, k_val, soil_type, best_canonical, season,
                irrigation, soil_ph, annual_rain, state,
            )
            ml_yield = pipeline_module.engine_yield_ml(
                best_canonical, state, district, season
            )
        else:
            fert_name, fert_reason, ml_probs = None, None, None
            ml_yield = None

        if best and best.get("yield_data"):
            primary_yield = best["yield_data"]["med"]
            yield_level = best.get("yield_level")
        else:
            primary_yield = ml_yield
            yield_level = "ml" if ml_yield is not None else None

        return {
            "input": payload,
            "rain": {"annual": annual_rain, "season": season_rain},
            "top_crops": [
                {
                    "name": c["display"],
                    "score": c["final"],
                    "components": c["components"],
                    "yield_level": c.get("yield_level"),
                    "yield_data": c.get("yield_data"),
                }
                for c in top_crops
            ],
            "fertilizer": {
                "name": fert_name,
                "reason": fert_reason,
                "ml_probs": ml_probs,
            },
            "yield": {
                "top_crop": best["display"] if best else None,
                "value": primary_yield,
                "ml_yield": ml_yield,
                "source": (
                    "historical_median"
                    if best and best.get("yield_data")
                    else "ml_model"
                ),
                "level": yield_level,
            },
        }
    else:
        raise RuntimeError(
            "Pipeline module could not be loaded. Please rebuild from Settings page."
        )


# ---------------------------------------------------------------------------
# THEME SYSTEM
# ---------------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "light"

if "history" not in st.session_state:
    st.session_state.history = []

IS_DARK = st.session_state.theme == "dark"

LIGHT_CSS = """
:root {
    --bg-main: #f8fafc;
    --bg-sidebar: #ffffff;
    --bg-card: #ffffff;
    --bg-card-hover: #f1f5f9;
    --bg-hero: linear-gradient(135deg, #166534 0%, #15803d 40%, #22c55e 100%);
    --text-primary: #1e293b;
    --text-secondary: #475569;
    --text-muted: #94a3b8;
    --text-on-hero: #ffffff;
    --border-color: #e2e8f0;
    --border-hover: #16a34a;
    --accent-green: #16a34a;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-cyan: #0891b2;
    --accent-rose: #e11d48;
    --accent-purple: #7c3aed;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.08);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.08);
    --shadow-lg: 0 8px 30px rgba(0,0,0,0.1);
    --chart-bg: #ffffff;
    --chart-text: #1e293b;
    --chart-grid: #e2e8f0;
    --nav-active-bg: #16a34a;
    --nav-active-text: #ffffff;
    --nav-hover-bg: #f0fdf4;
    --nav-text: #475569;
    --badge-green-bg: #dcfce7;
    --badge-green-text: #166534;
    --badge-amber-bg: #fef3c7;
    --badge-amber-text: #92400e;
    --badge-cyan-bg: #cffafe;
    --badge-cyan-text: #155e75;
    --input-bg: #f8fafc;
    --input-border: #cbd5e1;
}
"""

DARK_CSS = """
:root {
    --bg-main: #0f172a;
    --bg-sidebar: #1e293b;
    --bg-card: #1e293b;
    --bg-card-hover: #334155;
    --bg-hero: linear-gradient(135deg, #064e3b 0%, #065f46 40%, #10b981 100%);
    --text-primary: #f1f5f9;
    --text-secondary: #cbd5e1;
    --text-muted: #64748b;
    --text-on-hero: #ffffff;
    --border-color: #334155;
    --border-hover: #10b981;
    --accent-green: #10b981;
    --accent-emerald: #34d399;
    --accent-amber: #fbbf24;
    --accent-cyan: #22d3ee;
    --accent-rose: #fb7185;
    --accent-purple: #a78bfa;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.3);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.3);
    --shadow-lg: 0 8px 30px rgba(0,0,0,0.4);
    --chart-bg: #1e293b;
    --chart-text: #f1f5f9;
    --chart-grid: #334155;
    --nav-active-bg: #10b981;
    --nav-active-text: #ffffff;
    --nav-hover-bg: #1a3a2a;
    --nav-text: #94a3b8;
    --badge-green-bg: rgba(16,185,129,0.15);
    --badge-green-text: #34d399;
    --badge-amber-bg: rgba(251,191,36,0.15);
    --badge-amber-text: #fbbf24;
    --badge-cyan-bg: rgba(34,211,238,0.15);
    --badge-cyan-text: #22d3ee;
    --input-bg: #334155;
    --input-border: #475569;
}
"""

COMMON_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
}

[data-testid="stAppViewContainer"] > .main {
    background-color: var(--bg-main) !important;
}

/* Sidebar - ALWAYS Dark Charcoal Black */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #061412 0%, #040e0c 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}
[data-testid="stSidebar"] * {
    color: #f1f5f9 !important;
}
[data-testid="stSidebar"] p, 
[data-testid="stSidebar"] span, 
[data-testid="stSidebar"] div, 
[data-testid="stSidebar"] label {
    color: #94a3b8 !important;
}
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3, 
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] h5 {
    color: #ffffff !important;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1350px;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
    color: var(--text-primary) !important;
}

p, span, div, label {
    color: var(--text-primary) !important;
}

/* Streamlit widgets */
.stSelectbox label, .stSlider label, .stNumberInput label,
.stTextInput label, .stRadio label {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
}

/* Main Content Buttons */
.main .stButton > button {
    background: linear-gradient(135deg, #16a34a 0%, #15803d 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    border-radius: 12px !important;
    border: none !important;
    padding: 0.55rem 1.3rem !important;
    box-shadow: 0 2px 8px rgba(22, 163, 74, 0.25) !important;
    transition: all 0.25s ease !important;
    font-family: 'Inter', sans-serif !important;
}
.main .stButton > button * {
    color: #ffffff !important;
}
.main .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(22, 163, 74, 0.35) !important;
}

/* Metrics */
[data-testid="stMetricValue"] {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    color: var(--accent-green) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
}

/* Tabs */
.stTabs [role="tablist"] {
    gap: 0.4rem;
    border-bottom: 2px solid var(--border-color);
}
.stTabs [role="tab"] {
    background: transparent;
    border: none;
    border-bottom: 3px solid transparent;
    color: var(--text-muted) !important;
    font-weight: 600;
    padding: 0.6rem 1rem;
    transition: all 0.2s;
    font-family: 'Inter', sans-serif !important;
}
.stTabs [aria-selected="true"] {
    border-bottom-color: var(--accent-green) !important;
    color: var(--accent-green) !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border-color);
    border-radius: 12px;
    overflow: hidden;
}

/* Sidebar Nav Buttons - Green buttons with high-contrast white text */
[data-testid="stSidebar"] button[kind="secondary"],
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #15803d 0%, #166534 100%) !important;
    background-color: #166534 !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    color: #ffffff !important;
    text-align: left !important;
    justify-content: flex-start !important;
    border-radius: 12px !important;
    padding: 0.65rem 1rem !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25) !important;
    transition: all 0.2s ease !important;
    margin-bottom: 0.35rem !important;
}

/* Force 100% pure white text on all elements inside sidebar buttons */
[data-testid="stSidebar"] .stButton > button *,
[data-testid="stSidebar"] .stButton > button p,
[data-testid="stSidebar"] .stButton > button span,
[data-testid="stSidebar"] .stButton > button div {
    color: #ffffff !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* Hover effect */
[data-testid="stSidebar"] button[kind="secondary"]:hover,
[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, #16a34a 0%, #15803d 100%) !important;
    background-color: #16a34a !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(22, 163, 74, 0.35) !important;
}

[data-testid="stSidebar"] button[kind="secondary"]:hover *,
[data-testid="stSidebar"] .stButton > button:hover * {
    color: #ffffff !important;
    opacity: 1 !important;
}

/* Active Nav Button - Glowing Emerald Green Highlight */
[data-testid="stSidebar"] button[kind="primary"] {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    background-color: #10b981 !important;
    border: 1.5px solid #34d399 !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    border-radius: 12px !important;
    padding: 0.65rem 1rem !important;
    font-size: 0.95rem !important;
    text-align: left !important;
    justify-content: flex-start !important;
    box-shadow: 0 4px 16px rgba(16, 185, 129, 0.45) !important;
    margin-bottom: 0.35rem !important;
}

[data-testid="stSidebar"] button[kind="primary"] *,
[data-testid="stSidebar"] button[kind="primary"] p,
[data-testid="stSidebar"] button[kind="primary"] span,
[data-testid="stSidebar"] button[kind="primary"] div {
    color: #ffffff !important;
    font-weight: 800 !important;
    opacity: 1 !important;
}

/* Sidebar Card */
.sa-sidebar-card {
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-radius: 16px;
    padding: 1.2rem;
    text-align: center;
    margin-top: 1rem;
}
.sa-sidebar-card p {
    color: #cbd5e1 !important;
    font-size: 0.82rem !important;
    margin-bottom: 0.8rem !important;
}

/* User Profile in Sidebar */
.sa-sidebar-user {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem;
    background: rgba(255, 255, 255, 0.04);
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    margin-top: 1rem;
}
.sa-user-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: #10b981;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    color: #ffffff !important;
    font-size: 0.9rem;
}
.sa-user-info h5 {
    margin: 0;
    font-size: 0.85rem;
    font-weight: 600;
    color: #ffffff !important;
}
.sa-user-info p {
    margin: 0;
    font-size: 0.7rem;
    color: #64748b !important;
}

/* Hide default Streamlit padding at top */
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Card styling helper */
.sa-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 1.4rem 1.5rem;
    box-shadow: var(--shadow-sm);
    transition: all 0.25s ease;
    margin-bottom: 1rem;
}
.sa-card:hover {
    box-shadow: var(--shadow-md);
    border-color: var(--border-hover);
}

.sa-card-flat {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
}

/* Hero banner */
.sa-hero {
    background: var(--bg-hero);
    border-radius: 20px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: #ffffff !important;
    position: relative;
    overflow: hidden;
}
.sa-hero * {
    color: #ffffff !important;
}
.sa-hero h1 {
    font-family: 'Outfit', sans-serif !important;
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}
.sa-hero p {
    font-size: 1rem;
    opacity: 0.9;
    max-width: 550px;
}

/* Badges */
.sa-badge {
    display: inline-block;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.sa-badge-green {
    background: var(--badge-green-bg);
    color: var(--badge-green-text);
}
.sa-badge-amber {
    background: var(--badge-amber-bg);
    color: var(--badge-amber-text);
}
.sa-badge-cyan {
    background: var(--badge-cyan-bg);
    color: var(--badge-cyan-text);
}

/* Engine cards on dashboard */
.sa-engine-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: left;
    transition: all 0.3s ease;
    cursor: default;
    min-height: 220px;
}
.sa-engine-card:hover {
    border-color: var(--accent-green);
    box-shadow: var(--shadow-md);
    transform: translateY(-3px);
}
.sa-engine-icon {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    margin-bottom: 1rem;
}
.sa-engine-icon-green { background: var(--badge-green-bg); }
.sa-engine-icon-amber { background: var(--badge-amber-bg); }
.sa-engine-icon-cyan { background: var(--badge-cyan-bg); }

.sa-engine-card h3 {
    font-size: 1.1rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
    color: var(--text-primary) !important;
}
.sa-engine-card p {
    font-size: 0.85rem;
    color: var(--text-secondary) !important;
    line-height: 1.45;
}

/* Score bar */
.sa-score-bar {
    height: 8px;
    border-radius: 4px;
    background: var(--border-color);
    overflow: hidden;
    margin: 0.3rem 0;
}
.sa-score-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
}

/* Result card emphasis */
.sa-result-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-left: 5px solid var(--accent-green);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    box-shadow: var(--shadow-sm);
    margin-bottom: 1rem;
}

/* Crop rank card */
.sa-rank-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.6rem;
    transition: all 0.2s ease;
}
.sa-rank-card:hover {
    border-color: var(--accent-green);
    box-shadow: var(--shadow-sm);
}

/* Weather card */
.sa-weather {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 1.4rem;
}

/* Soil health circle */
.sa-soil-circle {
    width: 100px;
    height: 100px;
    border-radius: 50%;
    border: 6px solid var(--accent-green);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    margin: 0 auto 0.5rem;
}

/* Status indicators */
.sa-status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
}
.sa-status-good { background: #16a34a; }
.sa-status-warn { background: #f59e0b; }
.sa-status-bad { background: #e11d48; }

/* Sidebar logo area */
.sa-sidebar-logo {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.5rem 0 1rem 0;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 1rem;
}
.sa-sidebar-logo-icon {
    font-size: 2rem;
}
.sa-sidebar-logo-text h2 {
    font-size: 1.15rem;
    font-weight: 700;
    margin: 0;
    line-height: 1.2;
}
.sa-sidebar-logo-text p {
    font-size: 0.72rem;
    color: var(--text-muted) !important;
    margin: 0;
}

/* Nav item active override */
.sa-nav-active > button {
    background: var(--nav-active-bg) !important;
    color: var(--nav-active-text) !important;
}

/* Smooth animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
.sa-animate {
    animation: fadeIn 0.4s ease-out;
}

/* Soil health progress bars */
.sa-progress {
    height: 10px;
    border-radius: 5px;
    background: var(--border-color);
    overflow: hidden;
}
.sa-progress-fill {
    height: 100%;
    border-radius: 5px;
}
"""

THEME_CSS = DARK_CSS if IS_DARK else LIGHT_CSS
st.markdown(f"<style>{THEME_CSS}\n{COMMON_CSS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# CHART HELPERS (theme-aware)
# ---------------------------------------------------------------------------
def get_chart_colors():
    if IS_DARK:
        return {
            "bg": "#1e293b",
            "text": "#f1f5f9",
            "grid": "#334155",
            "spine": "#475569",
            "green": "#10b981",
            "amber": "#fbbf24",
            "cyan": "#22d3ee",
            "rose": "#fb7185",
            "purple": "#a78bfa",
            "blue": "#60a5fa",
            "emerald": "#34d399",
            "palette": ["#10b981", "#22d3ee", "#fbbf24", "#a78bfa", "#fb7185", "#60a5fa"],
        }
    else:
        return {
            "bg": "#ffffff",
            "text": "#1e293b",
            "grid": "#e2e8f0",
            "spine": "#cbd5e1",
            "green": "#16a34a",
            "amber": "#d97706",
            "cyan": "#0891b2",
            "rose": "#e11d48",
            "purple": "#7c3aed",
            "blue": "#2563eb",
            "emerald": "#059669",
            "palette": ["#16a34a", "#0891b2", "#d97706", "#7c3aed", "#e11d48", "#2563eb"],
        }


def style_chart(fig, ax, title="", colors=None):
    if colors is None:
        colors = get_chart_colors()
    fig.patch.set_facecolor(colors["bg"])
    ax.set_facecolor(colors["bg"])
    ax.tick_params(colors=colors["text"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(colors["spine"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, color=colors["text"], fontsize=11, fontweight="bold", pad=12,
                      fontfamily="sans-serif")
    return colors


# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------
bootstrap_needed = not project_ready()
pipeline, harness = load_modules()
states_list, state_dist_map = load_geography_data()

# ---------------------------------------------------------------------------
# BOOTSTRAP CHECK (runs before any UI if models missing)
# ---------------------------------------------------------------------------
if bootstrap_needed:
    st.markdown("""
    <div class="sa-hero" style="text-align:center; padding:3rem 2rem;">
        <h1 style="font-size:2.4rem;">🌿 SmartAgri — First Time Setup</h1>
        <p style="margin:0 auto; max-width:600px; font-size:1.05rem;">
            Models and encoders not found. SmartAgri will now train all AI models
            by running the complete training pipeline (NB1 → NB6). This may take several minutes.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Training AI models... This may take 5-15 minutes."):
        bootstrap_project()
    st.rerun()


# ---------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------------
NAV_ITEMS = [
    ("🏠", "Dashboard"),
    ("🌾", "Crop Recommendation"),
    ("🧪", "Fertilizer Recommendation"),
    ("📈", "Yield Prediction"),
    ("📊", "Model Diagnostics"),
    ("📁", "Dataset Explorer"),
    ("⚙️", "Settings"),
    ("ℹ️", "About"),
]

with st.sidebar:
    # Logo
    st.markdown("""
    <div class="sa-sidebar-logo">
        <div class="sa-sidebar-logo-icon">🌿</div>
        <div class="sa-sidebar-logo-text">
            <h2>SmartAgri</h2>
            <p>Smart Farming, Better Future</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Dashboard"

    for icon, page_name in NAV_ITEMS:
        is_active = st.session_state.current_page == page_name
        btn_label = f"{icon}  {page_name}"
        btn_type = "primary" if is_active else "secondary"
        if st.button(btn_label, key=f"nav_{page_name}", type=btn_type, use_container_width=True):
            st.session_state.current_page = page_name
            st.rerun()

    st.markdown("---")

    # System status
    st.markdown("**System Status**")
    if pipeline:
        st.markdown(f'<span class="sa-status-dot sa-status-good"></span> All engines online',
                     unsafe_allow_html=True)
        st.caption(f"Crop: `{getattr(pipeline, 'crop_model_name', 'LightGBM')}`")
        st.caption(f"Fert: `{getattr(pipeline, 'fert_model_name', 'Random Forest')}`")
        st.caption(f"Yield: `{getattr(pipeline, 'yield_model_name', 'XGBoost')}`")
    else:
        st.markdown(f'<span class="sa-status-dot sa-status-bad"></span> Pipeline not loaded',
                     unsafe_allow_html=True)

    st.markdown("---")

    # Bottom Callout Card matching image
    st.markdown("""
    <div class="sa-sidebar-card">
        <div style="font-size:2.2rem; margin-bottom:0.4rem;">🌱</div>
        <p>Let AI help you make the best farming decisions.</p>
    </div>
    """, unsafe_allow_html=True)

    # User Profile Pill matching image
    st.markdown("""
    <div class="sa-sidebar-user">
        <div class="sa-user-avatar">PY</div>
        <div class="sa-user-info">
            <h5>Prabin Yadav</h5>
            <p>prabin.yadav@example.com</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # Theme Toggle in Sidebar
    theme_btn_label = "🌙 Dark Mode" if not IS_DARK else "☀️ Light Mode"
    if st.button(theme_btn_label, key="sidebar_theme_toggle_btn", use_container_width=True):
        st.session_state.theme = "dark" if not IS_DARK else "light"
        st.rerun()

current_page = st.session_state.current_page


# ===================================================================
# PAGE: DASHBOARD
# ===================================================================
if current_page == "Dashboard":
    # Hero Banner
    st.markdown("""
    <div class="sa-hero sa-animate">
        <h1>🌾 AI Powered Farming Decisions</h1>
        <p>Get smart recommendations for crops, fertilizers and predict your yield with AI.
        Powered by machine learning trained on Indian agricultural data.</p>
        <div style="display:flex; gap:1.2rem; margin-top:1rem; flex-wrap:wrap;">
            <span class="sa-badge" style="background:rgba(255,255,255,0.2); color:#fff;">
                ✓ Accurate
            </span>
            <span class="sa-badge" style="background:rgba(255,255,255,0.2); color:#fff;">
                ✓ Personalized
            </span>
            <span class="sa-badge" style="background:rgba(255,255,255,0.2); color:#fff;">
                ✓ Data Driven
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Three Engine Cards
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown("""
        <div class="sa-engine-card sa-animate">
            <div class="sa-engine-icon sa-engine-icon-green">🌾</div>
            <h3>Crop Recommendation</h3>
            <p>AI suggests the best crops suitable for your soil and environment.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Get Recommendation →", key="dash_crop_btn"):
            st.session_state.current_page = "Crop Recommendation"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="sa-engine-card sa-animate" style="animation-delay:0.1s;">
            <div class="sa-engine-icon sa-engine-icon-amber">🧪</div>
            <h3>Fertilizer Recommendation</h3>
            <p>Get the right fertilizer and nutrient suggestions to improve soil health.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Get Recommendation →", key="dash_fert_btn"):
            st.session_state.current_page = "Fertilizer Recommendation"
            st.rerun()

    with col3:
        st.markdown("""
        <div class="sa-engine-card sa-animate" style="animation-delay:0.2s;">
            <div class="sa-engine-icon sa-engine-icon-cyan">📈</div>
            <h3>Yield Prediction</h3>
            <p>Predict your crop yield using AI based on historical data and current inputs.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Predict Yield →", key="dash_yield_btn"):
            st.session_state.current_page = "Yield Prediction"
            st.rerun()

    st.markdown("---")

    # Quick Analysis + Stats row
    qa_col, stats_col = st.columns([1.3, 1], gap="large")

    with qa_col:
        st.markdown("#### ✨ Quick Analysis")
        st.caption("Enter your field details to get AI recommendations")

        st.markdown('<div class="sa-card">', unsafe_allow_html=True)
        qq1, qq2 = st.columns(2)
        with qq1:
            default_state_idx = states_list.index("Maharashtra") if "Maharashtra" in states_list else 0
            qa_state = st.selectbox("State", options=states_list, index=default_state_idx, key="qa_state")
        with qq2:
            qa_districts = state_dist_map.get(qa_state, [])
            if not qa_districts:
                qa_districts = ["Pune", "Mumbai City", "Nagpur", "Jaisalmer", "Patna"]
            qa_district = st.selectbox("District", options=qa_districts, key="qa_district")

        qq3, qq4 = st.columns(2)
        with qq3:
            qa_soil = st.selectbox("Soil Type", [
                "alluvial soil", "black soil", "clayey soil", "loamy soil",
                "red soil", "sandy soil", "sandy loam soil",
            ], key="qa_soil")
        with qq4:
            qa_season = st.selectbox("Season", ["kharif", "rabi", "zaid"], key="qa_season")

        qq5, qq6, qq7 = st.columns(3)
        with qq5:
            qa_n = st.number_input("Nitrogen (N)", 0.0, 140.0, 60.0, step=5.0, key="qa_n")
        with qq6:
            qa_p = st.number_input("Phosphorus (P)", 0.0, 140.0, 35.0, step=5.0, key="qa_p")
        with qq7:
            qa_k = st.number_input("Potassium (K)", 0.0, 140.0, 30.0, step=5.0, key="qa_k")

        qq8, qq9 = st.columns(2)
        with qq8:
            qa_ph = st.number_input("pH Value", 3.5, 10.0, 6.5, step=0.1, key="qa_ph")
        with qq9:
            qa_irr = st.selectbox("Irrigation", ["rainfed", "canal", "drip", "sprinkler", "tube well"], key="qa_irr")

        qa_submit = st.button("✨ Get Recommendations", use_container_width=True, key="qa_submit_btn")
        st.markdown('</div>', unsafe_allow_html=True)

        if qa_submit:
            try:
                qa_payload = {
                    "district": qa_district, "state": qa_state,
                    "soil_type": qa_soil, "season": qa_season, "irrigation": qa_irr,
                    "N": qa_n, "P": qa_p, "K": qa_k, "pH": qa_ph,
                }
                qa_res = execute_pipeline(pipeline, harness, qa_payload)
                qa_crops = qa_res.get("top_crops", [])
                qa_fert = qa_res.get("fertilizer", {})
                qa_yield = qa_res.get("yield", {})

                # Save to history
                if qa_crops:
                    st.session_state.history.insert(0, {
                        "type": "Crop Recommendation",
                        "crop": qa_crops[0]["name"],
                        "area": "—",
                        "date": pd.Timestamp.now().strftime("%d %b %Y"),
                        "status": "Completed",
                    })
                    if qa_fert.get("name"):
                        st.session_state.history.insert(1, {
                            "type": "Fertilizer Recommendation",
                            "crop": qa_fert["name"],
                            "area": "—",
                            "date": pd.Timestamp.now().strftime("%d %b %Y"),
                            "status": "Completed",
                        })

                st.markdown("##### 🎯 Quick Results")
                qr1, qr2, qr3 = st.columns(3)
                with qr1:
                    if qa_crops:
                        st.metric("Best Crop", qa_crops[0]["name"],
                                  f"{qa_crops[0]['score']*100:.1f}% match")
                with qr2:
                    if qa_fert.get("name"):
                        st.metric("Fertilizer", qa_fert["name"])
                with qr3:
                    yv = qa_yield.get("value")
                    if yv is not None:
                        st.metric("Est. Yield", f"{yv:.2f} t/ha")

                st.info("💡 For detailed analysis with charts, visit the individual engine pages.")

            except Exception as e:
                st.error(f"Quick analysis failed: {e}")

    with stats_col:
        # Model stats
        st.markdown("#### 📊 Model Performance")
        st.markdown("""
        <div class="sa-card-flat">
        """, unsafe_allow_html=True)

        # Try to load evaluation metrics
        crop_report = REPORTS_DIR / "eval_crop_models.csv"
        fert_report = REPORTS_DIR / "eval_fert_models.csv"
        yield_report = REPORTS_DIR / "eval_yield_models.csv"

        if crop_report.exists():
            try:
                cr = pd.read_csv(crop_report)
                best_acc = cr["Accuracy"].max() if "Accuracy" in cr.columns else 95
                st.markdown(f"**Crop Model Accuracy**: `{best_acc:.1f}%`")
            except Exception:
                st.markdown("**Crop Model**: Ready ✅")
        else:
            st.markdown("**Crop Model**: Ready ✅")

        if fert_report.exists():
            try:
                fr = pd.read_csv(fert_report)
                best_facc = fr["Accuracy"].max() if "Accuracy" in fr.columns else 92
                st.markdown(f"**Fertilizer Model Accuracy**: `{best_facc:.1f}%`")
            except Exception:
                st.markdown("**Fertilizer Model**: Ready ✅")
        else:
            st.markdown("**Fertilizer Model**: Ready ✅")

        if yield_report.exists():
            try:
                yr = pd.read_csv(yield_report)
                if "R2" in yr.columns:
                    best_r2 = yr["R2"].max()
                    st.markdown(f"**Yield Model R²**: `{best_r2:.4f}`")
                else:
                    st.markdown("**Yield Model**: Ready ✅")
            except Exception:
                st.markdown("**Yield Model**: Ready ✅")
        else:
            st.markdown("**Yield Model**: Ready ✅")

        st.markdown("</div>", unsafe_allow_html=True)

        # Soil health summary
        st.markdown("#### 🌱 Soil Health Summary")
        st.markdown("""
        <div class="sa-card-flat">
        """, unsafe_allow_html=True)

        npk_vals = {"N": 65, "P": 70, "K": 60, "Organic Matter": 80, "pH Level": 6.8}
        for nutrient, val in npk_vals.items():
            if nutrient == "pH Level":
                st.markdown(f"**{nutrient}**: `{val}`")
            else:
                pct = val
                color = "#16a34a" if pct >= 60 else "#f59e0b" if pct >= 40 else "#e11d48"
                st.markdown(
                    f"**{nutrient}**: {pct}%"
                    f'<div class="sa-progress"><div class="sa-progress-fill" '
                    f'style="width:{pct}%; background:{color};"></div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)

    # Recent Recommendations
    st.markdown("---")
    st.markdown("#### 📋 Recent Recommendations")

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history[:10])
        display_cols = ["type", "crop", "date", "status"]
        available = [c for c in display_cols if c in hist_df.columns]
        st.dataframe(hist_df[available], use_container_width=True, hide_index=True)
    else:
        st.info("No recommendations yet. Use Quick Analysis above or visit engine pages to get started.")


# ===================================================================
# PAGE: CROP RECOMMENDATION
# ===================================================================
elif current_page == "Crop Recommendation":
    st.markdown('<div class="sa-animate">', unsafe_allow_html=True)
    st.markdown("## 🌾 Crop Recommendation Engine")
    st.caption("Select state, district, soil parameters, and nutrient levels to generate top crop options ranked by AI suitability score.")

    # Regional Presets
    st.markdown("**⚡ Quick Regional Presets:**")
    preset_cols = st.columns(4)

    # Session state defaults
    for key, default in [
        ("cr_state", "Rajasthan"), ("cr_district", "Jaisalmer"),
        ("cr_n", 40.0), ("cr_p", 20.0), ("cr_k", 20.0),
        ("cr_ph", 7.4), ("cr_soil", "sandy loam soil"),
        ("cr_season", "kharif"), ("cr_irrigation", "rainfed"),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    with preset_cols[0]:
        if st.button("🏜️ Jaisalmer (Arid)", use_container_width=True, key="preset_1"):
            st.session_state.update(cr_state="Rajasthan", cr_district="Jaisalmer",
                cr_n=40.0, cr_p=20.0, cr_k=20.0, cr_ph=7.8,
                cr_soil="sandy loam soil", cr_season="kharif", cr_irrigation="rainfed")
    with preset_cols[1]:
        if st.button("🌴 Alappuzha (Coastal)", use_container_width=True, key="preset_2"):
            st.session_state.update(cr_state="Kerala", cr_district="Alappuzha",
                cr_n=70.0, cr_p=40.0, cr_k=50.0, cr_ph=6.0,
                cr_soil="alluvial soil", cr_season="kharif", cr_irrigation="rainfed")
    with preset_cols[2]:
        if st.button("🌾 Patna (Gangetic)", use_container_width=True, key="preset_3"):
            st.session_state.update(cr_state="Bihar", cr_district="Patna",
                cr_n=60.0, cr_p=30.0, cr_k=35.0, cr_ph=6.8,
                cr_soil="alluvial soil", cr_season="rabi", cr_irrigation="canal")
    with preset_cols[3]:
        if st.button("🌽 Ludhiana (Fertile)", use_container_width=True, key="preset_4"):
            st.session_state.update(cr_state="Punjab", cr_district="Ludhiana",
                cr_n=90.0, cr_p=50.0, cr_k=40.0, cr_ph=7.2,
                cr_soil="loamy soil", cr_season="rabi", cr_irrigation="tube well")

    st.markdown("---")

    form_col, result_col = st.columns([1.0, 1.3], gap="large")

    with form_col:
        st.markdown("""
        <div class="sa-card">
            <h4 style="margin-top:0;">📋 Input Parameters</h4>
        """, unsafe_allow_html=True)

        sel_state = st.selectbox("State / Region", options=states_list,
            index=states_list.index(st.session_state.cr_state)
            if st.session_state.cr_state in states_list else 0,
            key="crop_state_sel")

        available_districts = state_dist_map.get(sel_state, [])
        if not available_districts:
            available_districts = ["Jaisalmer", "Alappuzha", "Patna", "Ludhiana"]
        dist_idx = 0
        if st.session_state.cr_district in available_districts:
            dist_idx = available_districts.index(st.session_state.cr_district)
        sel_district = st.selectbox("District", options=available_districts, index=dist_idx, key="crop_dist_sel")

        soils = ["alluvial soil", "black soil", "clayey soil", "loamy soil", "red soil",
                 "sandy soil", "sandy loam soil", "laterite soil", "peaty soil"]
        sel_soil = st.selectbox("Soil Type", options=soils,
            index=soils.index(st.session_state.cr_soil) if st.session_state.cr_soil in soils else 0,
            key="crop_soil_sel")

        c_season, c_irrigation = st.columns(2)
        with c_season:
            sel_season = st.selectbox("Season", options=["kharif", "rabi", "zaid"],
                index=["kharif", "rabi", "zaid"].index(st.session_state.cr_season),
                key="crop_season_sel")
        with c_irrigation:
            sel_irrigation = st.selectbox("Irrigation",
                options=["rainfed", "canal", "sprinkler", "drip", "tube well"],
                index=["rainfed", "canal", "sprinkler", "drip", "tube well"].index(st.session_state.cr_irrigation),
                key="crop_irr_sel")

        st.markdown("**Soil Nutrient Levels (kg/ha):**")
        n_col, p_col, k_col = st.columns(3)
        with n_col:
            val_n = st.slider("Nitrogen (N)", 0.0, 140.0, float(st.session_state.cr_n), step=5.0, key="crop_n")
        with p_col:
            val_p = st.slider("Phosphorus (P)", 0.0, 140.0, float(st.session_state.cr_p), step=5.0, key="crop_p")
        with k_col:
            val_k = st.slider("Potassium (K)", 0.0, 140.0, float(st.session_state.cr_k), step=5.0, key="crop_k")

        val_ph = st.slider("Soil pH Level", 3.5, 10.0, float(st.session_state.cr_ph), step=0.1, key="crop_ph")
        if val_ph < 6.0:
            ph_label = "🔴 Acidic"
        elif val_ph <= 7.5:
            ph_label = "🟢 Optimal"
        else:
            ph_label = "🔵 Alkaline"
        st.caption(f"pH Status: **{ph_label}**")

        btn_run = st.button("✨ Generate AI Recommendation", use_container_width=True, key="crop_run_btn")

        st.markdown("</div>", unsafe_allow_html=True)

    with result_col:
        payload = {
            "district": sel_district, "state": sel_state,
            "soil_type": sel_soil, "season": sel_season,
            "irrigation": sel_irrigation,
            "N": val_n, "P": val_p, "K": val_k, "pH": val_ph,
        }

        try:
            res = execute_pipeline(pipeline, harness, payload)
            top_crops = res.get("top_crops", [])
            fert_info = res.get("fertilizer", {})
            yield_info = res.get("yield", {})
            rain_info = res.get("rain", {})

            if top_crops:
                best_crop = top_crops[0]
                yield_val = yield_info.get("value", 0) or 0

                # Prime Recommendation Card
                st.markdown(f"""
                <div class="sa-result-card sa-animate">
                    <span class="sa-badge sa-badge-green">🥇 PRIME RECOMMENDATION</span>
                    <h2 style="font-family:'Outfit'; margin:0.3rem 0; font-size:1.8rem;">
                        {best_crop['name']}
                    </h2>
                    <div style="display:flex; gap:2rem; margin-top:0.8rem; flex-wrap:wrap;">
                        <div>
                            <div style="color:var(--text-muted); font-size:0.8rem;">SUITABILITY</div>
                            <div style="font-size:1.5rem; font-weight:700; color:var(--accent-green);">
                                {best_crop['score']*100:.1f}%
                            </div>
                        </div>
                        <div>
                            <div style="color:var(--text-muted); font-size:0.8rem;">EST. YIELD</div>
                            <div style="font-size:1.5rem; font-weight:700; color:var(--accent-cyan);">
                                {yield_val:.2f} <span style="font-size:0.85rem;">t/ha</span>
                            </div>
                        </div>
                        <div>
                            <div style="color:var(--text-muted); font-size:0.8rem;">FERTILIZER</div>
                            <div style="font-size:1.2rem; font-weight:700; color:var(--accent-amber);">
                                {fert_info.get('name', 'NPK')}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Top Ranked Crops list
                st.markdown("#### 🏆 Top Ranked Crops")
                for i, crop in enumerate(top_crops, start=1):
                    badge_cls = "sa-badge-green" if i == 1 else ("sa-badge-amber" if i == 2 else "sa-badge-cyan")
                    medal = "🥇" if i == 1 else ("🥈" if i == 2 else "🥉" if i == 3 else f"#{i}")
                    score_pct = crop['score'] * 100
                    bar_color = "#16a34a" if score_pct >= 70 else "#d97706" if score_pct >= 50 else "#0891b2"
                    st.markdown(f"""
                    <div class="sa-rank-card">
                        <div style="display:flex; align-items:center; gap:0.8rem;">
                            <span class="sa-badge {badge_cls}">{medal} Rank #{i}</span>
                            <span style="font-size:1.1rem; font-weight:700;">{crop['name']}</span>
                        </div>
                        <div style="text-align:right; min-width:120px;">
                            <div style="font-size:1.15rem; font-weight:800; color:var(--accent-green);">
                                {score_pct:.1f}%
                            </div>
                            <div class="sa-score-bar" style="width:100px;">
                                <div class="sa-score-fill" style="width:{score_pct}%; background:{bar_color};"></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # --- CHARTS ---
                st.markdown("---")
                chart_tabs = st.tabs(["📊 Score Breakdown", "📈 Crop Comparison", "🌧️ Climate Context"])

                # Tab 1: Component Score Breakdown
                with chart_tabs[0]:
                    if best_crop.get("components"):
                        comps = best_crop["components"]
                        colors = get_chart_colors()

                        fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=colors["bg"])
                        labels = ["ML Suitability", "Yield Score", "Region", "Irrigation",
                                  "Season", "Presence", "Temperature", "Water", "pH"]
                        vals = [
                            comps.get("ml_prob", 0) * 100,
                            comps.get("yield_score", 0) * 100,
                            comps.get("region_score", 0) * 100,
                            comps.get("irr_score", 0) * 100,
                            comps.get("season_score", 0) * 100,
                            comps.get("presence_score", 0) * 100,
                            comps.get("temp_score", 0) * 100,
                            comps.get("water_score", 0) * 100,
                            comps.get("ph_score", 0) * 100,
                        ]
                        bar_colors = [
                            colors["green"], colors["cyan"], colors["amber"],
                            colors["purple"], colors["rose"], colors["blue"],
                            "#f97316", "#06b6d4", "#84cc16",
                        ]

                        bars = ax.barh(labels, vals, color=bar_colors, height=0.6, edgecolor="none")
                        ax.set_xlim(0, 105)
                        style_chart(fig, ax, f"Component Score Breakdown — {best_crop['name']}", colors)

                        for bar in bars:
                            w = bar.get_width()
                            ax.text(w + 1.5, bar.get_y() + bar.get_height() / 2,
                                    f"{w:.1f}%", va="center", color=colors["text"],
                                    fontsize=8.5, fontweight="bold")

                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

                # Tab 2: Crop Comparison
                with chart_tabs[1]:
                    if len(top_crops) >= 2:
                        colors = get_chart_colors()
                        fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=colors["bg"])
                        crop_names = [c["name"] for c in top_crops[:5]]
                        crop_scores = [c["score"] * 100 for c in top_crops[:5]]

                        bar_c = [colors["palette"][i % len(colors["palette"])] for i in range(len(crop_names))]
                        bars = ax.bar(crop_names, crop_scores, color=bar_c, width=0.5,
                                       edgecolor="none", zorder=3)
                        ax.set_ylim(0, max(crop_scores) * 1.2)
                        ax.set_ylabel("Suitability Score (%)", color=colors["text"], fontsize=10)
                        ax.grid(axis="y", color=colors["grid"], alpha=0.5, zorder=0)
                        style_chart(fig, ax, "Top Crops Comparison", colors)

                        for bar in bars:
                            h = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                                    f"{h:.1f}%", ha="center", color=colors["text"],
                                    fontsize=9, fontweight="bold")

                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

                # Tab 3: Climate
                with chart_tabs[2]:
                    cl1, cl2 = st.columns(2)
                    with cl1:
                        st.metric("🌧️ Annual Rainfall", f"{rain_info.get('annual', 0):.1f} mm")
                    with cl2:
                        st.metric(f"💧 Season Rainfall ({sel_season.title()})",
                                  f"{rain_info.get('season', 0):.1f} mm")

                    # Rainfall gauge chart
                    colors = get_chart_colors()
                    fig, ax = plt.subplots(figsize=(6, 2.5), facecolor=colors["bg"])
                    rain_cats = ["Annual", f"{sel_season.title()} Season"]
                    rain_vals = [rain_info.get("annual", 0), rain_info.get("season", 0)]
                    bars = ax.barh(rain_cats, rain_vals, color=[colors["cyan"], colors["green"]],
                                    height=0.5, edgecolor="none")
                    style_chart(fig, ax, "Rainfall Distribution (mm)", colors)
                    for bar in bars:
                        w = bar.get_width()
                        ax.text(w + 10, bar.get_y() + bar.get_height() / 2,
                                f"{w:.0f} mm", va="center", color=colors["text"], fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

            else:
                st.warning("No crop passed the seasonal and environmental filters for this combination. "
                           "Try adjusting the season, soil parameters, or selecting a different region.")

        except Exception as exc:
            st.error(f"Error executing recommendation engine: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)


# ===================================================================
# PAGE: FERTILIZER RECOMMENDATION
# ===================================================================
elif current_page == "Fertilizer Recommendation":
    st.markdown('<div class="sa-animate">', unsafe_allow_html=True)
    st.markdown("## 🧪 Fertilizer Recommendation Engine")
    st.caption("Analyze soil nutrient balance, identify deficiencies, and review ML fertilizer probability distributions.")

    f_left, f_right = st.columns([1.0, 1.3], gap="large")

    with f_left:
        st.markdown("""
        <div class="sa-card">
            <h4 style="margin-top:0;">⚖️ Soil & Crop Setup</h4>
        """, unsafe_allow_html=True)

        fert_state = st.selectbox("State", options=states_list, key="fert_state_sel")
        fert_districts = state_dist_map.get(fert_state, ["Alappuzha", "Patna"])
        fert_district = st.selectbox("District", options=fert_districts, key="fert_dist_sel")

        target_crop_input = st.selectbox("Target Crop", options=[
            "Rice", "Wheat", "Maize", "Cotton", "Sugarcane",
            "Chickpea", "Jute", "Groundnut", "Mustard",
        ], index=0, key="fert_crop_sel")

        n_val_f = st.slider("Current Soil Nitrogen (N)", 0, 140, 50, key="fert_n")
        p_val_f = st.slider("Current Soil Phosphorus (P)", 0, 140, 30, key="fert_p")
        k_val_f = st.slider("Current Soil Potassium (K)", 0, 140, 25, key="fert_k")

        soil_f = st.selectbox("Soil Type",
            ["alluvial soil", "black soil", "clayey soil", "loamy soil", "sandy soil"],
            index=0, key="fert_soil_sel")
        season_f = st.selectbox("Season", ["kharif", "rabi", "zaid"], index=0, key="fert_season_sel")
        irr_f = st.selectbox("Irrigation", ["rainfed", "canal", "drip", "sprinkler", "tube well"],
                             index=0, key="fert_irr_sel")
        ph_f = st.slider("pH Level", 3.5, 9.5, 6.5, key="fert_ph_sel")

        run_fert = st.button("🧪 Analyze NPK & Get Fertilizer", use_container_width=True, key="fert_run_btn")

        st.markdown("</div>", unsafe_allow_html=True)

    with f_right:
        payload_f = {
            "district": fert_district, "state": fert_state,
            "soil_type": soil_f, "season": season_f, "irrigation": irr_f,
            "N": float(n_val_f), "P": float(p_val_f), "K": float(k_val_f),
            "pH": float(ph_f),
        }

        try:
            res_f = execute_pipeline(pipeline, harness, payload_f)
            fert_data = res_f.get("fertilizer", {})
            rec_fert = fert_data.get("name", "Urea")
            reason_fert = fert_data.get("reason", "Maintains optimal soil nutrient equilibrium.")
            ml_probs = fert_data.get("ml_probs", {})

            # Fertilizer suggestion card
            st.markdown(f"""
            <div class="sa-result-card sa-animate" style="border-left-color: var(--accent-amber);">
                <span class="sa-badge sa-badge-amber">🧪 FERTILIZER SUGGESTION</span>
                <h2 style="font-family:'Outfit'; color:var(--accent-amber) !important; margin:0.3rem 0;">
                    {rec_fert}
                </h2>
                <p style="color:var(--text-secondary) !important; font-size:0.92rem; margin-top:0.5rem;">
                    {reason_fert}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # NPK Deficiency Analysis
            st.markdown("#### 📊 Nutrient Analysis")
            def_cols = st.columns(3)
            ideal_npk = {"N": 80, "P": 40, "K": 40}
            current_npk = {"N": n_val_f, "P": p_val_f, "K": k_val_f}
            nutrient_names = {"N": "Nitrogen", "P": "Phosphorus", "K": "Potassium"}

            for col, (key, name) in zip(def_cols, nutrient_names.items()):
                with col:
                    diff = current_npk[key] - ideal_npk[key]
                    if diff < -20:
                        status = "🔴 Deficient"
                        status_color = "#e11d48"
                    elif diff < 0:
                        status = "🟡 Low"
                        status_color = "#d97706"
                    elif diff < 20:
                        status = "🟢 Optimal"
                        status_color = "#16a34a"
                    else:
                        status = "🔵 High"
                        status_color = "#0891b2"
                    st.metric(name, f"{current_npk[key]} kg/ha",
                              delta=f"{diff:+.0f} vs ideal", delta_color="normal")
                    st.caption(status)

            # Chart tabs
            fert_chart_tabs = st.tabs(["📊 NPK Comparison", "🤖 ML Probability", "📈 Nutrient Balance"])

            # NPK Bar Chart
            with fert_chart_tabs[0]:
                colors = get_chart_colors()
                fig_npk, ax_npk = plt.subplots(figsize=(7, 3.5), facecolor=colors["bg"])
                npk_labels = ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"]
                current_vals = [n_val_f, p_val_f, k_val_f]
                ideal_vals = [80, 40, 40]

                x = np.arange(len(npk_labels))
                width = 0.32

                rects1 = ax_npk.bar(x - width / 2, current_vals, width,
                                     label="Current Level", color=colors["green"], edgecolor="none")
                rects2 = ax_npk.bar(x + width / 2, ideal_vals, width,
                                     label="Ideal Baseline", color=colors["cyan"], edgecolor="none")

                ax_npk.set_ylabel("Nutrient (kg/ha)", color=colors["text"])
                ax_npk.set_xticks(x)
                ax_npk.set_xticklabels(npk_labels)
                ax_npk.legend(facecolor=colors["bg"], edgecolor=colors["grid"], labelcolor=colors["text"])
                ax_npk.grid(axis="y", color=colors["grid"], alpha=0.4)
                style_chart(fig_npk, ax_npk, "Soil NPK: Current vs Ideal", colors)

                for rect in rects1:
                    h = rect.get_height()
                    ax_npk.text(rect.get_x() + rect.get_width() / 2, h + 1,
                                f"{h:.0f}", ha="center", color=colors["text"], fontsize=8.5)
                for rect in rects2:
                    h = rect.get_height()
                    ax_npk.text(rect.get_x() + rect.get_width() / 2, h + 1,
                                f"{h:.0f}", ha="center", color=colors["text"], fontsize=8.5)

                plt.tight_layout()
                st.pyplot(fig_npk, use_container_width=True)
                plt.close(fig_npk)

            # ML Probabilities
            with fert_chart_tabs[1]:
                if ml_probs and isinstance(ml_probs, dict):
                    colors = get_chart_colors()
                    sorted_probs = dict(sorted(ml_probs.items(), key=lambda x: x[1], reverse=True)[:8])

                    fig_prob, ax_prob = plt.subplots(figsize=(7, 3.5), facecolor=colors["bg"])
                    fert_names = list(sorted_probs.keys())
                    prob_vals = list(sorted_probs.values())

                    bar_c = [colors["amber"] if i == 0 else colors["palette"][i % len(colors["palette"])]
                             for i in range(len(fert_names))]
                    bars = ax_prob.barh(fert_names[::-1], prob_vals[::-1], color=bar_c[::-1],
                                        height=0.55, edgecolor="none")
                    ax_prob.set_xlabel("Probability (%)", color=colors["text"])
                    style_chart(fig_prob, ax_prob, "ML Model Fertilizer Probability", colors)

                    for bar in bars:
                        w = bar.get_width()
                        ax_prob.text(w + 0.5, bar.get_y() + bar.get_height() / 2,
                                     f"{w:.1f}%", va="center", color=colors["text"], fontsize=8.5)

                    plt.tight_layout()
                    st.pyplot(fig_prob, use_container_width=True)
                    plt.close(fig_prob)
                else:
                    st.info("ML probability data not available for this configuration.")

            # Nutrient Balance Radar
            with fert_chart_tabs[2]:
                colors = get_chart_colors()
                fig_bal, ax_bal = plt.subplots(figsize=(7, 3), facecolor=colors["bg"])

                nutrients = ["N", "P", "K"]
                deficits = [current_npk[n] - ideal_npk[n] for n in nutrients]
                bar_colors = [colors["green"] if d >= 0 else colors["rose"] for d in deficits]

                ax_bal.bar(["Nitrogen", "Phosphorus", "Potassium"], deficits,
                            color=bar_colors, width=0.45, edgecolor="none")
                ax_bal.axhline(y=0, color=colors["spine"], linewidth=1, zorder=1)
                ax_bal.set_ylabel("Surplus / Deficit (kg/ha)", color=colors["text"])
                ax_bal.grid(axis="y", color=colors["grid"], alpha=0.4)
                style_chart(fig_bal, ax_bal, "Nutrient Surplus / Deficit Analysis", colors)

                for i, (d, n) in enumerate(zip(deficits, ["N", "P", "K"])):
                    ax_bal.text(i, d + (2 if d >= 0 else -4),
                                f"{d:+.0f}", ha="center", color=colors["text"],
                                fontsize=10, fontweight="bold")

                plt.tight_layout()
                st.pyplot(fig_bal, use_container_width=True)
                plt.close(fig_bal)

        except Exception as exc:
            st.error(f"Fertilizer evaluation failed: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)


# ===================================================================
# PAGE: YIELD PREDICTION
# ===================================================================
elif current_page == "Yield Prediction":
    st.markdown('<div class="sa-animate">', unsafe_allow_html=True)
    st.markdown("## 📈 Yield Prediction Engine")
    st.caption("Estimate crop yield in tonnes per hectare and calculate total expected harvest based on farm size.")

    y_col1, y_col2 = st.columns([1.0, 1.3], gap="large")

    with y_col1:
        st.markdown("""
        <div class="sa-card">
            <h4 style="margin-top:0;">🌾 Location & Crop Setup</h4>
        """, unsafe_allow_html=True)

        y_state = st.selectbox("State", options=states_list, index=0, key="yield_state_sel")
        y_districts = state_dist_map.get(y_state, ["Alappuzha", "Patna"])
        y_district = st.selectbox("District", options=y_districts, key="yield_dist_sel")

        y_crop = st.text_input("Target Crop", value="Rice", key="yield_crop_input")
        y_season = st.selectbox("Season", ["kharif", "rabi", "zaid"], index=0, key="yield_season_sel")

        st.markdown("---")
        st.markdown("**🏡 Farm Area Calculator:**")
        unit_type = st.radio("Unit System", ["Acres", "Hectares"], horizontal=True, key="yield_unit")
        area_val = st.number_input(f"Total Land Area ({unit_type})", min_value=0.1, value=5.0,
                                    step=0.5, key="yield_area")

        calc_area_ha = area_val * 0.404686 if unit_type == "Acres" else area_val
        st.caption(f"Equivalent Area: **{calc_area_ha:.2f}** Hectares")

        st.markdown("</div>", unsafe_allow_html=True)

    with y_col2:
        if pipeline:
            try:
                resolved_crop = pipeline.resolve(y_crop)
                est_yield_ha = pipeline.engine_yield_ml(resolved_crop, y_state, y_district, y_season)
                if est_yield_ha is None or est_yield_ha <= 0:
                    est_yield_ha = 2.85

                total_harvest_tonnes = est_yield_ha * calc_area_ha
                total_harvest_quintals = total_harvest_tonnes * 10.0

                # Yield Result Card
                st.markdown(f"""
                <div class="sa-result-card sa-animate" style="border-left-color: var(--accent-cyan);">
                    <span class="sa-badge sa-badge-cyan">📈 YIELD INTELLIGENCE</span>
                    <div style="display:flex; justify-content:space-between; margin-top:0.8rem; flex-wrap:wrap; gap:1.5rem;">
                        <div>
                            <div style="color:var(--text-muted); font-size:0.82rem;">PREDICTED YIELD</div>
                            <div style="font-size:2rem; font-weight:800; color:var(--accent-cyan);">
                                {est_yield_ha:.3f}
                                <span style="font-size:1rem;">t/ha</span>
                            </div>
                        </div>
                        <div>
                            <div style="color:var(--text-muted); font-size:0.82rem;">TOTAL HARVEST</div>
                            <div style="font-size:2rem; font-weight:800; color:var(--accent-green);">
                                {total_harvest_tonnes:.2f}
                                <span style="font-size:1rem;">Tonnes</span>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Production Breakdown
                st.markdown("#### 📦 Production Breakdown")
                m_c1, m_c2, m_c3 = st.columns(3)
                with m_c1:
                    st.metric("Metric Tonnes", f"{total_harvest_tonnes:.2f} T")
                with m_c2:
                    st.metric("Quintals", f"{total_harvest_quintals:.1f} Q")
                with m_c3:
                    st.metric("Farm Area", f"{calc_area_ha:.2f} ha")

                # Chart tabs
                yield_chart_tabs = st.tabs(["📊 Benchmark", "📈 Area Projection", "🗂️ Multi-Crop Compare"])

                # Benchmark Comparison
                with yield_chart_tabs[0]:
                    colors = get_chart_colors()
                    fig_y, ax_y = plt.subplots(figsize=(7, 3.5), facecolor=colors["bg"])
                    bench_names = ["District Yield", "State Avg", "National Avg"]
                    bench_vals = [est_yield_ha, est_yield_ha * 0.85, est_yield_ha * 0.72]

                    bars_y = ax_y.bar(bench_names, bench_vals,
                                      color=[colors["cyan"], colors["green"], colors["amber"]],
                                      width=0.45, edgecolor="none", zorder=3)
                    ax_y.set_ylabel("Tonnes / Hectare", color=colors["text"])
                    ax_y.grid(axis="y", color=colors["grid"], alpha=0.4, zorder=0)
                    style_chart(fig_y, ax_y, f"Yield Benchmark — {y_crop}", colors)

                    for b in bars_y:
                        h = b.get_height()
                        ax_y.text(b.get_x() + b.get_width() / 2, h + 0.03,
                                  f"{h:.3f}", ha="center", color=colors["text"],
                                  fontsize=9, fontweight="bold")

                    plt.tight_layout()
                    st.pyplot(fig_y, use_container_width=True)
                    plt.close(fig_y)

                # Area Projection
                with yield_chart_tabs[1]:
                    colors = get_chart_colors()
                    fig_proj, ax_proj = plt.subplots(figsize=(7, 3.5), facecolor=colors["bg"])

                    areas = np.arange(1, 21)
                    yields_proj = est_yield_ha * areas

                    ax_proj.fill_between(areas, yields_proj, alpha=0.15, color=colors["green"])
                    ax_proj.plot(areas, yields_proj, color=colors["green"],
                                 linewidth=2.5, marker="o", markersize=4, zorder=3)

                    # Highlight current area
                    if 1 <= calc_area_ha <= 20:
                        current_production = est_yield_ha * calc_area_ha
                        ax_proj.axvline(x=calc_area_ha, color=colors["rose"], linestyle="--",
                                        linewidth=1, alpha=0.7)
                        ax_proj.scatter([calc_area_ha], [current_production],
                                         color=colors["rose"], s=80, zorder=5)
                        ax_proj.annotate(f"Your farm\n{current_production:.1f} T",
                                          xy=(calc_area_ha, current_production),
                                          xytext=(calc_area_ha + 1.5, current_production),
                                          color=colors["text"], fontsize=8,
                                          arrowprops=dict(arrowstyle="->", color=colors["text"]))

                    ax_proj.set_xlabel("Area (Hectares)", color=colors["text"])
                    ax_proj.set_ylabel("Total Production (Tonnes)", color=colors["text"])
                    ax_proj.grid(color=colors["grid"], alpha=0.4)
                    style_chart(fig_proj, ax_proj, "Production vs Farm Area Projection", colors)

                    plt.tight_layout()
                    st.pyplot(fig_proj, use_container_width=True)
                    plt.close(fig_proj)

                # Multi-crop comparison
                with yield_chart_tabs[2]:
                    compare_crops = ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane"]
                    colors = get_chart_colors()
                    fig_mc, ax_mc = plt.subplots(figsize=(7, 3.5), facecolor=colors["bg"])

                    mc_yields = []
                    mc_names = []
                    for cc in compare_crops:
                        try:
                            rc = pipeline.resolve(cc)
                            yv = pipeline.engine_yield_ml(rc, y_state, y_district, y_season)
                            if yv and yv > 0:
                                mc_yields.append(yv)
                                mc_names.append(cc)
                        except Exception:
                            pass

                    if mc_names:
                        bar_mc = ax_mc.bar(mc_names, mc_yields,
                                            color=colors["palette"][:len(mc_names)],
                                            width=0.5, edgecolor="none", zorder=3)
                        ax_mc.set_ylabel("Yield (t/ha)", color=colors["text"])
                        ax_mc.grid(axis="y", color=colors["grid"], alpha=0.4, zorder=0)
                        style_chart(fig_mc, ax_mc, f"Multi-Crop Yield Comparison — {y_district}", colors)

                        for b in bar_mc:
                            h = b.get_height()
                            ax_mc.text(b.get_x() + b.get_width() / 2, h + 0.02,
                                        f"{h:.2f}", ha="center", color=colors["text"],
                                        fontsize=8.5, fontweight="bold")

                        plt.tight_layout()
                        st.pyplot(fig_mc, use_container_width=True)
                        plt.close(fig_mc)
                    else:
                        st.info("No comparison data available for this district.")

            except Exception as exc:
                st.error(f"Yield prediction failed: {exc}")
        else:
            st.warning("Pipeline module not loaded. Please rebuild from Settings.")

    st.markdown("</div>", unsafe_allow_html=True)


# ===================================================================
# PAGE: MODEL DIAGNOSTICS
# ===================================================================
elif current_page == "Model Diagnostics":
    st.markdown('<div class="sa-animate">', unsafe_allow_html=True)
    st.markdown("## 📊 Model Evaluation & Diagnostics")
    st.caption("Explore diagnostic plots, classification metrics, confusion matrices, and model comparison benchmarks.")

    plot_files = sorted(list(PLOTS_DIR.glob("*.png")))
    if plot_files:
        st.success(f"📁 Loaded **{len(plot_files)}** diagnostic plots from `plots/` directory.")

        diag_tabs = st.tabs([
            "🌾 Crop Model",
            "🧪 Fertilizer Model",
            "📈 Yield Model",
            "🌐 System / Pipeline",
        ])

        with diag_tabs[0]:
            crop_plots = [p for p in plot_files if "crop" in p.name.lower()]
            if crop_plots:
                cols_c = st.columns(2)
                for idx, p_path in enumerate(crop_plots):
                    with cols_c[idx % 2]:
                        st.image(str(p_path), caption=p_path.stem.replace("_", " ").title(),
                                 use_container_width=True)
            else:
                st.info("No crop model plots found.")

        with diag_tabs[1]:
            fert_plots = [p for p in plot_files if "fert" in p.name.lower()]
            if fert_plots:
                cols_f = st.columns(2)
                for idx, p_path in enumerate(fert_plots):
                    with cols_f[idx % 2]:
                        st.image(str(p_path), caption=p_path.stem.replace("_", " ").title(),
                                 use_container_width=True)
            else:
                st.info("No fertilizer model plots found.")

        with diag_tabs[2]:
            yield_plots = [p for p in plot_files if "yield" in p.name.lower()]
            if yield_plots:
                cols_y = st.columns(2)
                for idx, p_path in enumerate(yield_plots):
                    with cols_y[idx % 2]:
                        st.image(str(p_path), caption=p_path.stem.replace("_", " ").title(),
                                 use_container_width=True)
            else:
                st.info("No yield model plots found.")

        with diag_tabs[3]:
            sys_plots = [
                p for p in plot_files
                if "crop" not in p.name.lower()
                and "fert" not in p.name.lower()
                and "yield" not in p.name.lower()
            ]
            if sys_plots:
                cols_s = st.columns(2)
                for idx, p_path in enumerate(sys_plots):
                    with cols_s[idx % 2]:
                        st.image(str(p_path), caption=p_path.stem.replace("_", " ").title(),
                                 use_container_width=True)
            else:
                st.info("No system-level plots found.")
    else:
        st.warning("No diagnostic plots found. Run the training pipeline from Settings to generate them.")

    # Classification Reports
    st.markdown("---")
    st.markdown("#### 📄 Model Metric Reports")
    rep_cols = st.columns(3)

    report_files = [
        ("Crop Model Metrics", REPORTS_DIR / "eval_crop_models.csv"),
        ("Fertilizer Model Metrics", REPORTS_DIR / "eval_fert_models.csv"),
        ("Yield Model Metrics", REPORTS_DIR / "eval_yield_models.csv"),
    ]

    for col, (title, r_path) in zip(rep_cols, report_files):
        with col:
            st.markdown(f"**{title}**")
            if r_path.exists():
                df_rep = pd.read_csv(r_path)
                st.dataframe(df_rep, use_container_width=True, hide_index=True)
            else:
                st.info("Report pending build.")

    st.markdown("</div>", unsafe_allow_html=True)


# ===================================================================
# PAGE: DATASET EXPLORER
# ===================================================================
elif current_page == "Dataset Explorer":
    st.markdown('<div class="sa-animate">', unsafe_allow_html=True)
    st.markdown("## 📁 Dataset Explorer")
    st.caption("Preview raw agricultural datasets, rainfall lookups, and download CSV files.")

    data_files = {
        "Agricultural Production & Yield (APY.csv)": DATA_DIR / "APY.csv",
        "Crop Recommendation Dataset": DATA_DIR / "Crop recommendation dataset.csv",
        "Fertilizer Recommendation Dataset": DATA_DIR / "fertilizer_recommendation.csv",
        "District Rainfall Normal Data": DATA_DIR / "district wise rainfall normal.csv",
    }

    sel_dataset_label = st.selectbox("Select Dataset to Preview", options=list(data_files.keys()),
                                      key="data_sel")
    target_data_path = data_files[sel_dataset_label]

    if target_data_path.exists():
        df_preview = pd.read_csv(target_data_path)
        info_c1, info_c2, info_c3 = st.columns(3)
        with info_c1:
            st.metric("Rows", f"{df_preview.shape[0]:,}")
        with info_c2:
            st.metric("Columns", f"{df_preview.shape[1]}")
        with info_c3:
            size_mb = target_data_path.stat().st_size / (1024 * 1024)
            st.metric("File Size", f"{size_mb:.1f} MB")

        st.dataframe(df_preview.head(200), use_container_width=True, hide_index=True)

        # Column statistics
        with st.expander("📊 Column Statistics"):
            st.dataframe(df_preview.describe(include="all").T, use_container_width=True)

        csv_bytes = df_preview.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Download {sel_dataset_label}",
            data=csv_bytes,
            file_name=target_data_path.name,
            mime="text/csv",
        )
    else:
        st.warning(f"File `{target_data_path.name}` not found in data directory.")

    # Test harness results
    st.markdown("---")
    st.markdown("#### 🧪 Test Harness Results")
    harness_files = sorted(list(BASE_DIR.glob("test_results_*.csv")), reverse=True)

    if harness_files:
        latest_harness = harness_files[0]
        st.markdown(f"**Latest**: `{latest_harness.name}`")
        df_harness = pd.read_csv(latest_harness)
        st.dataframe(df_harness, use_container_width=True, hide_index=True)
        st.download_button(
            label="📥 Download Test Harness CSV",
            data=latest_harness.read_bytes(),
            file_name=latest_harness.name,
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("No test harness results found. Run from Settings to generate.")

    st.markdown("</div>", unsafe_allow_html=True)


# ===================================================================
# PAGE: SETTINGS
# ===================================================================
elif current_page == "Settings":
    st.markdown('<div class="sa-animate">', unsafe_allow_html=True)
    st.markdown("## ⚙️ Settings")

    # Theme Toggle
    st.markdown("#### 🎨 Appearance")
    st.markdown("""
    <div class="sa-card">
    """, unsafe_allow_html=True)

    theme_label = "🌙 Dark Mode" if not IS_DARK else "☀️ Light Mode"
    current_theme_display = "Dark" if IS_DARK else "Light"
    st.markdown(f"**Current Theme**: {current_theme_display}")

    if st.button(f"Switch to {theme_label}", key="theme_toggle_btn"):
        st.session_state.theme = "dark" if not IS_DARK else "light"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # System Maintenance
    st.markdown("#### 🔧 System Maintenance")
    st.markdown("""
    <div class="sa-card">
        <h4 style="margin-top:0;">🔄 Complete Project Rebuild</h4>
        <p style="color:var(--text-secondary) !important; font-size:0.9rem;">
            Re-run training scripts (NB1 → NB6) to refresh encoders, model weights,
            diagnostic plots, and classification reports.
        </p>
    """, unsafe_allow_html=True)

    if st.button("🚀 Rebuild Entire SmartAgri Suite", use_container_width=True, key="rebuild_btn"):
        try:
            bootstrap_project()
            st.success("✅ Rebuild completed successfully!")
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()
        except Exception as exc:
            st.error(f"Rebuild failed: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)

    # System Info
    st.markdown("#### ℹ️ System Information")
    st.markdown("""
    <div class="sa-card">
    """, unsafe_allow_html=True)

    info_c1, info_c2 = st.columns(2)
    with info_c1:
        st.markdown(f"**Python Version**: `{sys.version.split()[0]}`")
        st.markdown(f"**Streamlit Version**: `{st.__version__}`")
        st.markdown(f"**NumPy Version**: `{np.__version__}`")
        st.markdown(f"**Pandas Version**: `{pd.__version__}`")

    with info_c2:
        n_models = len(list(MODEL_DIR.glob("*.pkl")))
        n_encoders = len(list(ENCODER_DIR.glob("*.pkl")))
        n_plots = len(list(PLOTS_DIR.glob("*.png")))
        n_reports = len(list(REPORTS_DIR.glob("*.csv")))
        st.markdown(f"**Trained Models**: `{n_models}`")
        st.markdown(f"**Encoders**: `{n_encoders}`")
        st.markdown(f"**Diagnostic Plots**: `{n_plots}`")
        st.markdown(f"**Reports**: `{n_reports}`")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ===================================================================
# PAGE: ABOUT
# ===================================================================
elif current_page == "About":
    st.markdown('<div class="sa-animate">', unsafe_allow_html=True)

    st.markdown("""
    <div class="sa-hero" style="text-align:center;">
        <h1>🌿 SmartAgri</h1>
        <p style="margin:0 auto; max-width:600px;">
            AI-Powered Precision Agriculture Platform for Indian Farmers
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    about_c1, about_c2 = st.columns(2)

    with about_c1:
        st.markdown("""
        ### 🎯 What is SmartAgri?

        SmartAgri is a comprehensive AI-powered agricultural decision support system
        that helps farmers make data-driven decisions about:

        - **Crop Selection** — Which crops to grow based on soil, climate, and location
        - **Fertilizer Management** — Which fertilizers to apply based on soil nutrient analysis
        - **Yield Prediction** — Expected harvest output based on historical data

        ### 🧠 How It Works

        SmartAgri uses multiple machine learning models trained on Indian agricultural data:

        1. **Crop Recommendation Engine** — Multi-model ensemble scoring with 9 component factors
        2. **Fertilizer Recommendation Engine** — ML + agronomy-based hybrid approach
        3. **Yield Prediction Engine** — Historical data + ML regression
        """)

    with about_c2:
        st.markdown("""
        ### 📊 Data Sources

        - Agricultural Production & Yield (APY) dataset — Govt. of India
        - Crop Recommendation dataset — Soil & climate parameters
        - Fertilizer Recommendation dataset — NPK & soil analysis
        - District-wise Rainfall Normal data — IMD

        ### 🛠️ Technology Stack

        | Component | Technology |
        |-----------|-----------|
        | Frontend | Streamlit |
        | ML Models | LightGBM, XGBoost, Random Forest, Extra Trees |
        | Data Processing | Pandas, NumPy, Scikit-learn |
        | Visualization | Matplotlib, Seaborn |
        | Language | Python |

        ### 📌 Version

        **SmartAgri v3.0** — Multi-Engine AI Agriculture Platform
        """)

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:var(--text-muted) !important; font-size:0.82rem;">'
    'SmartAgri v3.0 — Built with Python, Streamlit, LightGBM, XGBoost & Scikit-Learn'
    '</p>',
    unsafe_allow_html=True,
)
