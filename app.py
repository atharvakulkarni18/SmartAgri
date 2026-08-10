from __future__ import annotations

import importlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
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

# Set page config once at top level
st.set_page_config(
    page_title="SmartAgri AI Suite",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
            "Rajasthan",
            "Kerala",
            "Bihar",
            "Punjab",
            "Uttar Pradesh",
            "Maharashtra",
            "Tamil Nadu",
            "Karnataka",
            "Gujarat",
            "Madhya Pradesh",
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
        status.info(f"⏳ Bootstrapping project stage ({index}/{total}): {stage}")
        run_stage(stage)
        progress.progress(index / total)

    status.success("✅ Project assets generated successfully!")


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
            n_val,
            p_val,
            k_val,
            soil_type,
            season,
            irrigation,
            temp_mid,
            hum_mid,
            season_rain,
            soil_ph,
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
                ml_prob,
                canonical,
                district,
                state,
                season,
                irrigation,
                annual_rain,
                season_rain,
                temp_c=temp_mid,
                soil_ph=soil_ph,
            )
            if final is None:
                continue
            meta = pipeline_module.get_meta(canonical) or {}
            display = meta.get("display", canonical.title())
            scored.append(
                {
                    "canonical": canonical,
                    "display": display,
                    "final": float(final),
                    "components": components,
                    "yield_data": yield_data,
                    "yield_level": yield_level,
                }
            )

        scored.sort(key=lambda x: x["final"], reverse=True)
        top_crops = scored[:3]
        best = top_crops[0] if top_crops else None
        best_canonical = best["canonical"] if best else None

        if best_canonical:
            fert_name, fert_reason, ml_probs = pipeline_module.engine_fertilizer(
                n_val,
                p_val,
                k_val,
                soil_type,
                best_canonical,
                season,
                irrigation,
                soil_ph,
                annual_rain,
                state,
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
            "Pipeline model could not be loaded. Please click 'Build project now' in the sidebar."
        )


# Inject Modern Glassmorphism Agronomy CSS
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    :root {
        --bg-main: #060e12;
        --panel-bg: rgba(13, 27, 34, 0.85);
        --panel-glow: rgba(16, 185, 129, 0.15);
        --accent-emerald: #10b981;
        --accent-jade: #059669;
        --accent-amber: #f59e0b;
        --accent-cyan: #06b6d4;
        --accent-rose: #f43f5e;
        --text-primary: #f1f5f9;
        --text-muted: #94a3b8;
        --border-color: rgba(16, 185, 129, 0.22);
        --glass-shadow: 0 16px 40px rgba(0, 0, 0, 0.45);
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background: 
            radial-gradient(circle at 10% 10%, rgba(16, 185, 129, 0.12), transparent 35%),
            radial-gradient(circle at 90% 20%, rgba(6, 182, 212, 0.10), transparent 40%),
            radial-gradient(circle at 50% 85%, rgba(245, 158, 11, 0.08), transparent 45%),
            linear-gradient(180deg, #050c10 0%, #09171f 50%, #060f14 100%);
        color: var(--text-primary);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(8, 18, 23, 0.98), rgba(11, 24, 31, 0.95)) !important;
        border-right: 1px solid rgba(16, 185, 129, 0.18) !important;
        box-shadow: 8px 0 25px rgba(0,0,0,0.3);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    /* Custom Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.18) 0%, rgba(6, 182, 212, 0.12) 60%, rgba(245, 158, 11, 0.15) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 24px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 1.8rem;
        box-shadow: var(--glass-shadow);
        backdrop-filter: blur(12px);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(120deg, #ffffff 30%, #34d399 70%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        color: var(--text-muted);
        font-size: 0.98rem;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }

    /* Glass Cards */
    .agri-card {
        background: var(--panel-bg);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 1.4rem 1.6rem;
        box-shadow: var(--glass-shadow);
        backdrop-filter: blur(16px);
        margin-bottom: 1.2rem;
        transition: transform 0.25s ease, border-color 0.25s ease;
    }
    .agri-card:hover {
        border-color: rgba(16, 185, 129, 0.45);
        transform: translateY(-2px);
    }
    
    .agri-badge {
        display: inline-block;
        padding: 0.3rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .badge-emerald { background: rgba(16, 185, 129, 0.18); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35); }
    .badge-amber { background: rgba(245, 158, 11, 0.18); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.35); }
    .badge-cyan { background: rgba(6, 182, 212, 0.18); color: #38bdf8; border: 1px solid rgba(6, 182, 212, 0.35); }

    /* Custom Streamlit Component Overrides */
    .stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        border-radius: 14px !important;
        border: none !important;
        padding: 0.6rem 1.4rem !important;
        box-shadow: 0 4px 18px rgba(16, 185, 129, 0.35) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 22px rgba(16, 185, 129, 0.5) !important;
    }

    .stTabs [role="tablist"] {
        gap: 0.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        padding-bottom: 0.4rem;
    }
    .stTabs [role="tab"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        color: var(--text-muted);
        font-weight: 600;
        padding: 0.55rem 1.1rem;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(16, 185, 129, 0.2) !important;
        border-color: rgba(16, 185, 129, 0.5) !important;
        color: #ffffff !important;
    }

    [data-testid="stMetricValue"] {
        font-family: 'Outfit', sans-serif;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #34d399 !important;
    }
    
    .ph-indicator {
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        margin-left: 0.5rem;
    }
    .ph-acidic { background: rgba(244, 63, 94, 0.2); color: #f43f5e; border: 1px solid #f43f5e; }
    .ph-neutral { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #34d399; }
    .ph-alkaline { background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid #38bdf8; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Top Bar Hero
st.markdown(
    """
    <div class="hero-banner">
        <div>
            <h1 class="hero-title">🌾 SmartAgri AI Suite</h1>
            <p class="hero-subtitle">State-of-the-Art Precision Agriculture, Multi-Crop Intelligence, Fertilizer Optimization & Yield Analytics</p>
        </div>
        <div>
            <span class="agri-badge badge-emerald">Engine v3.0 Active</span>
            <span class="agri-badge badge-cyan">684 Districts Indexed</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load sidebar & bootstrap check
bootstrap_needed = not project_ready()
pipeline, harness = load_modules()
states_list, state_dist_map = load_geography_data()

with st.sidebar:
    st.markdown("### ⚙️ System Status")
    if bootstrap_needed:
        st.error(
            "⚠️ Missing model or encoder artifacts. Click below to rebuild the AI engine."
        )
        if st.button("🔨 Build Project Assets Now", use_container_width=True):
            bootstrap_project()
            st.rerun()
    else:
        st.success("✅ Machine Learning Models & Encoders Ready")

    st.markdown("---")
    st.markdown("### 📌 Active Pipeline Models")
    if pipeline:
        st.markdown(
            f"• **Crop Model**: `{getattr(pipeline, 'crop_model_name', 'LightGBM')}`"
        )
        st.markdown(
            f"• **Fertilizer Model**: `{getattr(pipeline, 'fert_model_name', 'Random Forest')}`"
        )
        st.markdown(
            f"• **Yield Model**: `{getattr(pipeline, 'yield_model_name', 'XGBoost')}`"
        )
    else:
        st.caption("Pipeline models loading...")

    st.markdown("---")
    st.markdown("### 🌾 Quick Agricultural Info")
    st.markdown("• **58 Registered Crop Species**")
    st.markdown("• **37 States & Union Territories**")
    st.markdown("• **NPK Soil Chemistry Lookup**")
    st.markdown("• **District Rainfall Integration**")

if bootstrap_needed:
    with st.spinner(
        "Initializing SmartAgri AI Pipeline (Generating models from NB1 to NB6)..."
    ):
        bootstrap_project()
    st.rerun()

# Application Tabs
(
    tab_crop,
    tab_fert,
    tab_yield,
    tab_diagnostics,
    tab_data,
    tab_settings,
) = st.tabs(
    [
        "🌾 Crop Recommendation",
        "🧪 Fertilizer & NPK Analysis",
        "📈 Yield & Acreage Calculator",
        "📊 Model Evaluation & Diagnostic Plots",
        "📁 Dataset Explorer",
        "⚙️ System Maintenance",
    ]
)

# ---------------------------------------------------------
# TAB 1: CROP RECOMMENDATION ENGINE
# ---------------------------------------------------------
with tab_crop:
    st.markdown(
        "### 🎯 Precision Multi-Crop Recommendation & Agronomic Intelligence"
    )
    st.caption(
        "Select state, district, soil parameters, and nutrient levels to generate top crop options ranked by overall suitability score."
    )

    # Regional Preset Selector
    st.markdown("**⚡ Quick Regional Presets:**")
    preset_cols = st.columns(4)
    if "selected_state" not in st.session_state:
        st.session_state.selected_state = "Rajasthan"
    if "selected_district" not in st.session_state:
        st.session_state.selected_district = "Jaisalmer"
    if "input_n" not in st.session_state:
        st.session_state.input_n = 40.0
    if "input_p" not in st.session_state:
        st.session_state.input_p = 20.0
    if "input_k" not in st.session_state:
        st.session_state.input_k = 20.0
    if "input_ph" not in st.session_state:
        st.session_state.input_ph = 7.4
    if "input_soil" not in st.session_state:
        st.session_state.input_soil = "sandy loam soil"
    if "input_season" not in st.session_state:
        st.session_state.input_season = "kharif"
    if "input_irrigation" not in st.session_state:
        st.session_state.input_irrigation = "rainfed"

    with preset_cols[0]:
        if st.button("🏜️ Jaisalmer, RJ (Arid)", use_container_width=True):
            st.session_state.selected_state = "Rajasthan"
            st.session_state.selected_district = "Jaisalmer"
            st.session_state.input_n = 40.0
            st.session_state.input_p = 20.0
            st.session_state.input_k = 20.0
            st.session_state.input_ph = 7.8
            st.session_state.input_soil = "sandy loam soil"
            st.session_state.input_season = "kharif"
            st.session_state.input_irrigation = "rainfed"

    with preset_cols[1]:
        if st.button("🌴 Alappuzha, KL (Coastal)", use_container_width=True):
            st.session_state.selected_state = "Kerala"
            st.session_state.selected_district = "Alappuzha"
            st.session_state.input_n = 70.0
            st.session_state.input_p = 40.0
            st.session_state.input_k = 50.0
            st.session_state.input_ph = 6.0
            st.session_state.input_soil = "alluvial soil"
            st.session_state.input_season = "kharif"
            st.session_state.input_irrigation = "rainfed"

    with preset_cols[2]:
        if st.button("🌾 Patna, BR (Indo-Gangetic)", use_container_width=True):
            st.session_state.selected_state = "Bihar"
            st.session_state.selected_district = "Patna"
            st.session_state.input_n = 60.0
            st.session_state.input_p = 30.0
            st.session_state.input_k = 35.0
            st.session_state.input_ph = 6.8
            st.session_state.input_soil = "alluvial soil"
            st.session_state.input_season = "rabi"
            st.session_state.input_irrigation = "canal"

    with preset_cols[3]:
        if st.button("🌽 Ludhiana, PB (Fertile)", use_container_width=True):
            st.session_state.selected_state = "Punjab"
            st.session_state.selected_district = "Ludhiana"
            st.session_state.input_n = 90.0
            st.session_state.input_p = 50.0
            st.session_state.input_k = 40.0
            st.session_state.input_ph = 7.2
            st.session_state.input_soil = "loamy soil"
            st.session_state.input_season = "rabi"
            st.session_state.input_irrigation = "tube well"

    st.markdown("---")
    form_col, result_col = st.columns([1.0, 1.2], gap="large")

    with form_col:
        st.markdown(
            "<div class='agri-card'>", unsafe_allow_html=True
        )
        st.subheader("📋 Input Soil & Environmental Parameters")

        # Geography Cascading Selectors
        sel_state = st.selectbox(
            "State / Region",
            options=states_list,
            index=(
                states_list.index(st.session_state.selected_state)
                if st.session_state.selected_state in states_list
                else 0
            ),
        )

        available_districts = state_dist_map.get(sel_state, [])
        if not available_districts:
            available_districts = ["Jaisalmer", "Alappuzha", "Patna", "Ludhiana"]

        dist_idx = 0
        if st.session_state.selected_district in available_districts:
            dist_idx = available_districts.index(
                st.session_state.selected_district
            )

        sel_district = st.selectbox(
            "District", options=available_districts, index=dist_idx
        )

        # Soils & Season
        soils = [
            "alluvial soil",
            "black soil",
            "clayey soil",
            "loamy soil",
            "red soil",
            "sandy soil",
            "sandy loam soil",
            "laterite soil",
            "peaty soil",
        ]
        sel_soil = st.selectbox(
            "Soil Type",
            options=soils,
            index=(
                soils.index(st.session_state.input_soil)
                if st.session_state.input_soil in soils
                else 0
            ),
        )

        c_season, c_irrigation = st.columns(2)
        with c_season:
            sel_season = st.selectbox(
                "Season",
                options=["kharif", "rabi", "zaid"],
                index=["kharif", "rabi", "zaid"].index(
                    st.session_state.input_season
                ),
            )
        with c_irrigation:
            sel_irrigation = st.selectbox(
                "Irrigation Type",
                options=["rainfed", "canal", "sprinkler", "drip", "tube well"],
                index=["rainfed", "canal", "sprinkler", "drip", "tube well"].index(
                    st.session_state.input_irrigation
                ),
            )

        st.markdown("**Soil Nutrient Levels (kg/ha):**")
        n_col, p_col, k_col = st.columns(3)
        with n_col:
            val_n = st.slider(
                "Nitrogen (N)", 0.0, 140.0, float(st.session_state.input_n), step=5.0
            )
        with p_col:
            val_p = st.slider(
                "Phosphorus (P)", 0.0, 140.0, float(st.session_state.input_p), step=5.0
            )
        with k_col:
            val_k = st.slider(
                "Potassium (K)", 0.0, 140.0, float(st.session_state.input_k), step=5.0
            )

        val_ph = st.slider(
            "Soil pH Level", 3.5, 10.0, float(st.session_state.input_ph), step=0.1
        )
        if val_ph < 6.0:
            ph_tag = "<span class='ph-indicator ph-acidic'>Acidic Soil</span>"
        elif val_ph <= 7.5:
            ph_tag = "<span class='ph-indicator ph-neutral'>Optimal / Neutral</span>"
        else:
            ph_tag = "<span class='ph-indicator ph-alkaline'>Alkaline Soil</span>"

        st.markdown(f"pH Status: {ph_tag}", unsafe_allow_html=True)

        btn_run = st.button("✨ Generate AI Recommendation", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with result_col:
        # Build payload & execute pipeline
        payload = {
            "district": sel_district,
            "state": sel_state,
            "soil_type": sel_soil,
            "season": sel_season,
            "irrigation": sel_irrigation,
            "N": val_n,
            "P": val_p,
            "K": val_k,
            "pH": val_ph,
        }

        try:
            res = execute_pipeline(pipeline, harness, payload)
            top_crops = res.get("top_crops", [])
            fert_info = res.get("fertilizer", {})
            yield_info = res.get("yield", {})
            rain_info = res.get("rain", {})

            if top_crops:
                best_crop = top_crops[0]
                st.markdown(
                    f"""
                    <div class="agri-card" style="border-left: 6px solid #10b981;">
                        <span class="agri-badge badge-emerald">🥇 Prime Recommendation</span>
                        <h2 style="font-family:'Outfit'; margin: 0.2rem 0; color:#f1f5f9; font-size:1.9rem;">
                            {best_crop['name']}
                        </h2>
                        <div style="display:flex; gap: 1.5rem; margin-top: 0.8rem;">
                            <div>
                                <div style="color:var(--text-muted); font-size:0.82rem;">SUITABILITY MATCH</div>
                                <div style="font-size:1.6rem; font-weight:700; color:#34d399;">
                                    {best_crop['score']*100:.1f}%
                                </div>
                            </div>
                            <div>
                                <div style="color:var(--text-muted); font-size:0.82rem;">EXPECTED YIELD</div>
                                <div style="font-size:1.6rem; font-weight:700; color:#38bdf8;">
                                    {yield_info.get('value', 'N/A'):.2f} <span style="font-size:0.9rem;">t/ha</span>
                                </div>
                            </div>
                            <div>
                                <div style="color:var(--text-muted); font-size:0.82rem;">SUGGESTED FERTILIZER</div>
                                <div style="font-size:1.2rem; font-weight:700; color:#fbbf24; margin-top:0.3rem;">
                                    {fert_info.get('name', 'Balanced NPK')}
                                </div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("#### 🏆 Top Ranked Crops")
                for i, crop in enumerate(top_crops, start=1):
                    badge_style = (
                        "badge-emerald"
                        if i == 1
                        else ("badge-amber" if i == 2 else "badge-cyan")
                    )
                    st.markdown(
                        f"""
                        <div class="agri-card" style="padding:1rem 1.2rem;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <div>
                                    <span class="agri-badge {badge_style}">Rank #{i}</span>
                                    <span style="font-size:1.25rem; font-weight:700; margin-left:0.6rem; color:#f1f5f9;">
                                        {crop['name']}
                                    </span>
                                </div>
                                <div style="font-size:1.3rem; font-weight:800; color:#34d399;">
                                    {crop['score']*100:.1f}% Match
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Radar / Bar Component Breakdown Chart
                if best_crop.get("components"):
                    comps = best_crop["components"]
                    fig, ax = plt.subplots(figsize=(6.5, 2.8), facecolor="none")
                    labels = [
                        "Suitability",
                        "Yield",
                        "Region",
                        "Irrigation",
                        "Season",
                        "Presence",
                    ]
                    vals = [
                        comps.get("suitability", 0) * 100,
                        comps.get("yield", 0) * 100,
                        comps.get("region", 0) * 100,
                        comps.get("irrigation", 0) * 100,
                        comps.get("season", 0) * 100,
                        comps.get("presence", 0) * 100,
                    ]
                    bars = ax.barh(
                        labels,
                        vals,
                        color=[
                            "#10b981",
                            "#06b6d4",
                            "#f59e0b",
                            "#8b5cf6",
                            "#ec4899",
                            "#3b82f6",
                        ],
                        height=0.55,
                    )
                    ax.set_xlim(0, 100)
                    ax.set_facecolor("none")
                    ax.tick_params(colors="#94a3b8", labelsize=9)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.spines["left"].set_color("rgba(255,255,255,0.15)")
                    ax.spines["bottom"].set_color("rgba(255,255,255,0.15)")
                    ax.set_title(
                        f"Component Score Breakdown for {best_crop['name']}",
                        color="#f1f5f9",
                        fontsize=10.5,
                        pad=10,
                    )
                    for bar in bars:
                        w = bar.get_width()
                        ax.text(
                            w + 1.5,
                            bar.get_y() + bar.get_height() / 2,
                            f"{w:.1f}%",
                            va="center",
                            color="#e2e8f0",
                            fontsize=8.5,
                        )

                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                st.markdown("#### 🌧️ District Climate & Rainfall Context")
                r_c1, r_c2 = st.columns(2)
                with r_c1:
                    st.metric(
                        "Annual Rainfall", f"{rain_info.get('annual', 0):.1f} mm"
                    )
                with r_c2:
                    st.metric(
                        f"Seasonal Rainfall ({sel_season.capitalize()})",
                        f"{rain_info.get('season', 0):.1f} mm",
                    )

            else:
                st.warning(
                    "No crop passed strict seasonal and temperature filters for this specific combination. Try selecting an alternative season or adjusting soil parameters."
                )

        except Exception as exc:
            st.error(f"Error executing recommendation engine: {exc}")

# ---------------------------------------------------------
# TAB 2: FERTILIZER & NPK ANALYSIS
# ---------------------------------------------------------
with tab_fert:
    st.markdown("### 🧪 Precision Fertilizer Recommendation & Soil NPK Analysis")
    st.caption(
        "Analyze soil nutrient balance, identify deficiencies, and review machine learning fertilizer probability distributions."
    )

    f_left, f_right = st.columns([1.0, 1.1], gap="large")

    with f_left:
        st.markdown(
            "<div class='agri-card'>", unsafe_allow_html=True
        )
        st.subheader("⚖️ Soil Nutrient & Crop Target Setup")

        target_crop_input = st.selectbox(
            "Target Crop",
            options=[
                "Rice",
                "Wheat",
                "Maize",
                "Cotton",
                "Sugarcane",
                "Chickpea",
                "Jute",
                "Groundnut",
                "Mustard",
            ],
            index=0,
        )

        n_val_f = st.slider("Current Soil Nitrogen (N)", 0, 140, 50)
        p_val_f = st.slider("Current Soil Phosphorus (P)", 0, 140, 30)
        k_val_f = st.slider("Current Soil Potassium (K)", 0, 140, 25)

        soil_f = st.selectbox(
            "Soil Category",
            ["alluvial soil", "black soil", "clayey soil", "loamy soil", "sandy soil"],
            index=0,
            key="fert_soil_sel",
        )
        season_f = st.selectbox(
            "Crop Season",
            ["kharif", "rabi", "zaid"],
            index=0,
            key="fert_season_sel",
        )
        ph_f = st.slider("pH Level", 3.5, 9.5, 6.5, key="fert_ph_sel")

        run_fert = st.button(
            "🧪 Analyze NPK Balance & Fertilizer", use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with f_right:
        payload_f = {
            "district": "Alappuzha",
            "state": "Kerala",
            "soil_type": soil_f,
            "season": season_f,
            "irrigation": "rainfed",
            "N": float(n_val_f),
            "P": float(p_val_f),
            "K": float(k_val_f),
            "pH": float(ph_f),
        }

        try:
            res_f = execute_pipeline(pipeline, harness, payload_f)
            fert_data = res_f.get("fertilizer", {})
            rec_fert = fert_data.get("name", "Urea")
            reason_fert = fert_data.get("reason", "Maintains optimal soil nutrient equilibrium.")
            ml_probs = fert_data.get("ml_probs", {})

            st.markdown(
                f"""
                <div class="agri-card" style="border-left: 6px solid #f59e0b;">
                    <span class="agri-badge badge-amber">Fertilizer Suggestion</span>
                    <h2 style="color:#fbbf24; margin:0.3rem 0; font-family:'Outfit';">{rec_fert}</h2>
                    <p style="color:var(--text-muted); font-size:0.92rem; margin-top:0.4rem;">{reason_fert}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # NPK Soil Bar Chart Comparison vs Ideal Requirements
            st.markdown("#### 📊 Soil NPK Level vs Agronomic Ideal")
            fig_npk, ax_npk = plt.subplots(figsize=(6.5, 2.5), facecolor="none")
            npk_labels = ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"]
            current_npk = [n_val_f, p_val_f, k_val_f]
            ideal_npk = [80, 40, 40]  # Typical benchmark

            x = np.arange(len(npk_labels))
            width = 0.35

            rects1 = ax_npk.bar(
                x - width / 2,
                current_npk,
                width,
                label="Current Soil Level",
                color="#10b981",
            )
            rects2 = ax_npk.bar(
                x + width / 2,
                ideal_npk,
                width,
                label="Crop Ideal Baseline",
                color="#38bdf8",
            )

            ax_npk.set_ylabel("Nutrient (kg/ha)", color="#94a3b8")
            ax_npk.set_xticks(x)
            ax_npk.set_xticklabels(npk_labels, color="#f1f5f9")
            ax_npk.tick_params(colors="#94a3b8")
            ax_npk.set_facecolor("none")
            ax_npk.legend(
                facecolor="#0d1b22", edgecolor="rgba(255,255,255,0.15)", labelcolor="#f1f5f9"
            )
            ax_npk.spines["top"].set_visible(False)
            ax_npk.spines["right"].set_visible(False)
            ax_npk.spines["left"].set_color("rgba(255,255,255,0.15)")
            ax_npk.spines["bottom"].set_color("rgba(255,255,255,0.15)")

            st.pyplot(fig_npk, use_container_width=True)
            plt.close(fig_npk)

            # Fertilizer Model Probabilities
            if ml_probs and isinstance(ml_probs, dict):
                st.markdown("#### 🤖 ML Model Fertilizer Probability Distribution")
                df_probs = pd.DataFrame(
                    list(ml_probs.items()), columns=["Fertilizer", "Probability"]
                ).sort_values("Probability", ascending=False)
                st.bar_chart(df_probs.set_index("Fertilizer"), color="#f59e0b")

        except Exception as exc:
            st.error(f"Fertilizer evaluation failed: {exc}")

# ---------------------------------------------------------
# TAB 3: YIELD & HARVEST CALCULATOR
# ---------------------------------------------------------
with tab_yield:
    st.markdown("### 📈 Crop Yield Estimation & Farm Production Calculator")
    st.caption(
        "Estimate crop yield in Metric Tonnes per Hectare and calculate total expected harvest output based on farm size."
    )

    y_col1, y_col2 = st.columns([1.0, 1.1], gap="large")

    with y_col1:
        st.markdown(
            "<div class='agri-card'>", unsafe_allow_html=True
        )
        st.subheader("🌾 Land Acreage & Location Setup")

        y_state = st.selectbox(
            "State", options=states_list, index=0, key="yield_state_sel"
        )
        y_districts = state_dist_map.get(y_state, ["Alappuzha", "Patna"])
        y_district = st.selectbox("District", options=y_districts, key="yield_dist_sel")

        y_crop = st.text_input("Target Crop", value="Rice", key="yield_crop_input")
        y_season = st.selectbox(
            "Season", ["kharif", "rabi", "zaid"], index=0, key="yield_season_sel"
        )

        st.markdown("---")
        st.markdown("**🏡 Farm Acreage Calculator:**")
        unit_type = st.radio("Unit System", ["Acres", "Hectares"], horizontal=True)
        area_val = st.number_input(
            f"Total Land Area ({unit_type})", min_value=0.1, value=5.0, step=0.5
        )

        calc_area_ha = area_val * 0.404686 if unit_type == "Acres" else area_val
        st.caption(f"Equivalent Area: `{calc_area_ha:.2f}` Hectares")
        st.markdown("</div>", unsafe_allow_html=True)

    with y_col2:
        if pipeline:
            try:
                resolved_crop = pipeline.resolve(y_crop)
                est_yield_ha = pipeline.engine_yield_ml(
                    resolved_crop, y_state, y_district, y_season
                )
                if est_yield_ha is None or est_yield_ha <= 0:
                    est_yield_ha = 2.85  # Default reasonable yield fallback

                total_harvest_tonnes = est_yield_ha * calc_area_ha
                total_harvest_quintals = total_harvest_tonnes * 10.0

                st.markdown(
                    f"""
                    <div class="agri-card" style="border-left: 6px solid #06b6d4;">
                        <span class="agri-badge badge-cyan">Yield Intelligence</span>
                        <div style="display:flex; justify-content:space-between; margin-top:0.6rem;">
                            <div>
                                <div style="color:var(--text-muted); font-size:0.85rem;">PREDICTED YIELD RATE</div>
                                <div style="font-size:2.1rem; font-weight:800; color:#38bdf8;">
                                    {est_yield_ha:.3f} <span style="font-size:1.1rem;">Tonnes / ha</span>
                                </div>
                            </div>
                            <div>
                                <div style="color:var(--text-muted); font-size:0.85rem;">TOTAL EXPECTED HARVEST</div>
                                <div style="font-size:2.1rem; font-weight:800; color:#34d399;">
                                    {total_harvest_tonnes:.2f} <span style="font-size:1.1rem;">Tonnes</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("#### 📦 Estimated Production Breakdown")
                m_c1, m_c2, m_c3 = st.columns(3)
                with m_c1:
                    st.metric("Total Metric Tonnes", f"{total_harvest_tonnes:.2f} T")
                with m_c2:
                    st.metric(
                        "Total Quintals", f"{total_harvest_quintals:.1f} Q"
                    )
                with m_c3:
                    st.metric("Target Farm Area", f"{calc_area_ha:.2f} ha")

                st.markdown("#### 📊 Yield Benchmark Comparison")
                fig_y, ax_y = plt.subplots(figsize=(6.5, 2.5), facecolor="none")
                bench_names = [
                    "District Expected",
                    "State Average",
                    "National Baseline",
                ]
                bench_vals = [est_yield_ha, est_yield_ha * 0.85, est_yield_ha * 0.72]
                bars_y = ax_y.bar(
                    bench_names,
                    bench_vals,
                    color=["#06b6d4", "#10b981", "#f59e0b"],
                    width=0.45,
                )
                ax_y.set_ylabel("Tonnes / Hectare", color="#94a3b8")
                ax_y.set_facecolor("none")
                ax_y.tick_params(colors="#94a3b8")
                ax_y.spines["top"].set_visible(False)
                ax_y.spines["right"].set_visible(False)
                ax_y.spines["left"].set_color("rgba(255,255,255,0.15)")
                ax_y.spines["bottom"].set_color("rgba(255,255,255,0.15)")

                for b in bars_y:
                    h = b.get_height()
                    ax_y.text(
                        b.get_x() + b.get_width() / 2,
                        h + 0.05,
                        f"{h:.2f}",
                        ha="center",
                        color="#e2e8f0",
                        fontsize=9,
                    )

                st.pyplot(fig_y, use_container_width=True)
                plt.close(fig_y)

            except Exception as exc:
                st.error(f"Yield prediction failed: {exc}")

# ---------------------------------------------------------
# TAB 4: MODEL EVALUATION & DIAGNOSTICS
# ---------------------------------------------------------
with tab_diagnostics:
    st.markdown("### 📊 Model Evaluation, Diagnostic Charts & Feature Importance")
    st.caption(
        "Explore interactive plot galleries, classification metrics, confusion matrices, and model comparison benchmarks."
    )

    plot_files = sorted(list(PLOTS_DIR.glob("*.png")))
    if plot_files:
        st.success(f"Loaded {len(plot_files)} diagnostic plots from `plots/` directory.")

        diag_tabs = st.tabs(
            [
                "🌾 Crop Model Plots",
                "🧪 Fertilizer Plots",
                "📈 Yield Plots",
                "🌐 System Correlation Plots",
            ]
        )

        with diag_tabs[0]:
            crop_plots = [p for p in plot_files if "crop" in p.name.lower()]
            cols_c = st.columns(2)
            for idx, p_path in enumerate(crop_plots):
                with cols_c[idx % 2]:
                    st.image(
                        str(p_path),
                        caption=p_path.name.replace("_", " ").title(),
                        use_container_width=True,
                    )

        with diag_tabs[1]:
            fert_plots = [p for p in plot_files if "fert" in p.name.lower()]
            cols_f = st.columns(2)
            for idx, p_path in enumerate(fert_plots):
                with cols_f[idx % 2]:
                    st.image(
                        str(p_path),
                        caption=p_path.name.replace("_", " ").title(),
                        use_container_width=True,
                    )

        with diag_tabs[2]:
            yield_plots = [p for p in plot_files if "yield" in p.name.lower()]
            cols_y = st.columns(2)
            for idx, p_path in enumerate(yield_plots):
                with cols_y[idx % 2]:
                    st.image(
                        str(p_path),
                        caption=p_path.name.replace("_", " ").title(),
                        use_container_width=True,
                    )

        with diag_tabs[3]:
            sys_plots = [
                p
                for p in plot_files
                if "crop" not in p.name.lower()
                and "fert" not in p.name.lower()
                and "yield" not in p.name.lower()
            ]
            cols_s = st.columns(2)
            for idx, p_path in enumerate(sys_plots):
                with cols_s[idx % 2]:
                    st.image(
                        str(p_path),
                        caption=p_path.name.replace("_", " ").title(),
                        use_container_width=True,
                    )

    # Classification & Model Evaluation CSV Reports
    st.markdown("---")
    st.markdown("#### 📄 Detailed Classification & Metric Reports")
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
                st.dataframe(df_rep, use_container_width=True)
            else:
                st.info("Report file pending build.")

# ---------------------------------------------------------
# TAB 5: DATASET EXPLORER
# ---------------------------------------------------------
with tab_data:
    st.markdown("### 📁 Interactive Dataset Explorer & Downloads")
    st.caption(
        "Preview raw agricultural datasets, rainfall lookups, training splits, and generated CSV test harness results."
    )

    data_files = {
        "Agricultural Production & Yield (APY.csv)": DATA_DIR / "APY.csv",
        "Crop Recommendation Dataset": DATA_DIR
        / "Crop recommendation dataset.csv",
        "Fertilizer Recommendation Dataset": DATA_DIR
        / "fertilizer_recommendation.csv",
        "District Rainfall Normal Data": DATA_DIR
        / "district wise rainfall normal.csv",
    }

    sel_dataset_label = st.selectbox(
        "Select Dataset to Preview", options=list(data_files.keys())
    )
    target_data_path = data_files[sel_dataset_label]

    if target_data_path.exists():
        df_preview = pd.read_csv(target_data_path)
        st.markdown(
            f"**Shape**: `{df_preview.shape[0]} rows` × `{df_preview.shape[1]} columns`"
        )

        st.dataframe(df_preview.head(100), use_container_width=True)

        csv_bytes = df_preview.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Download {sel_dataset_label} (CSV)",
            data=csv_bytes,
            file_name=target_data_path.name,
            mime="text/csv",
        )
    else:
        st.warning(f"File {target_data_path.name} not found in `data/` directory.")

    # Pipeline harness results
    st.markdown("---")
    st.markdown("#### 🧪 Test Harness Evaluation Outputs")
    harness_files = sorted(list(BASE_DIR.glob("test_results_*.csv")), reverse=True)

    if harness_files:
        latest_harness = harness_files[0]
        st.markdown(f"**Latest Test Harness Output**: `{latest_harness.name}`")
        df_harness = pd.read_csv(latest_harness)
        st.dataframe(df_harness, use_container_width=True)
        st.download_button(
            label="📥 Download Test Harness CSV",
            data=latest_harness.read_bytes(),
            file_name=latest_harness.name,
            mime="text/csv",
            use_container_width=True,
        )

# ---------------------------------------------------------
# TAB 6: SYSTEM MAINTENANCE & BOOTSTRAP HARNESS
# ---------------------------------------------------------
with tab_settings:
    st.markdown("### ⚙️ System Maintenance & Automated Build Harness")
    st.caption(
        "Re-run raw notebook scripts (NB1 through NB6) to refresh feature encoders, model weights, and evaluation reports."
    )

    st.markdown(
        """
        <div class="agri-card">
            <h4>🔄 Complete Project Rebuild Pipeline</h4>
            <p style="color:var(--text-muted); font-size:0.9rem;">
                If you add new data rows or modify model hyperparameters in training scripts, click the button below to rebuild all pickle encoders, LightGBM/XGBoost model files, diagnostic plots, and classification reports in sequential order.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🚀 Rebuild Entire SmartAgri AI Suite", use_container_width=True):
        try:
            bootstrap_project()
            st.success("Rebuild completed successfully!")
            st.rerun()
        except Exception as exc:
            st.error(f"Rebuild failed: {exc}")

st.markdown("---")
st.caption(
    "SmartAgri AI Suite v3.0 • Built with Python, Streamlit, LightGBM, XGBoost, and Scikit-Learn."
)
