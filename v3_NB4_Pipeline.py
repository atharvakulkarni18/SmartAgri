import pickle, warnings, numpy as np, pandas as pd
from config import DATA_DIR, ENCODER_DIR, MODEL_DIR, ensure_dirs, resolve_data_file

warnings.filterwarnings("ignore")
ensure_dirs()


def L(name):
    with open(name, "rb") as f:
        return pickle.load(f)


CROP_REGISTRY = L(ENCODER_DIR / "CROP_REGISTRY.pkl")
ALIAS_MAP = L(ENCODER_DIR / "ALIAS_MAP.pkl")
DISTRICT_PROFILES = L(ENCODER_DIR / "DISTRICT_PROFILES.pkl")
STATE_PROFILES = L(ENCODER_DIR / "STATE_PROFILES.pkl")
NATIONAL_PROFILES = L(ENCODER_DIR / "NATIONAL_PROFILES.pkl")
DISTRICT_CROP_SET = L(ENCODER_DIR / "DISTRICT_CROP_SET.pkl")
FAMILY_YIELD_MAX = L(ENCODER_DIR / "FAMILY_YIELD_MAX.pkl")
FAMILY_YIELD_MIN = L(ENCODER_DIR / "FAMILY_YIELD_MIN.pkl")
rain_lookup = L(ENCODER_DIR / "rain_lookup_df.pkl")
DISTRICT_TEMP_OVERRIDE = L(ENCODER_DIR / "DISTRICT_TEMP_OVERRIDE.pkl")
SCORE_WEIGHTS = L(ENCODER_DIR / "SCORE_WEIGHTS.pkl")


crop_cat_enc = L(ENCODER_DIR / "crop_cat_enc.pkl")
crop_target_le = L(ENCODER_DIR / "crop_target_le.pkl")
crop_scaler = L(ENCODER_DIR / "crop_scaler.pkl")
CROP_FEATURES = L(ENCODER_DIR / "CROP_FEATURES.pkl")
fert_cat_enc = L(ENCODER_DIR / "fert_cat_enc.pkl")
fert_target_le = L(ENCODER_DIR / "fert_target_le.pkl")
fert_scaler = L(ENCODER_DIR / "fert_scaler.pkl")
yield_cat_enc = L(ENCODER_DIR / "yield_cat_enc.pkl")


STATE_NORM_MAP = L(ENCODER_DIR / "STATE_NORM_MAP.pkl")
SEASON_NORM_MAP = L(ENCODER_DIR / "SEASON_NORM_MAP.pkl")
IRRIGATION_NORM_MAP = L(ENCODER_DIR / "IRRIGATION_NORM_MAP.pkl")
SOIL_NORM_MAP = L(ENCODER_DIR / "SOIL_NORM_MAP.pkl")
REGION_CROP_MAP = L(ENCODER_DIR / "REGION_CROP_MAP.pkl")


with open(MODEL_DIR / "best_crop_model.txt") as f:
    crop_model_name = f.read().strip()
with open(MODEL_DIR / "best_fert_model.txt") as f:
    fert_model_name = f.read().strip()
with open(MODEL_DIR / "best_yield_model.txt") as f:
    yield_model_name = f.read().strip()

CROP_MDL = {
    "Random Forest": "crop_rf",
    "XGBoost": "crop_xgb",
    "LightGBM": "crop_lgbm",
    "Extra Trees": "crop_et",
    "Gradient Boosting": "crop_gb",
    "SVM (RBF)": "crop_svm",
    "Voting Ensemble": "crop_voting",
    "Stacking Ensemble": "crop_stack",
}
FERT_MDL = {
    "Random Forest": "fert_rf",
    "XGBoost": "fert_xgb",
    "LightGBM": "fert_lgbm",
    "Extra Trees": "fert_et",
    "SVM (RBF)": "fert_svm",
    "Voting Ensemble": "fert_vote",
}
YIELD_MDL = {
    "Random Forest": "yield_rf",
    "XGBoost": "yield_xgb",
    "LightGBM": "yield_lgbm",
    "Extra Trees": "yield_et",
}

crop_model = L(MODEL_DIR / f"{CROP_MDL[crop_model_name]}.pkl")
fert_model = L(MODEL_DIR / f'{FERT_MDL.get(fert_model_name, "fert_rf")}.pkl')
yield_model = L(MODEL_DIR / f'{YIELD_MDL.get(yield_model_name, "yield_xgb")}.pkl')

print(f"[OK] Crop  model : {crop_model_name}")
print(f"[OK] Fert  model : {fert_model_name}")
print(f"[OK] Yield model : {yield_model_name}")
print(f"[OK] District profiles: {len(DISTRICT_PROFILES)} districts")


W_SUITABILITY = 0.35
W_YIELD = 0.15
W_REGION = 0.10
W_IRRIGATION = 0.10
W_PRESENCE = 0.10
W_SEASON = 0.20
print(
    f"  Weights: suit={W_SUITABILITY} yield={W_YIELD} region={W_REGION} "
    f"irr={W_IRRIGATION} season={W_SEASON} pres={W_PRESENCE}"
)


def safe_enc(le, val, default=0):
    try:
        return int(le.transform([str(val)])[0])
    except:
        return default


def resolve(name):
    return ALIAS_MAP.get(name.lower().strip(), name.lower().strip())


def get_meta(name):
    return CROP_REGISTRY.get(resolve(name))


def normalize_state(state):
    state_clean = state.strip().title()
    return STATE_NORM_MAP.get(state_clean, state_clean)


def normalize_district(district):
    return district.strip().upper()


def normalize_season(season):
    season_clean = season.strip().lower()
    return SEASON_NORM_MAP.get(season_clean, season_clean)


def normalize_irrigation(irrigation):
    irr = irrigation.strip().lower()
    return IRRIGATION_NORM_MAP.get(irr, irr)


def normalize_soil(soil_type):
    soil = soil_type.strip().lower()
    return SOIL_NORM_MAP.get(soil, soil)


def normalize_input(input_dict):
    return {
        "district": normalize_district(input_dict["district"]),
        "state": normalize_state(input_dict["state"]),
        "soil_type": normalize_soil(input_dict["soil_type"]),
        "season": normalize_season(input_dict["season"]),
        "irrigation": normalize_irrigation(input_dict["irrigation"]),
        "N": float(input_dict["N"]),
        "P": float(input_dict["P"]),
        "K": float(input_dict["K"]),
        "pH": float(input_dict["pH"]),
    }


def lookup_rainfall(district, state=None):
    dist_u = normalize_district(district)
    q = rain_lookup["DISTRICT_CLEAN"] == dist_u
    if state:
        state_norm = normalize_state(state)
        rows = rain_lookup[
            q & (rain_lookup["STATE_NORM"].str.lower() == state_norm.lower())
        ]
    else:
        rows = rain_lookup[q]
    if rows.empty:
        rows = rain_lookup[
            rain_lookup["DISTRICT_CLEAN"].str.contains(dist_u, regex=False)
        ]
    if rows.empty:
        return None
    r = rows.iloc[0]
    return {
        "district": r["DISTRICT_CLEAN"],
        "state": r["STATE_NORM"],
        "annual": float(r["ANNUAL"]),
        "jun_sep": float(r["Jun-Sep"]),
        "oct_dec": float(r["Oct-Dec"]),
        "mar_may": float(r["Mar-May"]),
        "jan_feb": float(r["Jan-Feb"]),
    }


def drought_filter(canonical, annual_rain, season_rain):
    """Return False if rainfall is too low for water-hungry crops.
    Uses water_mm range from CROP_REGISTRY (replaces old water_need bucket)."""
    meta = CROP_REGISTRY.get(canonical, {})
    water_mm = meta.get("water_mm")
    if water_mm:
        min_need = water_mm[0]

        if season_rain < (min_need * 0.3) and annual_rain < (min_need * 0.5):
            return False

    elif season_rain < 200 or annual_rain < 400:
        return False
    return True


def get_region_crops(state):
    crops = REGION_CROP_MAP.get(state.lower().strip(), [])
    return {resolve(c) for c in crops}


def get_yield_top_crops(district, top_n=5):
    dist_u = normalize_district(district)
    if dist_u not in DISTRICT_PROFILES:
        return set()
    scores = []
    for apy_crop, seasons in DISTRICT_PROFILES[dist_u].items():
        med_vals = [s.get("med", 0) for s in seasons.values() if isinstance(s, dict)]
        if med_vals:
            scores.append((apy_crop, float(np.median(med_vals))))
    scores.sort(key=lambda x: x[1], reverse=True)
    return {resolve(crop) for crop, _ in scores[:top_n]}


def get_freq_top_crops(district, top_n=5):
    dist_u = normalize_district(district)
    if dist_u not in DISTRICT_PROFILES:
        return set()
    freq = []
    for apy_crop, seasons in DISTRICT_PROFILES[dist_u].items():
        n_years = sum(
            int(v.get("n", 0)) for v in seasons.values() if isinstance(v, dict)
        )
        if n_years:
            freq.append((apy_crop, n_years))
    freq.sort(key=lambda x: x[1], reverse=True)
    return {resolve(crop) for crop, _ in freq[:top_n]}


def engine_presence_score(canonical, district):
    """Presence score based on how often a crop appears in district history."""
    meta = CROP_REGISTRY.get(canonical, {})
    apy_name = meta.get("apy_name")
    dist_u = normalize_district(district)
    if not apy_name or dist_u not in DISTRICT_PROFILES:
        return 0.2
    crop_seasons = DISTRICT_PROFILES[dist_u].get(apy_name, {})
    crop_n = sum(
        int(v.get("n", 0)) for v in crop_seasons.values() if isinstance(v, dict)
    )
    if crop_n == 0:
        return 0.2
    max_n = 0
    for seasons in DISTRICT_PROFILES[dist_u].values():
        max_n = max(
            max_n,
            sum(int(v.get("n", 0)) for v in seasons.values() if isinstance(v, dict)),
        )
    if max_n == 0:
        return 0.2
    return float(np.clip(crop_n / max_n, 0, 1))


_STATE_SEASON_CORRECTION = {
    ("himachal pradesh", "kharif"): -10,
    ("himachal pradesh", "rabi"): -12,
    ("uttarakhand", "kharif"): -6,
    ("uttarakhand", "rabi"): -8,
    ("jammu and kashmir", "kharif"): -8,
    ("jammu and kashmir", "rabi"): -15,
    ("sikkim", "kharif"): -10,
    ("arunachal pradesh", "kharif"): -6,
    ("meghalaya", "kharif"): -8,
    ("rajasthan", "kharif"): +6,
    ("rajasthan", "zaid"): +8,
    ("gujarat", "kharif"): +3,
    ("kerala", "rabi"): +8,
    ("tamil nadu", "rabi"): +7,
    ("andhra pradesh", "rabi"): +5,
}
_DEFAULT_TEMP = {"kharif": 30.0, "rabi": 19.0, "zaid": 35.0}


def get_district_temp(district, state, season):
    """Return realistic temperature (°C) for given location + season.
    Priority: district override > state correction > season default."""
    key = (district.upper().strip(), season.lower().strip())
    if key in DISTRICT_TEMP_OVERRIDE:
        return DISTRICT_TEMP_OVERRIDE[key]
    base = _DEFAULT_TEMP.get(season.lower().strip(), 27.0)
    delta = _STATE_SEASON_CORRECTION.get(
        (state.lower().strip(), season.lower().strip()), 0
    )
    return base + delta


def temperature_filter(canonical, temp_c):
    """True if crop can grow at temp_c. Uses temp_range with ±2°C tolerance."""
    meta = CROP_REGISTRY.get(canonical, {})
    tr = meta.get("temp_range")
    if not tr:
        return True
    lo, hi = tr
    return (lo - 2) <= temp_c <= (hi + 2)


def ph_filter(canonical, soil_ph):
    """True if pH is within crop's acceptable range (±0.3 tolerance)."""
    meta = CROP_REGISTRY.get(canonical, {})
    pr = meta.get("ph_range")
    if not pr:
        return True
    lo, hi = pr
    return (lo - 0.3) <= soil_ph <= (hi + 0.3)


def water_filter(canonical, season_rain_mm):
    """True if seasonal rainfall can support crop's water requirement.
    Irrigated crops can be grown with 40% of min requirement."""
    meta = CROP_REGISTRY.get(canonical, {})
    wm = meta.get("water_mm")
    if not wm:
        return True
    lo, _hi = wm
    return season_rain_mm >= (lo * 0.4)


def engine_suitability(
    N,
    P,
    K,
    soil_type,
    season,
    irrigation,
    temp_mid,
    hum_mid,
    water_mid,
    soil_ph=6.5,
    dur_mid=120.0,
):
    """Returns dict of crop → ml_probability (top 15 candidates)"""
    irr_map = {
        "canal": "irrigated",
        "drip": "irrigated",
        "sprinkler": "irrigated",
        "borewell": "irrigated",
        "rainfed": "rainfed",
        "rainwater": "rainfed",
    }
    ws = irr_map.get(irrigation.lower(), "irrigated")

    soil_enc = safe_enc(crop_cat_enc["SOIL"], soil_type.lower().strip())
    seas_enc = safe_enc(crop_cat_enc["SEASON"], season.lower().strip())
    ws_enc = safe_enc(crop_cat_enc["WATER_SOURCE"], ws)

    x = np.array(
        [
            [
                soil_enc,
                seas_enc,
                ws_enc,
                soil_ph,
                temp_mid,
                hum_mid,
                water_mid,
                dur_mid,
                N,
                P,
                K,
                1.0,
                8.0,
                N * 0.3,
            ]
        ],
        dtype=np.float32,
    )

    if hasattr(crop_model, "predict_proba"):
        probs = crop_model.predict_proba(x)[0]
        top_idx = np.argsort(probs)[::-1][:15]
        return {crop_target_le.classes_[i]: float(probs[i]) for i in top_idx}
    else:
        pred = int(crop_model.predict(x)[0])
        return {crop_target_le.classes_[pred]: 1.0}


def engine_yield_score(canonical, district, state, season):
    """Returns (yield_score 0-1, yield_data_dict, data_level)"""
    meta = CROP_REGISTRY.get(canonical, {})
    apy_name = meta.get("apy_name")
    family = meta.get("family", "cereal")

    if not apy_name:
        return 0.3, None, "no_apy_data"

    dist_u = district.upper().strip()
    state_t = state.strip().title()
    season_l = season.lower().strip()
    SMAP = {
        "kharif": ["kharif"],
        "rabi": ["rabi", "winter"],
        "zaid": ["summer", "zaid"],
        "whole year": ["whole year", "kharif"],
    }
    v_seasons = SMAP.get(season_l, [season_l])

    data, level = None, None

    if dist_u in DISTRICT_PROFILES and apy_name in DISTRICT_PROFILES[dist_u]:
        cd = DISTRICT_PROFILES[dist_u][apy_name]
        for vs in v_seasons:
            if vs in cd:
                data, level = cd[vs], "district"
                break

    if data is None and state_t in STATE_PROFILES:
        if apy_name in STATE_PROFILES[state_t]:
            data, level = STATE_PROFILES[state_t][apy_name], "state"

    if data is None and apy_name in NATIONAL_PROFILES:
        data, level = NATIONAL_PROFILES[apy_name], "national"

    if data is None:
        return 0.2, None, "no_data"

    fam_max = FAMILY_YIELD_MAX.get(family, 5.0)
    fam_min = FAMILY_YIELD_MIN.get(family, 0.3)
    rng = max(fam_max - fam_min, 0.1)
    ys = float(np.clip((data["med"] - fam_min) / rng, 0, 1))
    if level == "district":
        ys = min(ys + 0.1, 1.0)

    return ys, data, level


def engine_region_score(canonical, district, state):
    meta = CROP_REGISTRY.get(canonical, {})
    st = state.strip().title()
    excl = [e.lower() for e in meta.get("regions_exclude", [])]
    if st.lower() in excl:
        return 0.0
    apy_name = meta.get("apy_name")
    dist_u = district.upper().strip()
    if apy_name:
        if dist_u in DISTRICT_PROFILES and apy_name in DISTRICT_PROFILES[dist_u]:
            return 1.0
        if st in STATE_PROFILES and apy_name in STATE_PROFILES[st]:
            return 0.7
        return 0.3
    return 0.5


def engine_irrigation_score(canonical, irrigation_type):
    """Score based on water_mm range vs irrigation source.
    Uses water_mm from CROP_REGISTRY instead of removed water_need bucket."""
    meta = CROP_REGISTRY.get(canonical, {})
    wm = meta.get("water_mm")
    irr = irrigation_type.lower().strip()

    if not wm:
        return 0.7

    min_need, _max_need = wm

    if irr in ("rainfed", "rainwater"):

        if min_need >= 1000:
            return 0.2
        if min_need >= 500:
            return 0.6
        return 1.0
    elif irr in ("drip", "sprinkler"):

        if min_need >= 1000:
            return 0.7
        if min_need >= 400:
            return 1.0
        return 0.8
    else:
        if min_need >= 1000:
            return 1.0
        if min_need >= 400:
            return 0.9
        return 0.7


def is_crop_in_season(canonical, season):
    meta = CROP_REGISTRY.get(canonical, {})
    valid = [s.lower().strip() for s in meta.get("seasons", [])]
    sl = season.lower().strip()
    equiv_map = {"zaid": "summer", "summer": "zaid", "winter": "rabi", "rabi": "winter"}
    equiv = equiv_map.get(sl)
    return sl in valid or (equiv is not None and equiv in valid)


def engine_season_score(canonical, season):
    return 1.0 if is_crop_in_season(canonical, season) else 0.0


def engine_score(
    suitability_prob,
    canonical,
    district,
    state,
    season,
    irrigation,
    annual_rain,
    season_rain,
    temp_c=None,
    soil_ph=None,
):
    """
    Full scoring: suitability + yield + region + irrigation + season + presence
    + NEW: temperature, pH, water range-based hard filters.
    Returns (final_score, component_scores, yield_data, yield_level)
    or (None, None, None, None) if hard-filtered.
    """
    meta = CROP_REGISTRY.get(canonical, {})
    ss = engine_season_score(canonical, season)
    rs = engine_region_score(canonical, district, state)

    if ss == 0.0 or rs == 0.0:
        return None, None, None, None
    if not drought_filter(canonical, annual_rain, season_rain):
        return None, None, None, None

    if temp_c is not None and not temperature_filter(canonical, temp_c):
        return None, None, None, None

    temp_score = 1.0

    ph_score = 1.0
    if soil_ph is not None and not ph_filter(canonical, soil_ph):
        ph_score = 0.5

    water_score = 1.0
    if not water_filter(canonical, season_rain):
        water_score = 0.3

    ys, yield_data, level = engine_yield_score(canonical, district, state, season)
    is_ = engine_irrigation_score(canonical, irrigation)
    ps = engine_presence_score(canonical, district)

    weight_sum = (
        W_SUITABILITY + W_YIELD + W_REGION + W_IRRIGATION + W_SEASON + W_PRESENCE
    )

    final = (
        W_SUITABILITY * suitability_prob
        + W_YIELD * ys
        + W_REGION * rs
        + W_IRRIGATION * is_
        + W_SEASON * ss
        + W_PRESENCE * ps
        + 0.10 * temp_score
        + 0.07 * water_score
        + 0.03 * ph_score
    )

    components = {
        "ml_prob": round(suitability_prob, 4),
        "yield_score": round(ys, 3),
        "region_score": round(rs, 3),
        "irr_score": round(is_, 3),
        "season_score": round(ss, 3),
        "presence_score": round(ps, 3),
        "temp_score": round(temp_score, 3),
        "water_score": round(water_score, 3),
        "ph_score": round(ph_score, 3),
        "final": round(final, 4),
    }
    return final, components, yield_data, level


FERT_FEAT = [
    "Soil_Type",
    "Soil_pH",
    "Soil_Moisture",
    "Organic_Carbon",
    "Electrical_Conductivity",
    "Nitrogen_Level",
    "Phosphorus_Level",
    "Potassium_Level",
    "Temperature",
    "Humidity",
    "Rainfall",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Previous_Crop",
    "Region",
]
FERT_CAT = [
    "Soil_Type",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Previous_Crop",
    "Region",
]

REGION_MAP = {
    "punjab": "North",
    "haryana": "North",
    "uttar pradesh": "North",
    "uttarakhand": "North",
    "himachal pradesh": "North",
    "jammu and kashmir": "North",
    "delhi": "North",
    "bihar": "North",
    "jharkhand": "East",
    "west bengal": "East",
    "odisha": "East",
    "assam": "East",
    "gujarat": "West",
    "maharashtra": "West",
    "goa": "West",
    "rajasthan": "West",
    "karnataka": "South",
    "kerala": "South",
    "tamil nadu": "South",
    "andhra pradesh": "South",
    "telangana": "South",
    "puducherry": "South",
    "madhya pradesh": "Central",
    "chhattisgarh": "Central",
}


def get_agronomy_fertilizer(canonical, N, P, K):
    meta = CROP_REGISTRY.get(canonical, {})
    n_fix = meta.get("n_fixing", False)
    display = meta.get("display", canonical)
    tend = meta.get("npk_tendency", "balanced")

    if n_fix:
        if P < 25:
            return (
                "SSP",
                f"{display} is a legume (fixes N). P={P:.0f} kg/ha is low — SSP (P+S) recommended.",
            )
        if P < 45:
            return "SSP", f"{display} (legume) needs P+S support — SSP recommended."
        return "DAP", f"{display} (legume) with adequate P — DAP for P+N balance."

    if N < 40:
        return "Urea", f"N={N:.0f} kg/ha critically low — Urea (46%N) for {display}."
    if P < 20:
        return "DAP", f"P={P:.0f} kg/ha critically low — DAP (46% P₂O₅)."
    if K < 30:
        return "MOP", f"K={K:.0f} kg/ha critically low — MOP (60% K₂O)."

    if tend == "high_n":
        if N < 80:
            return "Urea", f"{display} needs high N. N={N:.0f} below optimal — Urea."
        return "NPK", f"{display} — balanced NPK for maintenance."
    if tend == "high_n_high_k":
        if N < 80:
            return "Urea", f"{display} needs high N+K. N={N:.0f} low — Urea first."
        if K < 60:
            return "MOP", f"{display} needs K. K={K:.0f} low — MOP."
        return "NPK", f"{display} — NPK maintenance."
    if tend == "high_k":
        if K < 50:
            return "MOP", f"{display} is K-intensive (root/tuber). K={K:.0f} low — MOP."
        return "NPK", f"{display} — NPK maintenance."
    return "NPK", f"{display} — balanced NPK recommended."


def engine_fertilizer(
    N, P, K, soil_type, canonical, season, irrigation, soil_ph, annual_rain, state
):
    meta = CROP_REGISTRY.get(canonical, {})
    proxy = (meta.get("fert_proxy", "Rice") if meta else "Rice").title()
    if proxy not in fert_cat_enc["Crop_Type"].classes_:
        proxy = "Rice"

    region = REGION_MAP.get(state.lower().strip(), "South").title()
    hum_est = 75.0 if season.lower() == "kharif" else 55.0
    soil_mst = 35.0 if season.lower() == "kharif" else 25.0
    st_type = (
        soil_type.title()
        if soil_type.title() in fert_cat_enc["Soil_Type"].classes_
        else "Clayey"
    )
    irr_type = (
        irrigation.title()
        if irrigation.title() in fert_cat_enc["Irrigation_Type"].classes_
        else "Canal"
    )
    prev = "Rice" if season.lower() == "kharif" else "Wheat"
    seas_fert = (
        season.title()
        if season.title() in fert_cat_enc["Season"].classes_
        else "Kharif"
    )

    row_dict = {
        "Soil_Type": st_type,
        "Soil_pH": soil_ph,
        "Soil_Moisture": soil_mst,
        "Organic_Carbon": 0.5,
        "Electrical_Conductivity": 0.5,
        "Nitrogen_Level": int(np.clip(N, 20, 159)),
        "Phosphorus_Level": int(np.clip(P, 10, 89)),
        "Potassium_Level": int(np.clip(K, 10, 119)),
        "Temperature": 30.0 if season.lower() == "kharif" else 20.0,
        "Humidity": hum_est,
        "Rainfall": annual_rain,
        "Crop_Type": proxy,
        "Crop_Growth_Stage": "Vegetative",
        "Season": seas_fert,
        "Irrigation_Type": irr_type,
        "Previous_Crop": prev,
        "Region": region,
    }
    row = [
        float(safe_enc(fert_cat_enc[f], row_dict[f]) if f in FERT_CAT else row_dict[f])
        for f in FERT_FEAT
    ]
    X = np.array([row])
    if fert_model_name == "SVM (RBF)":
        X = fert_scaler.transform(X)

    ml_probs = {}
    if hasattr(fert_model, "predict_proba"):
        probs = fert_model.predict_proba(X)[0]
        ml_probs = {
            fert_target_le.classes_[i]: round(float(probs[i]) * 100, 1)
            for i in np.argsort(probs)[::-1]
        }

    agro_fert, agro_reason = get_agronomy_fertilizer(canonical, N, P, K)
    n_fix = meta.get("n_fixing", False) if meta else False
    critical = N < 40 or P < 20 or K < 30

    if n_fix or critical:
        return agro_fert, agro_reason, ml_probs
    if ml_probs and list(ml_probs.values())[0] >= 50:
        ml_top = list(ml_probs.keys())[0]
        ml_conf = list(ml_probs.values())[0]
        return ml_top, f"ML model ({ml_conf:.1f}% confidence). {agro_reason}", ml_probs
    return agro_fert, agro_reason + " [ML low confidence — agronomy used]", ml_probs


def engine_yield_ml(canonical, state, district, season, area=1000.0):
    """ML yield prediction as secondary estimate"""
    meta = get_meta(canonical)
    apy_name = meta.get("apy_name") if meta else canonical.title()
    if not apy_name:
        return None
    crop_enc = safe_enc(yield_cat_enc["Crop"], apy_name, -1)
    if crop_enc == -1:
        return None
    SMAP = {
        "kharif": "Kharif",
        "rabi": "Rabi",
        "zaid": "Summer",
        "whole year": "Whole Year",
        "summer": "Summer",
        "winter": "Winter",
    }
    yield_seas = SMAP.get(season.lower(), "Kharif")
    sea_enc = safe_enc(yield_cat_enc["Season"], yield_seas)
    st_enc = safe_enc(yield_cat_enc["State"], state.strip().title())
    dist_enc = safe_enc(yield_cat_enc["District"], district.upper().strip())
    X = np.array(
        [[crop_enc, st_enc, dist_enc, sea_enc, 2018, np.clip(area, 10, 100000)]],
        dtype=np.float32,
    )
    log_pred = float(yield_model.predict(X)[0])
    return float(np.expm1(max(log_pred, 0)))


def consistency_check(scored_crops, temp_c, soil_ph, season_rain_mm, season):
    """
    STRICT post-ranking validation — catches anything that slipped through
    the softer scoring filters. No tolerance applied.

    This is the "senior doctor approval" layer:
    - Scoring finds the best crops (probabilistic)
    - Consistency ensures they are valid (rule-based)
    """
    valid = []
    removed = []

    for c in scored_crops:
        canonical = c["canonical"]
        meta = CROP_REGISTRY.get(canonical, {})
        reason = None

        valid_seasons = meta.get("seasons", [])
        season_l = season.lower().strip()
        equiv_map = {"zaid": "summer", "summer": "zaid", "winter": "rabi"}
        equiv = equiv_map.get(season_l)
        if season_l not in valid_seasons and not (equiv and equiv in valid_seasons):
            reason = f"wrong season ({season_l} not in {valid_seasons})"

        if reason is None and temp_c is not None:
            tr = meta.get("temp_range")
            if tr:
                lo, hi = tr
                if temp_c < lo or temp_c > hi:
                    reason = f"temp {temp_c}°C outside range ({lo}–{hi}°C)"

        if reason is None and soil_ph is not None:
            pr = meta.get("ph_range")
            if pr:
                lo, hi = pr
                if soil_ph < lo or soil_ph > hi:
                    reason = f"pH {soil_ph} outside range ({lo}–{hi})"

        if reason is None:
            wm = meta.get("water_mm")
            if wm:
                min_need = wm[0]
                if season_rain_mm < (min_need * 0.3):
                    reason = f"rain {season_rain_mm:.0f}mm < 30% of {min_need}mm min"

        if reason:
            removed.append((c["display"], reason))
        else:
            valid.append(c)

    return valid, removed


def get_calendar(canonical, season):
    try:
        df = pd.read_csv(
            resolve_data_file(
                "Crop_recommendation_dataset.csv", "Crop recommendation dataset.csv"
            )
        )
        df["CROPS"] = df["CROPS"].str.lower().str.strip()
        m = df[df["CROPS"] == canonical.lower().strip()]
        if not m.empty:
            dur = int((m["CROPDURATION"].median() + m["CROPDURATION_MAX"].median()) / 2)
            return m["SOWN"].mode()[0], m["HARVESTED"].mode()[0], dur
    except:
        pass
    return {
        "kharif": ("Jun", "Nov", 150),
        "rabi": ("Nov", "Mar", 120),
        "zaid": ("Mar", "Jun", 90),
    }.get(season.lower(), ("Jun", "Nov", 150))


def run():
    W = 68
    print("\n" + "═" * W)
    print(f"{'🌾  SMART AGRICULTURE AI  v4  |  6-ENGINE SYSTEM':^{W}}")
    print("═" * W)

    print(f"\n📋 STEP 1 — FARM DETAILS")
    print("─" * W)
    district = input("  District name              : ").strip()
    state = input("  State name                 : ").strip()
    print("\n  Soil types: black cotton soil | alluvial soil | loamy soil")
    print("              red soil | clay loam soil | sandy loam soil | laterite soil")
    soil = input("  Soil Type                  : ").strip()
    season = input("\n  Season [kharif/rabi/zaid]  : ").strip().lower()
    if season not in ["kharif", "rabi", "zaid"]:
        season = "kharif"
        print("  → Defaulted to kharif")
    print("  Irrigation: canal / drip / sprinkler / rainfed / borewell")
    irrigation = input("  Irrigation                 : ").strip()
    print("\n  NPK from soil test (kg/ha):")
    try:
        N = float(input("  Nitrogen   (N)             : "))
        P = float(input("  Phosphorus (P)             : "))
        K = float(input("  Potassium  (K)             : "))
    except:
        N, P, K = 80.0, 40.0, 40.0
        print("  → Using defaults: N=80, P=40, K=40")
    try:
        soil_ph = float(input("  Soil pH    [Enter=6.5]     : ") or "6.5")
    except:
        soil_ph = 6.5

    print(f"\n{'─'*W}")
    print("📡 STEP 2 — LOCATION ENRICHMENT")
    print("─" * W)

    norm = normalize_input(
        {
            "district": district,
            "state": state,
            "soil_type": soil,
            "season": season,
            "irrigation": irrigation,
            "N": N,
            "P": P,
            "K": K,
            "pH": soil_ph,
        }
    )

    district = norm["district"]
    state = norm["state"]
    soil = norm["soil_type"]
    season = norm["season"]
    irrigation = norm["irrigation"]
    N = norm["N"]
    P = norm["P"]
    K = norm["K"]
    soil_ph = norm["pH"]

    rain = lookup_rainfall(district, state)
    if rain:
        annual_rain = rain["annual"]
        season_rain = {
            "kharif": rain["jun_sep"],
            "rabi": rain["oct_dec"],
            "zaid": rain["mar_may"],
        }.get(season, annual_rain * 0.5)
        print(f"  ✅ {rain['district']}, {rain['state']}")
        print(f"  🌧  Annual    : {annual_rain:.0f} mm")
        print(f"  🌧  {season.title():<8} : {season_rain:.0f} mm")
    else:
        print(f"  ⚠️  '{district}' not in rainfall database.")
        try:
            annual_rain = float(input("  Enter annual rainfall (mm) : ") or "800")
        except:
            annual_rain = 800.0
        season_rain = annual_rain * (0.6 if season == "kharif" else 0.25)

    TEMP_MAP_FLAT = {"kharif": 30.0, "rabi": 19.0, "zaid": 35.0}
    HUM_MAP = {"kharif": 75.0, "rabi": 55.0, "zaid": 40.0}
    temp_mid = get_district_temp(district, state, season)
    hum_mid = HUM_MAP.get(season, 60.0)
    print(f"  🌡  Temp (district-aware) : {temp_mid}°C")
    print(f"  💧 Humidity              : {hum_mid}%")

    print(f"\n{'─'*W}")
    print("🤖 STEP 3 — SUITABILITY ENGINE (ML)")
    print("─" * W)

    ml_scores = engine_suitability(
        N, P, K, soil, season, irrigation, temp_mid, hum_mid, season_rain, soil_ph
    )
    print(f"  ML model returned {len(ml_scores)} candidate crops")

    ml_candidates = set(ml_scores.keys())
    yield_candidates = get_yield_top_crops(district, top_n=5)
    freq_candidates = get_freq_top_crops(district, top_n=5)
    region_candidates = get_region_crops(state)
    candidate_crops = (
        {resolve(c) for c in ml_candidates}
        | yield_candidates
        | freq_candidates
        | region_candidates
    )
    print(f"  Candidate pool size: {len(candidate_crops)}")
    print(f"    • from ML suitability : {len(ml_candidates)}")
    print(f"    • from yield history  : {len(yield_candidates)}")
    print(f"    • from frequency      : {len(freq_candidates)}")
    print(f"    • from region priors  : {len(region_candidates)}")

    print(f"\n{'─'*W}")
    print("⚖️  STEP 4 — SCORING ENGINE (6 components)")
    print("─" * W)

    scored = []
    filter_stats = {
        "not_in_season": 0,
        "blocked_by_hard_filters": 0,
        "scored": 0,
    }
    for canonical in candidate_crops:

        meta = get_meta(canonical) or {}
        if not is_crop_in_season(canonical, season):
            filter_stats["not_in_season"] += 1
            continue

        ml_prob = float(ml_scores.get(canonical, 0.05))

        final, components, yield_data, yield_level = engine_score(
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
            filter_stats["blocked_by_hard_filters"] += 1
            continue

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
        filter_stats["scored"] += 1

    scored.sort(key=lambda x: x["final"], reverse=True)

    print(f"\n{'─'*W}")
    print("🛡️  STEP 4b — CONSISTENCY ENGINE (post-ranking)")
    print("─" * W)

    TOP_N = 4
    top_crops = scored[:TOP_N]

    print(f"\n  Scored {len(scored)} crops after filters. Top {TOP_N}:\n")
    print(f"  Filter diagnostics:")
    print(f"    • Not in requested season : {filter_stats['not_in_season']}")
    print(f"    • Blocked by climate/rule : {filter_stats['blocked_by_hard_filters']}")
    print(f"    • Passed and scored       : {filter_stats['scored']}")

    print(
        f"\n  {'#':<3} {'Crop':<24} {'Final':>7} {'ML%':>7} {'Yield':>7} {'Reg':>6} {'Irr':>6} {'Pres':>6} {'Tmp':>6} {'Wtr':>6} {'pH':>6} {'Data':>8}"
    )
    print(
        f"  {'─'*3} {'─'*24} {'─'*7} {'─'*7} {'─'*7} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*8}"
    )

    for i, c in enumerate(top_crops, 1):
        sc = c["components"]
        yd = c["yield_data"]
        yd_str = f"{yd['med']:.2f} t/ha" if yd else "—"
        print(
            f"  {i:<3} {c['display']:<24} {sc['final']:>7.4f} {sc['ml_prob']*100:>6.2f}% "
            f"{sc['yield_score']:>7.3f} {sc['region_score']:>6.3f} {sc['irr_score']:>6.3f} "
            f"{sc['presence_score']:>6.3f} {sc.get('temp_score', 0):>6.3f} "
            f"{sc.get('water_score', 0):>6.3f} {sc.get('ph_score', 0):>6.3f} {str(c['yield_level'] or '—'):>8}"
        )

    if not top_crops:
        print("  ⚠️  No crops passed strict season and climate filters for this input.")
        print(
            "  Try adjusting season/soil/NPK/irrigation to get valid seasonal matches."
        )
        return

    best = top_crops[0]
    best_canonical = best["canonical"]

    sown, harvested, duration = get_calendar(best_canonical, season)

    print(f"\n{'─'*W}")
    print("🧪 STEP 5 — FERTILIZER ENGINE")
    print("─" * W)

    fert_name, fert_reason, ml_probs = engine_fertilizer(
        N, P, K, soil, best_canonical, season, irrigation, soil_ph, annual_rain, state
    )

    meta_best = get_meta(best_canonical)
    proxy_note = ""
    if meta_best and meta_best.get("fert_proxy", "").lower() != best_canonical:
        proxy_note = f"\n  ℹ️  '{best['display']}' → '{meta_best['fert_proxy']}' for fertilizer model"

    if proxy_note:
        print(proxy_note)
    print(f"\n  🎯 PRIMARY : {fert_name}")
    print(f"     Reason  : {fert_reason}")

    if ml_probs:
        print(f"\n  📊 ML Model Probabilities:")
        for fn, fp in list(ml_probs.items())[:5]:
            bar = "█" * int(fp / 10)
            print(f"     {fn:<22} {fp:>5.1f}%  {bar}")

    n_st = (
        "🔴 Critical Low"
        if N < 40
        else "⚠️ Low" if N < 60 else "✅ Optimal" if N <= 110 else "🔴 High"
    )
    p_st = (
        "🔴 Critical Low"
        if P < 20
        else "⚠️ Low" if P < 30 else "✅ Optimal" if P <= 60 else "🔴 High"
    )
    k_st = (
        "🔴 Critical Low"
        if K < 30
        else "⚠️ Low" if K < 40 else "✅ Optimal" if K <= 80 else "🔴 High"
    )
    print(f"\n  🌿 NPK STATUS:")
    print(f"     N: {N:>5.1f} kg/ha  {n_st}")
    print(f"     P: {P:>5.1f} kg/ha  {p_st}")
    print(f"     K: {K:>5.1f} kg/ha  {k_st}")

    print(f"\n{'─'*W}")
    print("📈 STEP 6 — YIELD ENGINE (tonnes/hectare)")
    print("─" * W)
    print()
    print(
        f"  {'#':<3} {'Crop':<28} {'Hist (dist/state/nat)':>22}  {'Range (P10–P90)':>18}  {'ML Est':>9}  {'Confidence'}"
    )
    print(f"  {'─'*3} {'─'*28} {'─'*22}  {'─'*18}  {'─'*9}  {'─'*12}")

    for i, c in enumerate(top_crops, 1):
        yd = c["yield_data"]
        lvl = c["yield_level"]
        ml_y = engine_yield_ml(c["canonical"], state, district, season)

        if yd:
            conf = {
                "district": "🟢 High",
                "district_off_season": "🟡 Medium",
                "state": "🟡 Medium",
                "national": "🟠 Low",
                None: "—",
            }.get(lvl, "—")
            hist_str = f"{yd['med']:.2f} t/ha ({lvl[:5]})"
            range_str = f"{yd['p10']:.2f}–{yd['p90']:.2f} t/ha"
        else:
            hist_str, range_str, conf = "  no data", "—", "⚫ No data"

        ml_str = f"{ml_y:.2f} t/ha" if ml_y else "—"
        print(
            f"  {i:<3} {c['display']:<28} {hist_str:>22}  {range_str:>18}  {ml_str:>9}  {conf}"
        )

    best_yd = best["yield_data"]
    best_ml = engine_yield_ml(best_canonical, state, district, season)
    primary_yield = best_yd["med"] if best_yd else best_ml
    yield_src = best["yield_level"] if best_yd else ("ML model" if best_ml else None)

    print(f"\n{'═'*W}")
    print(f"{'🎯  FINAL RECOMMENDATION CARD':^{W}}")
    print(f"{'═'*W}")
    print(f"""
  📍 Location     : {district.title()}, {state.title()}
  🌧  Annual Rain  : {annual_rain:.0f} mm  |  Season Rain: {season_rain:.0f} mm
  🌡  Season       : {season.title()}

  ─────────────────── TOP RECOMMENDATIONS ───────────────────""")

    for i, c in enumerate(top_crops, 1):
        sc = c["components"]
        yd = c["yield_data"]
        yield_show = f"{yd['med']:.2f} t/ha" if yd else "—"
        why = []
        if sc["region_score"] >= 0.8:
            why.append("Proven in this district")
        if sc["ml_prob"] >= 0.15:
            why.append(f"Strong ML suitability ({sc['ml_prob']*100:.1f}%)")
        if sc["yield_score"] >= 0.6:
            why.append("High yield performance")
        if sc["irr_score"] >= 0.8:
            why.append("Matches irrigation type")

        print(f"\n  {i}. {c['display']} (score: {sc['final']:.4f})")
        print(f"     Yield    : {yield_show}")
        if why:
            print(f"     Why      : {' | '.join(why)}")

    print(f"""
  ─────────────────── PLANTING CALENDAR ─────────────────────
  {best['display']}: Sow {sown} → Harvest {harvested}  (~{duration} days)

  ─────────────────── FERTILIZER ─────────────────────────────
  Recommendation : {fert_name}
  Reason         : {fert_reason[:80]}

  NPK: N={N:.0f}({n_st[2:]}) | P={P:.0f}({p_st[2:]}) | K={K:.0f}({k_st[2:]})

  ─────────────────── EXPECTED YIELD ─────────────────────────
  {best['display']}: {f"{primary_yield:.2f} t/ha" if primary_yield else "Insufficient data"}
  Source: {yield_src or "—"}
  {'Range: ' + str(best_yd['p10']) + '–' + str(best_yd['p90']) + ' t/ha' if best_yd else ''}
""")
    print("═" * W)
    print(
        f"  Engines: ML({crop_model_name}) + Scoring(6-component) + Consistency + APY Profiles"
    )
    print(
        f"  Weights: suit={W_SUITABILITY} yield={W_YIELD} region={W_REGION} "
        f"irr={W_IRRIGATION} season={W_SEASON} pres={W_PRESENCE}"
    )
    print(f"  Filters: season=STRICT, drought=ON, temp/pH/water penalties=ON")
    print(
        f"  Candidate stats: total={len(candidate_crops)} | in-season={len(candidate_crops)-filter_stats['not_in_season']} | final={len(scored)}"
    )
    print(f"  Temp: {temp_mid}°C (district-aware) | pH: {soil_ph}")
    print("═" * W)

    again = input("\n  Another prediction? (y/n): ").strip().lower()
    if again == "y":
        run()


if __name__ == "__main__":
    run()
