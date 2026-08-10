import pandas as pd, numpy as np, pickle, warnings
from config import DATA_DIR, ENCODER_DIR, ensure_dirs, resolve_data_file

warnings.filterwarnings("ignore")
ensure_dirs()

apy = pd.read_csv(resolve_data_file("APY.csv"))
apy.columns = [c.strip() for c in apy.columns]
rain_df = pd.read_csv(
    resolve_data_file(
        "district_wise_rainfall_normal.csv", "district wise rainfall normal.csv"
    )
)
rain_df.columns = [c.strip() for c in rain_df.columns]
print(f"✅ APY loaded: {apy.shape}")

apy["State"] = apy["State"].str.strip().str.title()
apy["District"] = apy["District"].str.strip().str.upper()
apy["Crop"] = apy["Crop"].str.strip()
apy["Season"] = apy["Season"].str.strip()
apy = apy.dropna(subset=["Crop"])
apy = apy[apy["Yield"] > 0]
print(f"  After basic cleaning: {len(apy):,} rows")


jute_mask = apy["Crop"] == "Jute"
apy.loc[jute_mask, "Yield"] = apy.loc[jute_mask, "Yield"] / 10
print(
    f"  Jute yield after unit fix — median: {apy.loc[jute_mask,'Yield'].median():.2f} t/ha  (expect ~1.8–2.5)"
)


apy["Crop"] = apy["Crop"].str.rstrip()

STATE_NORM = {
    "Andaman And Nicobar Island": "Andaman and Nicobar Islands",
    "The Dadra And Nagar Haveli": "Dadra and Nagar Haveli",
    "Chatisgarh": "Chhattisgarh",
    "Uttaranchal": "Uttarakhand",
    "Pondicherry": "Puducherry",
    "Orissa": "Odisha",
    "Himachal": "Himachal Pradesh",
}
apy["State"] = apy["State"].replace(STATE_NORM)


cleaned = []
for crop_name, grp in apy.groupby("Crop"):
    p01, p99 = grp["Yield"].quantile(0.01), grp["Yield"].quantile(0.99)
    cleaned.append(grp[(grp["Yield"] >= p01) & (grp["Yield"] <= p99)])
apy_clean = pd.concat(cleaned, ignore_index=True)
print(f"  After per-crop outlier clipping: {len(apy_clean):,} rows")


CROP_REGISTRY = {
    "rice": {
        "display": "Rice",
        "apy_name": "Rice",
        "fert_proxy": "Rice",
        "family": "cereal",
        "n_fixing": False,
        "seasons": ["kharif"],
        "npk_tendency": "high_n",
        "regions_exclude": [],
        "temp_range": (22, 35),
        "ph_range": (5.5, 7.0),
        "water_mm": (1200, 2000),
    },
    "wheat": {
        "display": "Wheat",
        "apy_name": "Wheat",
        "fert_proxy": "Wheat",
        "family": "cereal",
        "n_fixing": False,
        "seasons": ["rabi", "winter"],
        "npk_tendency": "high_n",
        "regions_exclude": [
            "Kerala",
            "Tamil Nadu",
            "Andhra Pradesh",
            "Telangana",
            "Karnataka",
            "Goa",
        ],
        "temp_range": (10, 25),
        "ph_range": (6.0, 7.5),
        "water_mm": (450, 650),
    },
    "maize": {
        "display": "Maize",
        "apy_name": "Maize",
        "fert_proxy": "Maize",
        "family": "cereal",
        "n_fixing": False,
        "seasons": ["kharif", "zaid"],
        "npk_tendency": "high_n",
        "regions_exclude": [],
        "temp_range": (18, 35),
        "ph_range": (5.8, 7.0),
        "water_mm": (500, 800),
    },
    "sorghum": {
        "display": "Sorghum (Jowar)",
        "apy_name": "Jowar",
        "fert_proxy": "Maize",
        "family": "cereal",
        "n_fixing": False,
        "seasons": ["kharif", "rabi"],
        "npk_tendency": "balanced",
        "regions_exclude": [],
        "temp_range": (25, 40),
        "ph_range": (5.5, 7.5),
        "water_mm": (300, 500),
    },
    "ragi": {
        "display": "Ragi (Finger Millet)",
        "apy_name": "Ragi",
        "fert_proxy": "Maize",
        "family": "cereal",
        "n_fixing": False,
        "seasons": ["kharif"],
        "npk_tendency": "low_n",
        "regions_exclude": ["Punjab", "Haryana", "Rajasthan", "Uttar Pradesh"],
        "temp_range": (15, 30),
        "ph_range": (5.5, 7.5),
        "water_mm": (300, 600),
    },
    "pearl millet": {
        "display": "Pearl Millet (Bajra)",
        "apy_name": "Bajra",
        "fert_proxy": "Maize",
        "family": "cereal",
        "n_fixing": False,
        "seasons": ["kharif"],
        "npk_tendency": "balanced",
        "regions_exclude": [],
        "temp_range": (25, 42),
        "ph_range": (6.0, 8.0),
        "water_mm": (200, 400),
    },
    "kudiraivali": {
        "display": "Barnyard Millet",
        "apy_name": "Small millets",
        "fert_proxy": "Maize",
        "family": "cereal",
        "n_fixing": False,
        "seasons": ["kharif"],
        "npk_tendency": "low_n",
        "regions_exclude": [],
        "temp_range": (20, 35),
        "ph_range": (5.0, 7.5),
        "water_mm": (250, 500),
    },
    "thinai": {
        "display": "Foxtail Millet",
        "apy_name": "Small millets",
        "fert_proxy": "Maize",
        "family": "cereal",
        "n_fixing": False,
        "seasons": ["kharif"],
        "npk_tendency": "low_n",
        "regions_exclude": [],
        "temp_range": (20, 35),
        "ph_range": (5.0, 7.5),
        "water_mm": (250, 500),
    },
    "samai": {
        "display": "Little Millet",
        "apy_name": "Small millets",
        "fert_proxy": "Maize",
        "family": "cereal",
        "n_fixing": False,
        "seasons": ["kharif"],
        "npk_tendency": "low_n",
        "regions_exclude": [],
        "temp_range": (20, 35),
        "ph_range": (5.0, 7.5),
        "water_mm": (250, 500),
    },
    "panivaragu": {
        "display": "Proso Millet",
        "apy_name": "Small millets",
        "fert_proxy": "Maize",
        "family": "cereal",
        "n_fixing": False,
        "seasons": ["kharif"],
        "npk_tendency": "low_n",
        "regions_exclude": [],
        "temp_range": (20, 35),
        "ph_range": (5.0, 7.5),
        "water_mm": (250, 500),
    },
    "varagu": {
        "display": "Kodo Millet",
        "apy_name": "Small millets",
        "fert_proxy": "Maize",
        "family": "cereal",
        "n_fixing": False,
        "seasons": ["kharif"],
        "npk_tendency": "low_n",
        "regions_exclude": [],
        "temp_range": (20, 35),
        "ph_range": (5.0, 7.5),
        "water_mm": (250, 500),
    },
    "soyabean": {
        "display": "Soyabean",
        "apy_name": "Soyabean",
        "fert_proxy": "Wheat",
        "family": "oilseed",
        "n_fixing": True,
        "seasons": ["kharif"],
        "npk_tendency": "low_n_high_p",
        "regions_exclude": [],
        "temp_range": (20, 30),
        "ph_range": (6.0, 7.0),
        "water_mm": (450, 700),
    },
    "groundnut": {
        "display": "Groundnut",
        "apy_name": "Groundnut",
        "fert_proxy": "Wheat",
        "family": "oilseed",
        "n_fixing": True,
        "seasons": ["kharif", "zaid", "summer"],
        "npk_tendency": "low_n_high_p",
        "regions_exclude": [],
        "temp_range": (25, 35),
        "ph_range": (5.5, 7.0),
        "water_mm": (500, 700),
    },
    "sunflower": {
        "display": "Sunflower",
        "apy_name": "Sunflower",
        "fert_proxy": "Wheat",
        "family": "oilseed",
        "n_fixing": False,
        "seasons": ["kharif"],
        "npk_tendency": "high_n",
        "regions_exclude": ["Haryana"],
        "temp_range": (18, 35),
        "ph_range": (6.0, 7.5),
        "water_mm": (600, 1000),
    },
    "castor": {
        "display": "Castor",
        "apy_name": "Castor seed",
        "fert_proxy": "Wheat",
        "family": "oilseed",
        "n_fixing": False,
        "seasons": ["kharif"],
        "npk_tendency": "balanced",
        "regions_exclude": [],
        "temp_range": (20, 40),
        "ph_range": (5.5, 7.0),
        "water_mm": (300, 600),
    },
    "gingely": {
        "display": "Gingely (Sesame)",
        "apy_name": "Sesamum",
        "fert_proxy": "Wheat",
        "family": "oilseed",
        "n_fixing": False,
        "seasons": ["kharif"],
        "npk_tendency": "balanced",
        "regions_exclude": [],
        "temp_range": (25, 35),
        "ph_range": (5.5, 7.0),
        "water_mm": (300, 500),
    },
    "bengalgram": {
        "display": "Bengal Gram (Chickpea)",
        "apy_name": "Gram",
        "fert_proxy": "Wheat",
        "family": "pulse",
        "n_fixing": True,
        "seasons": ["rabi"],
        "npk_tendency": "low_n_high_p",
        "regions_exclude": [],
        "temp_range": (10, 25),
        "ph_range": (6.0, 8.0),
        "water_mm": (300, 500),
    },
    "redgram": {
        "display": "Red Gram (Tur/Pigeon Pea)",
        "apy_name": "Arhar/Tur",
        "fert_proxy": "Wheat",
        "family": "pulse",
        "n_fixing": True,
        "seasons": ["kharif"],
        "npk_tendency": "low_n_high_p",
        "regions_exclude": [],
        "temp_range": (18, 35),
        "ph_range": (5.5, 7.0),
        "water_mm": (400, 700),
    },
    "blackgram": {
        "display": "Black Gram (Urad)",
        "apy_name": "Urad",
        "fert_proxy": "Wheat",
        "family": "pulse",
        "n_fixing": True,
        "seasons": ["kharif"],
        "npk_tendency": "low_n_high_p",
        "regions_exclude": [],
        "temp_range": (25, 35),
        "ph_range": (5.5, 7.0),
        "water_mm": (300, 500),
    },
    "greengram": {
        "display": "Green Gram (Moong)",
        "apy_name": "Moong(Green Gram)",
        "fert_proxy": "Wheat",
        "family": "pulse",
        "n_fixing": True,
        "seasons": ["kharif", "zaid", "summer"],
        "npk_tendency": "low_n_high_p",
        "regions_exclude": [],
        "temp_range": (25, 35),
        "ph_range": (5.5, 7.0),
        "water_mm": (300, 500),
    },
    "horsegram": {
        "display": "Horse Gram",
        "apy_name": "Horse-gram",
        "fert_proxy": "Wheat",
        "family": "pulse",
        "n_fixing": True,
        "seasons": ["kharif", "rabi"],
        "npk_tendency": "low_n_high_p",
        "regions_exclude": [],
        "temp_range": (20, 35),
        "ph_range": (5.5, 7.0),
        "water_mm": (250, 500),
    },
    "cowpea": {
        "display": "Cowpea",
        "apy_name": "Cowpea(Lobia)",
        "fert_proxy": "Wheat",
        "family": "pulse",
        "n_fixing": True,
        "seasons": ["kharif", "zaid"],
        "npk_tendency": "low_n_high_p",
        "regions_exclude": [],
        "temp_range": (20, 35),
        "ph_range": (5.5, 7.5),
        "water_mm": (300, 600),
    },
    "peas": {
        "display": "Peas",
        "apy_name": "Peas & beans (Pulses)",
        "fert_proxy": "Wheat",
        "family": "pulse",
        "n_fixing": True,
        "seasons": ["rabi"],
        "npk_tendency": "low_n_high_p",
        "regions_exclude": [],
        "temp_range": (7, 24),
        "ph_range": (6.0, 7.5),
        "water_mm": (350, 600),
    },
    "cluster bean": {
        "display": "Cluster Bean (Guar)",
        "apy_name": "Guar seed",
        "fert_proxy": "Wheat",
        "family": "pulse",
        "n_fixing": True,
        "seasons": ["kharif"],
        "npk_tendency": "low_n_high_p",
        "regions_exclude": [],
        "temp_range": (25, 40),
        "ph_range": (7.0, 8.5),
        "water_mm": (200, 400),
    },
    "french bean": {
        "display": "French Bean",
        "apy_name": "Peas & beans (Pulses)",
        "fert_proxy": "Wheat",
        "family": "pulse",
        "n_fixing": True,
        "seasons": ["kharif", "rabi"],
        "npk_tendency": "low_n_high_p",
        "regions_exclude": [
            "Rajasthan",
            "Gujarat",
            "Punjab",
            "Haryana",
            "Uttar Pradesh",
            "Madhya Pradesh",
            "Bihar",
            "Andhra Pradesh",
            "Telangana",
        ],
        "temp_range": (15, 25),
        "ph_range": (6.0, 7.5),
        "water_mm": (400, 600),
    },
    "vegetable cowpea": {
        "display": "Veg. Cowpea",
        "apy_name": "Cowpea(Lobia)",
        "fert_proxy": "Wheat",
        "family": "pulse",
        "n_fixing": True,
        "seasons": ["kharif"],
        "npk_tendency": "low_n_high_p",
        "regions_exclude": [],
        "temp_range": (20, 35),
        "ph_range": (5.5, 7.5),
        "water_mm": (300, 600),
    },
    "cotton": {
        "display": "Cotton",
        "apy_name": "Cotton(lint)",
        "fert_proxy": "Cotton",
        "family": "fiber",
        "n_fixing": False,
        "seasons": ["kharif"],
        "npk_tendency": "high_n",
        "regions_exclude": [
            "Arunachal Pradesh",
            "Himachal Pradesh",
            "Uttarakhand",
            "Jammu And Kashmir",
            "Sikkim",
        ],
        "temp_range": (25, 40),
        "ph_range": (5.8, 8.0),
        "water_mm": (700, 1300),
    },
    "jute": {
        "display": "Jute",
        "apy_name": "Jute",
        "fert_proxy": "Rice",
        "family": "fiber",
        "n_fixing": False,
        "seasons": ["kharif"],
        "npk_tendency": "high_n",
        "regions_exclude": [
            "Maharashtra",
            "Gujarat",
            "Rajasthan",
            "Punjab",
            "Haryana",
            "Madhya Pradesh",
            "Karnataka",
            "Andhra Pradesh",
            "Telangana",
            "Tamil Nadu",
            "Kerala",
            "Himachal Pradesh",
            "Jammu And Kashmir",
        ],
        "temp_range": (25, 38),
        "ph_range": (6.0, 7.5),
        "water_mm": (1200, 2000),
    },
    "sugarcane": {
        "display": "Sugarcane",
        "apy_name": "Sugarcane",
        "fert_proxy": "Sugarcane",
        "family": "cash_crop",
        "n_fixing": False,
        "seasons": ["kharif", "whole year"],
        "npk_tendency": "high_n_high_k",
        "regions_exclude": [
            "Rajasthan",
            "Himachal Pradesh",
            "Arunachal Pradesh",
            "Jammu And Kashmir",
        ],
        "temp_range": (20, 38),
        "ph_range": (6.0, 7.5),
        "water_mm": (1500, 2500),
    },
    "tomato": {
        "display": "Tomato",
        "apy_name": None,
        "fert_proxy": "Tomato",
        "family": "vegetable",
        "n_fixing": False,
        "seasons": ["kharif", "rabi", "zaid"],
        "npk_tendency": "balanced",
        "regions_exclude": [],
        "temp_range": (18, 30),
        "ph_range": (6.0, 7.0),
        "water_mm": (400, 700),
    },
    "onion": {
        "display": "Onion",
        "apy_name": "Onion",
        "fert_proxy": "Tomato",
        "family": "vegetable",
        "n_fixing": False,
        "seasons": ["rabi", "kharif"],
        "npk_tendency": "balanced",
        "regions_exclude": [],
        "temp_range": (15, 30),
        "ph_range": (6.0, 7.5),
        "water_mm": (350, 600),
    },
    "small onion": {
        "display": "Small Onion",
        "apy_name": "Onion",
        "fert_proxy": "Tomato",
        "family": "vegetable",
        "n_fixing": False,
        "seasons": ["rabi"],
        "npk_tendency": "balanced",
        "regions_exclude": [
            "Punjab",
            "Haryana",
            "Uttar Pradesh",
            "Bihar",
            "Madhya Pradesh",
            "Rajasthan",
            "Uttarakhand",
            "Himachal Pradesh",
            "Jammu And Kashmir",
            "West Bengal",
            "Jharkhand",
        ],
        "temp_range": (15, 28),
        "ph_range": (6.0, 7.0),
        "water_mm": (300, 500),
    },
    "chillies": {
        "display": "Chillies",
        "apy_name": "Dry chillies",
        "fert_proxy": "Tomato",
        "family": "vegetable",
        "n_fixing": False,
        "seasons": ["kharif", "rabi"],
        "npk_tendency": "balanced",
        "regions_exclude": [],
        "temp_range": (20, 30),
        "ph_range": (6.0, 7.0),
        "water_mm": (400, 700),
    },
    "brinjal": {
        "display": "Brinjal",
        "apy_name": None,
        "fert_proxy": "Tomato",
        "family": "vegetable",
        "n_fixing": False,
        "seasons": ["kharif", "rabi"],
        "npk_tendency": "balanced",
        "regions_exclude": [],
        "temp_range": (20, 35),
        "ph_range": (5.5, 7.0),
        "water_mm": (400, 700),
    },
    "bhendi": {
        "display": "Bhendi (Okra)",
        "apy_name": None,
        "fert_proxy": "Tomato",
        "family": "vegetable",
        "n_fixing": False,
        "seasons": ["kharif"],
        "npk_tendency": "balanced",
        "regions_exclude": [],
        "temp_range": (24, 35),
        "ph_range": (6.0, 7.5),
        "water_mm": (400, 700),
    },
    "capsicum": {
        "display": "Capsicum",
        "apy_name": None,
        "fert_proxy": "Tomato",
        "family": "vegetable",
        "n_fixing": False,
        "seasons": ["rabi"],
        "npk_tendency": "balanced",
        "regions_exclude": [],
        "temp_range": (18, 28),
        "ph_range": (6.0, 7.0),
        "water_mm": (400, 600),
    },
    "cauliflower": {
        "display": "Cauliflower",
        "apy_name": None,
        "fert_proxy": "Tomato",
        "family": "vegetable",
        "n_fixing": False,
        "seasons": ["rabi"],
        "npk_tendency": "high_n",
        "regions_exclude": [],
        "temp_range": (15, 25),
        "ph_range": (6.0, 7.0),
        "water_mm": (350, 600),
    },
    "cabbage": {
        "display": "Cabbage",
        "apy_name": None,
        "fert_proxy": "Tomato",
        "family": "vegetable",
        "n_fixing": False,
        "seasons": ["rabi"],
        "npk_tendency": "high_n",
        "regions_exclude": [],
        "temp_range": (15, 25),
        "ph_range": (6.0, 7.0),
        "water_mm": (350, 600),
    },
    "carrot": {
        "display": "Carrot",
        "apy_name": None,
        "fert_proxy": "Tomato",
        "family": "vegetable",
        "n_fixing": False,
        "seasons": ["rabi"],
        "npk_tendency": "high_k",
        "regions_exclude": [],
        "temp_range": (15, 25),
        "ph_range": (6.0, 7.0),
        "water_mm": (350, 600),
    },
    "radish": {
        "display": "Radish",
        "apy_name": None,
        "fert_proxy": "Tomato",
        "family": "vegetable",
        "n_fixing": False,
        "seasons": ["rabi"],
        "npk_tendency": "balanced",
        "regions_exclude": [],
        "temp_range": (10, 22),
        "ph_range": (5.5, 7.0),
        "water_mm": (300, 500),
    },
    "pumpkin": {
        "display": "Pumpkin",
        "apy_name": None,
        "fert_proxy": "Tomato",
        "family": "vegetable",
        "n_fixing": False,
        "seasons": ["kharif"],
        "npk_tendency": "balanced",
        "regions_exclude": [],
        "temp_range": (20, 35),
        "ph_range": (6.0, 7.5),
        "water_mm": (400, 700),
    },
    "bitter gourd": {
        "display": "Bitter Gourd",
        "apy_name": None,
        "fert_proxy": "Tomato",
        "family": "vegetable",
        "n_fixing": False,
        "seasons": ["kharif", "zaid"],
        "npk_tendency": "balanced",
        "regions_exclude": [],
        "temp_range": (24, 38),
        "ph_range": (6.0, 7.0),
        "water_mm": (400, 700),
    },
    "bottle gourd": {
        "display": "Bottle Gourd",
        "apy_name": None,
        "fert_proxy": "Tomato",
        "family": "vegetable",
        "n_fixing": False,
        "seasons": ["kharif", "zaid"],
        "npk_tendency": "balanced",
        "regions_exclude": [],
        "temp_range": (24, 38),
        "ph_range": (6.0, 7.5),
        "water_mm": (400, 700),
    },
    "cucumber": {
        "display": "Cucumber",
        "apy_name": None,
        "fert_proxy": "Tomato",
        "family": "vegetable",
        "n_fixing": False,
        "seasons": ["zaid"],
        "npk_tendency": "balanced",
        "regions_exclude": [],
        "temp_range": (20, 35),
        "ph_range": (5.5, 7.0),
        "water_mm": (400, 600),
    },
    "watermelon": {
        "display": "Watermelon",
        "apy_name": None,
        "fert_proxy": "Tomato",
        "family": "vegetable",
        "n_fixing": False,
        "seasons": ["zaid"],
        "npk_tendency": "balanced",
        "regions_exclude": [],
        "temp_range": (24, 40),
        "ph_range": (6.0, 7.0),
        "water_mm": (400, 700),
    },
    "muskmelon": {
        "display": "Muskmelon",
        "apy_name": None,
        "fert_proxy": "Tomato",
        "family": "vegetable",
        "n_fixing": False,
        "seasons": ["zaid"],
        "npk_tendency": "balanced",
        "regions_exclude": [],
        "temp_range": (24, 38),
        "ph_range": (6.0, 7.5),
        "water_mm": (400, 600),
    },
    "ash gourd": {
        "display": "Ash Gourd",
        "apy_name": None,
        "fert_proxy": "Tomato",
        "family": "vegetable",
        "n_fixing": False,
        "seasons": ["kharif"],
        "npk_tendency": "balanced",
        "regions_exclude": [],
        "temp_range": (22, 35),
        "ph_range": (6.0, 7.5),
        "water_mm": (400, 700),
    },
    "beetroot": {
        "display": "Beetroot",
        "apy_name": None,
        "fert_proxy": "Tomato",
        "family": "vegetable",
        "n_fixing": False,
        "seasons": ["rabi"],
        "npk_tendency": "high_k",
        "regions_exclude": [],
        "temp_range": (15, 25),
        "ph_range": (6.0, 7.5),
        "water_mm": (350, 600),
    },
    "chowchow": {
        "display": "Chow Chow",
        "apy_name": None,
        "fert_proxy": "Tomato",
        "family": "vegetable",
        "n_fixing": False,
        "seasons": ["kharif"],
        "npk_tendency": "balanced",
        "regions_exclude": [],
        "temp_range": (18, 28),
        "ph_range": (6.0, 7.0),
        "water_mm": (500, 900),
    },
    "tinda": {
        "display": "Tinda",
        "apy_name": None,
        "fert_proxy": "Tomato",
        "family": "vegetable",
        "n_fixing": False,
        "seasons": ["zaid"],
        "npk_tendency": "balanced",
        "regions_exclude": [],
        "temp_range": (25, 40),
        "ph_range": (6.0, 7.5),
        "water_mm": (350, 600),
    },
    "ribbed gourd": {
        "display": "Ribbed Gourd",
        "apy_name": None,
        "fert_proxy": "Tomato",
        "family": "vegetable",
        "n_fixing": False,
        "seasons": ["kharif"],
        "npk_tendency": "balanced",
        "regions_exclude": [],
        "temp_range": (24, 38),
        "ph_range": (6.0, 7.5),
        "water_mm": (400, 700),
    },
    "snake gourd": {
        "display": "Snake Gourd",
        "apy_name": None,
        "fert_proxy": "Tomato",
        "family": "vegetable",
        "n_fixing": False,
        "seasons": ["kharif"],
        "npk_tendency": "balanced",
        "regions_exclude": [],
        "temp_range": (25, 38),
        "ph_range": (6.0, 7.5),
        "water_mm": (400, 700),
    },
    "annual moringa": {
        "display": "Moringa (Drumstick)",
        "apy_name": None,
        "fert_proxy": "Tomato",
        "family": "vegetable",
        "n_fixing": False,
        "seasons": ["kharif"],
        "npk_tendency": "balanced",
        "regions_exclude": [],
        "temp_range": (25, 40),
        "ph_range": (6.2, 7.0),
        "water_mm": (300, 600),
    },
    "potato": {
        "display": "Potato",
        "apy_name": "Potato",
        "fert_proxy": "Potato",
        "family": "tuber",
        "n_fixing": False,
        "seasons": ["rabi"],
        "npk_tendency": "high_k",
        "regions_exclude": [],
        "temp_range": (15, 25),
        "ph_range": (4.8, 5.5),
        "water_mm": (500, 700),
    },
    "sweet potato": {
        "display": "Sweet Potato",
        "apy_name": "Sweet potato",
        "fert_proxy": "Potato",
        "family": "tuber",
        "n_fixing": False,
        "seasons": ["kharif", "rabi"],
        "npk_tendency": "high_k",
        "regions_exclude": [],
        "temp_range": (20, 35),
        "ph_range": (5.5, 6.5),
        "water_mm": (800, 1500),
    },
    "tapoica": {
        "display": "Tapioca",
        "apy_name": "Tapioca",
        "fert_proxy": "Potato",
        "family": "tuber",
        "n_fixing": False,
        "seasons": ["kharif"],
        "npk_tendency": "high_k",
        "regions_exclude": [
            "Punjab",
            "Haryana",
            "Rajasthan",
            "Madhya Pradesh",
            "Uttar Pradesh",
            "Bihar",
        ],
        "temp_range": (25, 35),
        "ph_range": (5.5, 7.0),
        "water_mm": (1000, 1500),
    },
    "elephant foot yam": {
        "display": "Elephant Foot Yam",
        "apy_name": None,
        "fert_proxy": "Potato",
        "family": "tuber",
        "n_fixing": False,
        "seasons": ["kharif"],
        "npk_tendency": "high_k",
        "regions_exclude": [],
        "temp_range": (25, 38),
        "ph_range": (5.5, 7.0),
        "water_mm": (600, 1200),
    },
    "sugarbeet": {
        "display": "Sugar Beet",
        "apy_name": None,
        "fert_proxy": "Sugarcane",
        "family": "cash_crop",
        "n_fixing": False,
        "seasons": ["rabi"],
        "npk_tendency": "high_n_high_k",
        "regions_exclude": [],
        "temp_range": (15, 25),
        "ph_range": (6.5, 8.0),
        "water_mm": (450, 700),
    },
}


ALIAS_MAP = {}
for canonical, meta in CROP_REGISTRY.items():
    ALIAS_MAP[canonical.lower()] = canonical
    ALIAS_MAP[meta["display"].lower()] = canonical
    if meta.get("apy_name"):
        ALIAS_MAP[meta["apy_name"].lower()] = canonical

print(f"✅ CROP_REGISTRY: {len(CROP_REGISTRY)} crops")
print(f"✅ ALIAS_MAP    : {len(ALIAS_MAP)} entries")


DISTRICT_TEMP_OVERRIDE = {
    ("SHIMLA", "kharif"): 18.0,
    ("SHIMLA", "rabi"): 8.0,
    ("MANALI", "kharif"): 15.0,
    ("MANALI", "rabi"): 2.0,
    ("KULLU", "kharif"): 20.0,
    ("KULLU", "rabi"): 10.0,
    ("KANGRA", "kharif"): 22.0,
    ("KANGRA", "rabi"): 12.0,
    ("MANDI", "kharif"): 20.0,
    ("DEHRADUN", "kharif"): 26.0,
    ("DEHRADUN", "rabi"): 14.0,
    ("UTTARKASHI", "kharif"): 18.0,
    ("CHAMOLI", "kharif"): 17.0,
    ("DARJEELING", "kharif"): 17.0,
    ("DARJEELING", "rabi"): 8.0,
    ("KALIMPONG", "kharif"): 18.0,
    ("SRINAGAR", "kharif"): 22.0,
    ("SRINAGAR", "rabi"): 4.0,
    ("ANANTNAG", "kharif"): 20.0,
    ("ANANTNAG", "rabi"): 3.0,
    ("BARAMULLA", "kharif"): 20.0,
    ("LEH", "kharif"): 14.0,
    ("LEH", "rabi"): -5.0,
    ("KARGIL", "kharif"): 12.0,
    ("MUNNAR", "kharif"): 20.0,
    ("MUNNAR", "rabi"): 16.0,
    ("OOTY", "kharif"): 18.0,
    ("NILGIRIS", "kharif"): 18.0,
    ("KODAGU", "kharif"): 22.0,
    ("CHIKMAGALUR", "kharif"): 22.0,
    ("SHILLONG", "kharif"): 20.0,
    ("EAST KHASI HILLS", "kharif"): 20.0,
    ("AIZAWL", "kharif"): 22.0,
    ("BIKANER", "kharif"): 38.0,
    ("BIKANER", "zaid"): 42.0,
    ("JAISALMER", "kharif"): 40.0,
    ("BARMER", "kharif"): 39.0,
    ("JODHPUR", "kharif"): 36.0,
    ("PALI", "kharif"): 36.0,
    ("ERNAKULAM", "rabi"): 28.0,
    ("THIRUVANANTHAPURAM", "rabi"): 29.0,
    ("KOZHIKODE", "rabi"): 28.0,
    ("THRISSUR", "rabi"): 28.0,
    ("CHENNAI", "rabi"): 27.0,
    ("KANCHEEPURAM", "rabi"): 27.0,
    ("CUDDALORE", "rabi"): 27.0,
    ("VISAKHAPATNAM", "rabi"): 26.0,
}


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
    """Return realistic temperature (°C) for given location + season."""
    key = (district.upper().strip(), season.lower().strip())
    if key in DISTRICT_TEMP_OVERRIDE:
        return DISTRICT_TEMP_OVERRIDE[key]
    base = _DEFAULT_TEMP.get(season.lower(), 27.0)
    delta = _STATE_SEASON_CORRECTION.get(
        (state.lower().strip(), season.lower().strip()), 0
    )
    return base + delta


print("\n  Building district profiles from APY...")

profile_grp = (
    apy_clean.groupby(["State", "District", "Crop", "Season"])
    .agg(
        n_years=("Crop_Year", "nunique"),
        med_yield=("Yield", "median"),
        p10_yield=("Yield", lambda x: x.quantile(0.10)),
        p90_yield=("Yield", lambda x: x.quantile(0.90)),
        mean_area=("Area", "mean"),
    )
    .reset_index()
)

state_grp = (
    apy_clean.groupby(["State", "Crop"])
    .agg(
        n_years=("Crop_Year", "nunique"),
        med_yield=("Yield", "median"),
        p10_yield=("Yield", lambda x: x.quantile(0.10)),
        p90_yield=("Yield", lambda x: x.quantile(0.90)),
    )
    .reset_index()
)

national_grp = (
    apy_clean.groupby("Crop")
    .agg(
        n_years=("Crop_Year", "nunique"),
        med_yield=("Yield", "median"),
        p10_yield=("Yield", lambda x: x.quantile(0.10)),
        p90_yield=("Yield", lambda x: x.quantile(0.90)),
    )
    .reset_index()
)

valid_profiles = profile_grp[
    (profile_grp["n_years"] >= 3)
    & (profile_grp["med_yield"] > 0.05)
    & (profile_grp["mean_area"] > 10)
].copy()

print(f"  Valid district-crop-season profiles: {len(valid_profiles):,}")
print(f"  Districts with data                : {valid_profiles['District'].nunique()}")


DISTRICT_PROFILES = {}
for _, row in valid_profiles.iterrows():
    dist = row["District"].upper().strip()
    crop = row["Crop"]
    DISTRICT_PROFILES.setdefault(dist, {}).setdefault(crop, {})
    DISTRICT_PROFILES[dist][crop][row["Season"].lower().strip()] = {
        "med": round(row["med_yield"], 3),
        "p10": round(row["p10_yield"], 3),
        "p90": round(row["p90_yield"], 3),
        "n": int(row["n_years"]),
        "area": round(row["mean_area"], 0),
    }

STATE_PROFILES = {}
for _, row in state_grp.iterrows():
    STATE_PROFILES.setdefault(row["State"].strip(), {})[row["Crop"]] = {
        "med": round(row["med_yield"], 3),
        "p10": round(row["p10_yield"], 3),
        "p90": round(row["p90_yield"], 3),
        "n": int(row["n_years"]),
    }

NATIONAL_PROFILES = {}
for _, row in national_grp.iterrows():
    NATIONAL_PROFILES[row["Crop"]] = {
        "med": round(row["med_yield"], 3),
        "p10": round(row["p10_yield"], 3),
        "p90": round(row["p90_yield"], 3),
        "n": int(row["n_years"]),
    }

print(f"  DISTRICT_PROFILES : {len(DISTRICT_PROFILES)} districts")
print(f"  STATE_PROFILES    : {len(STATE_PROFILES)} states")
print(f"  NATIONAL_PROFILES : {len(NATIONAL_PROFILES)} crops")

DISTRICT_CROP_SET = {d: set(crops.keys()) for d, crops in DISTRICT_PROFILES.items()}


family_yields = {}
for canonical, meta in CROP_REGISTRY.items():
    apy_name = meta.get("apy_name")
    if apy_name and apy_name in NATIONAL_PROFILES:
        fam = meta["family"]
        family_yields.setdefault(fam, []).append(NATIONAL_PROFILES[apy_name]["med"])

FAMILY_YIELD_MAX = {f: max(v) for f, v in family_yields.items()}
FAMILY_YIELD_MIN = {f: min(v) for f, v in family_yields.items()}
print("\n  Family yield norms:")
for fam, mx in sorted(FAMILY_YIELD_MAX.items()):
    print(f"    {fam:<15} {FAMILY_YIELD_MIN.get(fam,0):.2f} – {mx:.2f} t/ha")


rain = rain_df.copy()
rain["DISTRICT_CLEAN"] = rain["DISTRICT"].str.upper().str.strip()
rain["STATE_NORM"] = (
    rain["STATE_UT_NAME"]
    .str.title()
    .str.strip()
    .replace(
        {
            "Chatisgarh": "Chhattisgarh",
            "Uttaranchal": "Uttarakhand",
            "Pondicherry": "Puducherry",
            "Orissa": "Odisha",
            "Himachal": "Himachal Pradesh",
        }
    )
)


def lookup_rainfall(district, state=None):
    dist_u = district.strip().upper()
    q = rain["DISTRICT_CLEAN"] == dist_u
    rows = (
        rain[q & (rain["STATE_NORM"].str.lower() == state.strip().lower())]
        if state
        else rain[q]
    )
    if rows.empty:
        rows = rain[rain["DISTRICT_CLEAN"].str.contains(dist_u, regex=False)]
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
        "monthly": {
            m: float(r[m])
            for m in [
                "JAN",
                "FEB",
                "MAR",
                "APR",
                "MAY",
                "JUN",
                "JUL",
                "AUG",
                "SEP",
                "OCT",
                "NOV",
                "DEC",
            ]
        },
    }


def temperature_filter(canonical, temp_c):
    """
    FIX: Uses explicit temp_range tuple instead of 3-bucket hot/cool/moderate.
    Allows ±2°C tolerance at boundaries (stress zone).
    Returns True if crop can grow at temp_c, False if hard-excluded.
    """
    meta = CROP_REGISTRY.get(canonical, {})
    lo, hi = meta.get("temp_range", (0, 50))
    return (lo - 2) <= temp_c <= (hi + 2)


def ph_filter(canonical, soil_ph):
    """
    FIX: ph_range was completely missing from v3.
    Returns True if pH is within acceptable range (±0.3 tolerance).
    """
    meta = CROP_REGISTRY.get(canonical, {})
    lo, hi = meta.get("ph_range", (4.0, 9.0))
    return (lo - 0.3) <= soil_ph <= (hi + 0.3)


def water_filter(canonical, season_rain_mm):
    """
    FIX: Uses explicit water_mm range instead of low/medium/high bucket.
    Returns True if seasonal rainfall can support crop's water requirement.
    Irrigation source check is done separately in irrigation_score.
    """
    meta = CROP_REGISTRY.get(canonical, {})
    lo, hi = meta.get("water_mm", (100, 5000))

    return season_rain_mm >= (lo * 0.4)


_W = dict(
    suitability=0.30,
    yield_=0.25,
    region=0.15,
    irrigation=0.10,
    season=0.10,
    presence=0.10,
)
_W_SUM = sum(_W.values())
assert abs(_W_SUM - 1.0) < 1e-9, f"Weights must sum to 1.0, got {_W_SUM}"

W_SUITABILITY = _W["suitability"]
W_YIELD = _W["yield_"]
W_REGION = _W["region"]
W_IRRIGATION = _W["irrigation"]
W_SEASON = _W["season"]
W_PRESENCE = _W["presence"]


STATE_NORM_MAP = {
    "Andaman And Nicobar Island": "Andaman and Nicobar Islands",
    "The Dadra And Nagar Haveli": "Dadra and Nagar Haveli",
    "Chatisgarh": "Chhattisgarh",
    "Uttaranchal": "Uttarakhand",
    "Pondicherry": "Puducherry",
    "Orissa": "Odisha",
    "Himachal": "Himachal Pradesh",
}
SEASON_NORM_MAP = {"summer": "zaid", "winter": "rabi"}
IRRIGATION_NORM_MAP = {
    "rain water": "rainfed",
    "rain-water": "rainfed",
    "rainfed": "rainfed",
    "rainwater": "rainfed",
    "drip": "drip",
    "sprinkler": "sprinkler",
    "canal": "canal",
    "borewell": "borewell",
    "bore well": "borewell",
    "tube well": "borewell",
}
SOIL_NORM_MAP = {
    "black soil": "black cotton soil",
    "black cotton": "black cotton soil",
    "red soil": "red soil",
    "red": "red soil",
    "alluvial": "alluvial soil",
    "alluvial soil": "alluvial soil",
    "sandy loam": "sandy loam soil",
    "sandy loam soil": "sandy loam soil",
    "loamy": "loamy soil",
    "loamy soil": "loamy soil",
    "clay loam": "clay loam soil",
    "clay loam soil": "clay loam soil",
    "laterite": "laterite soil",
    "laterite soil": "laterite soil",
}
REGION_CROP_MAP = {
    "rajasthan": ["pearl millet", "cluster bean", "castor", "gingely"],
    "gujarat": ["pearl millet", "groundnut", "castor", "cotton", "bengalgram"],
    "maharashtra": ["soyabean", "cotton", "sorghum", "bengalgram", "groundnut"],
    "karnataka": ["ragi", "groundnut", "sorghum", "maize", "cotton"],
    "tamil nadu": ["rice", "cotton", "groundnut", "ragi", "sorghum"],
}


to_save = {
    "CROP_REGISTRY": CROP_REGISTRY,
    "ALIAS_MAP": ALIAS_MAP,
    "DISTRICT_PROFILES": DISTRICT_PROFILES,
    "STATE_PROFILES": STATE_PROFILES,
    "NATIONAL_PROFILES": NATIONAL_PROFILES,
    "DISTRICT_CROP_SET": DISTRICT_CROP_SET,
    "FAMILY_YIELD_MAX": FAMILY_YIELD_MAX,
    "FAMILY_YIELD_MIN": FAMILY_YIELD_MIN,
    "STATE_NORM_MAP": STATE_NORM_MAP,
    "SEASON_NORM_MAP": SEASON_NORM_MAP,
    "IRRIGATION_NORM_MAP": IRRIGATION_NORM_MAP,
    "SOIL_NORM_MAP": SOIL_NORM_MAP,
    "REGION_CROP_MAP": REGION_CROP_MAP,
    "DISTRICT_TEMP_OVERRIDE": DISTRICT_TEMP_OVERRIDE,
    "SCORE_WEIGHTS": {
        "W_SUITABILITY": W_SUITABILITY,
        "W_YIELD": W_YIELD,
        "W_REGION": W_REGION,
        "W_IRRIGATION": W_IRRIGATION,
        "W_SEASON": W_SEASON,
        "W_PRESENCE": W_PRESENCE,
    },
}
for name, obj in to_save.items():
    with open(ENCODER_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(obj, f)
    print(f"  ✅ {name}.pkl")

with open(ENCODER_DIR / "rain_lookup_df.pkl", "wb") as f:
    pickle.dump(rain, f)
print("  ✅ rain_lookup_df.pkl")

print(f"\n✅ All knowledge saved to {ENCODER_DIR}")


print("\n=== FILTER SELF-TESTS ===")
tests = [
    ("maize", 18, "Shimla kharif", True, "was WRONG in v3 (blocked)"),
    ("maize", 38, "Bikaner kharif", False, "was WRONG in v3 (allowed)"),
    ("cotton", 22, "Shimla kharif", False, "correctly blocked"),
    ("wheat", 28, "Ernakulam rabi", False, "was WRONG in v3 (allowed)"),
    ("soyabean", 32, "hot kharif", False, "was WRONG in v3 (allowed)"),
    ("bengalgram", 28, "warm plains", False, "was WRONG in v3 (allowed)"),
    ("peas", 15, "Shimla rabi", True, "correctly allowed"),
    ("jute", 12, "Shimla kharif", False, "was WRONG in v3 (allowed)"),
]
pass_count = 0
for crop, temp, label, expected, note in tests:
    result = temperature_filter(crop, temp)
    status = "✅" if result == expected else "❌"
    if result == expected:
        pass_count += 1
    print(f"  {status}  {crop:<15} {temp}°C  {label:<22} → {str(result):<5}  {note}")
print(f"\n  {pass_count}/{len(tests)} tests passed")

print("\n✅ NB2 Knowledge Layer complete.")
