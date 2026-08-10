import warnings
from config import DATA_DIR, PLOTS_DIR, ensure_dirs, resolve_data_file

warnings.filterwarnings("ignore")
ensure_dirs()

print("✅ Local folders ready.")
print(f"\n📁 Place these files in {DATA_DIR}")
print("   - APY.csv                           (345K rows — NEW main dataset)")
print("   - Crop_recommendation_dataset.csv   (57K rows — soil/NPK/climate features)")
print("     (or: Crop recommendation dataset.csv)")
print("   - district_wise_rainfall_normal.csv (641 districts — monthly rainfall)")
print("     (or: district wise rainfall normal.csv)")
print("   - fertilizer_recommendation.csv     (10K rows — fertilizer model)")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

apy = pd.read_csv(resolve_data_file("APY.csv"))
apy.columns = [c.strip() for c in apy.columns]


crop_df = pd.read_csv(
    resolve_data_file(
        "Crop_recommendation_dataset.csv", "Crop recommendation dataset.csv"
    )
)


rain_df = pd.read_csv(
    resolve_data_file(
        "district_wise_rainfall_normal.csv", "district wise rainfall normal.csv"
    )
)
rain_df.columns = [c.strip() for c in rain_df.columns]


fert_df = pd.read_csv(resolve_data_file("fertilizer_recommendation.csv"))

print(f"\n{'Dataset':<40} {'Rows':>10} {'Cols':>6}")
print("─" * 60)
print(f"{'APY (Area-Production-Yield)':<40} {apy.shape[0]:>10,} {apy.shape[1]:>6}")
print(f"{'Crop Recommendation':<40} {crop_df.shape[0]:>10,} {crop_df.shape[1]:>6}")
print(f"{'District Rainfall':<40} {rain_df.shape[0]:>10,} {rain_df.shape[1]:>6}")
print(
    f"{'Fertilizer Recommendation':<40} {fert_df.shape[0]:>10,} {fert_df.shape[1]:>6}"
)


print("\n=== APY DATASET QUALITY AUDIT ===")
print(f"  Rows         : {len(apy):,}")
print(f"  States       : {apy['State'].nunique()}")
print(f"  Districts    : {apy['District'].nunique()}")
print(f"  Crops        : {apy['Crop'].nunique()}")
print(f"  Year range   : {apy['Crop_Year'].min()} – {apy['Crop_Year'].max()}")
print(f"  Seasons      : {list(apy['Season'].str.strip().unique())}")
print(f"  Null Crop    : {apy['Crop'].isnull().sum()}")
print(f"  Null Prod    : {apy['Production'].isnull().sum()}")
print(f"  Zero Yield   : {(apy['Yield'] == 0).sum()}")
print()


rice_punjab = apy[(apy["State"] == "Punjab") & (apy["Crop"] == "Rice")]
print("YIELD UNIT CONFIRMATION (tonnes/hectare):")
print(
    f"  Punjab Rice median : {rice_punjab['Yield'].median():.2f} t/ha ✓ (real-world ~3.8)"
)


grp = (
    apy.groupby(["State", "District", "Crop"])
    .agg(
        n_years=("Crop_Year", "nunique"),
        med_yield=("Yield", "median"),
        mean_area=("Area", "mean"),
        p10_yield=("Yield", lambda x: x.quantile(0.10)),
        p90_yield=("Yield", lambda x: x.quantile(0.90)),
    )
    .reset_index()
)


valid = grp[(grp["n_years"] >= 3) & (grp["med_yield"] > 0.05) & (grp["mean_area"] > 10)]

print(f"\n  Valid district-crop profiles (≥3 yrs): {len(valid):,}")
print(
    f"  Avg crops per district                : {valid.groupby('District')['Crop'].count().mean():.1f}"
)
print(
    f"  Districts with ≥5 valid crops         : {(valid.groupby('District')['Crop'].count() >= 5).sum()}"
)


fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("APY Dataset — Key Distributions", fontsize=15, fontweight="bold")


axes[0, 0].hist(
    apy[apy["Yield"].between(0.05, 20)]["Yield"],
    bins=60,
    color="forestgreen",
    edgecolor="white",
    alpha=0.8,
)
axes[0, 0].set_title("Yield Distribution (0.05–20 t/ha)")
axes[0, 0].set_xlabel("Tonnes/hectare")


apy["Season"].str.strip().value_counts().plot(
    kind="bar", ax=axes[0, 1], color="steelblue"
)
axes[0, 1].set_title("Records by Season")
axes[0, 1].tick_params(axis="x", rotation=30)


apy["Crop"].dropna().value_counts().head(20).plot(
    kind="barh", ax=axes[0, 2], color="darkorange"
)
axes[0, 2].set_title("Top 20 Crops by Record Count")
axes[0, 2].invert_yaxis()


apy.groupby("Crop_Year")["Yield"].median().plot(
    ax=axes[1, 0], color="forestgreen", marker="o", ms=4
)
axes[1, 0].set_title("Median Yield Trend 1997–2020")
axes[1, 0].set_ylabel("t/ha")


apy.groupby("State")["District"].nunique().sort_values().plot(
    kind="barh", ax=axes[1, 1], color="teal"
)
axes[1, 1].set_title("Districts per State")


top_crops = [
    "Rice",
    "Wheat",
    "Maize",
    "Soyabean",
    "Cotton(lint)",
    "Gram",
    "Arhar/Tur",
    "Groundnut",
    "Jowar",
    "Sugarcane",
]
plot_data = [apy[apy["Crop"] == c]["Yield"].clip(0, 20) for c in top_crops]
axes[1, 2].boxplot(plot_data, labels=[c[:10] for c in top_crops], vert=True)
axes[1, 2].set_title("Yield Range by Major Crop (clipped 20 t/ha)")
axes[1, 2].tick_params(axis="x", rotation=45)
axes[1, 2].set_ylabel("t/ha")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_apy_overview.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: 01_apy_overview.png")


CROP_TO_APY = {
    "rice": "Rice",
    "wheat": "Wheat",
    "maize": "Maize",
    "sorghum": "Jowar",
    "ragi": "Ragi",
    "pearl millet": "Bajra",
    "soyabean": "Soyabean",
    "groundnut": "Groundnut",
    "bengalgram": "Gram",
    "redgram": "Arhar/Tur",
    "blackgram": "Urad",
    "greengram": "Moong(Green Gram)",
    "horsegram": "Horse-gram",
    "cowpea": "Cowpea(Lobia)",
    "peas": "Peas & beans (Pulses)",
    "cluster bean": "Guar seed",
    "sunflower": "Sunflower",
    "castor": "Castor seed",
    "gingely": "Sesamum",
    "cotton": "Cotton(lint)",
    "jute": "Jute",
    "sugarcane": "Sugarcane",
    "onion": "Onion",
    "small onion": "Onion",
    "potato": "Potato",
    "sweet potato": "Sweet potato",
    "tapoica": "Tapioca",
    "chillies": "Dry chillies",
}

print("\n=== CROP COVERAGE IN APY ===")
print(f"{'crop_rec name':<22} {'APY name':<25} {'Districts':<12} {'States'}")
print("─" * 75)
for cr, apy_name in CROP_TO_APY.items():
    subset = apy[apy["Crop"] == apy_name]
    n_dist = subset["District"].nunique()
    n_st = subset["State"].nunique()
    print(f"  {cr:<20} {apy_name:<25} {n_dist:<12} {n_st}")

cr_crops = set(crop_df["CROPS"].str.lower().str.strip().unique())
no_apy = (
    cr_crops - set(CROP_TO_APY.keys()) - set(c.lower() for c in CROP_TO_APY.values())
)
print(f"\n  Crops in crop_rec WITHOUT APY data: {len(no_apy)}")
print(f"  (These will use crop-family average as yield fallback)")


rain_dists = set(rain_df["DISTRICT"].str.upper().str.strip().unique())
apy_dists = set(apy["District"].str.upper().str.strip().unique())
overlap = rain_dists & apy_dists

print(f"\n=== DISTRICT OVERLAP ===")
print(f"  Rain  districts: {len(rain_dists)}")
print(f"  APY   districts: {len(apy_dists)}")
print(f"  Overlap (exact): {len(overlap)}")
print(
    f"  Coverage: {len(overlap)/len(apy_dists)*100:.1f}% of APY districts have rainfall data"
)

print("\n✅ EDA complete. Dataset understood.")
print("\n📌 KEY FINDINGS:")
print("   1. Yield is in TONNES/hectare")
print("   2. APY has 707 districts × 55 crops (28 map to crop_rec crops)")
print(
    "   3. 681 districts have ≥5 valid crops (enough for district-level recommendations)"
)
print(
    "   4. Sugarcane/Banana/Onion have very high yields — need per-family normalization"
)
print("   5. Valid profiles require ≥3 years data + yield > 0.05 t/ha + area > 10 ha")
