import json
import csv
from datetime import datetime

import v3_NB4_Pipeline as pipeline


def run_pipeline(input_dict):
    norm = pipeline.normalize_input(input_dict)
    district = norm["district"]
    state = norm["state"]
    soil_type = norm["soil_type"]
    season = norm["season"]
    irrigation = norm["irrigation"]
    n_val = norm["N"]
    p_val = norm["P"]
    k_val = norm["K"]
    soil_ph = norm["pH"]

    rain = pipeline.lookup_rainfall(district, state)
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

    temp_mid = pipeline.get_district_temp(district, state, season)
    hum_mid = {"kharif": 75.0, "rabi": 55.0, "zaid": 40.0}.get(season, 60.0)

    ml_scores = pipeline.engine_suitability(
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
    yield_candidates = pipeline.get_yield_top_crops(district, top_n=5)
    freq_candidates = pipeline.get_freq_top_crops(district, top_n=5)
    region_candidates = pipeline.get_region_crops(state)
    candidate_crops = (
        {pipeline.resolve(c) for c in ml_candidates}
        | yield_candidates
        | freq_candidates
        | region_candidates
    )

    scored = []
    for canonical in candidate_crops:
        if not pipeline.is_crop_in_season(canonical, season):
            continue
        ml_prob = float(ml_scores.get(canonical, 0.05))
        final, components, yield_data, yield_level = pipeline.engine_score(
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
        meta = pipeline.get_meta(canonical) or {}
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

    if top_crops:
        best = top_crops[0]
        best_canonical = best["canonical"]
    else:
        best = None
        best_canonical = None

    if best_canonical:
        fert_name, fert_reason, ml_probs = pipeline.engine_fertilizer(
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
    else:
        fert_name, fert_reason, ml_probs = None, None, None

    if best_canonical:
        ml_yield = pipeline.engine_yield_ml(best_canonical, state, district, season)
    else:
        ml_yield = None

    if best and best.get("yield_data"):
        primary_yield = best["yield_data"]["med"]
        yield_level = best.get("yield_level")
    else:
        primary_yield = ml_yield
        yield_level = "ml" if ml_yield is not None else None

    return {
        "input": {
            "district": district,
            "state": state,
            "soil_type": soil_type,
            "season": season,
            "irrigation": irrigation,
            "N": n_val,
            "P": p_val,
            "K": k_val,
            "pH": soil_ph,
        },
        "rain": {
            "annual": annual_rain,
            "season": season_rain,
        },
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
            "source": yield_level,
        },
    }


def summarize(results):
    print("\nDistrict | Top Crop | Score | Fertilizer")
    print("-" * 60)
    for r in results:
        if r.get("error"):
            print(f"{r['district']} | ERROR | - | {r['error']}")
            continue
        top = r.get("top_crops", [{}])[0]
        fert = r.get("fertilizer", {})
        score = top.get("score", None)
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "-"
        print(
            f"{r['input']['district']} | {top.get('name', '-')} | {score_str} | {fert.get('name', '-')}"
        )


def main():
    test_cases = [
        {
            "district": "Jaisalmer",
            "state": "Rajasthan",
            "soil_type": "sandy loam soil",
            "season": "kharif",
            "irrigation": "rainfed",
            "N": 40,
            "P": 20,
            "K": 20,
            "pH": 7.8,
        },
        {
            "district": "Barmer",
            "state": "Rajasthan",
            "soil_type": "sandy loam soil",
            "season": "kharif",
            "irrigation": "rainfed",
            "N": 35,
            "P": 18,
            "K": 22,
            "pH": 8.0,
        },
        {
            "district": "Alappuzha",
            "state": "Kerala",
            "soil_type": "alluvial soil",
            "season": "kharif",
            "irrigation": "rainfed",
            "N": 70,
            "P": 40,
            "K": 50,
            "pH": 6.0,
        },
        {
            "district": "Cooch Behar",
            "state": "West Bengal",
            "soil_type": "alluvial soil",
            "season": "kharif",
            "irrigation": "canal",
            "N": 65,
            "P": 35,
            "K": 45,
            "pH": 6.2,
        },
        {
            "district": "Nagpur",
            "state": "Maharashtra",
            "soil_type": "black cotton soil",
            "season": "kharif",
            "irrigation": "rainfed",
            "N": 60,
            "P": 30,
            "K": 30,
            "pH": 6.8,
        },
        {
            "district": "Bhopal",
            "state": "Madhya Pradesh",
            "soil_type": "black cotton soil",
            "season": "kharif",
            "irrigation": "sprinkler",
            "N": 55,
            "P": 28,
            "K": 32,
            "pH": 7.0,
        },
        {
            "district": "Meerut",
            "state": "Uttar Pradesh",
            "soil_type": "loamy soil",
            "season": "rabi",
            "irrigation": "canal",
            "N": 80,
            "P": 40,
            "K": 35,
            "pH": 6.8,
        },
        {
            "district": "Karnal",
            "state": "Haryana",
            "soil_type": "alluvial soil",
            "season": "rabi",
            "irrigation": "borewell",
            "N": 85,
            "P": 42,
            "K": 38,
            "pH": 7.2,
        },
        {
            "district": "Kullu",
            "state": "Himachal Pradesh",
            "soil_type": "loamy soil",
            "season": "rabi",
            "irrigation": "rainfed",
            "N": 50,
            "P": 30,
            "K": 30,
            "pH": 6.5,
        },
        {
            "district": "Chamoli",
            "state": "Uttarakhand",
            "soil_type": "loamy soil",
            "season": "rabi",
            "irrigation": "rainfed",
            "N": 45,
            "P": 25,
            "K": 25,
            "pH": 6.3,
        },
        {
            "district": "Coimbatore",
            "state": "Tamil Nadu",
            "soil_type": "red soil",
            "season": "kharif",
            "irrigation": "drip",
            "N": 55,
            "P": 35,
            "K": 40,
            "pH": 6.7,
        },
        {
            "district": "Guntur",
            "state": "Andhra Pradesh",
            "soil_type": "black cotton soil",
            "season": "kharif",
            "irrigation": "canal",
            "N": 65,
            "P": 38,
            "K": 45,
            "pH": 7.1,
        },
        {
            "district": "Patna",
            "state": "Bihar",
            "soil_type": "alluvial soil",
            "season": "kharif",
            "irrigation": "rainfed",
            "N": 70,
            "P": 40,
            "K": 45,
            "pH": 6.5,
        },
        {
            "district": "Sundargarh",
            "state": "Odisha",
            "soil_type": "red soil",
            "season": "kharif",
            "irrigation": "rainfed",
            "N": 60,
            "P": 30,
            "K": 35,
            "pH": 6.2,
        },
        {
            "district": "Ahmednagar",
            "state": "Maharashtra",
            "soil_type": "sandy loam soil",
            "season": "kharif",
            "irrigation": "rainfed",
            "N": 20,
            "P": 10,
            "K": 10,
            "pH": 7.5,
        },
    ]

    results = []

    for case in test_cases:
        try:
            result = run_pipeline(case)
            results.append(result)
        except Exception as exc:
            results.append(
                {
                    "district": case.get("district"),
                    "state": case.get("state"),
                    "error": str(exc),
                }
            )

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = f"test_results_{timestamp}.json"
    csv_path = f"test_results_{timestamp}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "district",
                "state",
                "top1_crop",
                "top1_score",
                "top2_crop",
                "top2_score",
                "top3_crop",
                "top3_score",
                "fertilizer",
                "yield_top_crop",
                "yield_value",
                "yield_source",
                "ml_confidence_top1",
                "presence_score_top1",
            ]
        )
        for r in results:
            if r.get("error"):
                writer.writerow(
                    [r.get("district"), r.get("state"), "ERROR", r.get("error")]
                )
                continue
            top = r.get("top_crops", [])
            top1 = top[0] if len(top) > 0 else {}
            top2 = top[1] if len(top) > 1 else {}
            top3 = top[2] if len(top) > 2 else {}
            writer.writerow(
                [
                    r["input"]["district"],
                    r["input"]["state"],
                    top1.get("name"),
                    top1.get("score"),
                    top2.get("name"),
                    top2.get("score"),
                    top3.get("name"),
                    top3.get("score"),
                    r.get("fertilizer", {}).get("name"),
                    r.get("yield", {}).get("top_crop"),
                    r.get("yield", {}).get("value"),
                    r.get("yield", {}).get("source"),
                    (top1.get("components", {}) or {}).get("ml_prob"),
                    (top1.get("components", {}) or {}).get("presence_score"),
                ]
            )

    summarize(results)
    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
