from fastapi import FastAPI
from pydantic import BaseModel
import joblib, yaml, os
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

RISK_MODEL_PATH = os.getenv("RISK_MODEL_PATH", "risk_model.pkl")
risk_obj = joblib.load(RISK_MODEL_PATH)

risk_clf = risk_obj["risk_model"]
risk_features = risk_obj["feature_names"]
risk_classes = risk_obj["class_names"]
MODEL_VER = risk_obj.get("model_version", "risk_v1")

rules = yaml.safe_load(open("rules.yml"))

ACTIVITIES = [
    "walking",
    "prenatal_yoga",
    "swimming",
    "stationary_cycling",
    "pelvic_floor",
    "low_impact_aerobic",
    "pilates_prenatal",
    "strength_light_resistance",
    "stretching_gentle",
    "breathing_exercise",
    "tai_chi",
    "aqua_cycling",
    "balance_training",
    "resistance_band_workout",
    "seated_exercises",
    "water_walking",
    "light_dance",
    "treadmill_walk",
    "prenatal_pilates_ball",
    "modified_plank",
    "chair_yoga",
    "foam_rolling",
    "bodyweight_squat",
    "arm_circles",
    "elliptical_light",
    "side_leg_raise",
    "cat_cow_pose",
    "wall_push_up",
    "gentle_stairs_climb",
    "yoga_nidra",
    "padel"
]

app = FastAPI(
    title="Prenatal Exercise Recommender",
    version=MODEL_VER
)

@app.get("/healthz")
def health():
    return {"ok": True, "model": MODEL_VER}

class PredictInput(BaseModel):
    age: int
    gestational_age_weeks: int
    bmi: float
    blood_pressure_systolic: float
    blood_pressure_diastolic: float
    blood_sugar: float
    body_temp: float = 36.5
    heart_rate: float

    previous_complications: bool = False
    preexisting_diabetes: bool = False
    gestational_diabetes: bool = False
    mental_health_issue: bool = False

    placenta_position_restriction: bool | None = None

    low_impact_pref: bool = False
    water_access: bool = False
    back_pain: bool = False


def apply_rules(xd, ranked, risk_map):
    final_list = ranked.copy()
    ctx = xd.copy()
    ctx.update(risk_map)

    for rule in rules.get("high_risk_rules", []):
        cond = rule.get("condition")
        if cond and eval(cond, {}, ctx):
            banned = rule.get("ban", [])
            final_list = [r for r in final_list if r["activity"] not in banned]

    for rule in rules.get("medical_risk_rules", []):
        cond = rule.get("condition")
        if cond and eval(cond, {}, ctx):
            banned = rule.get("ban", [])
            final_list = [r for r in final_list if r["activity"] not in banned]

    for rule in rules.get("preference_rules", []):
        cond = rule.get("condition")
        if cond and eval(cond, {}, ctx):
            for act in rule.get("boost", []):
                for item in final_list:
                    if item["activity"] == act:
                        item["score"] += 0.05

            for act in rule.get("lower", []):
                for item in final_list:
                    if item["activity"] == act:
                        item["score"] -= 0.05

    final_list = sorted(final_list, key=lambda x: x["score"], reverse=True)
    return final_list


@app.post("/predict")
def predict(payload: PredictInput):
    xd = payload.dict()

    alias_map = {
        "blood_pressure_systolic": "systolic_bp",
        "blood_pressure_diastolic": "diastolic_bp",
        "mental_health_issue": "mental_health",
        "systolic_bp": "systolic_bp",
        "diastolic_bp": "diastolic_bp",
        "mental_health": "mental_health",
    }

    norm = {}
    for k, v in xd.items():
        key = alias_map.get(k, k)
        norm[key] = v

    for b in ["previous_complications", "preexisting_diabetes", "gestational_diabetes", "mental_health"]:
        if b in norm:
            norm[b] = int(bool(norm[b]))

    X_input = []
    missing = []
    for col in risk_features:
        if col not in norm or norm[col] is None:
            missing.append(col)
        else:
            X_input.append(norm[col])

    if missing:
        raise ValueError(f"Missing field for model: {', '.join(missing)}")

    X_array = np.array([X_input], dtype=float)


    probs = risk_clf.predict_proba(X_array)[0]
    risk_map = {}
    for cls_name, p in zip(risk_classes, probs):
        risk_map[f"risk_{cls_name.lower()}"] = float(p)

    risk_low = risk_map.get("risk_low", 0.0)
    risk_med = risk_map.get("risk_medium", 0.0)
    risk_high = risk_map.get("risk_high", 0.0)

    ranked = []
    for act in ACTIVITIES:
        base_score = (
            (0.7 * risk_low) +
            (0.25 * risk_med) +
            (0.05 * risk_high)
        )
        ranked.append({"activity": act, "score": round(base_score, 4)})

    final_list = apply_rules(xd, ranked, risk_map)

    return {
        "model_version": MODEL_VER,
        "risk_probabilities": risk_map,
        "recommendations": final_list[:10],
        "all_ranked": final_list
    }
