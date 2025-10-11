from fastapi import FastAPI
from pydantic import BaseModel
import joblib, yaml, os

# =========================
# 1. Load model dari file .pkl
# Kenapa? Karena ML dilatih di train.py, dan backend hanya konsumsi model yg sudah jadi.
# =========================
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
MODEL_VER = os.getenv("MODEL_VER", "v1")

obj = joblib.load(MODEL_PATH)
pipe = obj["model"]      # Pipeline sklearn yg siap prediksi
LABELS = obj["labels"]   # daftar aktivitas yg dikenali model

# =========================
# 2. Load rules dari YAML
# Kenapa? Agar aturan medis / pengecualian bisa diedit tanpa ganti kode.
# =========================
rules = yaml.safe_load(open("rules.yml"))

# =========================
# 3. Inisialisasi FastAPI
# Versi bisa diambil dari environment (.env)
# =========================
app = FastAPI(
    title="Prenatal Exercise Recommender",
    version=MODEL_VER
)

# =========================
# 4. Endpoint health check
# Dipakai oleh Docker, k8s, atau monitoring lain
# =========================
@app.get("/healthz")
def health():
    return {"ok": True, "model": MODEL_VER}

# =========================
# 5. Definisi schema input (validasi)
# Kenapa? Supaya request tidak bebas dan pasti terstruktur
# =========================
class XIn(BaseModel):
    age: int
    gestational_age_weeks: int
    bmi: float
    pre_pregnancy_activity_level: int = 0
    hypertension: bool = False
    gestational_diabetes: bool = False
    placenta_previa: bool = False
    pre_eclampsia: bool = False
    low_impact_pref: bool = False
    water_access: bool = False
    back_pain: bool = False

# =========================
# 6. Fungsi bantu: trimester
# Biar rules bisa pakai kondisi kehamilan (1/2/3)
# =========================
def trimester(w):
    if w <= 13: return 1
    if w <= 27: return 2
    return 3

# =========================
# 7. Penerapan rules (filter, ban, prefer)
# Kenapa? ML tidak tahu soal kondisi medis. Ini layer proteksi.
# =========================
def apply_rules(x):
    allow = None
    ban = set()
    prefer = set()
    
    # Konteks yang bisa dibaca rules
    ctx = dict(x)
    ctx["trimester"] = trimester(x["gestational_age_weeks"])

    # absolute_risk: kalau kondisi parah → batasi ke "allow only"
    for r in rules.get("absolute_risk", []):
        if eval(r["if"], {}, ctx):
            allow = set(r["allow"])

    # relative_risk: ban / prefer
    for r in rules.get("relative_risk", []):
        if eval(r["if"], {}, ctx):
            ban |= set(r.get("ban", []))
            prefer |= set(r.get("prefer", []))

    return allow, ban, prefer

# =========================
# 8. Endpoint utama: /predict
# Flow:
#   - validasi input
#   - prediksi probabilitas
#   - apply rules
#   - ranking hasil
# =========================
@app.post("/predict")
def predict(x: XIn):
    xd = x.model_dump()
    
    # Convert dict → DataFrame (2D)
    import pandas as pd
    df_input = pd.DataFrame([xd])
    
    # Prediksi
    probs = pipe.predict_proba(df_input)[0]
    
    # Apply rules
    allow, ban, prefer = apply_rules(xd)

    ranked = []
    for label, prob in zip(LABELS, probs):
        if allow and label not in allow:
            continue
        if label in ban:
            continue
        bonus = 0.05 if label in prefer else 0.0
        ranked.append({
            "activity": label,
            "score": round(float(prob) + bonus, 4)
        })
    
    ranked.sort(key=lambda x: x["score"], reverse=True)
    
    return {
        "model_version": MODEL_VER,
        "recommendations": ranked[:10],
        "all_ranked": ranked
    }
