import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.multiclass import OneVsRestClassifier

import os

# Dummy label & fitur (bisa diganti nanti)
LABELS = [
    "walking", "prenatal_yoga", "swimming", "stationary_cycling",
    "pelvic_floor", "low_impact_aerobic", "pilates_prenatal",
    "strength_light_resistance", "stretching_gentle"
]

# Cek kalau belum ada dataset, bikin dummy dataframe
dummy_data_path = "data/train.parquet"
if not os.path.exists(dummy_data_path):
    import pandas as pd
    from numpy.random import default_rng

    rng = default_rng(1)
    N = 50  # kecil dulu untuk dummy
    df = pd.DataFrame({
        "age": rng.integers(18, 45, N),
        "gestational_age_weeks": rng.integers(4, 40, N),
        "bmi": rng.normal(25, 3, N).round(1),
        "pre_pregnancy_activity_level": rng.integers(0, 3, N),
        "hypertension": rng.random(N) < 0.1,
        "gestational_diabetes": rng.random(N) < 0.1,
        "placenta_previa": rng.random(N) < 0.02,
        "pre_eclampsia": rng.random(N) < 0.02,
        "low_impact_pref": rng.random(N) < 0.5,
        "water_access": rng.random(N) < 0.4,
        "back_pain": rng.random(N) < 0.3,
    })
    # label dummy: binary random
    for lab in LABELS:
        df[lab] = (rng.random(N) < 0.3).astype(int)

    df.to_parquet(dummy_data_path, index=False)
    print("✅ Dummy dataset dibuat di:", dummy_data_path)

# Training model sederhana
df = pd.read_parquet(dummy_data_path)
X = df.drop(columns=LABELS)
Y = df[LABELS].astype(int)

num_cols = ["age","gestational_age_weeks","bmi","pre_pregnancy_activity_level"]
cat_cols = [c for c in X.columns if c not in num_cols]

preprocessor = ColumnTransformer([
    ("num", "passthrough", num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

model = Pipeline([
    ("pre", preprocessor),
    ("clf", OneVsRestClassifier(
        LogisticRegression(max_iter=300, class_weight="balanced")
    ))
])

model.fit(X, Y)

joblib.dump({
    "model": model,
    "labels": LABELS,
    "model_version": "v1"
}, "model.pkl")

print("✅ model.pkl berhasil dibuat")
