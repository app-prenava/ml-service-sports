import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="sklearn"
)

#menentukan lokasi dataset dan file output model & laporan
DATASET_PATH = os.getenv("DATASET_PATH", "data/maternal_health.csv")
MODEL_OUT    = os.getenv("RISK_MODEL_OUT", "risk_model.pkl")
REPORT_OUT   = os.getenv("RISK_REPORT_OUT", "risk_model_report.json")
SEED = 42

df = pd.read_csv(DATASET_PATH)

#menyederhanakan nama kolom agar mudah dipakai di kode dan API
colmap = {
    "Age": "age",
    "Systolic BP": "systolic_bp",
    "Diastolic": "diastolic_bp",
    "BS": "blood_sugar",
    "Body Temp": "body_temp",
    "BMI": "bmi",
    "Previous Complications": "previous_complications",
    "Preexisting Diabetes": "preexisting_diabetes",
    "Gestational Diabetes": "gestational_diabetes",
    "Mental Health": "mental_health",
    "Heart Rate": "heart_rate",
    "Risk Level": "risk_level",
}
df = df.rename(columns=colmap)

#missing value
required = list(colmap.values())
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in dataset: {missing}")

#konversi nilai string menjadi boolean
bool_like = [
    "previous_complications",
    "preexisting_diabetes",
    "gestational_diabetes",
    "mental_health",
]
for c in bool_like:
    df[c] = df[c].fillna(0).astype(int)

#Memastikan tidak ada missing value 
df = df.dropna(subset=[
    "age","systolic_bp","diastolic_bp","blood_sugar","body_temp",
    "bmi","previous_complications","preexisting_diabetes",
    "gestational_diabetes","mental_health","heart_rate","risk_level"
]).reset_index(drop=True)

#Menentukan fitur dan label
FEATURES = [
    "age","systolic_bp","diastolic_bp","blood_sugar","body_temp",
    "bmi","previous_complications","preexisting_diabetes",
    "gestational_diabetes","mental_health","heart_rate"
]
X = df[FEATURES].astype(float)
y = df["risk_level"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)


#Membuat pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=500,
        class_weight="balanced",    
        multi_class="ovr",          
        random_state=SEED
    ))
])

pipe.fit(X_train, y_train)


#Prediksi dan evaluasi
y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

print(f"Trained risk model. Accuracy: {acc:.3f}")
print("Classes:", list(pipe.named_steps["clf"].classes_))


#Simpan model
obj = {
    "risk_model": pipe,
    "feature_names": FEATURES,
    "class_names": list(pipe.named_steps["clf"].classes_),
    "model_version": "risk_v1"
}
joblib.dump(obj, MODEL_OUT)


#Simpan laporan json
Path(REPORT_OUT).write_text(json.dumps({
    "accuracy": acc,
    "classes": obj["class_names"],
    "report": report
}, indent=2))

print(f"Saved: {MODEL_OUT}")
print(f"Report: {REPORT_OUT}")
