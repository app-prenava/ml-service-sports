import os
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)
from sklearn.model_selection import train_test_split


# 1. Load Model & Dataset
DATASET_PATH = os.getenv("DATASET_PATH", "data/maternal_health.csv")
MODEL_PATH   = os.getenv("RISK_MODEL_PATH", "risk_model.pkl")
SEED = 42

df = pd.read_csv(DATASET_PATH)

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

bool_like = [
    "previous_complications",
    "preexisting_diabetes",
    "gestational_diabetes",
    "mental_health",
]
for c in bool_like:
    df[c] = df[c].fillna(0).astype(int)

df = df.dropna(subset=[
    "age","systolic_bp","diastolic_bp","blood_sugar","body_temp",
    "bmi","previous_complications","preexisting_diabetes",
    "gestational_diabetes","mental_health","heart_rate","risk_level"
]).reset_index(drop=True)

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

# Load model
obj = joblib.load(MODEL_PATH)
model = obj["risk_model"]
classes = obj["class_names"]

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# 2. Confusion Matrix

cm = confusion_matrix(y_test, y_pred, labels=classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# 3. ROC Curve

# Untuk kasus binary, ambil salah satu class sebagai "positif"
if len(classes) == 2:
    positive_class = classes[1]
    y_true_binary = (y_test == positive_class).astype(int)
    positive_index = list(classes).index(positive_class)

    fpr, tpr, _ = roc_curve(y_true_binary, y_proba[:, positive_index])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    plt.show()

# 4. Coefficient Plot
clf = model.named_steps["clf"]
coefs = clf.coef_
coef_df = pd.DataFrame(coefs, columns=FEATURES)

plt.figure(figsize=(10, 5))
if coef_df.shape[0] == 1:
    plt.bar(coef_df.columns, coef_df.iloc[0], alpha=0.7, label="Coefficients")
else:
    for i, cls in enumerate(classes):
        plt.bar(coef_df.columns, coef_df.iloc[i], alpha=0.6, label=f"Class: {cls}")

plt.title("Logistic Regression Coefficients")
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig("coef_plot.png")
plt.show()

print("Visualizations saved as:")
print("- confusion_matrix.png")
print("- roc_curve.png")
print("- coef_plot.png")
