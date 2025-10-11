Simple documentation to install, run, and use the project.

Build with FastAPI and Logistic Regression (OvR)  

---

## Clone Project

```bash
git clone <your-repo-url>
cd ml-service-sports```

## Use venv

```bash
# Mac / Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

## Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Train the model
```bash
python train_risk.py
```

## Run API
```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

## cURL API
```bash
GET http://localhost:8080/healthz

curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 28,
    "gestational_age_weeks": 20,
    "bmi": 26.8,
    "blood_pressure_systolic": 120,
    "blood_pressure_diastolic": 80,
    "blood_sugar": 100,
    "body_temp": 36.8,
    "heart_rate": 85,
    "previous_complications": false,
    "preexisting_diabetes": false,
    "gestational_diabetes": false,
    "mental_health_issue": false,
    "placenta_position_restriction": false,
    "low_impact_pref": true,
    "water_access": true,
    "back_pain": false
  }'
```

