## Rajasthan School Dropout Prediction - Backend

Production-ready FastAPI backend for student dropout prediction with automated preprocessing, training, prediction, and SHAP/LIME explainability. Designed to integrate with a React/Next.js dashboard.

### Folder Structure

```
/src
  ├── config.py
  ├── generate_dataset.py
  ├── train_model.py
  ├── predict.py
  ├── explain.py
  ├── api.py
/data
/models
/logs
```

### Quickstart (Local)

1. Install dependencies
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2. Generate synthetic dataset (Rajasthan context)
```
python -m src.generate_dataset --rows 5000 --out data/rajasthan_students.csv
```

3. Train model
```
python -m src.train_model --data data/rajasthan_students.csv
```

4. Run API
```
uvicorn src.api:app --reload --port 8000
```

5. Test endpoints
- POST `http://localhost:8000/predict`
- POST `http://localhost:8000/batch_predict`
- POST `http://localhost:8000/update_data`
- GET `http://localhost:8000/feature_importance`

### Request/Response Examples

- POST /predict (JSON)
```
{
  "attendance_rate": 0.85,
  "avg_marks": 0.7,
  "household_income": 0.4,
  "distance_to_school_km": 2.0,
  "teacher_student_ratio": 0.05,
  "electricity_availability": 1,
  "internet_availability": 0,
  "midday_meal": 1,
  "toilet_availability": 1,
  "books_supplied": 0.6,
  "transport_facility": 0,
  "caste_category": "OBC",
  "language_medium": "Hindi",
  "govt_schemes_access": "Scholarship",
  "parent_occupation": "Agriculture",
  "mother_education": "Secondary",
  "father_education": "Primary",
  "family_size": "Medium",
  "gender": "Female",
  "location": "Rural"
}
```
Response
```
{
  "dropout_risk": 0,
  "probability": 0.23,
  "top_features": [
    {"feature": "avg_marks", "contribution": -0.12},
    {"feature": "attendance_rate", "contribution": -0.09},
    {"feature": "distance_to_school_km", "contribution": 0.06},
    {"feature": "internet_availability", "contribution": 0.04},
    {"feature": "books_supplied", "contribution": -0.03}
  ]
}
```

### Docker
```
docker build -t dropout-backend .
docker run -p 8000:8000 -v $PWD/data:/app/data -v $PWD/models:/app/models -v $PWD/logs:/app/logs dropout-backend
```

### Frontend Integration Notes
- CORS is enabled for all origins by default. Restrict via `api.py` in production.
- Responses include probabilities and top feature contributions ready for charts.

### Retraining with New Data
Use `POST /update_data` with CSV/JSON file. The dataset is stored under `/data` with a timestamp, symlinked to `latest.csv`, and the model is retrained. Artifacts are updated under `/models`.

### Explainability
- Global: stored in `/models/global_importance.json` and served by `GET /feature_importance`.
- Local: returned in prediction responses using SHAP; if SHAP fails, permutation importance fallback is used for global.

### Logging
- App logs at `/logs/app.log`
- Prediction logs at `/logs/predictions.csv`

### Environment
- Python 3.11
- scikit-learn RandomForest or XGBoost

### Security & Production Tips
- Pin CORS origins, add auth (API key/JWT), and rate limiting.
- Persist `/data`, `/models`, `/logs` on durable storage.
- Monitor model/data drift; schedule periodic retraining.
