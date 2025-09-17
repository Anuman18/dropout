from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import (
	ALL_FEATURES,
	API_HOST,
	API_PORT,
	DATA_DIR,
	DEFAULT_DATASET_NAME,
	FEATURE_LIST_FILE,
	GLOBAL_IMPORTANCE_FILE,
	LATEST_DATA_SYMLINK,
	MODEL_FILE,
	PREPROCESSOR_FILE,
	STUDENTS_FILE,
)
from .data_pipeline import init_db, log_prediction, store_new_data, get_current_model_status
from .data import append_students, load_students
from .explain import load_global_importance, local_explanations, get_explanation
from .predict import Predictor, batch_predict
# Optional scheduler import
try:
	from .scheduler import RetrainScheduler  # type: ignore
	HAS_SCHEDULER = True
except Exception:
	RetrainScheduler = None  # type: ignore
	HAS_SCHEDULER = False
from .train_model import train
from .utils import ensure_feature_order, timestamp_str

app = FastAPI(title="Rajasthan Dropout Prediction API")

# CORS (allow all for demo; restrict in prod)
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

scheduler: Optional[object] = None


class StudentRecord(BaseModel):
	attendance_rate: Optional[float] = Field(None, example=0.85)
	avg_marks: Optional[float] = Field(None, example=0.7)
	household_income: Optional[float] = Field(None, example=0.4)
	distance_to_school_km: Optional[float] = Field(None, example=2.0)
	teacher_student_ratio: Optional[float] = Field(None, example=0.05)
	electricity_availability: Optional[float] = Field(None, example=1)
	internet_availability: Optional[float] = Field(None, example=0)
	midday_meal: Optional[float] = Field(None, example=1)
	toilet_availability: Optional[float] = Field(None, example=1)
	books_supplied: Optional[float] = Field(None, example=0.6)
	transport_facility: Optional[float] = Field(None, example=0)

	caste_category: Optional[str] = Field(None, example="OBC")
	language_medium: Optional[str] = Field(None, example="Hindi")
	govt_schemes_access: Optional[str] = Field(None, example="Scholarship")
	parent_occupation: Optional[str] = Field(None, example="Agriculture")
	mother_education: Optional[str] = Field(None, example="Secondary")
	father_education: Optional[str] = Field(None, example="Primary")
	family_size: Optional[str] = Field(None, example="Medium")
	gender: Optional[str] = Field(None, example="Female")
	location: Optional[str] = Field(None, example="Rural")


@app.on_event("startup")
async def startup_event() -> None:
	# DB and optional scheduler
	init_db()
	global scheduler
	if HAS_SCHEDULER and RetrainScheduler is not None:
		scheduler = RetrainScheduler()
		scheduler.start()
	# Warm up predictor if artifacts exist
	if PREPROCESSOR_FILE.exists() and MODEL_FILE.exists():
		app.state.predictor = Predictor()
	else:
		app.state.predictor = None


@app.post("/explain")
async def explain_endpoint(record: StudentRecord) -> Dict[str, Any]:
	try:
		data = {k: v for k, v in record.model_dump().items()}
		result = get_explanation(data, top_n=5)
		return result
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict_endpoint(record: StudentRecord) -> Dict[str, Any]:
	if not getattr(app.state, "predictor", None):
		raise HTTPException(status_code=400, detail="Model not trained")
	predictor: Predictor = app.state.predictor
	record_dict = {k: v for k, v in record.model_dump().items()}
	pred = predictor.predict_proba([record_dict])[0]
	try:
		loc_expl = local_explanations([record_dict])[0]
		top_features = loc_expl.get("top_features", [])
	except Exception:
		top_features = []
	# Human-friendly risk label
	risk_label = "NO" if pred["risk_tier"] == "green" else ("MAYBE" if pred["risk_tier"] == "yellow" else "YES")
	# Audit log in DB
	try:
		log_prediction({
			"input": record_dict,
			"predicted_label": 1 if pred["risk_tier"] == "red" else 0,
			"probability": pred["probability"],
			"risk_tier": pred["risk_tier"],
			"top_features": top_features,
		})
	except Exception:
		pass
	return {
		"dropout_risk": risk_label,
		"probability": pred["probability"],
		"risk_tier": pred["risk_tier"],
		"top_features": top_features,
	}


@app.post("/batch_predict")
async def batch_predict_endpoint(file: UploadFile = File(...)) -> Dict[str, Any]:
	if not getattr(app.state, "predictor", None):
		raise HTTPException(status_code=400, detail="Model not trained")
	predictor: Predictor = app.state.predictor

	content = await file.read()
	if file.filename.endswith(".csv"):
		df = pd.read_csv(io.BytesIO(content))
	elif file.filename.endswith(".json"):
		data = json.loads(content)
		df = pd.DataFrame(data)
	else:
		raise HTTPException(status_code=400, detail="Unsupported file type")

	out = batch_predict(predictor, df)
	try:
		explanations = local_explanations(df.to_dict(orient="records"))
	except Exception:
		explanations = [{"top_features": []} for _ in range(len(out))]
	out_dicts = out.to_dict(orient="records")
	for i in range(len(out_dicts)):
		out_dicts[i]["top_features"] = explanations[i].get("top_features", [])
		out_dicts[i]["dropout_risk"] = "NO" if out_dicts[i]["risk_tier"] == "green" else ("MAYBE" if out_dicts[i]["risk_tier"] == "yellow" else "YES")
		# Log each row to DB
		try:
			log_prediction({
				"input": df.iloc[i].to_dict(),
				"predicted_label": 1 if out_dicts[i]["risk_tier"] == "red" else 0,
				"probability": float(out_dicts[i]["probability"]),
				"risk_tier": out_dicts[i]["risk_tier"],
				"top_features": out_dicts[i]["top_features"],
			})
		except Exception:
			pass
	return {"results": out_dicts}


@app.post("/update_data")
async def update_data_endpoint(file: UploadFile = File(...)) -> Dict[str, Any]:
	# Append incoming data directly to students.csv and retrain
	content = await file.read()
	if file.filename.endswith(".csv"):
		df = pd.read_csv(io.BytesIO(content))
	elif file.filename.endswith(".json"):
		data = json.loads(content)
		df = pd.DataFrame(data)
	else:
		raise HTTPException(status_code=400, detail="Unsupported file type")
	append_students(df)
	# Auto-retrain
	train(STUDENTS_FILE)
	# Reload predictor
	app.state.predictor = Predictor()
	return {"status": "appended_and_retrained", "path": str(STUDENTS_FILE)}


@app.post("/retrain")
async def retrain_endpoint() -> Dict[str, Any]:
	train(STUDENTS_FILE)
	app.state.predictor = Predictor()
	return {"status": "retrained"}


@app.post("/train")
async def train_endpoint() -> Dict[str, Any]:
	# Backwards-compatible manual train
	train(STUDENTS_FILE)
	app.state.predictor = Predictor()
	return {"status": "trained"}


@app.get("/feature_importance")
async def feature_importance_endpoint() -> Dict[str, Any]:
	if not GLOBAL_IMPORTANCE_FILE.exists():
		raise HTTPException(status_code=404, detail="Global importance not found")
	return {"feature_importance": load_global_importance()}


@app.get("/model_status")
async def model_status_endpoint() -> Dict[str, Any]:
	return get_current_model_status()


@app.get("/")
async def root() -> Dict[str, Any]:
	return {"message": "Rajasthan Dropout Prediction API running"}
