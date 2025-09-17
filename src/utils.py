from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
	ALL_FEATURES,
	CATEGORICAL_FEATURES,
	DATA_DIR,
	FEATURE_LIST_FILE,
	LOGS_DIR,
	NUMERIC_FEATURES,
	PREDICTION_LOG_FILE,
)


def setup_logging(app_log_path: Path) -> None:
	LOGS_DIR.mkdir(parents=True, exist_ok=True)
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
		handlers=[
			logging.FileHandler(app_log_path),
			logging.StreamHandler(),
		],
	)


def timestamp_str() -> str:
	return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def save_json(path: Path, data: Dict) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w") as f:
		json.dump(data, f, indent=2)


def load_json(path: Path) -> Dict:
	with open(path, "r") as f:
		return json.load(f)


def create_preprocessor() -> ColumnTransformer:
	numeric_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
			("scaler", StandardScaler()),
		]
	)

	categorical_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			("onehot", OneHotEncoder(handle_unknown="ignore")),
		]
	)

	preprocessor = ColumnTransformer(
		transformers=[
			("num", numeric_pipeline, NUMERIC_FEATURES),
			("cat", categorical_pipeline, CATEGORICAL_FEATURES),
		]
	)
	return preprocessor


def ensure_feature_order(df: pd.DataFrame) -> pd.DataFrame:
	missing = [c for c in ALL_FEATURES if c not in df.columns]
	for c in missing:
		df[c] = np.nan
	return df[ALL_FEATURES]


def save_feature_list(feature_names: List[str]) -> None:
	joblib.dump(feature_names, FEATURE_LIST_FILE)


def load_feature_list() -> List[str]:
	return joblib.load(FEATURE_LIST_FILE)


def log_predictions(records: List[Dict]) -> None:
	PREDICTION_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
	columns = list(records[0].keys()) if records else []
	df = pd.DataFrame(records, columns=columns)
	mode = "a" if PREDICTION_LOG_FILE.exists() else "w"
	header = not PREDICTION_LOG_FILE.exists()
	df.to_csv(PREDICTION_LOG_FILE, index=False, mode=mode, header=header)


def derive_dropout_label(row: pd.Series) -> int:
	"""
	Rule-based label creation:
	- Higher risk with: low attendance, low marks, high distance, poor facilities,
	  low parental education, low income, disadvantaged caste, rural location, large family.
	"""
	risk = 0.0

	# Attendance and marks
	risk += 2.0 * (1 - (row.get("attendance_rate", 0) or 0))
	risk += 1.5 * (1 - (row.get("avg_marks", 0) or 0))

	# Distance to school
	dist = row.get("distance_to_school_km", 0) or 0
	risk += 0.1 * max(0, dist - 1.5)

	# Facilities (0/1 like values)
	for f in [
		"electricity_availability",
		"internet_availability",
		"midday_meal",
		"toilet_availability",
		"transport_facility",
	]:
		val = row.get(f, 0) or 0
		risk += 0.6 * (1 - val)

	# Books supplied scaled 0-1
	books = row.get("books_supplied", 0) or 0
	risk += 0.4 * (1 - books)

	# Income (assume normalized 0-1)
	income = row.get("household_income", 0) or 0
	risk += 1.2 * (1 - income)

	# Parental education
	mother = str(row.get("mother_education", "None"))
	father = str(row.get("father_education", "None"))
	low_edu = {"None", "Primary"}
	risk += 0.7 if mother in low_edu else 0
	risk += 0.7 if father in low_edu else 0

	# Caste and location
	caste = str(row.get("caste_category", "GEN"))
	location = str(row.get("location", "Rural"))
	if caste in {"SC", "ST"}:
		risk += 0.6
	if location == "Rural":
		risk += 0.4

	# Family size
	fs = str(row.get("family_size", "Medium"))
	if fs == "Large":
		risk += 0.3

	# Government schemes access lowers risk
	scheme = str(row.get("govt_schemes_access", "None"))
	if scheme in {"Scholarship", "Multiple"}:
		risk -= 0.5

	# Clamp and threshold
	risk = max(0.0, risk)
	return int(risk > 3.0)
