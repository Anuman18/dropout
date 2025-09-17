from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from .config import (
	ALL_FEATURES,
	FEATURE_LIST_FILE,
	MODEL_FILE,
	PREPROCESSOR_FILE,
	TARGET_COLUMN,
	THRESHOLD_FILE,
	RISK_GREEN_MAX,
	RISK_YELLOW_MAX,
)
from .utils import ensure_feature_order, log_predictions

logger = logging.getLogger("predict")


def _risk_tier(prob: float) -> str:
	if prob <= RISK_GREEN_MAX:
		return "green"
	if prob <= RISK_YELLOW_MAX:
		return "yellow"
	return "red"


class Predictor:
	def __init__(self) -> None:
		self.preprocessor = joblib.load(PREPROCESSOR_FILE)
		self.model = joblib.load(MODEL_FILE)
		self.feature_list: List[str] = joblib.load(FEATURE_LIST_FILE)
		self.threshold = 0.5
		if Path(THRESHOLD_FILE).exists():
			data = json.loads(Path(THRESHOLD_FILE).read_text())
			self.threshold = float(data.get("threshold", 0.5))

	def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
		df = ensure_feature_order(df)
		return df

	def predict_proba(self, records: List[Dict]) -> List[Dict]:
		df = pd.DataFrame(records)
		X = self._prepare_df(df)
		X_trans = self.preprocessor.transform(X)
		if hasattr(self.model, "predict_proba"):
			proba = self.model.predict_proba(X_trans)[:, 1]
		else:
			from scipy.special import expit
			scores = self.model.decision_function(X_trans)
			proba = expit(scores)
		preds = (proba >= self.threshold).astype(int)
		results: List[Dict] = []
		for i, row in enumerate(records):
			p = float(proba[i])
			res = {
				"prediction": int(preds[i]),
				"probability": p,
				"risk_tier": _risk_tier(p),
			}
			results.append(res)
		# Log
		rows_to_log = []
		for i, row in enumerate(records):
			row_copy = dict(row)
			row_copy["predicted_label"] = int(preds[i])
			row_copy["probability"] = float(proba[i])
			row_copy["risk_tier"] = _risk_tier(float(proba[i]))
			rows_to_log.append(row_copy)
		log_predictions(rows_to_log)
		return results


def batch_predict(predictor: Predictor, df: pd.DataFrame) -> pd.DataFrame:
	X = predictor._prepare_df(df)
	X_trans = predictor.preprocessor.transform(X)
	if hasattr(predictor.model, "predict_proba"):
		proba = predictor.model.predict_proba(X_trans)[:, 1]
	else:
		from scipy.special import expit
		scores = predictor.model.decision_function(X_trans)
		proba = expit(scores)
	preds = (proba >= predictor.threshold).astype(int)
	out = df.copy()
	out["prediction"] = preds
	out["probability"] = proba
	out["risk_tier"] = [ _risk_tier(float(p)) for p in proba ]
	# Log
	records = out.to_dict(orient="records")
	log_predictions(records)
	return out
