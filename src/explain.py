from __future__ import annotations

import json
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from .config import FEATURE_LIST_FILE, GLOBAL_IMPORTANCE_FILE, MODEL_FILE, PREPROCESSOR_FILE, RANDOM_STATE
from .utils import ensure_feature_order


def load_global_importance() -> Dict[str, float]:
	with open(GLOBAL_IMPORTANCE_FILE, "r") as f:
		return json.load(f)


def _final_feature_names(preprocessor, X: pd.DataFrame) -> list[str]:
	from sklearn.compose import ColumnTransformer
	from sklearn.preprocessing import OneHotEncoder
	from sklearn.pipeline import Pipeline

	ct: ColumnTransformer = preprocessor
	num_features = list(ct.transformers_[0][2])
	cat_cols = list(ct.transformers_[1][2])
	cat_transformer: Pipeline = ct.transformers_[1][1]
	onehot: OneHotEncoder = cat_transformer.named_steps["onehot"]
	onehot.fit(X[cat_cols])
	cat_feature_names = list(onehot.get_feature_names_out(cat_cols))
	return num_features + cat_feature_names


def local_explanations(records: List[Dict]) -> List[Dict]:
	preprocessor = joblib.load(PREPROCESSOR_FILE)
	model = joblib.load(MODEL_FILE)
	df = pd.DataFrame(records)
	X = ensure_feature_order(df)
	X_trans = preprocessor.transform(X)
	if hasattr(X_trans, "toarray"):
		X_trans = X_trans.toarray()
	feature_names = _final_feature_names(preprocessor, X)

	try:
		import shap
		if hasattr(model, "feature_importances_"):
			explainer = shap.TreeExplainer(model)
			sv = explainer.shap_values(X_trans)
			values = sv[-1] if isinstance(sv, list) else sv
			base_value = explainer.expected_value[-1] if isinstance(explainer.expected_value, list) else explainer.expected_value
		else:
			raise RuntimeError("not a tree model")
	except Exception:
		# Simple fallback: scale normalized feature_importances_ per sample signless
		if hasattr(model, "feature_importances_"):
			values = np.tile(model.feature_importances_, (X_trans.shape[0], 1))
			base_value = 0.0
		else:
			values = np.zeros_like(X_trans)
			base_value = 0.0

	outputs: List[Dict] = []
	for i in range(X_trans.shape[0]):
		pairs = list(zip(feature_names, values[i]))
		pairs.sort(key=lambda x: abs(x[1]), reverse=True)
		top5 = [{"feature": k, "contribution": float(v)} for k, v in pairs[:5]]
		outputs.append({
			"base_value": float(base_value),
			"top_features": top5,
		})
	return outputs
