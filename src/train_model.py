from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
	from xgboost import XGBClassifier  # type: ignore
	HAS_XGB = True
except Exception:
	HAS_XGB = False

from .config import (
	ALL_FEATURES,
	DATA_DIR,
	FEATURE_LIST_FILE,
	GLOBAL_IMPORTANCE_FILE,
	MODEL_FILE,
	MODEL_FILE_COMPAT,
	MODELS_DIR,
	PREPROCESSOR_FILE,
	RANDOM_STATE,
	TARGET_COLUMN,
	TEST_SIZE,
	THRESHOLD_FILE,
	METADATA_FILE,
)
from .utils import create_preprocessor, ensure_feature_order, save_feature_list
from .data_pipeline import get_latest_model_version, save_model_metadata


logger = logging.getLogger("train")


def choose_model() -> object:
	use_xgb = os.getenv("USE_XGB", "0") == "1"
	if HAS_XGB and use_xgb:
		model = XGBClassifier(
			n_estimators=500,
			max_depth=6,
			random_state=RANDOM_STATE,
			eval_metric="logloss",
			subsample=0.9,
			colsample_bytree=0.9,
			n_jobs=4,
		)
		return model
	# Default robust choice
	return RandomForestClassifier(n_estimators=600, max_depth=None, random_state=RANDOM_STATE, class_weight="balanced_subsample")


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


def compute_global_importance(model: object, preprocessor: object, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
	import shap
	X_trans = preprocessor.fit_transform(X)
	# Ensure dense for SHAP/permutation
	if hasattr(X_trans, "toarray"):
		X_trans = X_trans.toarray()
	feature_names = _final_feature_names(preprocessor, X)

	try:
		# Prefer TreeExplainer for tree models
		if hasattr(model, "feature_importances_"):
			explainer = shap.TreeExplainer(model)
			shap_values = explainer.shap_values(X_trans)
			if isinstance(shap_values, list):
				vals = np.abs(shap_values[-1]).mean(axis=0)
			else:
				vals = np.abs(shap_values).mean(axis=0)
		else:
			raise RuntimeError("Use permutation for non-tree models")
	except Exception:
		from sklearn.inspection import permutation_importance
		result = permutation_importance(model, X_trans, y, n_repeats=5, random_state=RANDOM_STATE, n_jobs=2)
		vals = result.importances_mean

	# Coerce to 1-D and align lengths
	vals = np.asarray(vals)
	while vals.ndim > 1:
		vals = vals.mean(axis=-1)
	vals = vals.ravel()
	k = min(len(feature_names), len(vals))
	importance = {feature_names[i]: float(vals[i]) for i in range(k)}
	return importance


def tune_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
	# Maximize F1 by sweeping thresholds
	thresholds = np.linspace(0.2, 0.8, 25)
	best_t, best_f1 = 0.5, -1.0
	for t in thresholds:
		pred = (prob >= t).astype(int)
		score = f1_score(y_true, pred)
		if score > best_f1:
			best_f1, best_t = score, t
	return float(best_t)


def train(data_path: Path) -> None:
	logger.info(f"Loading data from {data_path}")
	df = pd.read_csv(data_path)

	if TARGET_COLUMN not in df.columns:
		from .utils import derive_dropout_label
		df[TARGET_COLUMN] = df.apply(lambda r: derive_dropout_label(r), axis=1)

	dataset_rows = df.shape[0]
	X = ensure_feature_order(df[ALL_FEATURES].copy())
	y = df[TARGET_COLUMN].astype(int)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

	preprocessor = create_preprocessor()
	model = choose_model()
	pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

	logger.info("Training model...")
	pipeline.fit(X_train, y_train)

	logger.info("Evaluating...")
	proba_test = pipeline.predict_proba(X_test)[:, 1]
	best_threshold = tune_threshold(y_test.values, proba_test)
	pred_test = (proba_test >= best_threshold).astype(int)
	report = classification_report(y_test, pred_test, output_dict=True)
	auc = roc_auc_score(y_test, proba_test)
	logger.info(json.dumps(report, indent=2))
	logger.info(f"AUC: {auc:.4f}, Best threshold: {best_threshold:.3f}")

	# Versioning
	version = get_latest_model_version() + 1
	model_path = MODELS_DIR / f"model_v{version}.pkl"
	preproc_path = MODELS_DIR / f"preprocessor_v{version}.pkl"
	threshold_path = MODELS_DIR / f"threshold_v{version}.json"

	# Persist versioned
	joblib.dump(pipeline.named_steps["preprocessor"], preproc_path)
	joblib.dump(pipeline.named_steps["model"], model_path)
	with open(threshold_path, "w") as f:
		json.dump({"threshold": best_threshold}, f)
	save_feature_list(list(X.columns))

	# Update current symlinks
	for link, target in [
		(PREPROCESSOR_FILE, preproc_path),
		(MODEL_FILE, model_path),
		(THRESHOLD_FILE, threshold_path),
	]:
		try:
			if link.exists() or link.is_symlink():
				link.unlink()
			link.symlink_to(target)
		except Exception:
			pass
	# Also keep a compatibility copy at dropout_model.pkl
	try:
		if MODEL_FILE_COMPAT.exists() or MODEL_FILE_COMPAT.is_symlink():
			MODEL_FILE_COMPAT.unlink()
		MODEL_FILE_COMPAT.symlink_to(model_path)
	except Exception:
		pass

	# Global importance on train subset
	importance = compute_global_importance(
		pipeline.named_steps["model"], pipeline.named_steps["preprocessor"], X_train, y_train
	)
	with open(GLOBAL_IMPORTANCE_FILE, "w") as f:
		json.dump(importance, f, indent=2)

	# Save metadata
	meta = {
		"version": version,
		"trained_at": datetime.utcnow(),
		"accuracy": float(report["accuracy"]),
		"auc": float(auc),
		"dataset_rows": int(dataset_rows),
		"model_path": str(model_path),
		"preprocessor_path": str(preproc_path),
		"threshold": float(best_threshold),
	}
	save_model_metadata(meta)
	with open(METADATA_FILE, "w") as f:
		json.dump({k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in meta.items()}, f, indent=2)

	logger.info("Training complete.")


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--data", type=str, default=str(DATA_DIR / "rajasthan_students.csv"))
	args = parser.parse_args()

	logging.basicConfig(level=logging.INFO)
	train(Path(args.data))


if __name__ == "__main__":
	main()
