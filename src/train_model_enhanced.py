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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score, precision_recall_curve
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


def create_advanced_preprocessor() -> object:
	"""Enhanced preprocessing with feature engineering"""
	from sklearn.compose import ColumnTransformer
	from sklearn.impute import SimpleImputer
	from sklearn.pipeline import Pipeline
	from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
	
	numeric_pipeline = Pipeline([
		("imputer", SimpleImputer(strategy="median")),
		("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
		("scaler", StandardScaler()),
	])
	
	categorical_pipeline = Pipeline([
		("imputer", SimpleImputer(strategy="most_frequent")),
		("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
	])
	
	preprocessor = ColumnTransformer([
		("num", numeric_pipeline, [
			"attendance_rate", "avg_marks", "household_income", "distance_to_school_km",
			"teacher_student_ratio", "electricity_availability", "internet_availability",
			"midday_meal", "toilet_availability", "books_supplied", "transport_facility"
		]),
		("cat", categorical_pipeline, [
			"caste_category", "language_medium", "govt_schemes_access", "parent_occupation",
			"mother_education", "father_education", "family_size", "gender", "location"
		]),
	])
	return preprocessor


def create_ensemble_model() -> object:
	"""Create ensemble of best models"""
	models = []
	
	# RandomForest with optimized params
	rf = RandomForestClassifier(
		n_estimators=1000,
		max_depth=15,
		min_samples_split=5,
		min_samples_leaf=2,
		max_features="sqrt",
		class_weight="balanced_subsample",
		random_state=RANDOM_STATE,
		n_jobs=-1
	)
	models.append(("rf", rf))
	
	# GradientBoosting
	gb = GradientBoostingClassifier(
		n_estimators=500,
		learning_rate=0.05,
		max_depth=8,
		min_samples_split=10,
		min_samples_leaf=4,
		subsample=0.8,
		random_state=RANDOM_STATE
	)
	models.append(("gb", gb))
	
	# XGBoost if available
	if HAS_XGB:
		xgb = XGBClassifier(
			n_estimators=1000,
			max_depth=8,
			learning_rate=0.05,
			subsample=0.8,
			colsample_bytree=0.8,
			reg_alpha=0.1,
			reg_lambda=0.1,
			random_state=RANDOM_STATE,
			eval_metric="logloss",
			n_jobs=-1
		)
		models.append(("xgb", xgb))
	
	# Logistic Regression for linear patterns
	lr = LogisticRegression(
		C=0.1,
		class_weight="balanced",
		random_state=RANDOM_STATE,
		max_iter=1000
	)
	models.append(("lr", lr))
	
	# Voting ensemble
	ensemble = VotingClassifier(models, voting="soft")
	return ensemble


def compute_global_importance(model: object, preprocessor: object, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
	import shap
	X_trans = preprocessor.fit_transform(X)
	if hasattr(X_trans, "toarray"):
		X_trans = X_trans.toarray()
	
	# Get feature names after preprocessing
	from sklearn.compose import ColumnTransformer
	from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
	from sklearn.pipeline import Pipeline

	ct: ColumnTransformer = preprocessor
	num_features = list(ct.transformers_[0][2])
	cat_cols = list(ct.transformers_[1][2])
	cat_transformer: Pipeline = ct.transformers_[1][1]
	onehot: OneHotEncoder = cat_transformer.named_steps["onehot"]
	onehot.fit(X[cat_cols])
	cat_feature_names = list(onehot.get_feature_names_out(cat_cols))
	
	# Handle polynomial features
	poly_transformer: PolynomialFeatures = ct.transformers_[0][1].named_steps["poly"]
	poly_transformer.fit(X[num_features])
	poly_feature_names = poly_transformer.get_feature_names_out(num_features)
	
	all_feature_names = list(poly_feature_names) + cat_feature_names

	try:
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
	k = min(len(all_feature_names), len(vals))
	importance = {all_feature_names[i]: float(vals[i]) for i in range(k)}
	return importance


def tune_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
	"""Optimize threshold using precision-recall curve"""
	precision, recall, thresholds = precision_recall_curve(y_true, prob)
	f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
	best_idx = np.argmax(f1_scores)
	return float(thresholds[best_idx])


def train(data_path: Path) -> None:
	logger.info(f"Loading data from {data_path}")
	df = pd.read_csv(data_path)

	if TARGET_COLUMN not in df.columns:
		from .utils import derive_dropout_label
		df[TARGET_COLUMN] = df.apply(lambda r: derive_dropout_label(r), axis=1)

	dataset_rows = df.shape[0]
	X = ensure_feature_order(df[ALL_FEATURES].copy())
	y = df[TARGET_COLUMN].astype(int)

	# Stratified split
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
	)

	# Enhanced preprocessing
	preprocessor = create_advanced_preprocessor()
	
	# Create ensemble model
	model = create_ensemble_model()
	
	# Pipeline
	pipeline = Pipeline([
		("preprocessor", preprocessor),
		("model", model)
	])

	logger.info("Training enhanced ensemble model...")
	pipeline.fit(X_train, y_train)

	# Cross-validation
	cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1")
	logger.info(f"CV F1 scores: {cv_scores}")
	logger.info(f"CV F1 mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

	# Evaluation
	logger.info("Evaluating...")
	proba_test = pipeline.predict_proba(X_test)[:, 1]
	best_threshold = tune_threshold(y_test.values, proba_test)
	pred_test = (proba_test >= best_threshold).astype(int)
	
	report = classification_report(y_test, pred_test, output_dict=True)
	auc = roc_auc_score(y_test, proba_test)
	f1 = f1_score(y_test, pred_test)
	
	logger.info(json.dumps(report, indent=2))
	logger.info(f"AUC: {auc:.4f}, F1: {f1:.4f}, Best threshold: {best_threshold:.3f}")

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
		"f1": float(f1),
		"cv_f1_mean": float(cv_scores.mean()),
		"cv_f1_std": float(cv_scores.std()),
		"dataset_rows": int(dataset_rows),
		"model_path": str(model_path),
		"preprocessor_path": str(preproc_path),
		"threshold": float(best_threshold),
	}
	save_model_metadata(meta)
	with open(METADATA_FILE, "w") as f:
		json.dump({k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in meta.items()}, f, indent=2)

	logger.info("Enhanced training complete.")


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--data", type=str, default=str(DATA_DIR / "students.csv"))
	args = parser.parse_args()

	logging.basicConfig(level=logging.INFO)
	train(Path(args.data))


if __name__ == "__main__":
	main()
