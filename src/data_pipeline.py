from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import JSON, Column, DateTime, Float, Integer, MetaData, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import ALL_FEATURES, DATA_DIR, DB_URL, NEW_DATA_DIR


engine = create_engine(DB_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base(metadata=MetaData())


class PredictionLog(Base):
	__tablename__ = "prediction_logs"
	id = Column(Integer, primary_key=True, index=True)
	received_at = Column(DateTime, default=datetime.utcnow, index=True)
	input_record = Column(JSON)
	predicted_label = Column(Integer)
	probability = Column(Float)
	top_features = Column(JSON)


class ModelMetadata(Base):
	__tablename__ = "model_metadata"
	id = Column(Integer, primary_key=True)
	version = Column(Integer, index=True)
	trained_at = Column(DateTime, default=datetime.utcnow)
	accuracy = Column(Float)
	auc = Column(Float)
	dataset_rows = Column(Integer)
	model_path = Column(String)
	preprocessor_path = Column(String)
	threshold = Column(Float)


def init_db() -> None:
	Base.metadata.create_all(bind=engine)


def store_new_data(df: pd.DataFrame) -> Path:
	NEW_DATA_DIR.mkdir(parents=True, exist_ok=True)
	stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
	path = NEW_DATA_DIR / f"new_{stamp}.csv"
	df.to_csv(path, index=False)
	return path


def count_new_data_rows() -> int:
	total = 0
	for p in NEW_DATA_DIR.glob("*.csv"):
		try:
			total += sum(1 for _ in open(p)) - 1
		except Exception:
			pass
	return total


def clear_new_data() -> None:
	for p in NEW_DATA_DIR.glob("*.csv"):
		try:
			p.unlink()
		except Exception:
			pass


def log_prediction(record: Dict[str, Any]) -> None:
	session = SessionLocal()
	try:
		log = PredictionLog(
			input_record=record.get("input"),
			predicted_label=record.get("predicted_label"),
			probability=record.get("probability"),
			top_features=record.get("top_features"),
		)
		session.add(log)
		session.commit()
	finally:
		session.close()


def save_model_metadata(meta: Dict[str, Any]) -> None:
	session = SessionLocal()
	try:
		rec = ModelMetadata(
			version=meta["version"],
			trained_at=meta.get("trained_at", datetime.utcnow()),
			accuracy=meta.get("accuracy"),
			auc=meta.get("auc"),
			dataset_rows=meta.get("dataset_rows"),
			model_path=meta.get("model_path"),
			preprocessor_path=meta.get("preprocessor_path"),
			threshold=meta.get("threshold"),
		)
		session.add(rec)
		session.commit()
	finally:
		session.close()


def get_latest_model_version() -> int:
	session = SessionLocal()
	try:
		row = session.query(ModelMetadata).order_by(ModelMetadata.version.desc()).first()
		return int(row.version) if row else 0
	finally:
		session.close()


def get_current_model_status() -> Dict[str, Any]:
	session = SessionLocal()
	try:
		row = session.query(ModelMetadata).order_by(ModelMetadata.version.desc()).first()
		if not row:
			return {"trained": False}
		return {
			"trained": True,
			"version": row.version,
			"trained_at": row.trained_at.isoformat(),
			"accuracy": row.accuracy,
			"auc": row.auc,
			"dataset_rows": row.dataset_rows,
			"model_path": row.model_path,
			"preprocessor_path": row.preprocessor_path,
			"threshold": row.threshold,
		}
	finally:
		session.close()
