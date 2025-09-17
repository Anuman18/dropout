from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from .config import ALL_FEATURES, STUDENTS_FILE, TARGET_COLUMN
from .utils import derive_dropout_label, ensure_feature_order


def ensure_students_file() -> None:
	STUDENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
	if not STUDENTS_FILE.exists():
		df = pd.DataFrame(columns=ALL_FEATURES + [TARGET_COLUMN])
		df.to_csv(STUDENTS_FILE, index=False)


def append_students(df_new: pd.DataFrame) -> None:
	ensure_students_file()
	# Ensure columns exist, derive label if missing
	if TARGET_COLUMN not in df_new.columns:
		df_new[TARGET_COLUMN] = df_new.apply(lambda r: derive_dropout_label(r), axis=1)
	df_new = ensure_feature_order(df_new[ALL_FEATURES]).assign(**{TARGET_COLUMN: df_new[TARGET_COLUMN].values})
	# Append
	df_existing = pd.read_csv(STUDENTS_FILE) if STUDENTS_FILE.exists() else pd.DataFrame(columns=df_new.columns)
	df_concat = pd.concat([df_existing, df_new], ignore_index=True)
	df_concat.to_csv(STUDENTS_FILE, index=False)


def load_students() -> pd.DataFrame:
	ensure_students_file()
	return pd.read_csv(STUDENTS_FILE)
