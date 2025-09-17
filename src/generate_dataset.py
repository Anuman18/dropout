from __future__ import annotations

import argparse
import random
from typing import List

import numpy as np
import pandas as pd
from pathlib import Path

from .config import (
	ALL_FEATURES,
	CATEGORIES,
	DATA_DIR,
	DEFAULT_DATASET_NAME,
	TARGET_COLUMN,
)
from .utils import derive_dropout_label, timestamp_str


def _clip01(x: np.ndarray) -> np.ndarray:
	return np.clip(x, 0.0, 1.0)


def generate_rows(n: int, seed: int = 42) -> pd.DataFrame:
	rng = np.random.default_rng(seed)
	records: List[dict] = []

	caste_weights = {"GEN": 0.25, "OBC": 0.4, "SC": 0.2, "ST": 0.15}
	loc_probs = {"Rural": 0.7, "Urban": 0.3}
	lang_probs = {"Hindi": 0.8, "English": 0.15, "Urdu": 0.05}

	for _ in range(n):
		location = rng.choice(list(loc_probs.keys()), p=list(loc_probs.values()))
		caste = rng.choice(list(caste_weights.keys()), p=list(caste_weights.values()))
		gender = rng.choice(["Male", "Female"], p=[0.52, 0.48])
		family_size = rng.choice(["Small", "Medium", "Large"], p=[0.3, 0.5, 0.2])
		mother_edu = rng.choice(CATEGORIES["mother_education"], p=[0.25, 0.3, 0.25, 0.15, 0.05])
		father_edu = rng.choice(CATEGORIES["father_education"], p=[0.2, 0.3, 0.25, 0.2, 0.05])
		parent_occ = rng.choice(CATEGORIES["parent_occupation"], p=[0.35, 0.25, 0.15, 0.15, 0.1])
		language = rng.choice(list(lang_probs.keys()), p=list(lang_probs.values()))

		# Income distribution lower in rural and for SC/ST on average
		base_income = rng.beta(2, 3)
		if location == "Rural":
			base_income *= 0.8
		if caste in {"SC", "ST"}:
			base_income *= 0.85

		# Distance to school higher in rural
		distance = rng.normal(2.0 if location == "Rural" else 0.8, 0.6)
		distance = max(0.0, distance)

		# Facilities depend on location and schemes
		schemes = rng.choice(CATEGORIES["govt_schemes_access"], p=[0.35, 0.2, 0.15, 0.15, 0.15])
		electricity = 1.0 if location == "Urban" else float(rng.random() < 0.85)
		internet = float(rng.random() < (0.5 if location == "Rural" else 0.8))
		midday = 1.0
		toilet = float(rng.random() < (0.75 if location == "Rural" else 0.9))
		transport = float(rng.random() < (0.4 if location == "Rural" else 0.7))
		books = _clip01(base_income * 0.6 + rng.normal(0.3, 0.15))

		# Teacher-student ratio worse in rural
		tsr = _clip01(0.04 if location == "Urban" else 0.06 + rng.normal(0.0, 0.01))

		# Attendance and marks correlate with distance, facilities, income, parental edu
		attendance = _clip01(0.9 - 0.05 * distance + 0.05 * electricity + 0.05 * internet + 0.05 * (books) + 0.04 * transport + 0.06 * base_income + rng.normal(0, 0.05))
		marks = _clip01(0.65 - 0.03 * distance + 0.04 * electricity + 0.05 * internet + 0.05 * books + 0.05 * base_income + (0.05 if mother_edu in {"Graduate", "Postgraduate"} else 0) + (0.05 if father_edu in {"Graduate", "Postgraduate"} else 0) + rng.normal(0, 0.07))

		rec = {
			"attendance_rate": float(attendance),
			"avg_marks": float(marks),
			"household_income": float(_clip01(base_income)),
			"distance_to_school_km": float(distance),
			"teacher_student_ratio": float(tsr),
			"electricity_availability": float(electricity),
			"internet_availability": float(internet),
			"midday_meal": float(midday),
			"toilet_availability": float(toilet),
			"books_supplied": float(books),
			"transport_facility": float(transport),
			"caste_category": caste,
			"language_medium": language,
			"govt_schemes_access": schemes,
			"parent_occupation": parent_occ,
			"mother_education": mother_edu,
			"father_education": father_edu,
			"family_size": family_size,
			"gender": gender,
			"location": location,
		}
		rec[TARGET_COLUMN] = derive_dropout_label(pd.Series(rec))
		records.append(rec)

	df = pd.DataFrame.from_records(records)
	return df


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--rows", type=int, default=5000)
	parser.add_argument("--out", type=str, default=str(DATA_DIR / DEFAULT_DATASET_NAME))
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()

	df = generate_rows(args.rows, seed=args.seed)
	out_path = Path(args.out)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(out_path, index=False)
	print(f"Saved synthetic dataset to {out_path}")

	# Also write a timestamped copy
	stamp = timestamp_str()
	ts_path = DATA_DIR / f"dataset_{stamp}.csv"
	df.to_csv(ts_path, index=False)
	print(f"Saved timestamped copy to {ts_path}")


if __name__ == "__main__":
	main()
