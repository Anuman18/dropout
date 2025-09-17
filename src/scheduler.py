from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler

from .config import NEW_DATA_THRESHOLD
from .data_pipeline import clear_new_data, count_new_data_rows
from .train_model import train

logger = logging.getLogger("scheduler")


class RetrainScheduler:
	def __init__(self) -> None:
		self.scheduler = BackgroundScheduler()

	def start(self) -> None:
		self.scheduler.add_job(self.check_and_retrain, "interval", hours=24, id="retrain_job", replace_existing=True)
		self.scheduler.start()
		logger.info("Scheduler started with 24h interval job")

	def check_and_retrain(self) -> None:
		rows = count_new_data_rows()
		logger.info(f"New data rows pending: {rows}")
		if rows >= NEW_DATA_THRESHOLD:
			# Concatenate all new data files and retrain
			from .config import NEW_DATA_DIR
			dfs = []
			for p in NEW_DATA_DIR.glob("*.csv"):
				try:
					dfs.append(pd.read_csv(p))
				except Exception:
					pass
			if dfs:
				df_all = pd.concat(dfs, ignore_index=True)
				concat_path = NEW_DATA_DIR / "_concat_for_retrain.csv"
				df_all.to_csv(concat_path, index=False)
				logger.info("Triggering retraining from accumulated new data.")
				train(concat_path)
				clear_new_data()
