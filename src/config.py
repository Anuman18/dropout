import os
from pathlib import Path

# Directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
NEW_DATA_DIR = DATA_DIR / "new_data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for _dir in (DATA_DIR, NEW_DATA_DIR, MODELS_DIR, LOGS_DIR):
	_dir.mkdir(parents=True, exist_ok=True)

# Canonical students CSV
STUDENTS_FILE = DATA_DIR / "students.csv"

# File naming
DEFAULT_DATASET_NAME = "rajasthan_students.csv"
LATEST_DATA_SYMLINK = DATA_DIR / "latest.csv"
# Current model symlinks
MODEL_FILE = MODELS_DIR / "model_current.pkl"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor_current.pkl"
# Compatibility name expected by requirement
MODEL_FILE_COMPAT = MODELS_DIR / "dropout_model.pkl"
FEATURE_LIST_FILE = MODELS_DIR / "feature_list.pkl"
GLOBAL_IMPORTANCE_FILE = MODELS_DIR / "global_importance.json"
THRESHOLD_FILE = MODELS_DIR / "threshold.json"
METADATA_FILE = MODELS_DIR / "metadata.json"

# Logging
APP_LOG_FILE = LOGS_DIR / "app.log"
PREDICTION_LOG_FILE = LOGS_DIR / "predictions.csv"

# ML settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 300
MAX_DEPTH = None

# Risk tier thresholds
RISK_GREEN_MAX = float(os.getenv("RISK_GREEN_MAX", 0.33))  # <= green
RISK_YELLOW_MAX = float(os.getenv("RISK_YELLOW_MAX", 0.66))  # <= yellow, else red

# Feature definitions
NUMERIC_FEATURES = [
	"attendance_rate",
	"avg_marks",
	"household_income",
	"distance_to_school_km",
	"teacher_student_ratio",
	"electricity_availability",
	"internet_availability",
	"midday_meal",
	"toilet_availability",
	"books_supplied",
	"transport_facility",
]

CATEGORICAL_FEATURES = [
	"caste_category",
	"language_medium",
	"govt_schemes_access",
	"parent_occupation",
	"mother_education",
	"father_education",
	"family_size",
	"gender",
	"location",
]

TARGET_COLUMN = "dropout_label"

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Allowed categorical values (for dataset generator and validation)
CATEGORIES = {
	"caste_category": ["GEN", "OBC", "SC", "ST"],
	"language_medium": ["Hindi", "English", "Urdu"],
	"govt_schemes_access": ["None", "Scholarship", "Uniform", "Books", "Multiple"],
	"parent_occupation": ["Agriculture", "Labor", "Service", "Business", "Unemployed"],
	"mother_education": ["None", "Primary", "Secondary", "Graduate", "Postgraduate"],
	"father_education": ["None", "Primary", "Secondary", "Graduate", "Postgraduate"],
	"family_size": ["Small", "Medium", "Large"],
	"gender": ["Male", "Female"],
	"location": ["Rural", "Urban"],
}

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# Database
DB_URL = os.getenv("DB_URL", f"sqlite:///{(DATA_DIR / 'app.db').as_posix()}")

# Scheduler
NEW_DATA_THRESHOLD = int(os.getenv("NEW_DATA_THRESHOLD", 500))
RETRAIN_CRON = os.getenv("RETRAIN_CRON", "0 2 * * *")  # daily 2am by default
