# src/config.py

# Ordered list of all features expected in input
FEATURES = [
    "student_id",         # just for tracking, not used in model
    "age",
    "gender",
    "grade",
    "attendance",
    "marks",
    "income",
    "parent_education",
    "location",
    "prev_dropout",
    "financial_issue",
    "migration_flag",
    "health_issue",
    "school_dropout_rate"
]

# Which features are categorical
CATEGORICALS = [
    "gender",
    "parent_education",
    "location",
    "prev_dropout",
    "financial_issue",
    "migration_flag",
    "health_issue"
]

# Which features are numeric
NUMERICS = [
    "age",
    "grade",
    "attendance",
    "marks",
    "income",
    "school_dropout_rate"
]

# Dropout risk thresholds
RISK_THRESH = {
    "low": 0.33,
    "med": 0.66
}

# Paths (centralized so easy to change later)
DATA_PATH = "data/students_synth.csv"
PIPELINE_PATH = "artifacts/dropout_pipeline.joblib"
FEATURE_LIST_PATH = "artifacts/feature_list.json"
