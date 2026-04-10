import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "bank_marketing_model.joblib"

# Dataset Configuration
DATASET_ID = 222

# Training Configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model Hyperparameters
N_ESTIMATORS = 100
MAX_DEPTH = None

# API Configuration
PROJECT_NAME = "Bank Marketing ML API"
VERSION = "1.0.0"
