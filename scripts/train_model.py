import os
import sys
import logging
from pathlib import Path

# Add project root to path to allow importing from 'app'
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from app.config import (
    DATASET_ID,
    MODEL_PATH,
    MODEL_DIR,
    TEST_SIZE,
    RANDOM_STATE,
    N_ESTIMATORS,
    MAX_DEPTH
)

# Set up logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    logger.info(f"Fetching dataset with ID: {DATASET_ID}")
    try:
        bank_marketing = fetch_ucirepo(id=DATASET_ID)
    except Exception as e:
        logger.error(f"Error fetching dataset: {e}")
        return

    # Extract features and targets
    X = bank_marketing.data.features
    y = bank_marketing.data.targets

    logger.info("Identifying categorical and numerical columns dynamically")
    # Identify numerical and categorical features dynamically
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    logger.info(f"Found {len(numeric_features)} numerical and {len(categorical_features)} categorical features")

    # Define transformers
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    logger.info("Building preprocessing ColumnTransformer")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    logger.info(f"Building RandomForest pipeline (n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH})")
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=N_ESTIMATORS, 
            max_depth=MAX_DEPTH, 
            random_state=RANDOM_STATE
        ))
    ])

    logger.info("Splitting dataset into train and test sets")
    # Convert y to 1D array if it's a pandas DataFrame
    if isinstance(y, pd.DataFrame):
        y = y.squeeze()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    logger.info("Training the model...")
    pipeline.fit(X_train, y_train)

    logger.info("Evaluating the model on the test set...")
    y_pred = pipeline.predict(X_test)
    
    # Using 'weighted' average in case of multi-class or binary with string labels
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    logger.info(f"Metrics - Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

    logger.info(f"Ensuring model directory exists at {MODEL_DIR}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model pipeline and metrics to {MODEL_PATH}")
    artifact = {
        'model': pipeline,
        'metrics': {
            'accuracy': acc,
            'f1_score': f1
        }
    }
    
    try:
        joblib.dump(artifact, MODEL_PATH)
        logger.info("Model saved successfully.")
    except Exception as e:
        logger.error(f"Error saving model: {e}")

if __name__ == "__main__":
    main()
