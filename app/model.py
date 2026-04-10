import joblib
import pandas as pd
import logging
from typing import Dict

from app.config import MODEL_PATH
from app.schemas import BankMarketingInput

logger = logging.getLogger(__name__)

class ModelHandler:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelHandler, cls).__new__(cls)
            cls._instance.model_artifact = None
            cls._instance.is_loaded = False
        return cls._instance

    def load_model(self):
        """Loads the joblib artifact if it exists and hasn't been loaded yet."""
        if self.is_loaded:
            logger.info("Model is already loaded.")
            return

        try:
            logger.info(f"Loading model from {MODEL_PATH}...")
            self.model_artifact = joblib.load(MODEL_PATH)
            self.is_loaded = True
            logger.info("Model loaded successfully.")
        except FileNotFoundError:
            logger.error(f"Model file not found at {MODEL_PATH}.")
            self.model_artifact = None
            self.is_loaded = False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_artifact = None
            self.is_loaded = False

    def predict(self, input_data: BankMarketingInput) -> Dict:
        """Runs the data through the loaded model and returns prediction & probability."""
        if not self.is_loaded or self.model_artifact is None:
            raise RuntimeError("Model is not loaded.")

        pipeline = self.model_artifact['model']
        
        # Convert input Pydantic model to a Pandas DataFrame
        df = pd.DataFrame([input_data.model_dump(by_alias=True)])

        # Predict value and probabilities
        prediction_val = pipeline.predict(df)[0]
        prediction_probabilities = pipeline.predict_proba(df)[0]
        
        # Determine the probability of the 'yes' class
        class_idx = 1
        clf = pipeline.named_steps.get('classifier')
        if hasattr(clf, 'classes_'):
            classes = list(clf.classes_)
            if 'yes' in classes:
                class_idx = classes.index('yes')
        
        probability_val = float(prediction_probabilities[class_idx])
        
        # Normalize prediction output to strict literal "yes" or "no"
        pred_label = "yes"
        if prediction_val == "no" or (isinstance(prediction_val, bool) and not prediction_val) or (isinstance(prediction_val, int) and prediction_val == 0):
            pred_label = "no"

        return {
            "prediction": pred_label,
            "probability": probability_val
        }

    def get_metadata(self) -> Dict:
        """Returns the training metrics embedded in the joblib."""
        if not self.is_loaded or self.model_artifact is None:
            raise RuntimeError("Model is not loaded.")
        
        return self.model_artifact.get('metrics', {})
