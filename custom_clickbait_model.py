# custom_clickbait_model.py (New File)
import joblib
import pandas as pd
import numpy as np
import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)

# NOTE: The custom model requires external NLP/Vectorizer files, which are not provided, 
# but are essential for the '__nlp__.' features listed in model.pkl. 
# We are making a highly simplified placeholder.
MODEL_FILE_PATH = "clickbait_model.pkl" 
CLICKBAIT_MODEL = None

# Based on features listed in model.pkl 
BASIC_FEATURES = [
    'headline.char_count', 'headline.word_count', 'headline.capital_ratio', 
    'headline.lower_ratio', 'headline.digit_ratio', 'headline.special_ratio', 
    'headline.symbol_count..', 'headline.symbol_ratio..', 'headline.symbol_count.:', 
    'headline.symbol_ratio.:', 'headline.symbol_count. ', 'headline.symbol_ratio. ', 
    'headline.symbol_count.-', 'headline.symbol_ratio.-',
]
# There are 300+ '__nlp__' features that must also be included in the DataFrame.
# Since we cannot accurately recreate them without the original vectorizer, 
# we rely on the AutoGluon structure and list the non-basic ones found[cite: 1, 2, 3, 4, 5, 6, 7].
# In a real setup, this list MUST be comprehensive.
NLP_PLACEHOLDERS = [
    '__nlp__.000', '__nlp__.10', '__nlp__.100', '__nlp__.11', '__nlp__.12', 
    # ... (Add all 300+ features from model.pkl for a production environment)
    '__nlp__.about', '__nlp__.about the', '__nlp__.actually', '__nlp__.amazing', 
    '__nlp__.if you', '__nlp__.if you re', '__nlp__.what your', '__nlp__.you ve', 
    '__nlp__._total_', # Found as a feature [cite: 7]
]

def load_clickbait_model():
    """Attempts to load the AutoGluon XGBoost model using joblib/pickle."""
    global CLICKBAIT_MODEL
    try:
        # Load the raw model object.
        CLICKBAIT_MODEL = joblib.load(MODEL_FILE_PATH)
        logger.info("Custom AutoGluon XGBoost model object loaded successfully.")
    except Exception as e:
        logger.error(f"FATAL: Could not load the custom clickbait model (model.pkl). Error: {e}", exc_info=True)
        CLICKBAIT_MODEL = None

def _generate_caption_features(caption: str) -> dict:
    """
    Generates all features (basic stats + NLP vector components) required by the model.
    NOTE: The NLP part is a highly simplified placeholder and will cause prediction failure 
    if not replaced with the original AutoGluon feature generator/vectorizer.
    """
    features = {}
    if not caption:
        features = {f: 0 for f in (BASIC_FEATURES + NLP_PLACEHOLDERS)}
        features['headline.word_count'] = 0
        return features

    # 1. Basic Statistical Features (as seen in model.pkl )
    text_len = len(caption)
    word_count = len(caption.split())
    
    special_chars = re.sub(r'[a-zA-Z0-9\s]', '', caption)
    
    features['headline.char_count'] = text_len
    features['headline.word_count'] = word_count
    features['headline.capital_ratio'] = sum(1 for c in caption if c.isupper()) / text_len
    features['headline.lower_ratio'] = sum(1 for c in caption if c.islower()) / text_len
    features['headline.digit_ratio'] = sum(1 for c in caption if c.isdigit()) / text_len
    features['headline.special_ratio'] = len(special_chars) / text_len
    
    # Symbol counts (e.g., periods, colons, spaces, hyphens) 
    features['headline.symbol_count..'] = caption.count('.')
    features['headline.symbol_ratio..'] = caption.count('.') / text_len
    features['headline.symbol_count.:'] = caption.count(':')
    features['headline.symbol_ratio.:'] = caption.count(':') / text_len
    features['headline.symbol_count. '] = caption.count(' ')
    features['headline.symbol_ratio. '] = caption.count(' ') / text_len
    features['headline.symbol_count.-'] = caption.count('-')
    features['headline.symbol_ratio.-'] = caption.count('-') / text_len
    
    # 2. NLP Features (Placeholders)
    # The actual implementation requires the original Autogluon featurizer, which is missing.
    # We must include the keys, or prediction will fail due to mismatching feature count.
    for nlp_key in NLP_PLACEHOLDERS:
        features[nlp_key] = 0.0 # Default to 0, which is likely wrong but necessary

    return features


def predict_clickbait_binary(caption: str) -> int:
    """
    Predicts clickbait (1 or 0) from the caption using the custom model.
    Returns 1 for clickbait, 0 for not, or -1 if the model is unavailable or prediction fails.
    """
    global CLICKBAIT_MODEL
    if CLICKBAIT_MODEL is None:
        return -1 
    
    try:
        # 1. Generate the expected features from the caption
        features_dict = _generate_caption_features(caption)
        
        # 2. Create the DataFrame for prediction
        # The key names must EXACTLY match the model's expected features.
        input_df = pd.DataFrame([features_dict])
        
        # 3. Predict the binary score (assuming it's a model with a .predict method)
        # We rely on the model object having a standard `predict` method.
        # It's an XGBClassifier[cite: 53], which returns a binary label (0 or 1).
        prediction = CLICKBAIT_MODEL.predict(input_df)[0]
        return int(prediction)

    except Exception as e:
        logger.error(f"Error during custom clickbait prediction. Prediction failed for the model: {e}", exc_info=True)
        return -1