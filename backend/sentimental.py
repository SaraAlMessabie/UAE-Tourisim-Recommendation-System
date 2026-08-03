import joblib
import re
import os
import numpy as np

# Load the model and vectorizer once
_model = None
_vectorizer = None

# Resolve paths relative to this file, not a hardcoded local machine path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models_and_vectors", "sentiment_model_best.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models_and_vectors", "sentiment_vectorizer.joblib")


def load_sentiment_model():
    global _model, _vectorizer
    try:
        _model = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VECTORIZER_PATH)
        print("Sentiment model loaded successfully")
    except Exception as e:
        print(f"Error loading sentiment model: {e}")
        _model = None
        _vectorizer = None


def predict_sentiment(text):
    """Predict sentiment of a review text."""
    global _model, _vectorizer

    if _model is None or _vectorizer is None:
        load_sentiment_model()

    if _model is None or _vectorizer is None:
        return "Error"

    if not text or len(text.strip()) == 0:
        return "Error"

    try:
        # Clean text
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # Vectorize
        vectorized = _vectorizer.transform([text])

        # Predict
        prediction = _model.predict(vectorized)

        sentiment_labels = ["negative", "positive"]

        if prediction.ndim > 1:
            idx = int(np.argmax(prediction[0]))
        else:
            idx = int(prediction[0]) 

        if 0 <= idx < len(sentiment_labels):
            return sentiment_labels[idx]
        return "neutral"

    except Exception as e:
        print(f"Prediction error: {e}")
        return "neutral"