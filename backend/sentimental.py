import joblib
import re
import numpy as np

# Load the model and vectorizer once
_model = None
_vectorizer = None

def load_sentiment_model():
    global _model, _vectorizer
    try:
        _model = joblib.load("artifacts/sentiment_model.pkl")
        _vectorizer = joblib.load("artifacts/sentiment_vectorizer.pkl")
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
        return "neutral"
    
    if not text or len(text.strip()) == 0:
        return "neutral"
    
    try:
        # Clean text
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Vectorize
        vectorized = _vectorizer.transform([text])
        
        # Predict
        prediction = _model.predict(vectorized)
        
        # Map to sentiment label
        # Adjust based on your model's output
        sentiment_labels = ["negative", "neutral", "positive"]
        
        if len(prediction.shape) > 1:
            idx = np.argmax(prediction[0])
        else:
            idx = int(prediction[0] > 0.5)
        
        return sentiment_labels[idx] if idx < len(sentiment_labels) else "neutral"
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return "neutral"