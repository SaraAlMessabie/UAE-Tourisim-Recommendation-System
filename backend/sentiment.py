import joblib
import re
import os
import numpy as np

# Load the model and vectorizer once
_model = None
_vectorizer = None



def predict_sentiment(text):
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
        return "None"

    except Exception as e:
        print(f"Prediction error: {e}")
        return "None"