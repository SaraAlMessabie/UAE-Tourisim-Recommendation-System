import re
import numpy as np

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"<.*?>", " ", text)
    
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"n't", " not", text) 
    
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    
    text = re.sub(r"[^a-z0-9\s!?]", " ", text) 
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_sentiment(text, model, vectorizer):
    try:
        # Clean text
        text = clean_text(text)

        # Vectorize
        vectorized = vectorizer.transform([text])

        # Predict
        prediction = model.predict(vectorized)

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