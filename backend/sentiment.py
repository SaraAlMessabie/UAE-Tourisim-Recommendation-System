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


def predict_sentiment(text, model):
    try:
        cleaned = clean_text(text)

        prediction = model.predict([cleaned])

        sentiment_labels = ["negative", "positive"]
        idx = int(prediction[0])

        if 0 <= idx < len(sentiment_labels):
            return sentiment_labels[idx]
        return "None"

    except Exception as e:
        print(f"Prediction error: {e}")
        return "None"