# UAE Tourist Recommendation System

A tourist recommendation system for the UAE that generates personalized suggestions for **Events**, **Restaurants**, and **Attractions** based on a content-based recommendation model using cosine similarity. The system also collects user reviews and automatically assigns each one a **Positive** or **Negative** sentiment using a custom-trained sentiment analysis model.

**Live demo:** [uae-tourisim-recommendation-system.onrender.com](https://uae-tourisim-recommendation-system.onrender.com)
**Streamlit Frontend:** [https://iris-api-k8mmdmwepoer28vfzcnbaw.streamlit.app]

---

## Datasets Used

| Component | Source |
|---|---|
| Sentiment analysis model | [Yelp Review Dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset) (open license) |
| Attractions & Events | Web-scraped data, collected with AI assistance |
| Restaurants | Zomato dataset |

---

## Model Performance

The sentiment model was evaluated against a dummy baseline (most-frequent-class predictor):

| Model | F1 Score |
|---|---|
| Dummy baseline (most frequent class) | 76% |
| Our sentiment model | 97% |

---

## Limitations

- The sentiment model does not reliably detect sarcasm or Gen Z slang, since the Yelp dataset it was trained on predates much of that vocabulary.
- The Attractions/Events dataset would benefit from more data collection to support finer-grained personalization.

---

## How to Deploy This Project

### 1. Environment Variables
Add the following environment variables to your Render service configuration:

- `GOOGLE_SERVICE_ACCOUNT_CREDENTIALS` — the contents of your Google service account JSON key
- 'PYTHON_VERSION' - 

### 2. Google Sheets Setup
1. Create a new Google Sheet with tabs for `Hearts`, `Reviews`, `Trip_Preferences`, and `Recommendation_Log`.
2. Create a Google Cloud service account and enable the **Google Sheets API** and **Google Drive API**.
3. Share the Sheet with the service account's email (found in the credentials JSON) and grant **Editor** access.
4. Add the service account credentials to Render as an environment variable.

### 3. Deploy
1. Push the repository to GitHub and connect it to Render (or your preferred hosting platform), specifying the build and start commands for the FastAPI backend and Streamlit frontend.
2. Change any Path in the backend code files for attractions/events/restaurants to connect to your github csv.
3. You can add more data to the csv but any new columns will lead to adjustment needing to be made in the recommender and backend.
---

## Data Licensing

- The Yelp Review Dataset is used under its open license terms as published on Kaggle.
- Attraction/Event data was collected via AI-assisted web scraping for academic/demo purposes.
- The restaurant dataset is sourced from Zomato and used for academic/demo purposes only
