import pickle
import joblib
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException

from catalogs import load_events_catalog, load_restaurants_catalog, load_attractions_catalog
from models import (
    VisitorProfileRequest,
    build_event_profile,
    build_restaurant_profile,
    build_attraction_profile,
    RecommendationResponse,
    HeartRequest,
    ReviewRequest,
)
from event_recommender import recommend_events
from restaurant_recommender import recommend_restaurants
from attraction_recommender import recommend_attractions_with_sentiment
from sentiment import predict_sentiment
from sheets import get_sheet, get_sheet_as_df


app = FastAPI(title="UAE Tourist Recommendation API")


# ---------------------------------------------------------------------------
# Startup: load every catalog + saved model artifact ONCE, not per-request
# ---------------------------------------------------------------------------

@app.on_event("startup")
def load_resources():
    global events_df, event_tfidf, event_vectors
    global restaurants_df, restaurant_tfidf, restaurant_vectors
    global attractions_df, attraction_tfidf, attraction_vectors
    global sentiment_model, sentiment_vectorizer


    # --- Static catalogs (from GitHub) ---
    events_df = load_events_catalog()
    events_df['Start_Date'] = pd.to_datetime(events_df['Start_Date'])
    events_df['End_Date'] = pd.to_datetime(events_df['End_Date'])

    restaurants_df = load_restaurants_catalog()
    attractions_df = load_attractions_catalog()

    # --- Saved vectorizers / similarity matrices ---
    with open("models and vectors/events_tfidf_vectorizer.pkl", "rb") as f:
        event_tfidf = pickle.load(f)
    with open("models and vectors/all_event_vectors.pkl", "rb") as f:
        event_vectors = pickle.load(f)

    restaurant_tfidf = joblib.load("models and vectors/restaurants_vectorizer.joblib")
    restaurant_vectors = joblib.load("models and vectors/restaurants_matrix.joblib")

    attraction_tfidf = joblib.load("models and vectors/attractions_vectorizer.joblib")
    attraction_vectors = joblib.load("models and vectors/attractions_matrix.joblib")

    sentiment_model = joblib.load("models and vectors/sentiment_model_best.joblib")
    sentiment_vectorizer = joblib.load("models and vectors/sentiment_vectorizer.joblib")
    print("All resources loaded successfully.")


# ---------------------------------------------------------------------------
# Shared helper: log every recommendation shown, regardless of listing type
# ---------------------------------------------------------------------------

def log_recommendations(records, user_id, listing_type, id_field, name_field):
    """records: list of dicts (already-serialized recommendation rows)."""
    try:
        sheet = get_sheet("Recommendation_Log")
        rows = []
        for row in records:
            listing_id = row.get(id_field) or row.get(name_field, "")
            rows.append([
                user_id,
                listing_type,
                listing_id,
                row.get("fallback_stage", ""),
                row.get("similarity_score", ""),
                row.get("final_score", ""),
                row.get("rank", ""),
                str(datetime.now()),
            ])
        if rows:
            sheet.append_rows(rows)
    except Exception as e:
        # Logging should never break a recommendation response
        print(f"Warning: failed to write Recommendation_Log: {e}")


# ---------------------------------------------------------------------------
# /recommend-events — uses your event_recommender (already returns JSON-ready dict)
# ---------------------------------------------------------------------------

@app.post("/recommend-events", response_model=RecommendationResponse)
def get_event_recommendations(request: VisitorProfileRequest):
    visitor_profile = build_event_profile(request)

    result = recommend_events(events_df, visitor_profile, event_tfidf, event_vectors)

    log_recommendations(
        result["recommendations"],
        user_id=request.user_id,
        listing_type="Event",
        id_field="Event_ID",
        name_field="Name",
    )

    return result


# ---------------------------------------------------------------------------
# /recommend-restaurants — teammate's function returns a raw DataFrame,
# so this endpoint has to convert it to a JSON-safe dict itself
# ---------------------------------------------------------------------------

@app.post("/recommend-restaurants")
def get_restaurant_recommendations(request: VisitorProfileRequest):
    visitor_profile = build_restaurant_profile(request)

    result_df = recommend_restaurants(visitor_profile, restaurants_df, restaurant_tfidf, restaurant_vectors)

    score_cols = [c for c in ['similarity_score', 'final_score'] if c in result_df.columns]
    if score_cols:
        result_df[score_cols] = result_df[score_cols].round(3)

    records = result_df.to_dict(orient="records")

    log_recommendations(
        records,
        user_id=request.user_id,
        listing_type="Restaurant",
        id_field="restaurant_id",
        name_field="restaurant_name",
    )

    return {
        "visitor_id": request.user_id,
        "num_results": len(records),
        "recommendations": records,
    }


# ---------------------------------------------------------------------------
# /recommend-attractions — same pattern as restaurants
# ---------------------------------------------------------------------------

@app.post("/recommend-attractions")
def get_attraction_recommendations(request: VisitorProfileRequest):
    visitor_profile = build_attraction_profile(request)

    result_df = recommend_attractions_with_sentiment(
        visitor_profile, attractions_df, attraction_tfidf, attraction_vectors
    )

    score_cols = [c for c in ['similarity_score', 'final_score'] if c in result_df.columns]
    if score_cols:
        result_df[score_cols] = result_df[score_cols].round(3)

    records = result_df.to_dict(orient="records")

    log_recommendations(
        records,
        user_id=request.user_id,
        listing_type="Attraction",
        id_field="attraction_id",
        name_field="name",
    )

    return {
        "visitor_id": request.user_id,
        "num_results": len(records),
        "recommendations": records,
    }


# ---------------------------------------------------------------------------
# Hearts (likes) — shared across all three listing types
# ---------------------------------------------------------------------------

@app.post("/hearts")
def add_heart(heart: HeartRequest):
    try:
        sheet = get_sheet("Hearts")
        sheet.append_row([
            heart.user_id, heart.listing_type, heart.listing_id, str(datetime.now())
        ])
        return {"status": "saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save like: {e}")


@app.get("/hearts/{user_id}")
def get_hearts(user_id: str):
    try:
        df = get_sheet_as_df("Hearts")
        if df.empty:
            return []
        user_hearts = df[df["User_ID"] == user_id] if "User_ID" in df.columns else df[df["user_id"] == user_id]
        return user_hearts.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch likes: {e}")


# ---------------------------------------------------------------------------
# Reviews — with sentiment prediction wired in at write-time
# ---------------------------------------------------------------------------

@app.post("/reviews")
def add_review(review: ReviewRequest):
    try:
        sentiment = predict_sentiment(review.comment)
        sheet = get_sheet("Reviews")
        sheet.append_row([
            review.user_id, review.listing_type, review.listing_id,
            review.rating, review.comment, sentiment, str(datetime.now())
        ])
        return {"status": "saved", "sentiment": sentiment}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save review: {e}")


@app.get("/reviews/{listing_id}")
def get_reviews(listing_id: str):
    try:
        df = get_sheet_as_df("Reviews")
        if df.empty:
            return []
        listing_reviews = df[df["Listing_ID"] == listing_id] if "Listing_ID" in df.columns else df[df["listing_id"] == listing_id]
        return listing_reviews.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch reviews: {e}")