import joblib
from contextlib import asynccontextmanager
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import os
import uuid
from typing import Optional

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


# ---------------------------------------------------------------------------
# Resource container — replaces the old `global` variables from on_event
# ---------------------------------------------------------------------------

resources: dict = {}

# mapping for verification
LISTING_CATALOG_LOOKUP = {
    "event": ("events_df", "Event_ID"),
    "restaurant": ("restaurants_df", "restaurant_id"),
    "attraction": ("attractions_df", "attraction_id"),
}

# ---------------------------------------------------------------------------
# Lifespan: load every catalog + saved model artifact ONCE, not per-request
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models and vectors")

@asynccontextmanager
async def lifespan(app: FastAPI):
    events_df = load_events_catalog()
    events_df['Start_Date'] = pd.to_datetime(events_df['Start_Date'])
    events_df['End_Date'] = pd.to_datetime(events_df['End_Date'])

    restaurants_df = load_restaurants_catalog()
    attractions_df = load_attractions_catalog()

    # --- Saved vectorizers / similarity matrices ---
    event_tfidf = joblib.load(os.path.join(MODELS_DIR, "events_tfidf_vectorizer.pkl"))
    event_vectors = joblib.load(os.path.join(MODELS_DIR, "all_event_vectors.pkl"))

    restaurant_tfidf = joblib.load(os.path.join(MODELS_DIR, "restaurants_vectorizer.joblib"))
    restaurant_vectors = joblib.load(os.path.join(MODELS_DIR, "restaurants_matrix.joblib"))

    attraction_tfidf = joblib.load(os.path.join(MODELS_DIR, "attractions_vectorizer.joblib"))
    attraction_vectors = joblib.load(os.path.join(MODELS_DIR, "attractions_matrix.joblib"))

    sentiment_model = joblib.load(os.path.join(MODELS_DIR, "sentiment_model_best.joblib"))
    sentiment_vectorizer = joblib.load(os.path.join(MODELS_DIR, "sentiment_vectorizer.joblib"))

    resources.update({
        "events_df": events_df,
        "event_tfidf": event_tfidf,
        "event_vectors": event_vectors,
        "restaurants_df": restaurants_df,
        "restaurant_tfidf": restaurant_tfidf,
        "restaurant_vectors": restaurant_vectors,
        "attractions_df": attractions_df,
        "attraction_tfidf": attraction_tfidf,
        "attraction_vectors": attraction_vectors,
        "sentiment_model": sentiment_model,
        "sentiment_vectorizer": sentiment_vectorizer,
    })

    print("All resources loaded successfully.")

    yield

    resources.clear()


app = FastAPI(title="UAE Tourist Recommendation API", lifespan=lifespan)

#checks if the listing is real
def validate_listing_exists(listing_type: str, listing_id: str) -> str:
    listing_type_key = (listing_type or "").strip().lower()

    df_key, id_col = LISTING_CATALOG_LOOKUP[listing_type_key]
    catalog_df = resources.get(df_key)

    if catalog_df is not None and id_col in catalog_df.columns:
        valid_ids = catalog_df[id_col].astype(str).values
        if str(listing_id) not in valid_ids:
            raise HTTPException(
                status_code=400,
                detail=f"{listing_type} with ID {listing_id} does not exist.",
            )

    return listing_type_key

# ---------------------------------------------------------------------------
# Shared helper: log every recommendation shown, regardless of listing type
# ---------------------------------------------------------------------------

def log_recommendations(records, user_id, listing_type, id_field, name_field):
    """records: list of dicts (already-serialized recommendation rows)."""
    try:
        sheet = get_sheet("recommendation_log")
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



@app.post("/recommend-events", response_model=RecommendationResponse)
def get_event_recommendations(request: VisitorProfileRequest):
    visitor_profile = build_event_profile(request)

    result = recommend_events(
        resources["events_df"],
        visitor_profile,
        resources["event_tfidf"],
        resources["event_vectors"],
    )

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

    result_df = recommend_restaurants(
        visitor_profile,
        resources["restaurants_df"],
        resources["restaurant_tfidf"],
        resources["restaurant_vectors"],
    )

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
        visitor_profile,
        resources["attractions_df"],
        resources["attraction_tfidf"],
        resources["attraction_vectors"],
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
    validate_listing_exists(heart.listing_type, heart.listing_id)
    try:
        df = get_sheet_as_df("hearts")
        if not df.empty:
            user_col = "User_ID" if "User_ID" in df.columns else "user_id"
            type_col = "Listing_Type" if "Listing_Type" in df.columns else "listing_type"
            id_col = "Listing_ID" if "Listing_ID" in df.columns else "listing_id"

            df[id_col] = df[id_col].astype(str)

            duplicate = df[
                (df[user_col] == heart.user_id) &
                (df[type_col].str.lower() == heart.listing_type.lower()) &
                (df[id_col] == str(heart.listing_id))
            ]
            if not duplicate.empty:
                raise HTTPException(
                    status_code=409,
                    detail="This listing is already hearted by this user.",
                )

        heart_id = f"H-{uuid.uuid4().hex[:8]}"
        sheet = get_sheet("hearts")
        sheet.append_row([
            heart_id, heart.user_id, heart.listing_type, heart.listing_id, str(datetime.now())
        ])
        return {"status": "saved", "heart_id": heart_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save like: {e}")

@app.get("/hearts/{user_id}")
def get_hearts(user_id: str, listing_type: Optional[str] = None, listing_id: Optional[str] = None):
    try:
        df = get_sheet_as_df("hearts")
        if df.empty:
            return []

        user_col = "User_ID" if "User_ID" in df.columns else "user_id"
        type_col = "Listing_Type" if "Listing_Type" in df.columns else "listing_type"
        id_col = "Listing_ID" if "Listing_ID" in df.columns else "listing_id"

        df[id_col] = df[id_col].astype(str)

        user_hearts = df[df[user_col] == user_id]

        if listing_type is not None:
            user_hearts = user_hearts[user_hearts[type_col].str.lower() == listing_type.lower()]
        if listing_id is not None:
            user_hearts = user_hearts[user_hearts[id_col] == str(listing_id)]

        return user_hearts.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch likes: {e}")




@app.post("/reviews")
def add_review(review: ReviewRequest):
    comment = review.comment.strip() if review.comment else ""
    if not comment:
        raise HTTPException(status_code=400, detail="Comment cannot be empty.")

    validate_listing_exists(review.listing_type, review.listing_id)

    try:
        sentiment = predict_sentiment(comment, resources["sentiment_model"])
        review_id = f"R-{uuid.uuid4().hex[:8]}"
        sheet = get_sheet("reviews")
        sheet.append_row([
            review_id, review.user_id, review.listing_type, review.listing_id,
            review.rating, comment, sentiment, str(datetime.now())
        ])
        return {"status": "saved", "review_id": review_id, "sentiment": sentiment}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save review: {e}")

    

@app.get("/reviews/{listing_type}/{listing_id}")
def get_reviews(listing_type: str, listing_id: str):
    try:
        df = get_sheet_as_df("reviews")
        if df.empty:
            return []

        id_col = "Listing_ID" if "Listing_ID" in df.columns else "listing_id"
        type_col = "Listing_Type" if "Listing_Type" in df.columns else "listing_type"

        # Sheets read numbers as ints via get_all_records  compare as strings to be safe
        df[id_col] = df[id_col].astype(str)

        listing_reviews = df[
            (df[id_col] == str(listing_id)) &
            (df[type_col].str.lower() == listing_type.lower())
        ]
        return listing_reviews.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch reviews: {e}")



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
