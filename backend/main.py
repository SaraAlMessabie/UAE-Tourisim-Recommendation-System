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
    UserEmailRequest
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

#checks it writes back to google sheet
def append_row_safe(sheet, row: list, context: str = "row"):
    try:
        result = sheet.append_row(row)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write {context} to Google Sheets: {e}")


#checks if trip date is in range
def validate_trip_dates(request: VisitorProfileRequest):
    if request.trip_end_date < request.trip_start_date:
        raise HTTPException(
            status_code=400,
            detail="trip_end_date cannot be before trip_start_date.",
        )


import time

_sheet_cache: dict = {}
CACHE_TTL_SECONDS = 15  # how long a cached read stays valid before refetching


def get_sheet_as_df_cached(tab_name: str):
    now = time.time()
    cached = _sheet_cache.get(tab_name)

    if cached and (now - cached["time"] < CACHE_TTL_SECONDS):
        return cached["df"].copy()  # .copy() so callers mutating the df don't corrupt the cache

    df = get_sheet_as_df(tab_name)
    _sheet_cache[tab_name] = {"df": df, "time": now}
    return df.copy()


def invalidate_sheet_cache(tab_name: str):
    """Call this after a write to that tab, so the next read reflects the change immediately."""
    _sheet_cache.pop(tab_name, None)
    
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
            log_id = f"L-{uuid.uuid4().hex[:8]}"
            rows.append([
                log_id,
                user_id,
                listing_type,
                listing_id,
                row.get("fallback_stage", ""),
                row.get("final_score", ""),
                row.get("similarity_score", ""),
                row.get("rank", ""),
                str(datetime.now()),
            ])
        if rows:
            sheet.append_rows(rows)
            invalidate_sheet_cache("recommendation_log")
    except Exception as e:
        # Logging should never break a recommendation response
        print(f"Warning: failed to write Recommendation_Log: {e}")



@app.post("/recommend-events", response_model=RecommendationResponse)
def get_event_recommendations(request: VisitorProfileRequest):
    validate_trip_dates(request)
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

    df = get_sheet_as_df_cached("hearts")   # ← cached read
    if not df.empty:
        user_col = "User_ID" if "User_ID" in df.columns else "user_id"
        type_col = "Listing_Type" if "Listing_Type" in df.columns else "listing_type"
        id_col = "Listing_ID" if "Listing_ID" in df.columns else "listing_id"

        df[user_col] = df[user_col].astype(str)
        df[id_col] = df[id_col].astype(str)

        duplicate = df[
            (df[user_col] == str(heart.user_id)) &
            (df[type_col].str.lower() == heart.listing_type.lower()) &
            (df[id_col] == str(heart.listing_id))
        ]
        if not duplicate.empty:
            raise HTTPException(status_code=409, detail="This listing is already hearted by this user.")

    heart_id = f"H-{uuid.uuid4().hex[:8]}"
    sheet = get_sheet("hearts")

    append_row_safe(
        sheet,
        [heart_id, heart.user_id, heart.listing_type, heart.listing_id, str(datetime.now())],
        context="heart",
    )
    invalidate_sheet_cache("hearts")   # ← so the new heart shows up immediately, not after 15s

    return {"status": "saved", "heart_id": heart_id}


@app.get("/hearts/{user_id}")
def get_hearts(user_id: str, listing_type: Optional[str] = None, listing_id: Optional[str] = None):
    if listing_type is not None and listing_id is not None:
        validate_listing_exists(listing_type, listing_id)
    try:
        df = get_sheet_as_df_cached("hearts")   # ← cached read
        if df.empty:
            return []

        user_col = "User_ID" if "User_ID" in df.columns else "user_id"
        type_col = "Listing_Type" if "Listing_Type" in df.columns else "listing_type"
        id_col = "Listing_ID" if "Listing_ID" in df.columns else "listing_id"

        df[user_col] = df[user_col].astype(str)
        df[id_col] = df[id_col].astype(str)

        user_hearts = df[df[user_col] == str(user_id)]

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

    sentiment = predict_sentiment(comment, resources["sentiment_model"])
    review_id = f"R-{uuid.uuid4().hex[:8]}"
    sheet = get_sheet("reviews")

    append_row_safe(
        sheet,
        [review_id, review.user_id, review.listing_type, review.listing_id,
         review.rating, comment, sentiment, str(datetime.now())],
        context="review",
    )
    invalidate_sheet_cache("reviews")   # ← so this review shows up immediately

    return {"status": "saved", "review_id": review_id, "sentiment": sentiment}


@app.get("/reviews/{listing_type}/{listing_id}")
def get_reviews(listing_type: str, listing_id: str):
    validate_listing_exists(listing_type, listing_id)
    try:
        df = get_sheet_as_df_cached("reviews")   # ← cached read
        if df.empty:
            return []

        id_col = "Listing_ID" if "Listing_ID" in df.columns else "listing_id"
        type_col = "Listing_Type" if "Listing_Type" in df.columns else "listing_type"

        df[id_col] = df[id_col].astype(str).str.strip()
        df[type_col] = df[type_col].astype(str).str.strip()

        listing_reviews = df[
            (df[id_col] == str(listing_id).strip()) &
            (df[type_col].str.lower() == listing_type.strip().lower())
        ]
        return listing_reviews.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch reviews: {e}")
        

@app.get("/browse/{listing_type}")
def browse_listings(listing_type: str, limit: Optional[int] = None):
    """
    Returns the full, unfiltered catalog for a listing type.
    No quiz/profile matching — just the raw table, for users who skip the quiz.
    """
    listing_type_key = (listing_type or "").strip().lower()

    lookup = LISTING_CATALOG_LOOKUP.get(listing_type_key)
    if lookup is None:
        raise HTTPException(status_code=400, detail=f"Unknown listing_type: {listing_type}")

    df_key, _ = lookup
    catalog_df = resources.get(df_key)

    if catalog_df is None:
        raise HTTPException(status_code=500, detail=f"{listing_type} catalog is not loaded.")

    df = catalog_df.copy()
    if limit is not None:
        df = df.head(limit)
        
    df = df.astype(object).where(pd.notna(df), None)

    df = df.astype(object).where(pd.notna(df),None)

    records = df.to_dict(orient="records")

    return {
        "listing_type": listing_type_key,
        "num_results": len(records),
        "recommendations": records,
    }

def get_or_create_user(email: str) -> str:
    email = (email or "").strip().lower()
    df = get_sheet_as_df("users")

    if not df.empty:
        email_col = "user_email" if "user_email" in df.columns else "User_Email"
        id_col = "User_ID" if "User_ID" in df.columns else "user_id"
        df[email_col] = df[email_col].astype(str).str.strip().str.lower()

        match = df[df[email_col] == email]
        if not match.empty:
            return str(match.iloc[0][id_col])

    user_id = f"U-{uuid.uuid4().hex[:8]}"
    sheet = get_sheet("users")
    append_row_safe(sheet, [user_id, email], context="user")
    return user_id

@app.post("/users")
def register_user(payload: UserEmailRequest):
    email = (payload.email or "").strip().lower()
    if "@" not in email or "." not in email:
        raise HTTPException(status_code=400, detail="Invalid email address.")

    user_id = get_or_create_user(email)
    return {"status": "ok", "user_id": user_id, "email": email}



def log_trip_preferences(request: VisitorProfileRequest) -> str:
    preference_id = f"P-{uuid.uuid4().hex[:8]}"
    sheet = get_sheet("trip_preferencee")

    row = [
        preference_id,
        request.user_id,
        ", ".join(request.city),
        str(request.trip_start_date),
        str(request.trip_end_date) if request.trip_end_date else "",
        request.daily_food_budget,
        request.daily_attraction_budget,
        ", ".join(request.activity_preferences),
        request.activity_other or "",
        ", ".join(request.cuisine_preferences),
        request.cuisine_other or "",
        ", ".join(request.event_preferences),
        ", ".join(request.attraction_environment), 
        ", ".join(request.weather_preference),
        request.traveling_with_kids,
        request.num_recommendations,
        str(datetime.now()),
    ]

    append_row_safe(sheet, row, context="trip preference")
    return preference_id


@app.post("/trip-preferences")
def submit_trip_preferences(request: VisitorProfileRequest):
    if request.trip_end_date is not None and request.trip_end_date < request.trip_start_date:
        raise HTTPException(status_code=400, detail="trip_end_date cannot be before trip_start_date.")

    preference_id = log_trip_preferences(request)
    return {"status": "saved", "preference_id": preference_id}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
