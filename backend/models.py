from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional
from datetime import date
import pandas as pd


# --- Request schema (matches actual quiz form fields) ---
class VisitorProfileRequest(BaseModel):
    user_id: str
    trip_start_date: date
    trip_end_date: Optional[date] = None
    city: List[str]
    daily_food_budget: str
    daily_attraction_budget: str
    activity_preferences: List[str] = []
    activity_other: Optional[str] = None
    cuisine_preferences: List[str] = []
    cuisine_other: Optional[str] = None
    event_preferences: List[str] = []
    attraction_environment: List[str] = []
    weather_preference: List[str] = []
    crowdedness_preference: Optional[str] = None
    traveling_with_kids: bool
    num_recommendations: int = 5

    @field_validator("activity_other", "cuisine_other")

    @classmethod

    def clean_other_field(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        if len(value) > 100:
            raise ValueError("This field must be 100 characters or fewer.")
        return value

    @model_validator(mode="after")
    def check_trip_dates(self) -> "VisitorProfileRequest":
        if self.trip_end_date is not None and self.trip_end_date < self.trip_start_date:
            raise ValueError("Trip end date cannot be before trip start date.")
        return self
        
    additional_notes: Optional[str] = None


# --- Translation dict + builder functions (teammate's) ---
ACTIVITY_KEYWORDS = {
    "Culture focused (Museum/Galleries/..)": "culture museum galleries",
    "Nature focused (Beach/Desert/Parks/...)": "nature beach desert parks",
    "Shopping (Malls/Exhibitions/...)": "shopping malls exhibitions",
    "Adventure (Kayaking/ Desert Driving/ ...)": "adventure kayaking desert driving",
    "Kids Friendly": "kids friendly family",
}


def build_restaurant_profile(req: VisitorProfileRequest) -> dict:
    cuisines = list(req.cuisine_preferences)
    if req.cuisine_other:
        cuisines.append(req.cuisine_other)
    return {
        "city": req.city,
        "cuisines": cuisines,
        "budget": req.daily_food_budget,
        "top_n": req.num_recommendations,
    }


def build_event_profile(req: VisitorProfileRequest) -> dict:
    BUDGET_LEVELS = {
        "free": ["Free"],
        "low": ["Free", "Low"],
        "medium": ["Free", "Low", "Medium"],
        "high": ["Free", "Low", "Medium", "High"],
    }

    budget = BUDGET_LEVELS.get(
        req.daily_attraction_budget.lower(),
        ["Free", "Low", "Medium", "High"],
    )
    environment = (
        [e.capitalize() for e in req.attraction_environment]
        or ["Indoor", "Outdoor", "Both"]
    )
    trip_end = req.trip_end_date or req.trip_start_date

    return {
        "User_ID": req.user_id,
        "Trip_Start_Date": pd.Timestamp(req.trip_start_date),
        "Trip_End_Date": pd.Timestamp(trip_end),
        "City": req.city,
        "budget": budget,
        "Event_Preferences": req.event_preferences or None,
        "Activity_Preferences": req.activity_preferences,
        "Activity_Other": req.activity_other or "",
        "Cuisine_Preferences": req.cuisine_preferences,
        "Cuisine_Other": req.cuisine_other or "",
        "Environment": environment,
        "Family_Friendly": True if req.traveling_with_kids else None,
        "Temp_Pref": req.weather_preference or None,
        "Crowdedness_Preference": req.crowdedness_preference or "",
        "Num_Recommendations": req.num_recommendations,
        "special_note": req.additional_notes or "",
    }


def build_attraction_profile(req: VisitorProfileRequest) -> dict:
    selected_activities = req.activity_preferences
    keywords = [
        ACTIVITY_KEYWORDS.get(label, label)
        for label in selected_activities
    ]
    if req.activity_other:
        keywords.append(req.activity_other)
    environment = req.attraction_environment
    return {
        "city": req.city,
        "attraction_preference": keywords,
        "budget": req.daily_attraction_budget,
        "environment": environment,
        "with_kids": "Yes" if req.traveling_with_kids else "No",
        "top_n": req.num_recommendations,
    }


# --- Response schemas (yours) ---
class EventRecommendation(BaseModel):
    visitor_id: str
    Name: str
    Categories: list
    Location: str
    Start_Date: str
    End_Date: str
    Price_Range: str
    similarity_score: float
    fallback_stage: str
    final_score: float
    recommendation_reason: str
    Description: str


class RecommendationResponse(BaseModel):
    visitor_id: str
    num_results: int
    recommendations: List[EventRecommendation]


# --- Hearts / Reviews request schemas (yours) ---
class HeartRequest(BaseModel):
    user_id: str
    listing_type: str
    listing_id: str


class ReviewRequest(BaseModel):
    user_id: str
    listing_type: str
    listing_id: str
    rating: int
    comment: str
