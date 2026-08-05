import streamlit as st
import requests
import re
from datetime import date, datetime

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE_URL = "https://uae-tourisim-recommendation-system.onrender.com"  # <-- update to your real Render URL

LISTING_TYPES = ["Event", "Restaurant", "Attraction"]
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

st.set_page_config(page_title="UAE Tourist Recommendations", layout="wide")

# ---------------------------------------------------------------------------
# Session state setup
# ---------------------------------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "landing"
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "recommendations" not in st.session_state:
    st.session_state.recommendations = {"Event": [], "Restaurant": [], "Attraction": []}
if "hearted_ids" not in st.session_state:
    # local cache so heart buttons can reflect state without re-fetching every render
    st.session_state.hearted_ids = set()


def go_to(page: str):
    st.session_state.page = page


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def api_post(path: str, payload: dict):
    try:
        response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=30)
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            return None, detail
        return response.json(), None
    except requests.exceptions.RequestException as e:
        return None, str(e)


def api_get(path: str, params: dict = None):
    try:
        response = requests.get(f"{API_BASE_URL}{path}", params=params or {}, timeout=30)
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            return None, detail
        return response.json(), None
    except requests.exceptions.RequestException as e:
        return None, str(e)


def load_user_hearts(user_id: str):
    """
    Fetches every heart this user has ever saved (across all listing types)
    and populates hearted_ids, so cards correctly show 'Hearted' state as
    soon as they're rendered — not just for hearts made in this session.
    """
    data, error = api_get(f"/hearts/{user_id}")
    if error:
        # Non-fatal — just means heart buttons won't pre-fill; the app still works.
        st.session_state.hearted_ids = set()
        return

    hearted = set()
    for row in (data or []):
        listing_type = row.get("Listing_Type") or row.get("listing_type")
        listing_id = row.get("Listing_ID") or row.get("listing_id")
        if listing_type and listing_id is not None:
            hearted.add((listing_type, str(listing_id)))

    st.session_state.hearted_ids = hearted


def register_user(email: str):
    """
    Logs this email in the Users sheet (create-if-new). Non-fatal on
    failure — the app's actual identity model still just uses the email
    string as user_id everywhere else, this is purely a registry log.
    """
    result, error = api_post("/users", {"email": email})
    if error:
        st.warning(f"Could not register user (non-critical): {error}")


# ---------------------------------------------------------------------------
# Shared component: a single listing card (event, restaurant, or attraction)
# ---------------------------------------------------------------------------

def get_listing_id(listing: dict, listing_type: str):
    id_field_candidates = {
        "Event": ["Event_ID", "event_id", "EventID", "eventId", "id", "ID"],
        "Restaurant": ["restaurant_id", "Restaurant_ID", "restaurantId", "id", "ID"],
        "Attraction": ["attraction_id", "Attraction_ID", "attractionId", "id", "ID"],
    }
    for field in id_field_candidates.get(listing_type, []):
        value = listing.get(field)
        if value not in (None, ""):
            return value

    # Last resort: scan every key case-insensitively for anything containing "id"
    for key, value in listing.items():
        if "id" in key.lower() and value not in (None, ""):
            return value

    return None


def get_listing_name(listing: dict, listing_type: str):
    name_field_candidates = {
        "Event": ["Name", "name", "event_name", "Event_Name"],
        "Restaurant": ["restaurant_name", "Restaurant_Name", "name", "Name"],
        "Attraction": ["name", "Name", "attraction_name", "Attraction_Name"],
    }
    for field in name_field_candidates.get(listing_type, []):
        value = listing.get(field)
        if value not in (None, ""):
            return value
    return "Untitled"


def get_listing_description(listing: dict, listing_type: str):
    if listing_type == "Event":
        return listing.get("Description")
    if listing_type == "Restaurant":
        # Restaurants have no free-text description — build one from cuisines
        cuisines = listing.get("cuisines")
        return f"Cuisines: {cuisines}" if cuisines else None
    if listing_type == "Attraction":
        return listing.get("description")
    return None


def get_listing_details(listing: dict, listing_type: str):
    """Returns an ordered list of (label, value) detail pairs, per listing type,
    based on each catalog's real column names."""
    if listing_type == "Event":
        fields = [
            ("Location", "Location"),
            ("Environment", "Environment"),
            ("Price range", "Price_Range"),
            ("Dates", None),  # handled specially below
            ("Family friendly", "Family_Friendly"),
        ]
        details = []
        for label, key in fields:
            if key and listing.get(key) not in (None, "", "nan"):
                details.append(f"**{label}:** {listing[key]}")
        start = listing.get("Start_Date")
        end = listing.get("End_Date")
        if start and end:
            details.insert(3, f"**Dates:** {start} → {end}")
        return details

    if listing_type == "Restaurant":
        fields = [
            ("City", "city"),
            ("Locality", "locality"),
            ("Cost for two", "average_cost_for_two"),
            ("Currency", "currency"),
            ("Price range", "price_range"),
            ("Rating", "rating"),
            ("Rating", "rating_text"),
            ("Votes", "votes"),
        ]
        details = []
        for label, key in fields:
            if listing.get(key) not in (None, "", "nan"):
                details.append(f"**{label}:** {listing[key]}")
        return details

    if listing_type == "Attraction":
        fields = [
            ("City", "city"),
            ("Location", "location"),
            ("Categories", "categories"),
            ("Environment", "environment"),
            ("Price level", "price_level"),
            ("Rating", "rating"),
            ("Kids friendly", "kids_friendly"),
            ("Visitor sentiment", "normalized_sentiment"),
        ]
        details = []
        for label, key in fields:
            if listing.get(key) not in (None, "", "nan"):
                details.append(f"**{label}:** {listing[key]}")
        return details

    return []


def render_listing_card(listing: dict, listing_type: str, index: int = 0):
    listing_id = get_listing_id(listing, listing_type)
    name = get_listing_name(listing, listing_type)

    # Unique key suffix — combines listing_id (when available) with the
    # loop index, so cards never collide even if ID extraction fails
    # (e.g. a mismatched field name returning None for every row).
    key_suffix = f"{listing_type}_{listing_id}_{index}"

    if listing_id is None:
        st.warning(
            f"⚠️ Could not find an ID field for this {listing_type} listing. "
            f"Available fields: {list(listing.keys())}"
        )

    with st.container(border=True):
        col1, col2 = st.columns([5, 1])

        with col1:
            st.subheader(name)

            description = get_listing_description(listing, listing_type)
            if description:
                st.write(description)

            details = get_listing_details(listing, listing_type)
            if details:
                st.caption(" · ".join(details))

        with col2:
            heart_key = f"heart_{key_suffix}"
            already_hearted = (listing_type, str(listing_id)) in st.session_state.hearted_ids

            heart_label = "❤️ Hearted" if already_hearted else "🤍 Heart"
            if st.button(heart_label, key=heart_key, disabled=already_hearted):
                if not st.session_state.user_id:
                    st.warning("Enter your name on the landing page first.")
                else:
                    result, error = api_post("/hearts", {
                        "user_id": st.session_state.user_id,
                        "listing_type": listing_type,
                        "listing_id": str(listing_id),
                    })
                    if error:
                        st.error(f"Could not save heart: {error}")
                    else:
                        st.session_state.hearted_ids.add((listing_type, str(listing_id)))
                        st.rerun()

        # --- Review section (expandable so cards stay compact) ---
        with st.expander("Reviews"):
            existing, error = api_get(f"/reviews/{listing_type}/{listing_id}")
            if error:
                st.caption(f"Could not load reviews: {error}")
            elif existing:
                for r in existing:
                    rating = r.get("Rating") or r.get("rating")
                    comment = r.get("Comment") or r.get("comment")
                    sentiment = r.get("Sentiment") or r.get("sentiment")
                    st.write(f"⭐ {rating} — {comment}  _(sentiment: {sentiment})_")
            else:
                st.caption("No reviews yet — be the first!")

            st.markdown("---")
            with st.form(key=f"review_form_{key_suffix}", clear_on_submit=True):
                rating = st.slider("Your rating", 1, 5, 5, key=f"rating_{key_suffix}")
                comment = st.text_area("Your comment", key=f"comment_{key_suffix}")
                submitted = st.form_submit_button("Submit review")

                if submitted:
                    if not st.session_state.user_id:
                        st.warning("Enter your name on the landing page first.")
                    elif not comment.strip():
                        st.warning("Comment cannot be empty.")
                    else:
                        result, error = api_post("/reviews", {
                            "user_id": st.session_state.user_id,
                            "listing_type": listing_type,
                            "listing_id": str(listing_id),
                            "rating": rating,
                            "comment": comment.strip(),
                        })
                        if error:
                            st.error(f"Could not save review: {error}")
                        else:
                            st.success(f"Review saved! Sentiment detected: {result.get('sentiment')}")
                            st.rerun()


# ---------------------------------------------------------------------------
# Page: Landing
# ---------------------------------------------------------------------------

def render_landing_page():
    st.title("UAE Tourist Recommendation System")
    st.write("Plan your perfect trip across Dubai and Abu Dhabi — events, restaurants, and attractions, personalized to you.")

    email_input = st.text_input("Enter your email to get started", value=st.session_state.user_id or "")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 Start Quiz", use_container_width=True, type="primary"):
            if not EMAIL_PATTERN.match(email_input.strip()):
                st.warning("Please enter a valid email address.")
            else:
                st.session_state.user_id = email_input.strip().lower()
                register_user(st.session_state.user_id)
                load_user_hearts(st.session_state.user_id)
                go_to("quiz")
                st.rerun()

    with col2:
        if st.button("🔍 Skip to Browse", use_container_width=True):
            if not EMAIL_PATTERN.match(email_input.strip()):
                st.warning("Please enter a valid email address.")
            else:
                st.session_state.user_id = email_input.strip().lower()
                register_user(st.session_state.user_id)
                load_user_hearts(st.session_state.user_id)
                load_browse_results()
                go_to("results")
                st.rerun()


# ---------------------------------------------------------------------------
# Page: Quiz
# ---------------------------------------------------------------------------

def render_quiz_page():
    st.title("Tell us about your trip")

    with st.form("trip_quiz"):
        col1, col2 = st.columns(2)
        with col1:
            trip_start_date = st.date_input("Trip start date", value=date.today())
        with col2:
            trip_end_date = st.date_input("Trip end date", value=date.today())

        city = st.multiselect("Which cities?", ["Dubai", "Abu Dhabi"], default=["Dubai"])

        col3, col4 = st.columns(2)
        with col3:
            daily_food_budget = st.select_slider(
                "Daily food budget", options=["Free", "Low", "Medium", "High"], value="Medium"
            )
        with col4:
            daily_attraction_budget = st.select_slider(
                "Daily attraction budget", options=["Free", "Low", "Medium", "High"], value="Medium"
            )

        activity_preferences = st.multiselect(
            "Activity preferences",
            ["Festival", "Cultural", "Sports", "Entertainment", "Exhibition", "Concert"],
        )
        activity_other = st.text_input("Other activities (optional)")

        cuisine_preferences = st.multiselect(
            "Cuisine preferences",
            ["Emirati", "International", "Italian", "Indian", "Japanese", "Middle Eastern"],
        )
        cuisine_other = st.text_input("Other cuisines (optional)")

        event_preferences = st.multiselect(
            "Event preferences",
            ["Entertainment", "Exhibition", "Concert", "Sports", "Cultural", "Festival"],
        )

        attraction_environment = st.multiselect(
            "Preferred environment", ["Indoor", "Outdoor", "Both"]
        )

        weather_preference = st.multiselect(
            "Weather preference", ["Cold", "Normal", "Warm"]
        )

        traveling_with_kids = st.toggle("Traveling with kids?")

        num_recommendations = st.slider("How many recommendations per category?", 1, 10, 5)

        additional_notes = st.text_area("Anything else we should know? (optional)")

        submitted = st.form_submit_button("Get my recommendations", type="primary")

    if submitted:
        if trip_end_date < trip_start_date:
            st.error("Trip end date cannot be before the start date.")
            return

        payload = {
            "user_id": st.session_state.user_id,
            "trip_start_date": trip_start_date.strftime("%Y-%m-%d"),
            "trip_end_date": trip_end_date.strftime("%Y-%m-%d"),
            "city": city,
            "daily_food_budget": daily_food_budget.lower(),
            "daily_attraction_budget": daily_attraction_budget.lower(),
            "activity_preferences": activity_preferences,
            "activity_other": activity_other,
            "cuisine_preferences": cuisine_preferences,
            "cuisine_other": cuisine_other,
            "event_preferences": event_preferences,
            "attraction_environment": attraction_environment,
            "weather_preference": weather_preference,
            "traveling_with_kids": traveling_with_kids,
            "num_recommendations": num_recommendations,
            "additional_notes": additional_notes,
        }

        with st.spinner("Finding your perfect trip..."):
            api_post("/trip-preferences", payload)  # logs this submission; non-blocking on failure
            load_recommend_results(payload)

        go_to("results")
        st.rerun()

    if st.button("← Back"):
        go_to("landing")
        st.rerun()


# ---------------------------------------------------------------------------
# Data loading for the results page — two possible sources
# ---------------------------------------------------------------------------

def load_recommend_results(payload: dict):
    endpoint_map = {
        "Event": "/recommend-events",
        "Restaurant": "/recommend-restaurants",
        "Attraction": "/recommend-attractions",
    }
    results = {}
    for listing_type, endpoint in endpoint_map.items():
        data, error = api_post(endpoint, payload)
        if error:
            st.error(f"Could not load {listing_type} recommendations: {error}")
            results[listing_type] = []
        else:
            results[listing_type] = data.get("recommendations", [])
    st.session_state.recommendations = results


def load_browse_results():
    results = {}
    for listing_type in LISTING_TYPES:
        data, error = api_get(f"/browse/{listing_type.lower()}")
        if error:
            st.error(f"Could not load {listing_type} listings: {error}")
            results[listing_type] = []
        else:
            results[listing_type] = data.get("recommendations", [])
    st.session_state.recommendations = results


# ---------------------------------------------------------------------------
# Page: Results / Browse (same page, different data source)
# ---------------------------------------------------------------------------

def render_results_page():
    st.title(f"Welcome, {st.session_state.user_id} 👋")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 Take the quiz"):
            go_to("quiz")
            st.rerun()
    with col2:
        if st.button("🔄 Refresh browse (unfiltered)"):
            load_browse_results()
            st.rerun()

    st.divider()

    tab_events, tab_restaurants, tab_attractions = st.tabs(["🎉 Events", "🍽️ Restaurants", "🏛️ Attractions"])

    with tab_events:
        listings = st.session_state.recommendations.get("Event", [])
        if not listings:
            st.info("No events to show yet.")
        for i, listing in enumerate(listings):
            render_listing_card(listing, "Event", index=i)

    with tab_restaurants:
        listings = st.session_state.recommendations.get("Restaurant", [])
        if not listings:
            st.info("No restaurants to show yet.")
        for i, listing in enumerate(listings):
            render_listing_card(listing, "Restaurant", index=i)

    with tab_attractions:
        listings = st.session_state.recommendations.get("Attraction", [])
        if not listings:
            st.info("No attractions to show yet.")
        for i, listing in enumerate(listings):
            render_listing_card(listing, "Attraction", index=i)


# ---------------------------------------------------------------------------
# Sidebar — persistent nav, visible once the user has entered their email
# ---------------------------------------------------------------------------

def render_sidebar():
    if not st.session_state.user_id:
        return

    with st.sidebar:
        st.write(f"👤 {st.session_state.user_id}")
        st.divider()

        if st.button("🏠 Home / Browse", use_container_width=True):
            load_browse_results()
            go_to("results")
            st.rerun()

        if st.button("📝 Quiz", use_container_width=True):
            go_to("quiz")
            st.rerun()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

render_sidebar()

if st.session_state.page == "landing":
    render_landing_page()
elif st.session_state.page == "quiz":
    render_quiz_page()
elif st.session_state.page == "results":
    render_results_page()