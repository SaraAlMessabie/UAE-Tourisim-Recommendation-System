import streamlit as st
import requests

# Point this at your deployed Render backend URL
API_BASE_URL = "https://uae-tourisim-recommendation-system.onrender.com"

st.set_page_config(page_title="UAE Tourist Recommendations", layout="wide")
st.title("UAE Tourist Recommendation System")

st.header("Get Event Recommendations")

with st.form("event_form"):
    user_id = st.text_input("User ID", value="test_user_001")
    trip_start_date = st.date_input("Trip Start Date")
    trip_end_date = st.date_input("Trip End Date")
    city = st.multiselect("City", ["Dubai", "Abu Dhabi"], default=["Dubai"])
    activity_preferences = st.multiselect(
        "Activity Preferences", ["Festival", "Cultural", "Sports", "Entertainment"]
    )
    num_recommendations = st.slider("Number of Recommendations", 1, 10, 5)
    submitted = st.form_submit_button("Get Recommendations")

if submitted:
    payload = {
        "user_id": user_id,
        "trip_start_date": trip_start_date.strftime("%Y-%m-%d"),
        "trip_end_date": trip_end_date.strftime("%Y-%m-%d"),
        "city": city,
        "daily_food_budget": "medium",
        "daily_attraction_budget": "medium",
        "activity_preferences": activity_preferences,
        "activity_other": "",
        "cuisine_preferences": [],
        "cuisine_other": "",
        "event_preferences": [],
        "attraction_environment": [],
        "weather_preference": [],
        "crowdedness_preference": "moderate",
        "traveling_with_kids": False,
        "num_recommendations": num_recommendations,
        "additional_notes": "",
    }

    try:
        response = requests.post(f"{API_BASE_URL}/recommend-events", json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        st.success(f"Found {data.get('num_results', len(data.get('recommendations', [])))} recommendations")
        for rec in data.get("recommendations", []):
            with st.container(border=True):
                st.subheader(rec.get("Name", "Unknown Event"))
                st.write(rec.get("Description", ""))
                st.caption(f"{rec.get('Location', '')} · {rec.get('Start_Date', '')} to {rec.get('End_Date', '')}")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch recommendations: {e}")