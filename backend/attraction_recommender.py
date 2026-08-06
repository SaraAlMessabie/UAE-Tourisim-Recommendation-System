import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def add_new_candidates(candidates_df,new_candidates,fallback_stage,required_count):
    needed = required_count - len(candidates_df)

    new_candidates = new_candidates.copy()
    new_candidates["fallback_stage"] = fallback_stage

    new_candidates = new_candidates[
        ~new_candidates["attraction_id"].isin(
            candidates_df["attraction_id"]
        )
    ]

    new_candidates = new_candidates.sort_values(
        by="similarity_score",
        ascending=False
    )

    candidates_df = pd.concat([
        candidates_df,
        new_candidates.head(needed)
    ])

    return candidates_df


def attraction_reason_with_sentiment(row, user_profile):

    reasons = []

    if row["fallback_stage"] == "strict_match":
        reasons.append(
            "matches your city, environment and budget preferences"
        )

    elif row["fallback_stage"] == "relaxed_environment":
        reasons.append(
            "matches your city and budget preferences"
        )
        reasons.append(
            "environment preference was relaxed"
        )

    elif row["fallback_stage"] == "relaxed_budget_environment":
        reasons.append(
            "matches your city preference"
        )
        reasons.append(
            "budget and environment preferences were relaxed"
        )

    elif row["fallback_stage"] == "expanded_city":
        reasons.append(
            "the search was expanded to other cities"
        )
        reasons.append(
            "city, budget and environment preferences were relaxed"
        )

    if user_profile["with_kids"].lower() == "yes":
        if row["kids_friendly"].lower() == "yes":
            reasons.append(
                "is suitable for kids"
            )

    if row["rating"] >= 0.90:
        reasons.append(
            "has excellent ratings"
        )
    else:
        reasons.append(
            "has good ratings"
        )

    if row["sentiment_score"] > 0.5:
        reasons.append(
            "has mostly positive reviews"
        )

    elif row["sentiment_score"] > 0:
        reasons.append(
            "has positive reviews"
        )

    elif row["sentiment_score"] == 0:
        reasons.append(
            "has neutral reviews"
        )

    return "; ".join(reasons) + "."


def recommend_attractions_with_sentiment(user_profile, attractions, vectorizer, attractions_matrix):

    required_count = user_profile["top_n"]

    user_query = " ".join([
        " ".join(user_profile["city"]),
        " ".join(user_profile["attraction_preference"])
    ])

    # Vectorize
    user_matrix = vectorizer.transform([user_query])

    # Calculate similarity
    similarity_scores = cosine_similarity(
        user_matrix,
        attractions_matrix
    ).flatten()

    attractions_copy = attractions.copy()
    attractions_copy["similarity_score"] = similarity_scores

    cities = [
        city.lower()
        for city in user_profile["city"]
    ]

    environments = [
        environment.lower()
        for environment in user_profile["environment"]
    ]

    budget = user_profile["budget"].lower()

    if budget == "low":
        budget_levels = ["free", "low"]

    elif budget == "medium":
        budget_levels = ["free", "low", "medium"]

    elif budget == "high":
        budget_levels = ["free", "low", "medium", "high"]

    else:
        budget_levels = ["free", "low", "medium", "high"]

    # Strict match
    candidates_df = attractions_copy[
        attractions_copy["city"].str.lower().isin(cities) &
        attractions_copy["environment"].str.lower().isin(environments) &
        attractions_copy["price_level"].str.lower().isin(budget_levels)
    ].copy()

    if user_profile["with_kids"].lower() == "yes":
        candidates_df = candidates_df[
            candidates_df["kids_friendly"].str.lower() == "yes"
        ].copy()

    candidates_df["fallback_stage"] = "strict_match"

    # Relax environment
    if len(candidates_df) < required_count:

        relaxed_environment_df = attractions_copy[
            attractions_copy["city"].str.lower().isin(cities) &
            attractions_copy["price_level"].str.lower().isin(
                budget_levels
            )
        ].copy()

        if user_profile["with_kids"].lower() == "yes":
            relaxed_environment_df = relaxed_environment_df[
                relaxed_environment_df["kids_friendly"].str.lower() == "yes"
            ].copy()

        candidates_df = add_new_candidates(
            candidates_df,
            relaxed_environment_df,
            "relaxed_environment",
            required_count
        )

    # Relax budget and environment
    if len(candidates_df) < required_count:

        relaxed_budget_environment_df = attractions_copy[
            attractions_copy["city"].str.lower().isin(cities)
        ].copy()

        if user_profile["with_kids"].lower() == "yes":
            relaxed_budget_environment_df = (
                relaxed_budget_environment_df[
                    relaxed_budget_environment_df[
                        "kids_friendly"
                    ].str.lower() == "yes"
                ].copy()
            )

        candidates_df = add_new_candidates(
            candidates_df,
            relaxed_budget_environment_df,
            "relaxed_budget_environment",
            required_count
        )

    # Expand city
    if len(candidates_df) < required_count:

        expanded_city_df = attractions_copy.copy()

        if user_profile["with_kids"].lower() == "yes":
            expanded_city_df = expanded_city_df[
                expanded_city_df["kids_friendly"].str.lower() == "yes"
            ].copy()

        candidates_df = add_new_candidates(
            candidates_df,
            expanded_city_df,
            "expanded_city",
            required_count
        )

    # Calculate final ranking score
    candidates_df["final_score"] = (
        0.75 * candidates_df["similarity_score"] +
        0.15 * candidates_df["rating"] +
        0.10 * candidates_df["normalized_sentiment"]
    )

    final_recommendations = candidates_df.sort_values(
        by="final_score",
        ascending=False
    ).head(required_count).copy()

    final_recommendations["rank"] = range(
        1,
        len(final_recommendations) + 1
    )

    final_recommendations["recommendation_reason"] = (
        final_recommendations.apply(
            lambda row: attraction_reason_with_sentiment(
                row,
                user_profile
            ),
            axis=1
        )
    )

    return final_recommendations[ 
        [
            "attraction_id", 
            "name",
            "description",
            "location",
            "price_level",
            "city",
            "categories",
            "environment",
            "kids_friendly",
            "similarity_score",
            "rating",
            "normalized_sentiment",
            "final_score",
            "fallback_stage",
            "recommendation_reason",
            "rank"
        ]
    ]
