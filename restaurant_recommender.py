import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def add_new_candidates(current_df, new_df, stage_name, required_count):

    new_df = new_df[
        ~new_df["restaurant_name"].isin(
            current_df["restaurant_name"]
        )
    ].copy()

    needed = required_count - len(current_df)

    if needed > 0:

        new_df = (
            new_df
            .sort_values(
                by="similarity_score",
                ascending=False
            )
            .head(needed)
        )

        new_df["fallback_stage"] = stage_name

        current_df = pd.concat(
            [current_df, new_df],
            ignore_index=True
        )

    return current_df


def recommendation_reason(row, user_profile):

    reasons = []

    if row["fallback_stage"] == "strict_match":
        reasons.append(
            "matches your city, cuisine and budget preferences"
        )

    elif row["fallback_stage"] == "relaxed_budget":
        reasons.append(
            "matches your city and cuisine preferences"
        )
        reasons.append(
            "budget preference was relaxed"
        )

    elif row["fallback_stage"] == "relaxed_cuisine":
        reasons.append(
            "matches your city and budget preferences"
        )
        reasons.append(
            "cuisine preference was relaxed"
        )

    elif row["fallback_stage"] == "relaxed_budget_cuisine":
        reasons.append(
            "matches your city preference"
        )
        reasons.append(
            "budget and cuisine preferences were relaxed"
        )

    elif row["fallback_stage"] == "relaxed_city":
        reasons.append(
            "the search was expanded to other cities"
        )
        reasons.append(
            "city preference was relaxed"
        )

    if row["rating_score"] >= 0.90:
        reasons.append(
            "has excellent ratings"
        )
    else:
        reasons.append(
            "has good ratings"
        )

    if row["popularity_score"] >= 0.70:
        reasons.append(
            "is popular with visitors"
        )
    else:
        reasons.append(
            "is worth trying"
        )

    return "; ".join(reasons) + "."


def recommend_restaurants(user_profile, zomato, vectorizer, item_matrix):
    required_count = user_profile['top_n']

    user_query = " ".join([
        " ".join(user_profile["city"]),
        " ".join(user_profile["cuisines"])
    ])

    # Calculating Similarity

    user_matrix = vectorizer.transform([user_query])

    similarity_scores = cosine_similarity(
        user_matrix,
        item_matrix
    ).flatten()

    zomato["similarity_score"] = similarity_scores

    budget = user_profile["budget"].lower()

    if budget == "low":
        allowed_prices = ["low"]

    elif budget == "medium":
        allowed_prices = ["low", "medium"]

    else:
        allowed_prices = ["low", "medium", "high"]

    # Strict match
    candidates_df = zomato[
        zomato["city"].str.lower().isin(
            [city.lower() for city in user_profile["city"]]
        )
    ].copy()

    selected_cuisines = [
        cuisine.lower()
        for cuisine in user_profile["cuisines"]
    ]

    candidates_df = candidates_df[
        candidates_df["cuisines"].apply(
            lambda x: any(cuisine in x for cuisine in selected_cuisines)
        )
    ]

    candidates_df = candidates_df[
        candidates_df["price_level"].isin(allowed_prices)
    ].copy()

    candidates_df["fallback_stage"] = "strict_match"

    # Relax budget
    if len(candidates_df) < required_count:

        relaxed_budget_df = zomato[
            zomato["city"].str.lower().isin(
                [city.lower() for city in user_profile["city"]]
            )
        ].copy()

        relaxed_budget_df = relaxed_budget_df[
            relaxed_budget_df["cuisines"].apply(
                lambda x: any(cuisine in x for cuisine in selected_cuisines)
            )
        ].copy()

        candidates_df = add_new_candidates(
            candidates_df,
            relaxed_budget_df,
            "relaxed_budget",
            required_count
        )

    # Relax cuisine
    if len(candidates_df) < required_count:

        relaxed_cuisine_df = zomato[
            zomato["city"].str.lower().isin(
                [city.lower() for city in user_profile["city"]]
            )
        ].copy()

        relaxed_cuisine_df = relaxed_cuisine_df[
            relaxed_cuisine_df["price_level"].isin(
                allowed_prices
            )
        ].copy()

        candidates_df = add_new_candidates(
            candidates_df,
            relaxed_cuisine_df,
            "relaxed_cuisine",
            required_count
        )

    # Relax budget and expand cuisines
    if len(candidates_df) < required_count:

        relaxed_budget_cuisine_df = zomato[
            zomato["city"].str.lower().isin(
                [city.lower() for city in user_profile["city"]]
            )
        ].copy()

        candidates_df = add_new_candidates(
            candidates_df,
            relaxed_budget_cuisine_df,
            "relaxed_budget_cuisine",
            required_count
        )

    # Expand city
    if len(candidates_df) < required_count:

        relaxed_city_df = zomato.copy()

        candidates_df = add_new_candidates(
            candidates_df,
            relaxed_city_df,
            "relaxed_city",
            required_count
        )

    candidates_df["final_score"] = (
        0.75 * candidates_df["similarity_score"] +
        0.15 * candidates_df["rating_score"] +
        0.10 * candidates_df["popularity_score"]
    )

    candidates_df = candidates_df.sort_values(
        by="final_score",
        ascending=False
    ).head(required_count)

    candidates_df["rank"] = range(
        1,
        len(candidates_df) + 1
    )

    candidates_df["recommendation_reason"] = (
        candidates_df.apply(
            lambda row: recommendation_reason(
                row,
                user_profile
            ),
            axis=1
        )
    )

    candidates_df = candidates_df[
        [
            "restaurant_name",
            "city",
            "cuisines",
            "price_level",
            "rating_score",
            "popularity_score",
            "final_score",
            "rank",
            "fallback_stage",
            "recommendation_reason"
        ]
    ]

    return candidates_df
