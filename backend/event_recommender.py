import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


REQUIRED_PROFILE_KEYS = ['Num_Recommendations', 'Trip_Start_Date', 'Trip_End_Date', 'User_ID']


def parse_categories(categories_value):
    if isinstance(categories_value, str):
        return [c.strip().lower() for c in categories_value.split(',') if c.strip()]
    if isinstance(categories_value, (list, tuple, set)):
        return [str(c).strip().lower() for c in categories_value]
    return []


def filtering_fallback(catalog_df, n_recommendation, Start_Date, End_Date, environment=None, city=None, budget=None, family_friendly=None, temp_pref=None, event_preference=None):

    base = catalog_df[
        (catalog_df['Start_Date'] <= End_Date) &
        (catalog_df['End_Date'] >= Start_Date)
    ]

    if family_friendly is not None:
        base = base[base['Family_Friendly'] == family_friendly]

    parsed_categories = base['Categories'].apply(parse_categories)

    stage_names = ['strict_match', 'relaxed_weather', 'relaxed_budget', 'expanded_city', 'relaxed_event']

    stage_frames = []
    matched_index = pd.Index([])

    if event_preference is not None:
        event_preference_lower = [e.strip().lower() for e in event_preference]
    else:
        event_preference_lower = None

    for stage_name in stage_names:
        if len(matched_index) >= n_recommendation:
            break

        remaining = base[~base.index.isin(matched_index)]
        mask = pd.Series(True, index=remaining.index)

        if environment is not None:
            mask &= remaining['Environment'].apply(
                lambda env: env == 'Both' or env in environment
            )

        if city is not None and stage_name not in ['expanded_city', 'relaxed_event']:
            mask &= remaining['City_Key'].isin(city)

        if budget is not None and stage_name in ['strict_match', 'relaxed_weather']:
            mask &= remaining['Price_Range'].isin(budget)

        # Weather: one consistent field (temp_pref) checked against both max/min category columns
        if temp_pref is not None and stage_name == 'strict_match':
            mask &= (
                remaining['Temp_max_Category'].isin(temp_pref) |
                remaining['Temp_min_Category'].isin(temp_pref)
            )

        # Exact category matching against the parsed list, not substring matching
        # on the raw comma-separated string.
        if event_preference_lower is not None and stage_name != 'relaxed_event':
            row_categories = parsed_categories.loc[remaining.index]
            mask &= row_categories.apply(
                lambda cats: any(e in cats for e in event_preference_lower)
            )

        matches = remaining[mask].copy()
        if matches.empty:
            continue

        matches['fallback_stage'] = stage_name
        stage_frames.append(matches)
        matched_index = matched_index.union(matches.index)

    if not stage_frames:
        return base.iloc[0:0].assign(fallback_stage=pd.Series(dtype=object))

    candidates_df = pd.concat(stage_frames)
    return candidates_df.head(n_recommendation)


def build_visitor_text(visitor_profile):
    activities = ', '.join(visitor_profile.get('Activity_Preferences', []))
    activity_other = visitor_profile.get('Activity_Other', '') or ''
    cuisines = ', '.join(visitor_profile.get('Cuisine_Preferences', []))
    cuisine_other = visitor_profile.get('Cuisine_Other', '') or ''
    crowd_pref = visitor_profile.get('Crowdedness_Preference', '')
    family = visitor_profile.get('Family_Friendly', False)
    environments = ', '.join(visitor_profile.get('Environment', []))
    special_note = visitor_profile.get('special_note', '') or ''

    visitor_text = (
        f"I am interested in {activities}"
        f"{' and ' + activity_other if activity_other else ''} activities. "
        f"I enjoy {cuisines}"
        f"{' and ' + cuisine_other if cuisine_other else ''} food. "
        f"I prefer {environments} settings with a {crowd_pref.lower() if crowd_pref else ''} crowd level. "
        f"{'I am traveling with kids and prefer family-friendly options.' if family else 'I am not traveling with kids.'} "
        f"{special_note}"
    )
    return visitor_text


def compute_similarity(candidates_df, visitor_profile, tfidf_vectorizer, all_event_vectors):
    visitor_text = build_visitor_text(visitor_profile)
    visitor_vector = tfidf_vectorizer.transform([visitor_text])

    if candidates_df.empty:
        candidates_df = candidates_df.copy()
        candidates_df['similarity_score'] = pd.Series(dtype=float)
        return candidates_df

    original_index = candidates_df.attrs.get('source_index', None)
    if original_index is not None:
        positions = original_index.get_indexer(candidates_df.index)
        if (positions == -1).any():
            raise ValueError(
                "compute_similarity: some candidate rows could not be matched "
                "back to the original catalog index used to build all_event_vectors."
            )
        candidate_vectors = all_event_vectors[positions]
    else:
        candidate_vectors = all_event_vectors[candidates_df.index]

    similarity_scores = cosine_similarity(candidate_vectors, visitor_vector).flatten()

    candidates_df = candidates_df.copy()
    candidates_df['similarity_score'] = similarity_scores
    return candidates_df


def build_recommendation_reason(row, visitor_profile):
    reasons = {
        'strict_match': 'Matches your city, budget, and weather preference',
        'relaxed_weather': 'Matches your city and budget (weather preference relaxed)',
        'relaxed_budget': 'Matches your city (budget relaxed)',
        'expanded_city': 'Recommended from other locations to meet your request',
        'relaxed_event': 'Recommended based on your general preferences',
    }
    reason = reasons.get(row['fallback_stage'], 'Recommended based on your profile')

    extras = []
    if row.get('Family_Friendly') is True:
        extras.append('family-friendly')

    categories = set(parse_categories(row['Categories']))
    prefs = set(p.strip().lower() for p in visitor_profile.get('Activity_Preferences', []))
    matched_activities = prefs & categories
    if matched_activities:
        extras.append(f"matches your interest in {', '.join(matched_activities)}")

    if row.get('date_in_range', True):
        extras.append('falls within your trip dates')

    if extras:
        reason += " (" + ", ".join(extras) + ")"

    return reason


def recommend_events(catalog_df, visitor_profile, tfidf_vectorizer, all_event_vectors):
    missing_keys = [k for k in REQUIRED_PROFILE_KEYS if k not in visitor_profile]
    if missing_keys:
        raise ValueError(f"visitor_profile is missing required keys: {missing_keys}")

    n_recommendation = visitor_profile['Num_Recommendations']

    # 1. Filter candidates using fallback stages
    candidates_df = filtering_fallback(
        catalog_df,
        n_recommendation=n_recommendation,
        Start_Date=visitor_profile['Trip_Start_Date'],
        End_Date=visitor_profile['Trip_End_Date'],
        environment=visitor_profile.get('Environment'),
        city=visitor_profile.get('City'),
        budget=visitor_profile.get('budget'),
        family_friendly=visitor_profile.get('Family_Friendly'),
        temp_pref=visitor_profile.get('Temp_Pref'),
        event_preference=visitor_profile.get('Event_Preferences')
    )

    # Track the original catalog index so compute_similarity can map candidate
    # rows back to positional indices in all_event_vectors safely.
    candidates_df.attrs['source_index'] = catalog_df.index

    if candidates_df.empty:
        return {
            "visitor_id": visitor_profile['User_ID'],
            "num_results": 0,
            "recommendations": [],
        }

    # 2. Compute similarity score
    candidates_df = compute_similarity(candidates_df, visitor_profile, tfidf_vectorizer, all_event_vectors)

    # 3. Apply fallback-stage penalty
    stage_penalty = {
        'strict_match': 1.0,
        'relaxed_weather': 0.80,
        'relaxed_budget': 0.75,
        'expanded_city': 0.6,
        'relaxed_event': 0.5
    }
    candidates_df['final_score'] = candidates_df['similarity_score'] * candidates_df['fallback_stage'].map(stage_penalty)

    # 4. Select Top-N (deterministic tie-breaking by final_score, then Name)
    sort_columns = ['final_score']
    sort_ascending = [False]
    if 'Name' in candidates_df.columns:
        sort_columns.append('Name')
        sort_ascending.append(True)

    final_recommendations = candidates_df.sort_values(
        by=sort_columns, ascending=sort_ascending, kind='mergesort'
    ).head(n_recommendation).copy()

    # 5. Add visitor_id and recommendation reason
    final_recommendations['visitor_id'] = visitor_profile['User_ID']
    final_recommendations['recommendation_reason'] = final_recommendations.apply(
        lambda row: build_recommendation_reason(row, visitor_profile), axis=1
    )

    # 6. Round scoring columns
    final_recommendations[['similarity_score', 'final_score']] = final_recommendations[['similarity_score', 'final_score']].round(3)

    # 7. Select and order required columns
    required_columns = [
        'visitor_id', 'Name', 'Categories', 'Location',
        'Start_Date', 'End_Date', 'Price_Range', 'similarity_score',
        'fallback_stage', 'final_score', 'recommendation_reason', 'Description'
    ]
    final_recommendations = final_recommendations[required_columns].sort_values(
        by=sort_columns, ascending=sort_ascending, kind='mergesort'
    ).reset_index(drop=True)

    # 8. Format for FastAPI: convert dates to strings, return a JSON-safe dict
    final_recommendations['Start_Date'] = final_recommendations['Start_Date'].dt.strftime('%Y-%m-%d')
    final_recommendations['End_Date'] = final_recommendations['End_Date'].dt.strftime('%Y-%m-%d')

    records = final_recommendations.to_dict(orient='records')

    return {
        "visitor_id": visitor_profile['User_ID'],
        "num_results": len(records),
        "recommendations": records,
    }