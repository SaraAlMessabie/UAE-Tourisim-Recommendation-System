import pandas as pd
import ast

# Raw GitHub URLs for each static catalog
EVENTS_URL = "https://github.com/SaraAlMessabie/UAE-Tourisim-Recommendation-System/blob/main/data/events.csv"
RESTAURANTS_URL = "https://github.com/SaraAlMessabie/UAE-Tourisim-Recommendation-System/blob/main/data/restaurants.csv"
ATTRACTIONS_URL = "https://github.com/SaraAlMessabie/UAE-Tourisim-Recommendation-System/commit/549440a96490103061b0e51ef8a38c4e91525e43"


def _parse_list_column(value):
    """Converts a stringified list like "['Festival', 'Cultural']" back into a real Python list.
    Falls back to an empty list if the value is missing or malformed."""
    if isinstance(value, list):
        return value
    if pd.isna(value) or value == '':
        return []
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


def get_catalog_from_github(csv_url, list_columns=None):
    """Reads a static catalog CSV directly from a GitHub raw URL into a DataFrame.
    list_columns: names of columns that were saved as stringified lists (e.g. 'Categories')
    and need to be converted back into real Python lists."""
    df = pd.read_csv(csv_url)

    if list_columns:
        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(_parse_list_column)

    return df


def load_events_catalog():
    return get_catalog_from_github(EVENTS_URL, list_columns=['Categories'])


def load_restaurants_catalog():
    return get_catalog_from_github(RESTAURANTS_URL, list_columns=['Cuisine_Types'])


def load_attractions_catalog():
    return get_catalog_from_github(ATTRACTIONS_URL, list_columns=['Categories'])