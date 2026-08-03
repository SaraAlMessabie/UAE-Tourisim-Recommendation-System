import gspread
import pandas as pd

# Connects using the service account credentials — never commit credentials.json to GitHub
gc = gspread.service_account(filename="credentials.json")
spreadsheet = gc.open("TouristAppData")

def get_sheet(tab_name):
    """Returns a live connection to one specific sheet tab (e.g. 'Hearts', 'Reviews')."""
    return spreadsheet.worksheet(tab_name)

def get_sheet_as_df(tab_name):
    """Pulls all rows from a sheet tab and returns them as a pandas DataFrame."""
    sheet = get_sheet(tab_name)
    records = sheet.get_all_records()
    return pd.DataFrame(records)