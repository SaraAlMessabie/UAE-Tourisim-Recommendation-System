import os
import json
import base64
import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def _get_gspread_client():
    creds_b64 = os.environ.get("GOOGLE_CREDENTIALS_B64")
    if creds_b64:
        creds_json = base64.b64decode(creds_b64).decode("utf-8")
        creds_dict = json.loads(creds_json)
        creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        return gspread.authorize(creds)
    else:
        return gspread.service_account(filename="credentials.json")


gc = _get_gspread_client()
spreadsheet = gc.open("TouristAppData")


def get_sheet(tab_name):
    return spreadsheet.worksheet(tab_name)


def get_sheet_as_df(tab_name):
    sheet = get_sheet(tab_name)
    records = sheet.get_all_records()
    return pd.DataFrame(records)