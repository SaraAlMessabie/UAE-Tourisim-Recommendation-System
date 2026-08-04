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


if __name__ == "__main__":
    from datetime import datetime
    import uuid

    print(f"Auth method: {'env var (GOOGLE_CREDENTIALS_B64)' if os.environ.get('GOOGLE_CREDENTIALS_B64') else 'local credentials.json'}")
    print(f"Connected to spreadsheet: {spreadsheet.title}")

    tab_names = [ws.title for ws in spreadsheet.worksheets()]
    print(f"Tabs found: {tab_names}")

    TEST_USER_ID = "smoke_test_user"
    TEST_LISTING_ID = "smoke_test_listing_001"

    # -----------------------------------------------------------------
    # 1. Read + write test on "hearts"
    # Columns: Heart_ID | User_ID | Listing_Type | Listing_ID | Submitted_Date
    # -----------------------------------------------------------------
    if "hearts" in tab_names:
        before_df = get_sheet_as_df("hearts")
        print(f"\n'hearts' shape before write: {before_df.shape}")
        print(f"'hearts' columns: {before_df.columns.tolist()}")

        hearts_ws = get_sheet("hearts")
        heart_id = f"H-{uuid.uuid4().hex[:8]}"
        hearts_ws.append_row([
            heart_id, TEST_USER_ID, "Restaurant", TEST_LISTING_ID, str(datetime.now())
        ])

        after_df = get_sheet_as_df("hearts")
        print(f"'hearts' shape after write: {after_df.shape}")
        print("Last row written:")
        print(after_df.tail(1))
    else:
        print("\n'hearts' tab not found — check the tab name.")

    # -----------------------------------------------------------------
    # 2. Read + write test on "reviews"
    # Columns: Review_ID | User_ID | Listing_Type | Listing_ID | Rating | Comment | Sentiment | Submitted_Date
    # -----------------------------------------------------------------
    if "reviews" in tab_names:
        before_df = get_sheet_as_df("reviews")
        print(f"\n'reviews' shape before write: {before_df.shape}")
        print(f"'reviews' columns: {before_df.columns.tolist()}")

        reviews_ws = get_sheet("reviews")
        review_id = f"R-{uuid.uuid4().hex[:8]}"
        reviews_ws.append_row([
            review_id, TEST_USER_ID, "Restaurant", TEST_LISTING_ID,
            5, "This is a smoke test review — safe to delete.",
            "positive", str(datetime.now())
        ])

        after_df = get_sheet_as_df("reviews")
        print(f"'reviews' shape after write: {after_df.shape}")
        print("Last row written:")
        print(after_df.tail(1))
    else:
        print("\n'reviews' tab not found — check the tab name.")

    print(f"\nDone. Search both tabs for '{TEST_USER_ID}' to find and delete the test rows when you're finished.")