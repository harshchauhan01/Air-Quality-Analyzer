import pandas as pd
from datetime import datetime
import os
import requests
from io import StringIO

FILE_NAME = "weather_data_v3.csv"

def fetch_and_store():
    url = (
        "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
        "?api-key=579b464db66ec23bdd000001d2c2499fa4944ae76ca8d20687c6d3e7"
        "&format=csv&limit=4000"
    )

    headers = {
        "User-Agent": "GitHubActions/WeatherFetcher"
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
    except Exception as e:
        print("Request failed:", e)
        return

    # ❌ API error
    if response.status_code != 200:
        print("API failed with status:", response.status_code)
        return

    # ❌ Empty response
    if not response.text.strip():
        print("Empty response received. Skipping update.")
        return

    try:
        df = pd.read_csv(StringIO(response.text))
    except Exception as e:
        print("Failed to parse CSV:", e)
        return

    # ❌ Empty dataframe
    if df.empty:
        print("No data returned. Skipping write.")
        return

    df["fetched_at"] = datetime.now()

    write_header = not os.path.exists(FILE_NAME)

    df.to_csv(FILE_NAME, mode="a", header=write_header, index=False)

    print("Saved to:", FILE_NAME, "at:", datetime.now())

if __name__ == "__main__":
    fetch_and_store()
