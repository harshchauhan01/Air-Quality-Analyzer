import pandas as pd
import os
from datetime import datetime
import requests

def fetch_and_store():
    url = "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69?api-key=579b464db66ec23bdd000001d2c2499fa4944ae76ca8d20687c6d3e7&format=csv&limit=4000"
    
    df = pd.read_csv(url)
    df["fetched_at"] = datetime.now()

    # Save inside repo
    file_path = "weather_data.csv"
    if os.path.exists(file_path):
        df.to_csv(file_path, mode="a", header=False, index=False)
    else:
        df.to_csv(file_path, index=False)

    print("Fetched at:", datetime.now())

if __name__ == "__main__":
    fetch_and_store()
