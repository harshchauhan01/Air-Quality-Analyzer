import pandas as pd
from datetime import datetime
import os

FILE_NAME = "weather_data_v2.csv"  # single file to store all data

def fetch_and_store():
    url = "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69?api-key=579b464db66ec23bdd000001d2c2499fa4944ae76ca8d20687c6d3e7&format=csv&limit=4000"
    
    # Fetch data
    df = pd.read_csv(url)
    df["fetched_at"] = datetime.now()
    
    # Determine if header should be written
    write_header = not os.path.exists(FILE_NAME)
    
    # Append data to the single CSV file
    df.to_csv(FILE_NAME, mode="a", header=write_header, index=False)
    
    print("Saved to:", FILE_NAME, "at:", datetime.now())

if __name__ == "__main__":
    fetch_and_store()
