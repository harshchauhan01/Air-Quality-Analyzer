import pandas as pd
import os
from datetime import datetime
import requests

MAX_SIZE_MB = 90                     # maximum CSV size (change if needed)
MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024


def get_latest_file():
    """Finds the latest versioned CSV file."""
    version = 1
    while True:
        filename = f"weather_data_v{version}.csv"
        if not os.path.exists(filename):
            # previous version is the latest existing file
            return f"weather_data_v{version}.csv"
        version += 1


def rotate_file_if_needed(filepath):
    """Checks file size and returns a new path if it exceeds max size."""
    if not os.path.exists(filepath):
        return filepath  # no rotation needed

    size = os.path.getsize(filepath)
    if size < MAX_SIZE_BYTES:
        return filepath  # still within limit

    # rotate → move to next version
    version = int(filepath.split("_v")[1].split(".")[0])
    new_version = version + 1
    new_filepath = f"weather_data_v{new_version}.csv"
    return new_filepath


def fetch_and_store():
    url = "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69?api-key=579b464db66ec23bdd000001d2c2499fa4944ae76ca8d20687c6d3e7&format=csv&limit=4000"
    
    df = pd.read_csv(url)
    df["fetched_at"] = datetime.now()

    # Determine file to write
    latest_file = get_latest_file()
    final_file = rotate_file_if_needed(latest_file)

    # If new version file is created, write header; else append
    write_header = not os.path.exists(final_file)

    df.to_csv(final_file, mode="a", header=write_header, index=False)

    print("Saved to:", final_file, "at:", datetime.now())


if __name__ == "__main__":
    fetch_and_store()
