import os
import pandas as pd
import shutil

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

REQUIRED_COLUMNS = [
    'trans_date_trans_time', 'cc_num', 'merchant', 'category', 'amt',
    'first', 'last', 'gender', 'street', 'city', 'state', 'zip',
    'lat', 'long', 'city_pop', 'job', 'dob', 'trans_num',
    'unix_time', 'merch_lat', 'merch_long', 'is_fraud'
]

def validate_data(df: pd.DataFrame) -> bool:
    # Check if all required columns are present
    if not all(col in df.columns for col in REQUIRED_COLUMNS):
        print("❌ Missing required columns")
        return False

    # Check for null values in required fields
    if df[REQUIRED_COLUMNS].isnull().any().any():
        print("❌ Null values found in required fields")
        return False

    return True

def ingest_files():
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    for filename in os.listdir(RAW_DIR):
        if filename.endswith(".csv"):
            file_path = os.path.join(RAW_DIR, filename)
            print(f"🔍 Processing: {filename}")

            try:
                df = pd.read_csv(file_path)
                if validate_data(df):
                    shutil.move(file_path, os.path.join(PROCESSED_DIR, filename))
                    print(f"✅ Valid file moved: {filename}")
                else:
                    print(f"❌ Invalid file: {filename}")
            except Exception as e:
                print(f"❌ Error reading {filename}: {e}")

if __name__ == "__main__":
    ingest_files()
