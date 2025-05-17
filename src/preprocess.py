import os
import pandas as pd
from datetime import datetime

PROCESSED_DIR = "data/processed"
FINAL_DIR = "data/final"
FEATURES_TO_KEEP = [
    'amt',
    'age',
    'gender',
    'trans_hour',
    'trans_day_of_week',
    'category_grocery_pos',
    'category_shopping_net',
    'category_misc_net',
    'is_fraud'
]

def calculate_age(dob_str, trans_time_str):
    dob = datetime.strptime(dob_str, "%Y-%m-%d")
    trans_time = datetime.strptime(trans_time_str, "%Y-%m-%d %H:%M:%S")
    age = trans_time.year - dob.year - ((trans_time.month, trans_time.day) < (dob.month, dob.day))
    return age

def preprocess_data():
    if not os.path.exists(FINAL_DIR):
        os.makedirs(FINAL_DIR)

    for filename in os.listdir(PROCESSED_DIR):
        if filename.endswith(".csv"):
            file_path = os.path.join(PROCESSED_DIR, filename)
            print(f"🛠️ Preprocessing: {filename}")
            
            df = pd.read_csv(file_path)

            # Create 'age'
            df['age'] = df.apply(lambda row: calculate_age(row['dob'], row['trans_date_trans_time']), axis=1)

            # Extract transaction hour and day of week
            df['trans_hour'] = pd.to_datetime(df['trans_date_trans_time']).dt.hour
            df['trans_day_of_week'] = pd.to_datetime(df['trans_date_trans_time']).dt.dayofweek

            # One-hot encode 'category' and keep specific ones
            dummies = pd.get_dummies(df['category'], prefix='category')
            for col in ['category_grocery_pos', 'category_shopping_net', 'category_misc_net']:
                if col not in dummies.columns:
                    dummies[col] = 0  # Handle missing categories
            df = pd.concat([df, dummies[['category_grocery_pos', 'category_shopping_net', 'category_misc_net']]], axis=1)

            # Select only relevant columns
            df_final = df[FEATURES_TO_KEEP]

            # Save the cleaned file
            df_final.to_csv(os.path.join(FINAL_DIR, filename), index=False)
            print(f"✅ Preprocessed and saved: {filename}")

if __name__ == "__main__":
    preprocess_data()
