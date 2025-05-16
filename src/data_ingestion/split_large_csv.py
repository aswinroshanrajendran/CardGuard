import pandas as pd
import os

# Define paths
input_file = r"C:\Users\aswin\Downloads\archive\split_files\part_1.csv" # Adjust name if different
output_folder = 'data/raw'
chunk_size = 2000

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Load dataset
df = pd.read_csv(input_file)

# Split and save chunks
for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i + chunk_size]
    chunk.to_csv(f'{output_folder}/chunk_{i//chunk_size + 1}.csv', index=False)

print(f"✅ Successfully split {len(df)} rows into {len(range(0, len(df), chunk_size))} files in '{output_folder}'")
