import os
import glob
import pandas as pd

# Root dataset path (where Layout1, Layout2, etc. are stored)
DATASET_PATH = r"C:/Users/Smart/Desktop/AI Agent/IOB"

# Get all CSVs inside subfolders
all_files = glob.glob(os.path.join(DATASET_PATH, "*", "*.csv"))

print(f"Found {len(all_files)} CSV files.")

for file_path in all_files:
    df = pd.read_csv(file_path)

    # Rename first two columns
    cols = list(df.columns)
    if len(cols) < 2:
        print(f"⚠ Skipping {file_path}, not enough columns")
        continue

    df.rename(columns={cols[0]: "Text", cols[1]: "Tag"}, inplace=True)

    # Overwrite same file
    df.to_csv(file_path, index=False, encoding="utf-8")
    print(f"✔ Fixed and saved: {file_path}")

print("✅ All CSVs updated with first col = Text, second col = Tag.")
