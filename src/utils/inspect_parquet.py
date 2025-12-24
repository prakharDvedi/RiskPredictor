import pandas as pd
import os

def inspect():
    processed_dir = "data/processed/embeddings"
    files = [f for f in os.listdir(processed_dir) if f.endswith('.parquet') and "synthetic" not in f]
    
    if not files:
        print("❌ No embedding files found.")
        return

    latest_file = os.path.join(processed_dir, max(files))
    print(f"🧐 Inspecting: {latest_file}")
    
    df = pd.read_parquet(latest_file)
    
    print("\n--- COLUMNS ---")
    print(df.columns.tolist())
    
    print("\n--- FIRST 5 ROWS (Subset) ---")
    # Print Day and a few others to check validity
    cols_to_show = [c for c in ['Day', 'GlobalEventID', 'text_content'] if c in df.columns]
    print(df[cols_to_show].head())
    
    print("\n--- DAY COLUMN STATS ---")
    if 'Day' in df.columns:
        print(df['Day'].describe())
        print(f"Nulls: {df['Day'].isnull().sum()}")
    else:
        print("❌ 'Day' column is MISSING!")

if __name__ == "__main__":
    inspect()