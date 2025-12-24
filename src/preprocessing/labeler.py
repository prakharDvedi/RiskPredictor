import pandas as pd
import numpy as np
import os
from datetime import timedelta

class EventLabeler:
    def __init__(self, processed_dir="data/processed"):
        self.processed_dir = processed_dir
        self.output_dir = "data/processed/training_sets"
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        """Loads and aligns the different data sources."""
        # 1. Load Embeddings (The Rich Text Signal)
        emb_dir = os.path.join(self.processed_dir, "embeddings")
        files = [os.path.join(emb_dir, f) for f in os.listdir(emb_dir) if f.endswith('.parquet')]
        if not files:
            raise FileNotFoundError("No embedding files found. Run text_embedder.py first.")
        
        # Load most recent embedding file
        latest_file = max(files, key=os.path.getctime)
        print(f"Loading text signals from {latest_file}...")
        df_text = pd.read_parquet(latest_file)
        
        # 2. Load Finance (The Economic Context)
        fin_dir = "data/raw/finance"
        fin_files = [os.path.join(fin_dir, f) for f in os.listdir(fin_dir) if f.endswith('.csv')]
        if not fin_files:
            raise FileNotFoundError("No finance data found. Run finance_loader.py first.")
        
        latest_fin = max(fin_files, key=os.path.getctime)
        print(f"Loading economic context from {latest_fin}...")
        df_fin = pd.read_csv(latest_fin)
        
        return df_text, df_fin

    def create_daily_features(self, df_text):
        """Aggregates 15-min updates into Daily Hexagon Summaries."""
        print("Aggregating text signals by (Hexagon, Day)...")
        
        # Ensure correct types
        df_text['Day'] = df_text['Day'].astype(str)
        
        # 1. Pivot/Group to get daily vectors per location
        # We average the embeddings of all news in that city for that day
        # (In production, use LSTM/GRU, but for v1 Mean Pooling is fine)
        
        # Expand embedding lists into columns for averaging is expensive,
        # so we do a trick: stack and mean.
        # Note: This is memory intensive. For competition, we sample.
        
        def mean_vector(vecs):
            # vecs is a Series of lists/arrays
            if len(vecs) == 0: return np.zeros(384)
            return np.mean(np.vstack(vecs.values), axis=0)

        # Group by Hex and Day
        # We also count 'Mentions' to see how loud the signal is
        daily_stats = df_text.groupby(['h3_hex', 'Day']).agg({
            'NumMentions': 'sum',
            'AvgTone': 'mean',         
            'embedding': mean_vector 
        }).reset_index()
        
        return daily_stats

    def generate_labels(self, df_daily, horizon_days=1):
        """
        Look ahead X days. If 'Protest' (Code 14) mentions spike, Label=1.
        """
        print(f"Generating training labels (Prediction Horizon: {horizon_days} day)...")
        
        # Sort by location and time
        df_daily = df_daily.sort_values(['h3_hex', 'Day'])
        
        # Define the Event Definition (Weak Supervision)
        # Threshold: > 50 mentions of conflict in a day = Significant Event
        # In a real system, we filter by EventCode 14*, here we use Mention volume as proxy for demo
        df_daily['is_event_today'] = (df_daily['NumMentions'] > 10).astype(int)
        
        # SHIFT to create Target
        # The label for TODAY is "Will an event happen TOMORROW?"
        # So we shift the 'is_event_today' column BACKWARDS by 1 day
        df_daily['target_label'] = df_daily.groupby('h3_hex')['is_event_today'].shift(-horizon_days)
        
        # Drop NaN (the last day has no future)
        df_daily = df_daily.dropna(subset=['target_label'])
        
        return df_daily

    def merge_finance(self, df_daily, df_fin):
        """Attaches global economic state to every local hex."""
        print("Merging economic context...")
        
        # Prepare finance: Pivot so columns are 'Brent_Oil_volatility', 'Wheat_price', etc.
        # The raw finance data is long-format (Asset, Price). We need wide.
        
        # For simplicity, let's just take the first asset (Brent_Oil) volatility as a global stress feature
        # (In v2 we use all of them)
        oil_data = df_fin[df_fin['asset'] == 'Brent_Oil'].copy()
        
        # Convert GDELT 'YYYYMMDD' string to datetime for merging
        df_daily['date_obj'] = pd.to_datetime(df_daily['Day'], format='%Y%m%d')
        oil_data['date_obj'] = pd.to_datetime(oil_data['Date'])
        
        # Merge on date
        merged = pd.merge(df_daily, oil_data[['date_obj', 'volatility_7d', 'price']], 
                          on='date_obj', how='left')
        
        # Forward fill finance data (for weekends)
        merged['volatility_7d'] = merged['volatility_7d'].ffill().fillna(0)
        
        return merged

    def run(self):
        df_text, df_fin = self.load_data()
        
        # 1. Aggregate Text (Spatial)
        df_daily = self.create_daily_features(df_text)
        
        # 2. Generate Targets (Temporal)
        df_labeled = self.generate_labels(df_daily)
        
        # 3. Fuse Finance (Contextual)
        df_final = self.merge_finance(df_labeled, df_fin)
        
        # Save
        save_path = os.path.join(self.output_dir, "train_multimodal.parquet")
        # Convert embedding back to list for parquet compatibility if needed, 
        # or keep as numpy. Parquet handles lists well.
        
        df_final.to_parquet(save_path)
        print(f"✅ Training Set Ready: {len(df_final)} samples saved to {save_path}")
        print("Sample Data:")
        print(df_final[['Day', 'h3_hex', 'target_label', 'volatility_7d']].head())

if __name__ == "__main__":
    Labeler = EventLabeler()
    Labeler.run()