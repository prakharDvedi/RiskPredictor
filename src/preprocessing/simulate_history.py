import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

class SmartHistorySimulator:
    def __init__(self, days=14):
        self.days = days
        self.processed_dir = "data/processed/embeddings"
        
    def run(self):
        # 1. Load the real file
        files = [os.path.join(self.processed_dir, f) for f in os.listdir(self.processed_dir) if f.endswith('.parquet')]
        latest_file = max(files, key=os.path.getctime)
        print(f"🧬 Loading real data from {latest_file}...")
        
        df_real = pd.read_parquet(latest_file).dropna(subset=['Day'])
        
        # 2. Assign "Roles" to locations (50% Safe, 50% Risky)
        # This gives the model contrasting examples to learn from
        unique_hexes = df_real['h3_hex'].unique()
        risky_hexes = np.random.choice(unique_hexes, size=len(unique_hexes)//2, replace=False)
        
        print(f"   Creating synthetic history: {len(unique_hexes)} locations ({len(risky_hexes)} simulated as RISKY)")
        
        base_date = datetime.now()
        all_data = []
        
        # 3. Generate 14 days of history
        for i in range(self.days):
            df_day = df_real.copy()
            
            # Set Date
            current_date = base_date - timedelta(days=i)
            df_day['Day'] = int(current_date.strftime("%Y%m%d"))
            
            # --- INJECT PATTERNS ---
            
            # Logic: If a location is in 'risky_hexes', we make it look dangerous
            is_risky = df_day['h3_hex'].isin(risky_hexes)
            
            # A. Text Signal (Embeddings)
            # We add MORE noise to risky locations to simulate "chaotic/breaking news"
            # (In a real system, the vector direction would shift, here noise is a proxy for change)
            embeddings = np.vstack(df_day['embedding'].values)
            
            # Safe noise (small)
            noise_safe = np.random.normal(0, 0.01, embeddings.shape)
            # Risky noise (large - simulates turbulent news cycle)
            noise_risky = np.random.normal(0, 0.05, embeddings.shape)
            
            final_embs = embeddings + noise_safe
            final_embs[is_risky] = embeddings[is_risky] + noise_risky[is_risky]
            
            df_day['embedding'] = final_embs.tolist()
            
            # B. Event Volume (Mentions)
            # Risky places get 5x to 10x more mentions (Viral news)
            df_day.loc[is_risky, 'NumMentions'] = df_day.loc[is_risky, 'NumMentions'] * np.random.randint(5, 15, size=sum(is_risky))
            
            # C. Sentiment (AvgTone)
            # Risky places get negative sentiment (lower is worse)
            df_day.loc[is_risky, 'AvgTone'] = df_day.loc[is_risky, 'AvgTone'] - 5.0
            
            all_data.append(df_day)
            
        # 4. Save
        df_history = pd.concat(all_data)
        output_path = os.path.join(self.processed_dir, "synthetic_history.parquet")
        df_history.to_parquet(output_path, index=False)
        print(f"✅ Created SMART synthetic dataset at {output_path}")

if __name__ == "__main__":
    sim = SmartHistorySimulator()
    sim.run()