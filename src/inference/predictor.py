import torch
import pandas as pd
import numpy as np
from src.models.event_transformer import EventPredictor
from src.utils.geo_utils import GeoGrid

class RiskEngine:
    def __init__(self, model_path="outputs/model_v1.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.geo = GeoGrid(resolution=5)
        
        # 1. Load Architecture
        self.model = EventPredictor().to(self.device)
        
        # 2. Load Weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval() # Set to evaluation mode (turns off Dropout)
        print("✅ Prediction Engine Loaded.")

    def preprocess_window(self, df_window):
        """Converts a raw DataFrame of last 7 days into a Tensor."""
        # Ensure we have 7 days
        if len(df_window) < 7:
            print("⚠️ Not enough history for prediction (Need 7 days).")
            return None, None

        # Sort by date
        df_window = df_window.sort_values('Day')
        
        # Extract features
        embeddings = np.stack(df_window['embedding'].values) # (7, 384)
        
        # Normalize volatility (using same stats as training - simplified here)
        econ_raw = df_window['volatility_7d'].values.reshape(-1, 1)
        econ = (econ_raw - econ_raw.mean()) / (econ_raw.std() + 1e-6)

        # To Tensor & Add Batch Dimension (1, 7, Dim)
        txt_tensor = torch.FloatTensor(embeddings).unsqueeze(0).to(self.device)
        eco_tensor = torch.FloatTensor(econ).unsqueeze(0).to(self.device)
        
        return txt_tensor, eco_tensor

    def predict(self, df_history, location_hex):
        """
        Input: DataFrame containing history for all locations.
        Output: Probability of unrest tomorrow.
        """
        # Filter for specific location
        df_loc = df_history[df_history['h3_hex'] == location_hex].copy()
        
        # Get last 7 days
        df_loc = df_loc.sort_values('Day').tail(7)
        
        if len(df_loc) < 7:
            return 0.0 # Not enough data
            
        # Prepare inputs
        txt, eco = self.preprocess_window(df_loc)
        
        # Forward Pass
        with torch.no_grad():
            probability = self.model(txt, eco).item()
            
        return probability

if __name__ == "__main__":
    # Test Run
    engine = RiskEngine()
    
    # Load data (Simulating "Live Database")
    df = pd.read_parquet("data/processed/training_sets/train_multimodal.parquet")
    
    # Pick a random location from the dataset to test
    sample_hex = df['h3_hex'].iloc[0]
    
    print(f"\n🔮 Analyzing Risk for Location: {sample_hex} ...")
    risk = engine.predict(df, sample_hex)
    
    print(f"⚡ CIVIL UNREST PROBABILITY (24H): {risk:.2%}")
    
    if risk > 0.5:
        print("🚨 ALERT: High risk of disruption detected!")
    else:
        print("✅ Status: Stable.")