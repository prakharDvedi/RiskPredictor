import torch
import pandas as pd
import numpy as np
from src.models.event_transformer import EventPredictor
from src.utils.geo_utils import GeoGrid
import os

class RiskEngine:
    def __init__(self, model_path="outputs/model_v1.pth", load_weights=True):
        self.device = torch.device("cpu")
        # V3 Model Params (Hybrid)
        self.model = EventPredictor(d_model=128, n_layers=2, dropout=0.3).to(self.device)
        self.geo = GeoGrid()
        
        if load_weights and os.path.exists(model_path):
            print(f"✅ Loading V3 Model from {model_path}...")
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
            except Exception as e:
                print(f"⚠️ Load Error: {e}. Starting fresh.")
        else:
            print("🆕 Initializing fresh model.")

    def preprocess_window(self, df_window):
        df_window = df_window.sort_values('Day')
        text_embs = np.stack(df_window['embedding'].values)
        econ_vals = df_window['volatility_7d'].values.reshape(-1, 1)
        
        txt_tensor = torch.FloatTensor(text_embs).unsqueeze(0).to(self.device)
        eco_tensor = torch.FloatTensor(econ_vals).unsqueeze(0).to(self.device)
        return txt_tensor, eco_tensor

    def predict(self, df, hex_id):
        """
        Returns: 
        1. risk_score (0-1)
        2. attention_weights (list of 7 values)
        3. top_contributing_day_index (0-6)
        """
        df_loc = df[df['h3_hex'] == hex_id].sort_values('Day')
        
        if len(df_loc) < 7:
            return 0.0, None, None
            
        recent_window = df_loc.tail(7)
        txt, eco = self.preprocess_window(recent_window)
        
        self.model.eval()
        with torch.no_grad():
            # Request Attention Weights
            risk_score, attn_weights = self.model(txt, eco, return_attention=True)
            
            # Process Attention:
            # attn_weights shape is (Batch, TargetLen, SourceLen) -> (1, 7, 7) for self-attention
            # We want to know which input day influenced the LAST day (Prediction Day) the most.
            # So we look at the last row of the attention matrix.
            final_step_attn = attn_weights[0, -1, :].cpu().numpy()
            
            # Find the day with max impact
            top_day_idx = np.argmax(final_step_attn)
            
        return risk_score.item(), final_step_attn, top_day_idx