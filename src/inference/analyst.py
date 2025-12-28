import pandas as pd
import torch
import os
import numpy as np
from src.inference.predictor import RiskEngine
from dotenv import load_dotenv

# Load .env file to get keys
load_dotenv()

# 🔑 SETUP: Get Groq API Key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class IntelligenceAnalyst:
    def __init__(self):
        self.engine = RiskEngine()
        self.raw_data_path = "data/processed/embeddings/synthetic_history.parquet"
        
        # Load raw text data for retrieval
        if os.path.exists(self.raw_data_path):
            self.df_raw = pd.read_parquet(self.raw_data_path)
            self.df_raw['Day'] = self.df_raw['Day'].astype(int)
        else:
            print("❌ Raw history not found. Cannot retrieve headlines.")
            self.df_raw = None

    def get_critical_headlines(self, hex_id, critical_day_offset):
        """Retrieves news from the specific trigger day."""
        if self.df_raw is None: return [], 0
        
        hex_data = self.df_raw[self.df_raw['h3_hex'] == hex_id].sort_values('Day')
        unique_days = hex_data['Day'].unique()
        
        if len(unique_days) < 7: return ["Insufficient history."], 0
            
        target_date_idx = -1 - critical_day_offset
        if abs(target_date_idx) > len(unique_days): return ["Date out of range."], 0
            
        target_date = unique_days[target_date_idx]
        daily_news = hex_data[hex_data['Day'] == target_date]
        
        # Get top 5 unique headlines
        headlines = daily_news['text_content'].unique()[:5].tolist()
        return headlines, target_date

    def generate_briefing(self, hex_id):
        print(f"🕵️‍♀️ Analyst processing Hex: {hex_id}...")
        
        # 1. Get Prediction & Attention
        df_train = pd.read_parquet("data/processed/training_sets/train_multimodal.parquet")
        df_loc = df_train[df_train['h3_hex'] == hex_id].sort_values('Day').tail(7)
        
        if len(df_loc) < 7:
            print("❌ Insufficient data.")
            return

        txt, eco = self.engine.preprocess_window(df_loc)
        
        self.engine.model.eval()
        with torch.no_grad():
            pred, attn = self.engine.model(txt, eco, return_attention=True)
        
        risk_score = pred.item()
        
        # 2. Find Trigger Day
        weights = attn[0, -1, :].cpu().numpy()
        max_idx = np.argmax(weights)
        days_ago = 6 - max_idx 
        
        print(f"⚡ Risk: {risk_score:.2%} | Critical Trigger: {days_ago} days ago")
        
        # 3. Retrieve Headlines
        headlines, date = self.get_critical_headlines(hex_id, days_ago)
        
        # 4. Generate Prompt
        context = f"""
        **SITUATION REPORT**
        Location ID: {hex_id}
        Date of Interest: {date} (T-{days_ago})
        Threat Level: {'CRITICAL' if risk_score > 0.8 else 'ELEVATED'} ({risk_score:.1%})
        
        **DETECTED SIGNALS (Raw Intelligence):**
        {chr(10).join(['- ' + h for h in headlines])}
        """
        
        if GROQ_API_KEY:
            self._call_groq(context)
        else:
            print("⚠️ No GROQ_API_KEY found. Showing raw context only.")
            self._simulate_llm(context)

    def _call_groq(self, context):
        """Call Groq API (Llama 3)"""
        from groq import Groq
        
        client = Groq(api_key=GROQ_API_KEY)
        
        print("\n📝 Querying Groq (Llama-3) for synthesis...\n")
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile", # Or 'mixtral-8x7b-32768'
                messages=[
                    {"role": "system", "content": "You are a senior intelligence analyst. Write a concise, urgent warning briefing based on the provided raw signals. Explain WHY the risk is high. Use professional, military-style language."},
                    {"role": "user", "content": context}
                ],
                temperature=0.5,
                max_tokens=1024
            )
            print("--- 📢 INTELLIGENCE BRIEFING (BY GROQ) 📢 ---")
            print(response.choices[0].message.content)
            print("-------------------------------------------")
        except Exception as e:
            print(f"❌ Groq API Error: {e}")

    def _simulate_llm(self, context):
        """Fallback simulation"""
        print("\n--- 📢 INTELLIGENCE BRIEFING (SIMULATION) 📢 ---")
        print("Subject: IMMEDIATE ALERT - CIVIL UNREST DETECTED")
        print("Assessment: Groq API key missing. Simulated briefing based on headlines.")
        print(context)
        print("---------------------------------------------")

if __name__ == "__main__":
    analyst = IntelligenceAnalyst()
    
    # Use your high-risk hex from earlier
    df = pd.read_parquet("data/processed/training_sets/train_multimodal.parquet")
    # Dynamically find a risky hex so it always works
    risky_locations = df[df['target_label'] == 1]['h3_hex'].unique()
    target_hex = risky_locations[0] if len(risky_locations) > 0 else "8506134bfffffff"
    
    analyst.generate_briefing(target_hex)