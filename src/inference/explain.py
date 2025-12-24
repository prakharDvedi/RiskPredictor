import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.inference.predictor import RiskEngine
from src.utils.geo_utils import GeoGrid

def explain_prediction(target_hex):
    print(f"🕵️‍♀️ Explaining Risk for Hex: {target_hex}")
    
    # 1. Load Engine
    engine = RiskEngine()
    df = pd.read_parquet("data/processed/training_sets/train_multimodal.parquet")
    
    # 2. Get Data for Location
    df_loc = df[df['h3_hex'] == target_hex].sort_values('Day').tail(7)
    
    if len(df_loc) < 7:
        print("❌ Not enough history.")
        return

    # 3. Preprocess
    txt, eco = engine.preprocess_window(df_loc)
    
    # 4. Forward Pass with Attention
    # We manually call the model to get the weights
    engine.model.eval()
    with torch.no_grad():
        prediction, attn_matrix = engine.model(txt, eco, return_attention=True)
    
    risk = prediction.item()
    print(f"⚡ Risk Probability: {risk:.2%}")

    # 5. Process Attention Weights
    # attn_matrix shape: (Batch, Target_Len, Source_Len) -> (1, 7, 7)
    # We care about the LAST row: "How much did Day 7 attend to Days 1-7?"
    # We average across heads if multi-head, but here it comes out combined or per-head depending on torch version.
    # Usually it returns (Batch, Target, Source) averaged across heads.
    
    weights = attn_matrix[0, -1, :].cpu().numpy() # Get last row (7 values)
    
    # Normalize for plotting
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-6)

    # 6. Visualize
    days = [f"T-{6-i}" for i in range(7)] # T-6, T-5 ... T-0
    
    plt.figure(figsize=(10, 5))
    
    # Plot Volatility (Context)
    volatility = df_loc['volatility_7d'].values
    # Normalize vol for comparison
    vol_norm = (volatility - volatility.min()) / (volatility.max() - volatility.min() + 1e-6)
    
    plt.plot(days, vol_norm, label="Market Volatility (Normalized)", color='gray', linestyle='--')
    
    # Plot Attention ( The "Why")
    plt.bar(days, weights, color='red', alpha=0.6, label="Model Attention (Importance)")
    
    plt.title(f"Why did the AI predict {risk:.1%} risk?\nSignal Contribution Analysis")
    plt.xlabel("Timeline (Days before prediction)")
    plt.ylabel("Normalized Magnitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = "outputs/explanation_plot.png"
    plt.savefig(output_file)
    print(f"✅ Explanation saved to {output_file}")
    print("   (Higher Red Bar = This day contained the trigger event)")

if __name__ == "__main__":
    # Pick a risky location to explain
    df = pd.read_parquet("data/processed/training_sets/train_multimodal.parquet")
    # Find a hex with high volatility to make it interesting
    risky_hex ="8506134bfffffff"
    
    explain_prediction(risky_hex)