from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import torch
from src.inference.predictor import RiskEngine
from src.utils.geo_utils import GeoGrid
import os

# 1. Initialize App & Components
app = FastAPI(title="Event Prediction Engine API", version="1.0")
engine = None
geo = None
latest_data = None

@app.on_event("startup")
def load_artifacts():
    """Loads the model and data once when server starts."""
    global engine, geo, latest_data
    print("🚀 API Starting up...")
    
    # Load Model
    engine = RiskEngine(model_path="outputs/model_v1.pth")
    
    # Load Geo Utils
    geo = GeoGrid(resolution=5)
    
    # Load Latest Data (Simulating a Database connection)
    data_path = "data/processed/training_sets/train_multimodal.parquet"
    if os.path.exists(data_path):
        print(f"📂 Loading data cache from {data_path}...")
        latest_data = pd.read_parquet(data_path)
    else:
        print("⚠️ WARNING: No data found. Predictions will fail.")

@app.get("/")
def health_check():
    return {"status": "online", "model_loaded": engine is not None}

@app.get("/predict/hex/{hex_id}")
def predict_risk_by_hex(hex_id: str):
    """
    Get the Civil Unrest probability for a specific H3 Hexagon.
    """
    if latest_data is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
        
    # Check if hex exists in our data
    if hex_id not in latest_data['h3_hex'].values:
        raise HTTPException(status_code=404, detail="Location not found in monitored regions")
    
    # Predict
    try:
        risk_score = engine.predict(latest_data, hex_id)
        
        # Determine Status
        status = "SAFE"
        if risk_score > 0.4: status = "WARNING"
        if risk_score > 0.75: status = "DANGER"
        
        return {
            "hex_id": hex_id,
            "risk_probability": round(risk_score, 4),
            "status": status,
            "coordinates": geo.hex_to_lat_lon(hex_id)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/geo")
def predict_risk_by_coords(lat: float, lon: float):
    """
    Get risk for a specific Latitude/Longitude.
    Auto-converts to H3 Hexagon.
    """
    # Convert Lat/Lon -> Hex
    hex_id = geo.lat_lon_to_hex(lat, lon)
    
    # Reuse the hex logic
    return predict_risk_by_hex(hex_id)

# Run with: uvicorn src.api.main:app --reload