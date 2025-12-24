import folium
import pandas as pd
from src.inference.predictor import RiskEngine
from src.utils.geo_utils import GeoGrid

def generate_map():
    print("🌍 Generating Global Risk Map...")
    
    # Init Engine & Data
    engine = RiskEngine()
    geo = GeoGrid()
    df = pd.read_parquet("data/processed/training_sets/train_multimodal.parquet")
    
    # Get list of unique locations
    unique_hexes = df['h3_hex'].unique()
    
    # Create Map centered on World
    m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB dark_matter")
    
    alerts = 0
    
    print(f"Scanning {len(unique_hexes)} locations...")
    for hex_id in unique_hexes[:50]: # Limit to 50 for speed demo
        # Predict
        risk = engine.predict(df, hex_id)
        
        # Only plot if risk is notable (> 40%)
        if risk > 0.4:
            lat, lon = geo.hex_to_lat_lon(hex_id)
            
            # Color Code
            color = "orange" if risk < 0.7 else "red"
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                color=color,
                fill=True,
                fill_color=color,
                popup=f"Risk: {risk:.1%}<br>Hex: {hex_id}"
            ).add_to(m)
            alerts += 1
            
    output_file = "outputs/risk_map.html"
    m.save(output_file)
    print(f"✅ Dashboard generated: {output_file} ({alerts} active alerts)")

if __name__ == "__main__":
    generate_map()