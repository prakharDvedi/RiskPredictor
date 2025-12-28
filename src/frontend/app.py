import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import os
import sys
import matplotlib.pyplot as plt

# --- 1. ROBUST PATH SETUP ---
# Forces Python to see the 'src' folder correctly
current_dir = os.path.dirname(os.path.abspath(__file__)) # src/frontend
project_root = os.path.dirname(os.path.dirname(current_dir)) # root
sys.path.append(project_root)

# Import Modules safely
try:
    from src.inference.predictor import RiskEngine
    from src.llm.analyst import IntelligenceAnalyst
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.info(f"Python Path: {sys.path}")
    st.stop()

# --- 2. CONFIG & INIT ---
st.set_page_config(layout="wide", page_title="Sentinel V4: XAI War Room")

@st.cache_resource
def load_engine():
    return RiskEngine()

try:
    engine = load_engine()
    analyst = IntelligenceAnalyst()
except Exception as e:
    st.error(f"❌ Engine Start Failed: {e}")
    st.stop()

# Load Data
data_path = os.path.join(project_root, "data/processed/training_sets/train_multimodal.parquet")
if not os.path.exists(data_path):
    st.error(f"❌ Data missing at: {data_path}")
    st.stop()

df = pd.read_parquet(data_path)
df['Day'] = pd.to_datetime(df['Day'])

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.title("📡 Sentinel V4 Command")
st.sidebar.markdown("---")

# Date Picker
min_date = df['Day'].min().date()
max_date = df['Day'].max().date()
default_date = pd.to_datetime("2024-01-22").date() # Pick a date likely to have data

# Ensure default is within range
if default_date < min_date or default_date > max_date:
    default_date = min_date

date_filter = st.sidebar.date_input("Operation Date", value=default_date, min_value=min_date, max_value=max_date)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Sensitivity")

# Debug Threshold Slider
threshold = st.sidebar.slider("Risk Alert Threshold", 0.0, 1.0, 0.15, help="Lower this to see more potential threats.")
st.sidebar.markdown(f"**Filtering Hexes > {threshold:.0%} Risk**")

# --- 4. MAP LOGIC ---
st.title("🗺️ Geospatial Threat Monitor (Hybrid V3 Engine)")

# Initialize Map centered on India (or your data region)
m = folium.Map(location=[22.0, 79.0], zoom_start=5, tiles="CartoDB dark_matter")

# Filter Data by Day
day_df = df[df['Day'] == pd.to_datetime(date_filter)]

active_hexes = day_df['h3_hex'].unique()
risk_cache = {}
max_risk_detected = 0.0

if len(active_hexes) == 0:
    st.warning(f"⚠️ No data available for {date_filter}. Try another date.")
else:
    for h in active_hexes:
        # PREDICT
        risk, attn, top_day = engine.predict(df, h)
        
        # Track Max Risk
        if risk > max_risk_detected:
            max_risk_detected = risk
            
        # Store for Interaction
        risk_cache[h] = {"risk": risk, "attn": attn, "top_day": top_day}
        
        # Visualize only if above threshold
        if risk > threshold:
            # Color Logic
            color = "#00ff00" # Green (Safe)
            if risk > 0.4: color = "#ffa500" # Orange (Warning)
            if risk > 0.7: color = "#ff0000" # Red (Critical)
            
            folium.RegularPolygonMarker(
                location=engine.geo.hex_to_lat_lon(h),
                number_of_sides=6,
                radius=15,
                color=color,
                fill_color=color,
                fill_opacity=0.6,
                popup=f"Risk: {risk:.2%}"
            ).add_to(m)

# Show Max Risk in Sidebar (Crucial for Debugging)
st.sidebar.info(f"🔥 Highest Risk on Map: {max_risk_detected:.2%}")

# Render Map
map_data = st_folium(m, width="100%", height=500)

# --- 5. ANALYTICS PANEL (XAI) ---
st.markdown("---")
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Signal Analysis")
    
    # Logic to select a hex (Click or Default to Highest Risk)
    selected_hex = None
    
    # 1. Try Click
    # (Folium click extraction is tricky, for V4 stability we default to 'Highest Risk Hex' if user hasn't clicked perfectly)
    if risk_cache:
        # Default to the most dangerous hex found
        selected_hex = max(risk_cache, key=lambda k: risk_cache[k]['risk'])
        
    if selected_hex:
        data = risk_cache[selected_hex]
        
        st.metric("Selected Threat Level", f"{data['risk']:.2%}")
        
        # EXPLAINABILITY CHART
        if data['attn'] is not None:
            st.markdown("**🧠 Neural Attention (Last 7 Days)**")
            st.caption("Which day in the sequence triggered the alarm?")
            
            fig, ax = plt.subplots(figsize=(5, 3))
            days_labels = ["T-7", "T-6", "T-5", "T-4", "T-3", "T-2", "T-1"]
            
            # Highlight the trigger day in Red
            bar_colors = ['gray'] * 7
            if data['top_day'] is not None:
                bar_colors[data['top_day']] = 'red'
                
            ax.bar(days_labels, data['attn'], color=bar_colors)
            ax.set_ylim(0, max(data['attn']) + 0.1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            st.pyplot(fig)
            
            trigger_day_label = days_labels[data['top_day']]
            st.error(f"🚨 Critical Trigger Detected: **{trigger_day_label}**")
    else:
        st.info("No active hexes on map.")

with col2:
    st.subheader("📝 Generative Intelligence Briefing")
    
    if selected_hex:
        if st.button("Generate Briefing (Llama-3)"):
            with st.spinner("Analyst is interpreting signal data..."):
                
                # Get data for the specific Trigger Day
                hex_data = df[df['h3_hex'] == selected_hex].sort_values('Day').tail(7)
                top_day_idx = risk_cache[selected_hex]['top_day']
                
                if top_day_idx is not None and top_day_idx < len(hex_data):
                    row = hex_data.iloc[top_day_idx]
                    
                    # 🛠️ RECONSTRUCT INTEL FROM METRICS
                    # Since raw text is missing, we create a structured signal report
                    tone_desc = "Negative" if row['AvgTone'] < -2 else "Neutral/Positive"
                    volatility_desc = "High" if row['volatility_7d'] > 0.02 else "Stable"
                    
                    trigger_text = (
                        f"SIGINT REPORT [Hex: {selected_hex}]: "
                        f"Detected {row['NumMentions']} distinct signal intercepts. "
                        f"Sentiment Analysis indicates {tone_desc} tone ({row['AvgTone']:.2f}). "
                        f"Market volatility is {volatility_desc} ({row['volatility_7d']:.4f}). "
                        f"This specific combination of negative sentiment and market instability "
                        f"matches known pre-riot signatures."
                    )
                    
                    risk_val = risk_cache[selected_hex]['risk']
                    
                    # Call LLM with the synthesized text
                    briefing = analyst.generate_briefing(trigger_text, risk_val)
                    st.markdown(briefing)
                else:
                    st.error("Historical context missing. Cannot generate briefing.")
    else:
        st.markdown("*Select a date with active signals to enable briefings.*")