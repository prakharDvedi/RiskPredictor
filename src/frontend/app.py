import streamlit as st
import pandas as pd
import pydeck as pdk
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

# --- ENHANCED CSS STYLING ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0a0e27 100%);
        background-attachment: fixed;
    }
    
    /* Headers */
    h1 {
        font-family: 'Courier New', monospace;
        color: #00ff41 !important;
        text-shadow: 0 0 20px rgba(0,255,65,0.8), 0 0 40px rgba(0,255,65,0.4);
        letter-spacing: 3px;
        font-size: 2.5rem !important;
        text-align: center;
        padding: 20px 0;
    }
    
    h2 {
        font-family: 'Courier New', monospace;
        color: #00d4ff !important;
        text-shadow: 0 0 10px rgba(0,212,255,0.6);
        letter-spacing: 2px;
    }
    
    h3 {
        color: #00ff41 !important;
        font-family: 'Courier New', monospace;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 3px solid #00ff41;
        box-shadow: 5px 0 20px rgba(0,255,65,0.2);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #00ff41 !important;
        text-shadow: 0 0 10px rgba(0,255,65,0.5);
    }
    
    .stMetric {
        background: rgba(0, 255, 65, 0.05);
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #00ff41;
        box-shadow: 0 0 20px rgba(0,255,65,0.3);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #ff3030 0%, #ff6b6b 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        font-weight: bold;
        border-radius: 8px;
        box-shadow: 0 0 20px rgba(255,48,48,0.5);
        transition: all 0.3s ease;
        font-family: 'Courier New', monospace;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        box-shadow: 0 0 30px rgba(255,48,48,0.8);
        transform: translateY(-2px);
    }
    
    /* Map Container */
    .stDeckGlJsonChart {
        border: 3px solid #00ff41;
        border-radius: 10px;
        box-shadow: 0 0 30px rgba(0,255,65,0.3);
        background: #000;
    }
    
    /* Info boxes */
    .stAlert {
        background: rgba(0, 212, 255, 0.1);
        border-left: 5px solid #00d4ff;
        color: white;
    }
    
    /* Selectbox */
    .stSelectbox {
        background: rgba(0, 255, 65, 0.05);
    }
    
    /* Text color */
    p, label, .stMarkdown {
        color: #e0e0e0 !important;
    }
    
    /* Divider */
    hr {
        border-color: #00ff41;
        opacity: 0.3;
    }
</style>
""", unsafe_allow_html=True)

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

try:
    from src.inference.predictor import RiskEngine
    from src.llm.analyst import IntelligenceAnalyst
except ImportError as e:
    st.error(f"❌ System Path Error: {e}")
    st.stop()

# --- INIT ---
st.set_page_config(
    layout="wide", 
    page_title="Sentinel V7: 3D Command", 
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# --- HEADER ---
col_h1, col_h2, col_h3 = st.columns([1, 2, 1])
with col_h2:
    st.markdown("""
    # 🛡️ SENTINEL V7: 3D COMMAND
    """)

current_time = datetime.now().strftime("%H:%M:%S UTC")
st.markdown(f"""
<div style='text-align: center; background: rgba(0,255,65,0.1); padding: 15px; 
            border-radius: 10px; margin-bottom: 20px; border: 1px solid #00ff41;'>
    <span style='color: #00ff41; font-size: 1.2rem; font-family: monospace;'>
        ⏰ SYSTEM TIME: {current_time} | STATUS: <span style='color: #ff3030;'>● ACTIVE</span>
    </span>
</div>
""", unsafe_allow_html=True)

# Intelligence Quotes
quotes = [
    "\"The best defense is a good prediction.\" — Intelligence Doctrine",
    "\"Data reveals what the enemy tries to hide.\" — Strategic Analysis",
    "\"In 3D space, threats have nowhere to hide.\" — Sentinel Protocol",
    "\"Neural networks see patterns humans cannot.\" — AI Warfare Manual"
]
import random
st.markdown(f"""
<div style='background: rgba(0, 212, 255, 0.1); padding: 15px; border-left: 5px solid #00d4ff; 
            border-radius: 5px; margin-bottom: 20px;'>
    <p style='color: #00d4ff; font-style: italic; margin: 0; font-size: 1.1rem;'>{random.choice(quotes)}</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    return RiskEngine()

engine = load_engine()
analyst = IntelligenceAnalyst()

# Load Data
data_path = os.path.join(project_root, "data/processed/training_sets/train_multimodal.parquet")
if not os.path.exists(data_path):
    st.error("❌ Critical Data Failure.")
    st.stop()

df = pd.read_parquet(data_path)
df['Day'] = pd.to_datetime(df['Day'])

# --- SIDEBAR ---
st.sidebar.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h1 style='color: #ff3030; font-size: 3rem; margin: 0;'>⚡</h1>
    <h2 style='color: #00ff41; margin: 10px 0;'>SENTINEL V7</h2>
    <p style='color: #00d4ff; font-family: monospace; margin: 0;'>Neural Threat Engine</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# System Status
st.sidebar.markdown("""
<div style='background: rgba(0, 255, 65, 0.15); padding: 15px; border-radius: 8px; 
            border: 2px solid #00ff41; margin-bottom: 20px;'>
    <p style='color: #00ff41; margin: 0; font-size: 1rem; text-align: center;'>
        <b>🎯 SYSTEM STATUS</b><br>
        <span style='font-size: 0.9rem;'>All Neural Cores Online</span>
    </p>
</div>
""", unsafe_allow_html=True)

# Auto-select the last available date
min_date = df['Day'].min().date()
max_date = df['Day'].max().date()
default_date = max_date 

date_filter = st.sidebar.date_input(
    "📅 Operation Date", 
    value=default_date, 
    min_value=min_date, 
    max_value=max_date
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ THREAT PARAMETERS")

# Lower default threshold
threshold = st.sidebar.slider(
    "Risk Threshold", 
    0.0, 1.0, 0.0, 
    help="Adjust to filter threat levels on the map"
)

# Map Style Selector
st.sidebar.markdown("---")
st.sidebar.markdown("### 🗺️ MAP STYLE")
map_style_option = st.sidebar.radio(
    "Choose Base Map",
    ["Satellite", "Streets (Light)", "Streets (Dark)", "Terrain"],
    index=0,
    help="Select map style for better geographic visibility"
)

# Map style mapping
map_styles = {
    "Satellite": "mapbox://styles/mapbox/satellite-streets-v12",
    "Streets (Light)": "mapbox://styles/mapbox/streets-v12",
    "Streets (Dark)": "mapbox://styles/mapbox/dark-v11",
    "Terrain": "mapbox://styles/mapbox/outdoors-v12"
}
selected_map_style = map_styles[map_style_option]

st.sidebar.markdown(f"""
<div style='background: rgba(255, 48, 48, 0.15); padding: 12px; border-radius: 5px; 
            border: 1px solid #ff3030; margin-top: 10px;'>
    <p style='color: #ff3030; margin: 0; text-align: center;'>
        ⚠️ Displaying Threats ≥ <b>{threshold:.0%}</b>
    </p>
</div>
""", unsafe_allow_html=True)

# --- MAIN LOGIC: PRE-CALCULATE RISK ---
day_df = df[df['Day'] == pd.to_datetime(date_filter)]
active_hexes = day_df['h3_hex'].unique()

map_data = []
max_risk_display = 0.0
prediction_cache = {} 

if len(active_hexes) == 0:
    st.warning(f"⚠️ No intelligence data available for {date_filter}. Try picking another date.")
else:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, h in enumerate(active_hexes):
        status_text.text(f"🔍 Scanning sector {idx + 1}/{len(active_hexes)}...")
        progress_bar.progress((idx + 1) / len(active_hexes))
        
        # Predict
        risk, attn, top_day = engine.predict(df, h)
        
        # Store in cache
        prediction_cache[h] = {"risk": risk, "attn": attn, "top_day": top_day}
        
        if risk > max_risk_display:
            max_risk_display = risk

        # Filter for map
        if risk >= threshold:
            # Enhanced color mapping with more opacity for visibility
            if risk > 0.7:
                color = [255, 0, 0, 230]  # Bright Red - highly visible
            elif risk > 0.4:
                color = [255, 140, 0, 210]  # Orange
            elif risk > 0.2:
                color = [255, 215, 0, 190]  # Gold/Yellow
            else:
                color = [50, 255, 50, 170]  # Bright Green
            
            # Get coordinates
            lat, lon = engine.geo.hex_to_lat_lon(h)
            
            # Elevation based on risk (more dramatic scaling)
            height = int((risk ** 0.7) * 80000 + 5000)  # Power scaling for visibility
            
            map_data.append({
                "hex": h,
                "lat": lat,
                "lon": lon,
                "risk": risk,
                "height": height,
                "fill_color": color,
                "risk_label": f"{risk:.1%}",
                "threat_level": "CRITICAL" if risk > 0.7 else "HIGH" if risk > 0.4 else "MODERATE"
            })
    
    progress_bar.empty()
    status_text.empty()

map_df = pd.DataFrame(map_data)

# --- 3D MAP VISUALIZATION (ENHANCED) ---
st.markdown("---")
st.markdown("## 🌐 3D GLOBAL THREAT MONITOR")

# Display stats
col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
with col_stat1:
    st.metric("📊 Total Sectors", len(active_hexes))
with col_stat2:
    st.metric("🎯 Active Threats", len(map_df))
with col_stat3:
    st.metric("🔥 Max Risk", f"{max_risk_display:.1%}")
with col_stat4:
    critical_count = len(map_df[map_df['risk'] > 0.7]) if not map_df.empty else 0
    st.metric("🚨 Critical Alerts", critical_count)

st.markdown("---")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style='background: rgba(255, 48, 48, 0.2); padding: 15px; border-radius: 8px; 
            border: 2px solid #ff3030;'>
    <p style='color: white; margin: 0; font-size: 0.85rem; text-align: center;'>
        <b>MAXIMUM THREAT</b>
    </p>
    <h2 style='color: #ff3030; margin: 10px 0 0 0; text-align: center;'>
        🔥 {max_risk_display:.2%}
    </h2>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='background: rgba(0, 212, 255, 0.1); padding: 12px; border-radius: 5px;'>
    <p style='color: #00d4ff; margin: 0; font-size: 0.75rem;'><b>📡 INTEL SOURCES</b></p>
    <ul style='color: white; font-size: 0.7rem; margin: 8px 0 0 0; padding-left: 15px;'>
        <li>Hybrid V3 Neural Engine</li>
        <li>H3 Geospatial Intelligence</li>
        <li>Multi-Modal Signal Analysis</li>
        <li>Llama-3 Analyst Core</li>
    </ul>
</div>
""", unsafe_allow_html=True)

if not map_df.empty:
    # Calculate center point
    center_lat = map_df['lat'].mean()
    center_lon = map_df['lon'].mean()
    
    # Enhanced H3 Hexagon Layer with better visibility
    layer = pdk.Layer(
        "H3HexagonLayer",
        map_df,
        get_hexagon="hex",
        get_fill_color="fill_color",
        get_elevation="height",
        elevation_scale=1.8,  # Increased for maximum visibility
        elevation_range=[0, 8000],  # Increased range
        pickable=True,
        extruded=True,
        opacity=0.9,  # Higher opacity
        wireframe=False,  # Removed wireframe for cleaner look on satellite
        line_width_min_pixels=2,
        material={
            "ambient": 0.8,
            "diffuse": 0.8,
            "shininess": 64,
            "specularColor": [60, 60, 60]
        }
    )

    # View State with better angle for geographic context
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=5,  # Better zoom for country visibility
        pitch=45,  # Balanced 3D angle
        bearing=0,
        height=650
    )

    # Enhanced deck with better styling and visible map
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={
            "html": "<b>Threat Level:</b> {threat_level}<br/><b>Risk Score:</b> {risk_label}<br/><b>Hex ID:</b> {hex}",
            "style": {
                "backgroundColor": "rgba(0, 0, 0, 0.9)",
                "color": "#00ff41",
                "border": "2px solid #00ff41",
                "borderRadius": "5px",
                "padding": "10px",
                "fontFamily": "monospace"
            }
        },
        map_style=selected_map_style  # Use selected map style
    )

    st.pydeck_chart(deck, use_container_width=True)
    
    # Map Legend
    st.markdown("""
    <div style='background: rgba(0,0,0,0.7); padding: 15px; border-radius: 8px; 
                border: 1px solid #00ff41; margin-top: 10px;'>
        <p style='color: #00ff41; margin: 0 0 10px 0; font-weight: bold;'>🎨 THREAT LEVEL LEGEND</p>
        <div style='display: flex; justify-content: space-around;'>
            <span style='color: #00ff64;'>● LOW (0-20%)</span>
            <span style='color: #ffff00;'>● MODERATE (20-40%)</span>
            <span style='color: #ff8c00;'>● HIGH (40-70%)</span>
            <span style='color: #ff0000;'>● CRITICAL (70-100%)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("🔍 No threats detected at current threshold. Lower the slider to reveal low-risk sectors.")

# --- DEEP DIVE SECTION ---
st.markdown("---")
st.markdown("## 🔬 SECTOR INTELLIGENCE ANALYSIS")

if prediction_cache:
    # Sort by risk
    sorted_hexes = sorted(prediction_cache.keys(), key=lambda x: prediction_cache[x]['risk'], reverse=True)
    
    # Format selector options
    hex_options = [f"{h} (Risk: {prediction_cache[h]['risk']:.1%})" for h in sorted_hexes]
    selected_option = st.selectbox("🎯 Select Sector for Deep Analysis", hex_options)
    selected_hex = sorted_hexes[hex_options.index(selected_option)]

    if selected_hex:
        data = prediction_cache[selected_hex]
        
        # Risk metric display
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("🎯 Threat Level", f"{data['risk']:.2%}")
        with col_m2:
            threat_cat = "CRITICAL" if data['risk'] > 0.7 else "HIGH" if data['risk'] > 0.4 else "MODERATE"
            st.metric("⚠️ Classification", threat_cat)
        with col_m3:
            st.metric("📍 Sector ID", f"...{selected_hex[-8:]}")
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📈 Historical Signal Timeline")
            st.caption("*14-day trend analysis: News volume vs. sentiment*")
            
            # Get historical data
            hist_data = df[df['h3_hex'] == selected_hex].sort_values('Day')
            mask = (hist_data['Day'] <= pd.to_datetime(date_filter)) & \
                   (hist_data['Day'] > pd.to_datetime(date_filter) - timedelta(days=14))
            chart_data = hist_data.loc[mask, ['Day', 'AvgTone', 'NumMentions']].set_index('Day')
            
            if not chart_data.empty:
                fig, ax1 = plt.subplots(figsize=(7, 4))
                fig.patch.set_facecolor('#0a0e27')
                ax1.set_facecolor('#0a0e27')
                
                # Volume bars
                ax1.bar(chart_data.index, chart_data['NumMentions'], 
                       color='#00d4ff', alpha=0.4, label='News Volume', width=0.8)
                ax1.set_ylabel('Volume', color='#00d4ff', fontsize=11)
                ax1.tick_params(axis='y', labelcolor='#00d4ff')
                ax1.tick_params(axis='x', colors='white', rotation=45)
                ax1.grid(True, alpha=0.2, color='#00ff41')
                
                # Sentiment line
                ax2 = ax1.twinx()
                ax2.plot(chart_data.index, chart_data['AvgTone'], 
                        color='#ff3030', linewidth=2.5, marker='o', 
                        markersize=6, label='Sentiment', markerfacecolor='#ff6b6b')
                ax2.set_ylabel('Sentiment Score', color='#ff3030', fontsize=11)
                ax2.tick_params(axis='y', labelcolor='#ff3030')
                ax2.axhline(y=-2, color='yellow', linestyle='--', 
                           alpha=0.6, linewidth=2, label='Danger Threshold')
                ax2.legend(loc='upper right', fontsize=8)
                
                plt.title('Threat Evolution Pattern', color='white', fontsize=12, pad=10)
                st.pyplot(fig)
            else:
                st.warning("⚠️ Insufficient historical data for trend visualization.")

        with col2:
            st.subheader("🧠 Neural Attention Matrix")
            st.caption("*XAI: Temporal trigger point identification*")
            
            if data['attn'] is not None:
                fig_attn, ax_attn = plt.subplots(figsize=(7, 3))
                fig_attn.patch.set_facecolor('#0a0e27')
                ax_attn.set_facecolor('#0a0e27')
                
                days = ["T-7", "T-6", "T-5", "T-4", "T-3", "T-2", "T-1"]
                colors = ['#00d4ff'] * 7
                if data['top_day'] is not None:
                    colors[data['top_day']] = '#ff3030'
                
                bars = ax_attn.bar(days, data['attn'], color=colors, 
                                  edgecolor='white', linewidth=1.5, alpha=0.8)
                ax_attn.set_ylim(0, max(data['attn']) * 1.2)
                ax_attn.spines['top'].set_visible(False)
                ax_attn.spines['right'].set_visible(False)
                ax_attn.spines['bottom'].set_color('#00ff41')
                ax_attn.spines['left'].set_color('#00ff41')
                ax_attn.tick_params(colors='white')
                ax_attn.set_ylabel('Attention Weight', color='white', fontsize=11)
                ax_attn.grid(True, alpha=0.2, axis='y', color='#00ff41')
                
                st.pyplot(fig_attn)
                
                if data['top_day'] is not None:
                    st.markdown(f"""
                    <div style='background: rgba(255, 48, 48, 0.2); padding: 15px; 
                                border-left: 5px solid #ff3030; border-radius: 5px; margin-top: 10px;'>
                        <p style='color: #ff3030; margin: 0; font-size: 1.1rem;'>
                            🚨 <b>CRITICAL TRIGGER:</b> {days[data['top_day']]}
                        </p>
                        <p style='color: white; margin: 5px 0 0 0; font-size: 0.9rem;'>
                            Neural network identified this day as the primary threat indicator
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

        # --- BRIEFING SECTION ---
        st.markdown("---")
        st.subheader("📝 AI-Generated Situation Report")
        st.caption("*Powered by Llama-3 Intelligence Analyst*")
        
        if st.button("🤖 GENERATE CLASSIFIED BRIEFING", use_container_width=True):
            with st.spinner("🔐 Establishing secure uplink to AI analyst..."):
                hex_data = df[df['h3_hex'] == selected_hex].sort_values('Day').tail(7)
                top_day_idx = data['top_day']
                
                if top_day_idx is not None and top_day_idx < len(hex_data):
                    row = hex_data.iloc[top_day_idx]
                    
                    context = (
                        f"CLASSIFIED INTEL REPORT\n"
                        f"SECTOR: {selected_hex}\n"
                        f"SIGNAL INTERCEPTS: {int(row['NumMentions'])} unique sources detected\n"
                        f"SENTIMENT ANALYSIS: {row['AvgTone']:.2f} (Critical threshold: -2.0)\n"
                        f"MARKET VOLATILITY INDEX: {row['volatility_7d']:.4f}\n"
                        f"PATTERN: Concurrent negative sentiment spike with elevated message volume.\n"
                        f"ASSESSMENT: Pre-incident behavioral signature detected."
                    )
                    
                    briefing = analyst.generate_briefing(context, data['risk'])
                    
                    st.markdown(f"""
                    <div style='background: rgba(0, 255, 65, 0.05); padding: 25px; 
                                border-radius: 10px; border: 2px solid #00ff41; margin-top: 15px;'>
                        <p style='color: #00ff41; margin: 0 0 15px 0; font-size: 1.1rem;'>
                            <b>🔐 CLASSIFIED INTELLIGENCE BRIEFING</b>
                        </p>
                        <div style='color: white; line-height: 1.8;'>
                            {briefing}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("⚠️ Data integrity error: Cannot generate briefing from incomplete records.")
else:
    st.info("🔍 No active sectors detected for selected date. Adjust filters or choose a different operational timeframe.")

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 25px; background: rgba(0,0,0,0.5); 
            border-radius: 10px; border: 1px solid #00ff41;'>
    <p style='color: #00ff41; font-size: 1.2rem; margin: 0 0 10px 0; font-family: monospace;'>
        🛡️ SENTINEL V7 | Neural Threat Detection Platform
    </p>
    <p style='font-size: 0.85rem; color: #00d4ff; margin: 0;'>
        Powered by Hybrid V3 Engine • H3 Geospatial Intelligence • Llama-3 Analysis Core
    </p>
    <p style='font-size: 0.9rem; color: #00ff41; margin: 15px 0 5px 0; font-weight: bold;'>
        💻 Made by Aditya Sharma (IIIT Bhopal)
    </p>
    <p style='font-size: 0.75rem; color: gray; margin: 10px 0 0 0;'>
        Classification: TOP SECRET // COMPARTMENTED // NOFORN
    </p>
</div>
""", unsafe_allow_html=True)
