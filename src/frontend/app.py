import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime

# --- CUSTOM CSS STYLING ---
st.markdown("""
<style>
    /* Main Background with Grid Pattern */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
        background-image: 
            linear-gradient(rgba(0, 255, 65, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 65, 0.03) 1px, transparent 1px);
        background-size: 50px 50px;
    }
    
    /* Title Styling */
    h1 {
        color: #00ff41 !important;
        font-family: 'Courier New', monospace;
        text-shadow: 0 0 10px rgba(0, 255, 65, 0.5);
        letter-spacing: 2px;
    }
    
    h2, h3 {
        color: #00d4ff !important;
        font-family: 'Courier New', monospace;
        text-shadow: 0 0 5px rgba(0, 212, 255, 0.3);
    }
    
    /* Sidebar Styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 2px solid #00ff41;
    }
    
    /* Metrics */
    .stMetric {
        background: rgba(0, 255, 65, 0.1);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #00ff41;
        box-shadow: 0 0 20px rgba(0, 255, 65, 0.2);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #ff3030 0%, #ff6b6b 100%);
        color: white;
        border: none;
        padding: 10px 30px;
        font-weight: bold;
        border-radius: 5px;
        box-shadow: 0 0 15px rgba(255, 48, 48, 0.4);
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        box-shadow: 0 0 25px rgba(255, 48, 48, 0.6);
        transform: scale(1.05);
    }
    
    /* Warning/Info Boxes */
    .stAlert {
        background: rgba(0, 212, 255, 0.1);
        border-left: 4px solid #00d4ff;
    }
    
    /* Code blocks */
    code {
        color: #00ff41;
        background: rgba(0, 255, 65, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- 1. ROBUST PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
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
st.set_page_config(
    layout="wide", 
    page_title="Sentinel V4: XAI War Room",
    page_icon="🛡️"
)

# --- HEADER WITH QUOTE ---
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("""
    # 🛡️ SENTINEL V4: XAI WAR ROOM
    ### *Neural Threat Detection & Predictive Intelligence System*
    """)
    
with col_h2:
    current_time = datetime.now().strftime("%H:%M:%S")
    st.markdown(f"""
    <div style='text-align: right; color: #00ff41; font-family: monospace;'>
        <h3>⏰ {current_time}</h3>
        <p style='color: #00d4ff;'>SYSTEM ACTIVE</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Rotating Intelligence Quotes
quotes = [
    "\"In the age of information, ignorance is a choice.\" — Unknown",
    "\"The supreme art of war is to subdue the enemy without fighting.\" — Sun Tzu",
    "\"Knowledge is power. Intelligence is the key to victory.\" — Anonymous",
    "\"Prevention is better than cure, prediction is better than prevention.\" — AI Proverb",
    "\"Data sees what the eye cannot.\" — Intelligence Doctrine"
]

import random
selected_quote = random.choice(quotes)

st.markdown(f"""
<div style='background: rgba(0, 212, 255, 0.1); padding: 15px; border-left: 4px solid #00d4ff; 
            border-radius: 5px; margin-bottom: 20px;'>
    <p style='color: #00d4ff; font-style: italic; margin: 0;'>{selected_quote}</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    return RiskEngine()

try:
    engine = load_engine()
    analyst = IntelligenceAnalyst()
    st.sidebar.success("✅ Neural Engine: ONLINE")
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
st.sidebar.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h1 style='color: #ff3030; font-size: 48px;'>⚡</h1>
    <h2>COMMAND CENTER</h2>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Mission Status
st.sidebar.markdown("""
<div style='background: rgba(0, 255, 65, 0.1); padding: 10px; border-radius: 5px; margin-bottom: 15px;'>
    <p style='color: #00ff41; margin: 0;'>🎯 <b>MISSION STATUS</b></p>
    <p style='color: white; margin: 5px 0 0 0; font-size: 12px;'>Real-time Threat Monitoring</p>
</div>
""", unsafe_allow_html=True)

# Date Picker
min_date = df['Day'].min().date()
max_date = df['Day'].max().date()
default_date = pd.to_datetime("2024-01-22").date()

if default_date < min_date or default_date > max_date:
    default_date = min_date

date_filter = st.sidebar.date_input(
    "📅 Operation Date", 
    value=default_date, 
    min_value=min_date, 
    max_value=max_date
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ THREAT SENSITIVITY")

threshold = st.sidebar.slider(
    "Risk Alert Threshold", 
    0.0, 1.0, 0.15, 
    help="Lower threshold reveals more potential threats"
)

st.sidebar.markdown(f"""
<div style='background: rgba(255, 48, 48, 0.1); padding: 10px; border-radius: 5px; margin-top: 10px;'>
    <p style='color: #ff3030; margin: 0; font-size: 14px;'>
        ⚠️ Filtering Hexes > <b>{threshold:.0%}</b> Risk
    </p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# System Info
st.sidebar.markdown("""
<div style='background: rgba(0, 212, 255, 0.1); padding: 10px; border-radius: 5px;'>
    <p style='color: #00d4ff; margin: 0; font-size: 12px;'><b>📡 INTELLIGENCE SOURCES</b></p>
    <ul style='color: white; font-size: 11px; margin: 5px 0 0 0;'>
        <li>Hybrid V3 Neural Engine</li>
        <li>Multi-Modal Signal Analysis</li>
        <li>H3 Geospatial Intelligence</li>
        <li>Llama-3 Analyst Integration</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# --- 4. MAP LOGIC ---
st.markdown("## 🗺️ GEOSPATIAL THREAT MONITOR")
st.caption("*Real-time neural threat detection across operational theater*")

# Initialize Map
m = folium.Map(
    location=[22.0, 79.0], 
    zoom_start=5, 
    tiles="CartoDB dark_matter"
)

# Filter Data by Day
day_df = df[df['Day'] == pd.to_datetime(date_filter)]

active_hexes = day_df['h3_hex'].unique()
risk_cache = {}
max_risk_detected = 0.0

if len(active_hexes) == 0:
    st.warning(f"⚠️ No intelligence data available for {date_filter}. Adjust operation date.")
else:
    for h in active_hexes:
        risk, attn, top_day = engine.predict(df, h)
        
        if risk > max_risk_detected:
            max_risk_detected = risk
            
        risk_cache[h] = {"risk": risk, "attn": attn, "top_day": top_day}
        
        if risk > threshold:
            color = "#00ff00"
            if risk > 0.4: color = "#ffa500"
            if risk > 0.7: color = "#ff0000"
            
            folium.RegularPolygonMarker(
                location=engine.geo.hex_to_lat_lon(h),
                number_of_sides=6,
                radius=15,
                color=color,
                fill_color=color,
                fill_opacity=0.6,
                popup=f"<b>Risk Level:</b> {risk:.2%}<br><b>Hex:</b> {h}"
            ).add_to(m)

# Display Max Risk
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style='background: rgba(255, 48, 48, 0.2); padding: 15px; border-radius: 5px; border: 2px solid #ff3030;'>
    <p style='color: white; margin: 0; font-size: 12px;'>MAXIMUM DETECTED THREAT</p>
    <h2 style='color: #ff3030; margin: 10px 0 0 0;'>🔥 {max_risk_detected:.2%}</h2>
</div>
""", unsafe_allow_html=True)

# Render Map
map_data = st_folium(m, width="100%", height=500)

# --- 5. ANALYTICS PANEL (XAI) ---
st.markdown("---")
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("## 📊 SIGNAL ANALYSIS")
    st.caption("*Neural attention patterns & threat decomposition*")
    
    selected_hex = None
    
    if risk_cache:
        selected_hex = max(risk_cache, key=lambda k: risk_cache[k]['risk'])
        
    if selected_hex:
        data = risk_cache[selected_hex]
        
        st.metric(
            "🎯 SELECTED THREAT LEVEL", 
            f"{data['risk']:.2%}",
            delta=f"{'CRITICAL' if data['risk'] > 0.7 else 'ELEVATED' if data['risk'] > 0.4 else 'MODERATE'}"
        )
        
        st.markdown(f"""
        <div style='background: rgba(0, 212, 255, 0.1); padding: 10px; border-radius: 5px; margin-top: 10px;'>
            <p style='color: #00d4ff; margin: 0; font-size: 12px;'>Target Hex: <code>{selected_hex}</code></p>
        </div>
        """, unsafe_allow_html=True)
        
        if data['attn'] is not None:
            st.markdown("### 🧠 Neural Attention Weights")
            st.caption("*Temporal sequence analysis (T-7 to T-1)*")
            
            fig, ax = plt.subplots(figsize=(5, 3), facecolor='#0a0e27')
            ax.set_facecolor('#0a0e27')
            
            days_labels = ["T-7", "T-6", "T-5", "T-4", "T-3", "T-2", "T-1"]
            
            bar_colors = ['#00d4ff'] * 7
            if data['top_day'] is not None:
                bar_colors[data['top_day']] = '#ff3030'
                
            ax.bar(days_labels, data['attn'], color=bar_colors, edgecolor='white', linewidth=0.5)
            ax.set_ylim(0, max(data['attn']) + 0.1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#00ff41')
            ax.spines['left'].set_color('#00ff41')
            ax.tick_params(colors='white')
            ax.set_ylabel('Attention Weight', color='white')
            
            st.pyplot(fig)
            
            trigger_day_label = days_labels[data['top_day']]
            st.markdown(f"""
            <div style='background: rgba(255, 48, 48, 0.2); padding: 10px; border-left: 4px solid #ff3030; border-radius: 5px;'>
                <p style='color: #ff3030; margin: 0;'>🚨 <b>CRITICAL TRIGGER:</b> {trigger_day_label}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("🔍 No active threats detected in current operational window.")

with col2:
    st.markdown("## 📝 INTELLIGENCE BRIEFING")
    st.caption("*AI-generated threat assessment & recommendations*")
    
    if selected_hex:
        if st.button("🤖 GENERATE BRIEFING (Llama-3)", use_container_width=True):
            with st.spinner("🔄 Analyst is processing signal intelligence..."):
                
                hex_data = df[df['h3_hex'] == selected_hex].sort_values('Day').tail(7)
                top_day_idx = risk_cache[selected_hex]['top_day']
                
                if top_day_idx is not None and top_day_idx < len(hex_data):
                    row = hex_data.iloc[top_day_idx]
                    
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
                    
                    briefing = analyst.generate_briefing(trigger_text, risk_val)
                    
                    st.markdown(f"""
                    <div style='background: rgba(0, 255, 65, 0.05); padding: 20px; border-radius: 10px; 
                                border: 1px solid #00ff41;'>
                        {briefing}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("⚠️ Insufficient historical data for briefing generation.")
    else:
        st.markdown("""
        <div style='background: rgba(0, 212, 255, 0.1); padding: 20px; border-radius: 10px; text-align: center;'>
            <p style='color: #00d4ff; margin: 0;'>
                🔒 <b>BRIEFING SYSTEM STANDBY</b><br>
                <span style='font-size: 12px;'>Select an operational date with active signals to enable AI analysis</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; color: #00d4ff; font-family: monospace;'>
    <p>🛡️ SENTINEL V4 | Neural Threat Detection Platform</p>
    <p style='font-size: 10px; color: gray;'>Powered by Hybrid V3 Engine • H3 Geospatial Intelligence • Llama-3 Analysis</p>
</div>
""", unsafe_allow_html=True)