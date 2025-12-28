#!/bin/bash

# 1. Simulate/Scrape Data (In a real V3, this would be live GDELT)
echo "🔄 [V3 AUTO] Ingesting & Simulating Data..."
python3 -m src.preprocessing.simulate_history

# 2. Refresh Labels
echo "🏷️ [V3 AUTO] Refreshing Labels..."
python3 -m src.preprocessing.labeler

# 3. Retrain Model (Active Learning)
# We retrain quickly (1 epoch) to adapt to new patterns
echo "🧠 [V3 AUTO] Retraining V2 Deep Transformer..."
python3 -m src.training.train

# 4. Launch the War Room (The persistent process)
echo "🚀 [V3 AUTO] Launching Sentinel Dashboard..."
python3 -m streamlit run src/frontend/app.py --server.port 8501 --server.address 0.0.0.0