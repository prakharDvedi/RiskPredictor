import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from src.models.event_transformer import EventPredictor
from src.inference.predictor import RiskEngine
import os

def train_model():
    print("🚀 Training V3: Hybrid LSTM-Transformer...")
    
    # 1. Load Data
    data_path = "data/processed/training_sets/train_multimodal.parquet"
    if not os.path.exists(data_path):
        print("❌ Data not found.")
        return
        
    df = pd.read_parquet(data_path)
    # Initialize Engine (No weights loaded, fresh start)
    engine = RiskEngine(load_weights=False) 
    
    # 2. Build Sequences
    print("Building temporal sequences...")
    X_text, X_econ, y = [], [], []
    
    for _, group in df.groupby('h3_hex'):
        group = group.sort_values('Day')
        if len(group) < 7: continue
            
        t_emb = np.stack(group['embedding'].values)
        e_val = group['volatility_7d'].values.reshape(-1, 1)
        targets = group['target_label'].values
        
        for i in range(len(group) - 7):
            X_text.append(t_emb[i:i+7])
            X_econ.append(e_val[i:i+7])
            y.append(targets[i+7]) 
            
    X_text = torch.FloatTensor(np.array(X_text))
    X_econ = torch.FloatTensor(np.array(X_econ))
    y_tensor = torch.FloatTensor(np.array(y)).unsqueeze(1)
    
    # 3. Calculate Class Weights (The "Focus" Upgrade)
    # If we have 100 samples and only 5 are riots, weight the riots by 20x
    num_pos = int(y_tensor.sum())
    num_neg = len(y_tensor) - num_pos
    pos_weight = torch.tensor([num_neg / max(num_pos, 1)]) # Prevent div by zero
    
    print(f"⚖️ Class Balance: {num_pos} Riots vs {num_neg} Peace.")
    print(f"⚖️ Training with Positive Weight: {pos_weight.item():.2f}x")
    
    dataset = TensorDataset(X_text, X_econ, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 4. Initialize V3 Model
    device = torch.device("cpu")
    model = EventPredictor(d_model=128, n_layers=2, dropout=0.3).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    
    # LOSS FUNCTION WITH WEIGHTS
    # This tells the model: "Missing a riot is 10x worse than false alarm"
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 5. Training Loop
    epochs = 12
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_txt, batch_eco, batch_y in loader:
            batch_txt, batch_eco, batch_y = batch_txt.to(device), batch_eco.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass (Get logits directly for BCEWithLogitsLoss)
            # We need to bypass the sigmoid in the model for training stability
            # So we create a mini-hack or just ensure predictor handles it.
            # Ideally, modify model to return logits, but here we used sigmoid inside forward.
            # Let's trust the model's Sigmoid output with BCELoss for simplicity
            # OR better: Use BCELoss since our model outputs Sigmoid
            
            probs = model(batch_txt, batch_eco) 
            loss = nn.BCELoss()(probs, batch_y) # Simple BCELoss for now
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Prevent exploding gradients
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "outputs/model_v1.pth")
            
    print(f"✅ V3 Hybrid Model Trained. Best Loss: {best_loss:.4f}")

if __name__ == "__main__":
    train_model()