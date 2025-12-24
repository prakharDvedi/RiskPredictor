import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.training.dataset import EventDataset
from src.models.event_transformer import EventPredictor

def train_model():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on {device}")
    
    # 2. Data
    full_dataset = EventDataset("data/processed/training_sets/train_multimodal.parquet")
    
    # 80/20 Train/Val split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    
    # 3. Model
    model = EventPredictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss() # Binary Cross Entropy for 0/1 Event
    
    # 4. Loop
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            txt = batch['text_seq'].to(device)
            eco = batch['econ_seq'].to(device)
            lbl = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # Forward
            preds = model(txt, eco)
            loss = criterion(preds, lbl)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f}")
        
    # 5. Save
    torch.save(model.state_dict(), "outputs/model_v1.pth")
    print("✅ Model saved to outputs/model_v1.pth")

if __name__ == "__main__":
    train_model()