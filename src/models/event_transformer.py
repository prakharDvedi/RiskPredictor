import torch
import torch.nn as nn

class EventPredictor(nn.Module):
    def __init__(self, text_dim=384, econ_dim=1, d_model=128, n_heads=4, n_layers=2, dropout=0.3):
        super().__init__()
        
        # 1. Feature Projectors
        self.text_fc = nn.Sequential(
            nn.Linear(text_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.econ_fc = nn.Sequential(
            nn.Linear(econ_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Fusion
        self.fusion_norm = nn.LayerNorm(d_model * 2)
        
        # 2. LSTM Layer (The "Trend" Engine)
        # Captures the sequential build-up of tension
        self.lstm = nn.LSTM(
            input_size=d_model * 2,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
            bidirectional=True # Looks forward and backward
        )
        
        # 3. Transformer Layer (The "Trigger" Engine)
        # Looks for specific events in the sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model * 2, # *2 because LSTM is bidirectional
            nhead=n_heads, 
            dim_feedforward=512, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 4. Attention Head (For Explainability)
        self.attn_head = nn.MultiheadAttention(
            embed_dim=d_model * 2, 
            num_heads=n_heads, 
            batch_first=True
        )
        
        # 5. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1) # No Sigmoid here, we use BCEWithLogitsLoss for stability
        )
        
    def forward(self, text_seq, econ_seq, return_attention=False):
        # A. Project & Fuse
        t_emb = self.text_fc(text_seq) 
        e_emb = self.econ_fc(econ_seq)
        x = torch.cat([t_emb, e_emb], dim=2) 
        x = self.fusion_norm(x)
        
        # B. Hybrid Processing
        # 1. LSTM captures the "Flow"
        lstm_out, _ = self.lstm(x) 
        
        # 2. Transformer captures the "Context"
        trans_out = self.transformer(lstm_out)
        
        # C. Attention (Explainability)
        attn_out, attn_weights = self.attn_head(trans_out, trans_out, trans_out)
        
        # D. Predict
        # We take the LAST state of the sequence
        last_state = attn_out[:, -1, :]
        logits = self.classifier(last_state)
        
        if return_attention:
            return torch.sigmoid(logits), attn_weights
            
        return torch.sigmoid(logits)