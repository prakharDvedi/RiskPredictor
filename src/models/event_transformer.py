import torch
import torch.nn as nn

class EventPredictor(nn.Module):
    def __init__(self, text_dim=384, econ_dim=1, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        
        # 1. Feature Projectors
        self.text_fc = nn.Linear(text_dim, d_model)
        self.econ_fc = nn.Linear(econ_dim, d_model)
        
        self.fusion_norm = nn.LayerNorm(d_model * 2)
        
        # 2. Manual Self-Attention Layer (To capture weights)
        # We use a single layer for the demo to make extraction easy
        self.attention = nn.MultiheadAttention(embed_dim=d_model * 2, num_heads=n_heads, batch_first=True)
        
        # 3. Prediction Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, text_seq, econ_seq, return_attention=False):
        # Input Shapes: (Batch, Window, Dim)
        
        # A. Project & Fuse
        t_emb = torch.relu(self.text_fc(text_seq)) 
        e_emb = torch.relu(self.econ_fc(econ_seq))
        
        x = torch.cat([t_emb, e_emb], dim=2) # (B, W, 128)
        x = self.fusion_norm(x)
        
        # B. Self-Attention with Weights
        # attn_output: The processed features
        # attn_weights: The "Importance Map" (Batch, TargetSeq, SourceSeq)
        attn_output, attn_weights = self.attention(x, x, x)
        
        # C. Predict based on the last time step
        last_state = attn_output[:, -1, :] 
        prediction = self.classifier(last_state)
        
        if return_attention:
            # Return average attention across heads for the LAST time step
            # We want to know: "When predicting T=7, which days did it look at?"
            # Shape: (Batch, Window)
            return prediction, attn_weights
            
        return prediction