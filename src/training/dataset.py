import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class EventDataset(Dataset):
    def __init__(self, parquet_path, window_size=7):
        """
        Converts flat history into sequences:
        X = [Day_t-7, ..., Day_t-1]
        Y = Label_t
        """
        self.window_size = window_size
        
        # Load and Sort
        print(f"Loading dataset from {parquet_path}...")
        self.df = pd.read_parquet(parquet_path)
        self.df = self.df.sort_values(['h3_hex', 'Day'])
        
        # Normalize volatility (Critical for convergence)
        # Simple MinMax scaling or standard scaling
        v_mean = self.df['volatility_7d'].mean()
        v_std = self.df['volatility_7d'].std() + 1e-6
        self.df['volatility_7d'] = (self.df['volatility_7d'] - v_mean) / v_std
        
        self.sequences = []
        self._build_sequences()
        
    def _build_sequences(self):
        """Groups by Hexagon and creates sliding windows."""
        print("Building temporal sequences...")
        
        # Group by location
        for hex_id, group in self.df.groupby('h3_hex'):
            # We need at least window_size days to predict the next one
            if len(group) <= self.window_size:
                continue
                
            # Extract features as numpy arrays to speed up indexing
            embeddings = np.stack(group['embedding'].values) # (N, 384)
            econ = group['volatility_7d'].values.reshape(-1, 1) # (N, 1)
            labels = group['target_label'].values # (N,)
            
            # Slide the window
            # Range: stop at len - 1 because we need the NEXT day as target
            for i in range(len(group) - self.window_size):
                # X: days i to i+7
                emb_seq = embeddings[i : i + self.window_size]
                econ_seq = econ[i : i + self.window_size]
                
                # Y: day i+7 (The prediction target)
                target = labels[i + self.window_size]
                
                self.sequences.append((emb_seq, econ_seq, target))
                
        print(f"✅ Created {len(self.sequences)} sequences.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        emb_seq, econ_seq, target = self.sequences[idx]
        
        # Convert to Float Tensors
        return {
            'text_seq': torch.FloatTensor(emb_seq),  # Shape: (7, 384)
            'econ_seq': torch.FloatTensor(econ_seq), # Shape: (7, 1)
            'label': torch.FloatTensor([target])     # Shape: (1,)
        }