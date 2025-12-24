import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import re
import os

class TextEmbedder:
    def __init__(self,model_name='paraphrase-multilingual-MiniLM-L12-v2',device=None):
        self.device=device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model {model_name} on {self.device}")
        self.model=SentenceTransformer(model_name,device=self.device)

    def extract_slug(self,url):
       """
       Extracts meaningful text from a URL
       """
       if not isinstance(url,str):
           return ""
       parts = url.split('/')
       candidates = [p for p in parts if len(p) > 5]
       if not candidates:
           return ""
       
       slug=candidates[-1]
       slug = re.sub(r'[-_]', ' ', slug)
       slug = re.sub(r'\.(html|htm|php|asp|aspx)', '', slug)
       return slug
    
    def process_file(self,input_path):
        """
        Loads a GDELT CSV, computes embeddings, and saves the result
        """
        print(f"Loading {input_path}...")
        df=pd.read_csv(input_path)

        if 'SOURCEURL' not in df.columns:
            print("No sourceurl column found")
            return None
        
        #1. Extract text signal
        print("Processing URL slugs...")
        df['text_content']=df['SOURCEURL'].apply(self.extract_slug)

        df = df[df['text_content'].str.len() > 3].copy()

        #2. Compute embeddings
        print(f"Computing embeddings for {len(df)} rows..")
        texts=df['text_content'].tolist()

        embeddings = self.model.encode(texts, batch_size=64, show_progress_bar=True)
        
        df['embedding'] = [x.tolist() for x in embeddings]
        filename = os.path.basename(input_path).replace('.csv', '.parquet')
        output_dir="data/processed/embeddings"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)

        df.to_parquet(output_path, index=False)
        print(f"✅ Saved embeddings to {output_path}")
        return df
    
if __name__ == "__main__":
    embedder = TextEmbedder()
    
    # Find the most recent GDELT file in raw/gdelt
    gdelt_dir = "data/raw/gdelt"
    files = [os.path.join(gdelt_dir, f) for f in os.listdir(gdelt_dir) if f.endswith('.csv')]
    
    if files:
        latest_file = max(files, key=os.path.getctime)
        embedder.process_file(latest_file)
    else:
        print("⚠️ No GDELT files found. Run gdelt_loader.py first.")