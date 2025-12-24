import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

class FinanceLoader:
    def __init__(self, output_dir="data/raw/finance"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Key Economic Indicators (The "Precursors")
        self.tickers = {
            "Brent_Oil": "BZ=F",       # Energy costs (high oil -> transport protests)
            "Wheat": "ZW=F",           # Food security (high wheat -> bread riots)
            "Gold": "GC=F",            # Fear index / Safe haven
            "S&P500": "^GSPC",         # US Market health
            "USD_Index": "DX-Y.NYB"    # Dollar strength (impacts emerging markets)
        }

    def fetch_data(self, days=30):
        """Fetches last N days of data for all tickers."""
        print(f"Fetching financial data for last {days} days...")
        
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        data_frames = []

        for name, ticker in self.tickers.items():
            try:
                # Download daily data
                df = yf.download(ticker, start=start_date, progress=False)
                
                if not df.empty:
                    # Keep only Close price and rename
                    # yfinance returns multi-index columns sometimes, flatten them
                    if isinstance(df.columns, pd.MultiIndex):
                        df = df['Close']
                    else:
                        df = df[['Close']]
                        
                    df.columns = ['price']
                    
                    # Feature Engineering: Daily Return & Volatility
                    df['pct_change'] = df['price'].pct_change()
                    df['volatility_7d'] = df['pct_change'].rolling(window=7).std()
                    
                    df['asset'] = name
                    df.reset_index(inplace=True)
                    data_frames.append(df)
                    print(f"✅ Fetched {name}")
                else:
                    print(f"⚠️ No data for {name}")
                    
            except Exception as e:
                print(f"❌ Error fetching {name}: {e}")

        if data_frames:
            final_df = pd.concat(data_frames)
            
            # Save
            timestamp = datetime.now().strftime("%Y%m%d")
            save_path = f"{self.output_dir}/finance_{timestamp}.csv"
            final_df.to_csv(save_path, index=False)
            print(f"💾 Saved financial data to {save_path}")
            return final_df
        return None

if __name__ == "__main__":
    loader = FinanceLoader()
    df = loader.fetch_data(days=60) # Get 2 months of history
    if df is not None:
        print(df.head())