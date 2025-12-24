import requests
import pandas as pd
import io
import zipfile
import os
from datetime import datetime
from src.utils.geo_utils import GeoGrid

class RobustGDELTLoader:
    def __init__(self, output_dir="data/raw/gdelt"):
        self.output_dir = output_dir
        self.master_url = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
        self.geo = GeoGrid(resolution=5)
        os.makedirs(output_dir, exist_ok=True)

    def get_latest_urls(self, n=5):
        print("Fetching master file list...")
        try:
            response = requests.get(self.master_url)
            lines = response.text.strip().split('\n')
            event_urls = [line.split(' ')[-1] for line in lines if "export.CSV.zip" in line]
            return event_urls[-n:]
        except Exception as e:
            print(f"Error fetching list: {e}")
            return []

    def process_url(self, url):
        print(f"Processing {url}...")
        try:
            r = requests.get(url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            csv_filename = z.namelist()[0]
            
            # READ RAW: No header, no names. Just indices.
            # GDELT 2.0 Standard Indices:
            # 0: GlobalEventID, 1: Day (Integer), 56: Lat, 57: Long, 60: SourceURL
            # We explicitly ask for these columns
            df = pd.read_csv(
                z.open(csv_filename), 
                sep='\t', 
                header=None,
                usecols=[0, 1, 26, 30, 33, 34, 56, 57, 60], # Key indices
                names=[
                    "GlobalEventID", "Day", "EventCode", "NumMentions", 
                    "AvgTone", "Actor1Geo_Type", 
                    "ActionGeo_Lat", "ActionGeo_Long", "SOURCEURL"
                ],
                dtype={'GlobalEventID': str, 'Day': int, 'EventCode': str}
            )
            
            # Filter bad rows immediately
            df = df.dropna(subset=['ActionGeo_Lat', 'ActionGeo_Long', 'Day', 'SOURCEURL'])
            
            # Add H3 Hexagon
            df['h3_hex'] = df.apply(
                lambda row: self.geo.lat_lon_to_hex(row['ActionGeo_Lat'], row['ActionGeo_Long']), 
                axis=1
            )
            
            return df
            
        except Exception as e:
            print(f"⚠️ Error processing {url}: {e}")
            return pd.DataFrame()

    def run(self):
        urls = self.get_latest_urls(n=5) # Get last ~75 mins
        all_data = []
        
        for url in urls:
            df = self.process_url(url)
            if not df.empty:
                all_data.append(df)
        
        if all_data:
            final_df = pd.concat(all_data)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"{self.output_dir}/gdelt_{timestamp}.csv"
            final_df.to_csv(save_path, index=False)
            print(f"✅ Saved {len(final_df)} clean events to {save_path}")
            return final_df
        else:
            print("❌ No data collected.")
            return None

if __name__ == "__main__":
    loader = RobustGDELTLoader()
    df = loader.run()
    if df is not None:
        print(df.head())