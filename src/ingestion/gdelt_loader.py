import requests
import pandas as pd
import io
import zipfile
import os
from datetime import datetime
from src.utils.geo_utils import GeoGrid
from src.ingestion.gdelt_columns import GDELT_COLUMNS,RELEVANT_COLUMNS

class GDELTLoader:
    def __init__(self,output_dir="data/raw/gdelt"):
        self.output_dir=output_dir
        self.master_url="http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
        self.geo=GeoGrid(resolution=5)
        os.makedirs(output_dir,exist_ok=True)

    def get_latest_urls(self,n=5):
        """
        Fetches the last n file URLs from GDELT master list
        """
        print("Fetching master file list...")
        response=requests.get(self.master_url)
        lines=response.text.strip().split('\n')
        event_urls = [line.split(' ')[-1] for line in lines if "export.CSV.zip" in line]
        return event_urls[-n:]
    
    def process_url(self,url):
        """
        Downloads, unzipsm and filters a single GDELT file
        """
        filename = url.split('/')[-1]
        print(f"Processing {filename}...")
        try:
            r=requests.get(url)
            z=zipfile.ZipFile(io.BytesIO(r.content))
            csv_filename=z.namelist()[0]
            df = pd.read_csv(z.open(csv_filename), sep='\t', header=None, names=GDELT_COLUMNS)
            df=df[RELEVANT_COLUMNS].copy()
            df=df.dropna(subset=['ActionGeo_Lat', 'ActionGeo_Long'])

            df['h3_hex']=df.apply(
                lambda row: self.geo.lat_lon_to_hex(row['ActionGeo_Lat'], row['ActionGeo_Long']), 
                axis=1      
            )
            return df
        except Exception as e:
            print(f"Error processing {url}:{e}")
            return pd.DataFrame()
        
    def run(self):
        urls=self.get_latest_urls(n=3)
        all_data=[]
        for url in urls:
            df=self.process_url(url)
            if not df.empty:
                all_data.append(df)
        
        if all_data:
            final_df=pd.concat(all_data)
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path=f"{self.output_dir}/gdelt_{timestamp}.csv"
            final_df.to_csv(save_path,index=False)
            print(f"✅ Saved {len(final_df)} events to {save_path}")
            return final_df
        else:
            print("No data collected.")
            return None
        
if __name__ == "__main__":
    loader = GDELTLoader()
    df = loader.run()
    if df is not None:
        print(df[['Day', 'EventCode', 'h3_hex', 'SOURCEURL']].head())