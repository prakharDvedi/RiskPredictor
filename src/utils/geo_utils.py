import h3
import numpy as np
from typing import List, Tuple

class GeoGrid:
    """
    Handles conversion between Lat/Long and H3 Hexagons.
    UPDATED for H3 v4.0+ API.
    """
    def __init__(self, resolution: int = 5):
        self.resolution = resolution

    def lat_lon_to_hex(self, lat: float, lon: float) -> str:
        """Converts a coordinate to an H3 Hexagon ID."""
        # v4 API: geo_to_h3 -> latlng_to_cell
        return h3.latlng_to_cell(lat, lon, self.resolution)

    def hex_to_lat_lon(self, hex_id: str) -> Tuple[float, float]:
        """Returns the center coordinate (lat, lon) of a hexagon."""
        # v4 API: h3_to_geo -> cell_to_latlng
        return h3.cell_to_latlng(hex_id)

    def get_neighbors(self, hex_id: str, k: int = 1) -> List[str]:
        """Get k-ring neighbors for spatial convolution."""
        # v4 API: k_ring -> grid_disk
        return list(h3.grid_disk(hex_id, k))

# Quick Test
if __name__ == "__main__":
    geo = GeoGrid(resolution=5)
    # Example: New York City
    nyc_hex = geo.lat_lon_to_hex(40.7128, -74.0060)
    print(f"NYC H3 Index: {nyc_hex}")
    
    coords = geo.hex_to_lat_lon(nyc_hex)
    print(f"Center Coords: {coords}")