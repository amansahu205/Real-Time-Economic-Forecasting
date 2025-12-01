"""
Satellite Feature Extraction Module

Extracts economic indicators from satellite imagery detection results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SatelliteFeatureExtractor:
    """Extract economic features from satellite detection results."""
    
    def __init__(self, annotations_dir: Path):
        """
        Initialize the feature extractor.
        
        Args:
            annotations_dir: Path to annotations directory
        """
        self.annotations_dir = Path(annotations_dir)
        
    def load_location_data(self, location: str, dataset: str = "google_earth_tiled") -> pd.DataFrame:
        """
        Load detection summary for a location.
        
        Args:
            location: Location name (e.g., 'Port_of_LA')
            dataset: Dataset name (e.g., 'google_earth_tiled')
            
        Returns:
            DataFrame with detection data
        """
        summary_path = self.annotations_dir / dataset / location / "all_years_summary.csv"
        
        if not summary_path.exists():
            logger.warning(f"Summary not found: {summary_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(summary_path)
        df['location'] = location
        
        return df
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract economic features from detection data.
        
        Args:
            df: DataFrame with detection counts
            
        Returns:
            DataFrame with extracted features
        """
        if df.empty:
            return df
        
        features = df.copy()
        
        # Ensure numeric columns
        numeric_cols = ['total_detections', 'total_ship', 'total_storage-tank', 
                       'total_harbor', 'total_large-vehicle', 'total_small-vehicle']
        for col in numeric_cols:
            if col in features.columns:
                features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)
        
        # Calculate derived features
        if 'total_ship' in features.columns:
            features['ship_count'] = features['total_ship']
        
        if 'total_storage-tank' in features.columns:
            features['storage_tank_count'] = features['total_storage-tank']
        
        if 'total_harbor' in features.columns:
            features['harbor_count'] = features['total_harbor']
        
        # Port activity index (weighted sum)
        features['port_activity_index'] = (
            features.get('total_ship', 0) * 0.4 +
            features.get('total_storage-tank', 0) * 0.3 +
            features.get('total_harbor', 0) * 0.2 +
            features.get('total_large-vehicle', 0) * 0.1
        )
        
        # Year-over-year growth (if multiple years)
        if len(features) > 1:
            features = features.sort_values('year')
            features['yoy_ship_growth'] = features['total_ship'].pct_change() * 100
            features['yoy_activity_growth'] = features['port_activity_index'].pct_change() * 100
        
        # Capacity utilization proxy (ships / harbor)
        if 'total_harbor' in features.columns and 'total_ship' in features.columns:
            features['capacity_utilization'] = (
                features['total_ship'] / features['total_harbor'].replace(0, np.nan)
            ).fillna(0)
        
        return features
    
    def process_all_locations(self, locations: List[str], 
                             dataset: str = "google_earth_tiled") -> pd.DataFrame:
        """
        Process all locations and combine features.
        
        Args:
            locations: List of location names
            dataset: Dataset name
            
        Returns:
            Combined DataFrame with all features
        """
        all_features = []
        
        for location in locations:
            logger.info(f"Processing {location}...")
            df = self.load_location_data(location, dataset)
            
            if not df.empty:
                features = self.extract_features(df)
                all_features.append(features)
        
        if not all_features:
            return pd.DataFrame()
        
        combined = pd.concat(all_features, ignore_index=True)
        logger.info(f"Processed {len(locations)} locations, {len(combined)} records")
        
        return combined
    
    def save_features(self, features: pd.DataFrame, output_path: Path) -> None:
        """Save extracted features to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        features.to_csv(output_path, index=False)
        logger.info(f"Saved features to {output_path}")


def main():
    """Example usage."""
    from src.config import ANNOTATIONS_DIR, FEATURES_DIR, LOCATIONS
    
    extractor = SatelliteFeatureExtractor(ANNOTATIONS_DIR)
    
    # Process port locations
    port_features = extractor.process_all_locations(
        LOCATIONS['ports'], 
        dataset="google_earth_tiled"
    )
    
    if not port_features.empty:
        extractor.save_features(
            port_features, 
            FEATURES_DIR / "satellite_port_features.csv"
        )
        print(f"\n📊 Port Features Summary:")
        print(port_features[['location', 'year', 'total_ship', 'port_activity_index']].to_string())


if __name__ == "__main__":
    main()
