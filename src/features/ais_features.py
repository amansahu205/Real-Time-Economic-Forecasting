"""
AIS Feature Extraction Module

Extracts economic indicators from AIS maritime data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AISFeatureExtractor:
    """Extract economic features from AIS maritime data."""
    
    # AIS Vessel Type Codes
    VESSEL_TYPES = {
        'cargo': list(range(70, 80)),
        'tanker': list(range(80, 90)),
        'passenger': list(range(60, 70)),
        'fishing': [30],
        'tug': [31, 32, 52],
    }
    
    def __init__(self, ais_dir: Path):
        """
        Initialize the feature extractor.
        
        Args:
            ais_dir: Path to processed AIS data directory
        """
        self.ais_dir = Path(ais_dir)
        
    def load_daily_metrics(self, location: str = "Port_of_LA") -> pd.DataFrame:
        """
        Load daily AIS metrics for a location.
        
        Args:
            location: Location name
            
        Returns:
            DataFrame with daily metrics
        """
        metrics_path = self.ais_dir / f"{location}_ais_daily.csv"
        
        if not metrics_path.exists():
            logger.warning(f"AIS metrics not found: {metrics_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(metrics_path, parse_dates=['date'])
        df['location'] = location
        
        return df
    
    def extract_features(self, df: pd.DataFrame, 
                        windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
        """
        Extract economic features from daily AIS data.
        
        Args:
            df: DataFrame with daily AIS metrics
            windows: Moving average window sizes
            
        Returns:
            DataFrame with extracted features
        """
        if df.empty:
            return df
        
        features = df.copy()
        features = features.sort_values('date')
        
        # Ensure numeric columns
        numeric_cols = ['unique_ships', 'cargo_ships', 'tanker_ships', 
                       'passenger_ships', 'fishing_ships', 'tug_ships']
        for col in numeric_cols:
            if col in features.columns:
                features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)
        
        # Port Activity Index
        features['port_activity_index'] = (
            features.get('cargo_ships', 0) * 0.4 +
            features.get('tanker_ships', 0) * 0.3 +
            features.get('unique_ships', 0) * 0.2 +
            features.get('tug_ships', 0) * 0.1
        )
        
        # Moving Averages
        for window in windows:
            features[f'ships_ma{window}'] = features['unique_ships'].rolling(
                window, min_periods=1
            ).mean()
            features[f'cargo_ma{window}'] = features['cargo_ships'].rolling(
                window, min_periods=1
            ).mean()
            features[f'activity_ma{window}'] = features['port_activity_index'].rolling(
                window, min_periods=1
            ).mean()
        
        # Growth Rates
        features['ships_wow_growth'] = features['unique_ships'].pct_change(7) * 100
        features['ships_mom_growth'] = features['unique_ships'].pct_change(30) * 100
        features['cargo_wow_growth'] = features['cargo_ships'].pct_change(7) * 100
        
        # Volatility
        features['ships_volatility_7d'] = features['unique_ships'].rolling(7).std()
        features['ships_volatility_30d'] = features['unique_ships'].rolling(30).std()
        
        # Cargo Ratio
        features['cargo_ratio'] = (
            features['cargo_ships'] / features['unique_ships'].replace(0, np.nan)
        ).fillna(0)
        
        # Activity Level (vs 30-day average)
        features['activity_level'] = (
            features['unique_ships'] / features['ships_ma30'].replace(0, np.nan)
        ).fillna(1)
        
        # Trend indicator
        features['trend_7d'] = np.where(
            features['ships_ma7'] > features['ships_ma30'], 1, -1
        )
        
        # Year and month for aggregation
        features['year'] = features['date'].dt.year
        features['month'] = features['date'].dt.month
        features['day_of_week'] = features['date'].dt.dayofweek
        
        return features
    
    def aggregate_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate daily features to monthly.
        
        Args:
            df: DataFrame with daily features
            
        Returns:
            DataFrame with monthly aggregates
        """
        if df.empty:
            return df
        
        monthly = df.groupby(['location', 'year', 'month']).agg({
            'unique_ships': 'mean',
            'cargo_ships': 'mean',
            'tanker_ships': 'mean',
            'port_activity_index': 'mean',
            'cargo_ratio': 'mean',
            'ships_volatility_7d': 'mean',
            'activity_level': 'mean'
        }).reset_index()
        
        monthly.columns = [
            'location', 'year', 'month', 
            'avg_ships', 'avg_cargo_ships', 'avg_tanker_ships',
            'avg_activity_index', 'avg_cargo_ratio', 
            'avg_volatility', 'avg_activity_level'
        ]
        
        return monthly
    
    def aggregate_yearly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate daily features to yearly (for satellite comparison).
        
        Args:
            df: DataFrame with daily features
            
        Returns:
            DataFrame with yearly aggregates
        """
        if df.empty:
            return df
        
        yearly = df.groupby(['location', 'year']).agg({
            'unique_ships': ['mean', 'sum', 'std'],
            'cargo_ships': ['mean', 'sum'],
            'tanker_ships': ['mean', 'sum'],
            'port_activity_index': 'mean',
            'cargo_ratio': 'mean'
        }).reset_index()
        
        # Flatten column names
        yearly.columns = [
            'location', 'year',
            'avg_daily_ships', 'total_ship_days', 'ships_std',
            'avg_daily_cargo', 'total_cargo_days',
            'avg_daily_tanker', 'total_tanker_days',
            'avg_activity_index', 'avg_cargo_ratio'
        ]
        
        return yearly
    
    def save_features(self, features: pd.DataFrame, output_path: Path) -> None:
        """Save extracted features to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        features.to_csv(output_path, index=False)
        logger.info(f"Saved features to {output_path}")


def main():
    """Example usage."""
    from src.config import AIS_PROCESSED_DIR, FEATURES_DIR
    
    extractor = AISFeatureExtractor(AIS_PROCESSED_DIR)
    
    # Load and process Port of LA
    daily = extractor.load_daily_metrics("Port_of_LA")
    
    if not daily.empty:
        # Extract features
        features = extractor.extract_features(daily)
        extractor.save_features(features, FEATURES_DIR / "ais_daily_features.csv")
        
        # Monthly aggregates
        monthly = extractor.aggregate_monthly(features)
        extractor.save_features(monthly, FEATURES_DIR / "ais_monthly_features.csv")
        
        # Yearly aggregates (for satellite comparison)
        yearly = extractor.aggregate_yearly(features)
        extractor.save_features(yearly, FEATURES_DIR / "ais_yearly_features.csv")
        
        print(f"\n📊 AIS Features Summary:")
        print(f"Daily records: {len(features)}")
        print(f"Monthly records: {len(monthly)}")
        print(f"Yearly records: {len(yearly)}")


if __name__ == "__main__":
    main()
