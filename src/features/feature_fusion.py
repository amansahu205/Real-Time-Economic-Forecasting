"""
Feature Fusion Module

Merges satellite, AIS, and other data sources into unified features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureFusion:
    """Fuse multiple data sources into unified features."""
    
    def __init__(self, features_dir: Path):
        """
        Initialize the feature fusion module.
        
        Args:
            features_dir: Path to features directory
        """
        self.features_dir = Path(features_dir)
        
    def load_satellite_features(self) -> pd.DataFrame:
        """Load satellite-derived features."""
        path = self.features_dir / "satellite_port_features.csv"
        if path.exists():
            return pd.read_csv(path)
        logger.warning(f"Satellite features not found: {path}")
        return pd.DataFrame()
    
    def load_ais_yearly_features(self) -> pd.DataFrame:
        """Load AIS yearly aggregated features."""
        path = self.features_dir / "ais_yearly_features.csv"
        if path.exists():
            return pd.read_csv(path)
        logger.warning(f"AIS yearly features not found: {path}")
        return pd.DataFrame()
    
    def load_ais_daily_features(self) -> pd.DataFrame:
        """Load AIS daily features."""
        path = self.features_dir / "ais_daily_features.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=['date'])
            return df
        logger.warning(f"AIS daily features not found: {path}")
        return pd.DataFrame()
    
    def fuse_yearly_features(self) -> pd.DataFrame:
        """
        Fuse satellite and AIS features at yearly level.
        
        Returns:
            DataFrame with unified yearly features
        """
        satellite = self.load_satellite_features()
        ais = self.load_ais_yearly_features()
        
        if satellite.empty and ais.empty:
            logger.error("No features to fuse")
            return pd.DataFrame()
        
        if satellite.empty:
            logger.warning("Only AIS features available")
            return ais
        
        if ais.empty:
            logger.warning("Only satellite features available")
            return satellite
        
        # Merge on location and year
        merged = pd.merge(
            satellite,
            ais,
            on=['location', 'year'],
            how='outer',
            suffixes=('_sat', '_ais')
        )
        
        # Create unified features
        merged['unified_ship_count'] = merged['total_ship'].fillna(
            merged['avg_daily_ships']
        )
        
        merged['unified_activity_index'] = (
            merged['port_activity_index_sat'].fillna(0) * 0.5 +
            merged['avg_activity_index'].fillna(0) * 0.5
        )
        
        # Validation: compare satellite vs AIS
        if 'total_ship' in merged.columns and 'avg_daily_ships' in merged.columns:
            merged['sat_ais_correlation'] = (
                merged['total_ship'] / merged['avg_daily_ships'].replace(0, np.nan)
            ).fillna(1)
        
        logger.info(f"Fused {len(merged)} yearly records")
        return merged
    
    def fuse_daily_features(self, satellite_dates: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Fuse features at daily level.
        
        For days with satellite images, include satellite features.
        For all days, include AIS features.
        
        Args:
            satellite_dates: Optional DataFrame with satellite observation dates
            
        Returns:
            DataFrame with unified daily features
        """
        ais = self.load_ais_daily_features()
        
        if ais.empty:
            logger.warning("No AIS daily features available")
            return pd.DataFrame()
        
        unified = ais.copy()
        
        # Add satellite observation flag
        if satellite_dates is not None and not satellite_dates.empty:
            satellite_dates['has_satellite'] = True
            unified = pd.merge(
                unified,
                satellite_dates[['date', 'has_satellite']],
                on='date',
                how='left'
            )
            unified['has_satellite'] = unified['has_satellite'].fillna(False)
        else:
            unified['has_satellite'] = False
        
        # Create data quality score
        unified['data_quality'] = np.where(
            unified['has_satellite'], 1.0, 0.7
        )
        
        logger.info(f"Created {len(unified)} daily unified records")
        return unified
    
    def create_forecasting_dataset(self, 
                                   target_col: str = 'port_activity_index',
                                   lag_periods: List[int] = [1, 7, 14, 30]) -> pd.DataFrame:
        """
        Create dataset ready for forecasting.
        
        Args:
            target_col: Column to forecast
            lag_periods: Lag periods for features
            
        Returns:
            DataFrame ready for model training
        """
        daily = self.load_ais_daily_features()
        
        if daily.empty:
            logger.error("No daily features for forecasting")
            return pd.DataFrame()
        
        df = daily.copy()
        df = df.sort_values('date')
        
        # Create lag features
        for lag in lag_periods:
            df[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)
            df[f'ships_lag{lag}'] = df['unique_ships'].shift(lag)
        
        # Create rolling features
        for window in [7, 14, 30]:
            df[f'{target_col}_roll_mean{window}'] = df[target_col].rolling(window).mean()
            df[f'{target_col}_roll_std{window}'] = df[target_col].rolling(window).std()
        
        # Time features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Drop rows with NaN from lag creation
        df = df.dropna()
        
        logger.info(f"Created forecasting dataset with {len(df)} records")
        return df
    
    def save_unified_features(self, features: pd.DataFrame, name: str) -> Path:
        """Save unified features."""
        output_path = self.features_dir / f"unified_{name}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(output_path, index=False)
        logger.info(f"Saved unified features to {output_path}")
        return output_path


def main():
    """Example usage."""
    from src.config import FEATURES_DIR
    
    fusion = FeatureFusion(FEATURES_DIR)
    
    # Fuse yearly features
    yearly = fusion.fuse_yearly_features()
    if not yearly.empty:
        fusion.save_unified_features(yearly, "yearly_features")
        print(f"\n📊 Yearly Features: {len(yearly)} records")
    
    # Fuse daily features
    daily = fusion.fuse_daily_features()
    if not daily.empty:
        fusion.save_unified_features(daily, "daily_features")
        print(f"📊 Daily Features: {len(daily)} records")
    
    # Create forecasting dataset
    forecast_df = fusion.create_forecasting_dataset()
    if not forecast_df.empty:
        fusion.save_unified_features(forecast_df, "forecasting_dataset")
        print(f"📊 Forecasting Dataset: {len(forecast_df)} records")


if __name__ == "__main__":
    main()
