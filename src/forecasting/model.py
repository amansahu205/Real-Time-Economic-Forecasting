"""
Economic Forecasting Model

Time series forecasting for economic indicators using satellite and AIS features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import pickle
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import ML libraries
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Install with: pip install scikit-learn")


class EconomicForecaster:
    """Economic forecasting using ML models."""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the forecaster.
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'linear', 'ridge')
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required. Install with: pip install scikit-learn")
        
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = None
        self.metrics = {}
        
    def _create_model(self):
        """Create the ML model based on type."""
        if self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'linear':
            return LinearRegression()
        elif self.model_type == 'ridge':
            return Ridge(alpha=1.0)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def prepare_data(self, df: pd.DataFrame, 
                    target_col: str,
                    feature_cols: Optional[List[str]] = None,
                    test_size: float = 0.2) -> Tuple:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with features
            target_col: Target column name
            feature_cols: Feature column names (auto-detect if None)
            test_size: Fraction for test set
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        self.target_column = target_col
        
        # Auto-detect feature columns
        if feature_cols is None:
            exclude_cols = ['date', 'location', target_col, 'year', 'month']
            feature_cols = [c for c in df.columns 
                          if c not in exclude_cols 
                          and df[c].dtype in ['int64', 'float64']]
        
        self.feature_columns = feature_cols
        
        # Prepare X and y
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Time series split (maintain temporal order)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
        """
        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        logger.info(f"Model trained: {self.model_type}")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.model.predict(X_test)
        
        self.metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / np.where(y_test == 0, 1, y_test))) * 100
        }
        
        logger.info(f"Model Performance:")
        logger.info(f"  MAE: {self.metrics['mae']:.4f}")
        logger.info(f"  RMSE: {self.metrics['rmse']:.4f}")
        logger.info(f"  R²: {self.metrics['r2']:.4f}")
        logger.info(f"  MAPE: {self.metrics['mape']:.2f}%")
        
        return self.metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features (unscaled)
            
        Returns:
            Predictions
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance (for tree-based models).
        
        Returns:
            DataFrame with feature importance
        """
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model doesn't support feature importance")
            return pd.DataFrame()
        
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance
    
    def save_model(self, path: Path) -> None:
        """Save model to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'model_type': self.model_type,
            'metrics': self.metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Path) -> None:
        """Load model from file."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.target_column = model_data['target_column']
        self.model_type = model_data['model_type']
        self.metrics = model_data['metrics']
        
        logger.info(f"Model loaded from {path}")


def train_forecasting_model(features_path: Path, 
                           target_col: str = 'port_activity_index',
                           model_type: str = 'random_forest') -> EconomicForecaster:
    """
    Train a forecasting model on the unified features.
    
    Args:
        features_path: Path to unified features CSV
        target_col: Target column to predict
        model_type: Type of model to use
        
    Returns:
        Trained EconomicForecaster
    """
    # Load data
    df = pd.read_csv(features_path, parse_dates=['date'] if 'date' in pd.read_csv(features_path, nrows=1).columns else None)
    
    logger.info(f"Loaded {len(df)} records from {features_path}")
    
    # Initialize forecaster
    forecaster = EconomicForecaster(model_type=model_type)
    
    # Prepare data
    X_train, X_test, y_train, y_test = forecaster.prepare_data(
        df, target_col=target_col
    )
    
    # Train
    forecaster.train(X_train, y_train)
    
    # Evaluate
    metrics = forecaster.evaluate(X_test, y_test)
    
    # Feature importance
    importance = forecaster.get_feature_importance()
    if not importance.empty:
        logger.info("\nTop 10 Important Features:")
        print(importance.head(10).to_string())
    
    return forecaster


def main():
    """Example usage."""
    from src.config import FEATURES_DIR, MODELS_DIR
    
    features_path = FEATURES_DIR / "unified_forecasting_dataset.csv"
    
    if not features_path.exists():
        logger.error(f"Features not found: {features_path}")
        logger.info("Run feature extraction first: python -m src.features.feature_fusion")
        return
    
    # Train model
    forecaster = train_forecasting_model(
        features_path,
        target_col='port_activity_index',
        model_type='random_forest'
    )
    
    # Save model
    model_path = MODELS_DIR / "forecasting" / "economic_forecaster.pkl"
    forecaster.save_model(model_path)
    
    print(f"\n✅ Model trained and saved to {model_path}")
    print(f"📊 Performance: R² = {forecaster.metrics['r2']:.4f}, MAPE = {forecaster.metrics['mape']:.2f}%")


if __name__ == "__main__":
    main()
