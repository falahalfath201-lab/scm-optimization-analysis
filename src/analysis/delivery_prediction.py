"""
Delivery Time Analysis and Prediction Module
Uses Prophet for time series forecasting and ML models for prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path

# ML and forecasting libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet not available. Time series forecasting disabled.")

from ..config import (
    FORECAST_HORIZON, TEST_SIZE, CV_FOLDS,
    DELIVERY_RESULTS_FILE, RANDOM_SEED
)

logger = logging.getLogger(__name__)

class DeliveryPredictor:
    """
    Predicts delivery times and analyzes delivery patterns
    """

    def __init__(self):
        """Initialize delivery predictor"""
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = [
            'Order Item Quantity', 'Product Price', 'Sales',
            'order_month', 'order_dayofweek', 'order_hour',
            'shipping_month', 'shipping_dayofweek', 'Days for shipment (scheduled)'
        ]
        self.target_column = 'Days for shipping (real)'

    def analyze_delivery_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze delivery patterns and performance

        Args:
            data: Preprocessed delivery data

        Returns:
            Dictionary with delivery analysis results
        """
        analysis = {}

        # Basic statistics
        if self.target_column in data.columns:
            analysis['delivery_stats'] = {
                'mean_delivery_time': data[self.target_column].mean(),
                'median_delivery_time': data[self.target_column].median(),
                'std_delivery_time': data[self.target_column].std(),
                'min_delivery_time': data[self.target_column].min(),
                'max_delivery_time': data[self.target_column].max()
            }

            # Late delivery analysis
            if 'is_late_delivery' in data.columns:
                late_delivery_rate = data['is_late_delivery'].mean() * 100
                analysis['late_delivery_rate'] = late_delivery_rate

        # Delivery by category
        category_cols = ['Category Name', 'Department Name', 'Customer Segment']
        analysis['delivery_by_category'] = {}

        for col in category_cols:
            if col in data.columns and self.target_column in data.columns:
                category_stats = data.groupby(col)[self.target_column].agg([
                    'mean', 'median', 'count', 'std'
                ]).round(2)
                analysis['delivery_by_category'][col] = category_stats.to_dict()

        # Time-based analysis
        if 'order_month' in data.columns and self.target_column in data.columns:
            monthly_stats = data.groupby('order_month')[self.target_column].agg([
                'mean', 'count'
            ]).round(2)
            analysis['monthly_delivery_stats'] = monthly_stats.to_dict()

        return analysis

    def train_prediction_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train multiple ML models for delivery time prediction

        Args:
            data: Training data with features and target

        Returns:
            Dictionary with trained models and performance metrics
        """
        # Prepare data
        X, y = self._prepare_features(data)

        if X is None or y is None:
            return {"error": "Insufficient data for training"}

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Define models
        models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(
                n_estimators=100, random_state=RANDOM_SEED
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100, random_state=RANDOM_SEED
            )
        }

        results = {}

        for name, model in models.items():
            try:
                # Train model
                model.fit(X_train_scaled, y_train)

                # Predictions
                y_pred = model.predict(X_test_scaled)

                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)

                # Cross-validation score
                cv_scores = cross_val_score(
                    model, X_train_scaled, y_train,
                    cv=CV_FOLDS, scoring='neg_mean_absolute_error'
                )
                cv_mae = -cv_scores.mean()

                results[name] = {
                    'model': model,
                    'metrics': {
                        'mae': mae,
                        'mse': mse,
                        'rmse': rmse,
                        'r2_score': r2,
                        'cv_mae': cv_mae
                    },
                    'feature_importance': self._get_feature_importance(model, X.columns)
                }

                logger.info(f"Trained {name} model - MAE: {mae:.2f}, R²: {r2:.2f}")

            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                results[name] = {"error": str(e)}

        self.models = {name: result['model'] for name, result in results.items()
                      if 'model' in result}

        return results

    def forecast_delivery_times(self, historical_data: pd.DataFrame,
                               periods: int = FORECAST_HORIZON) -> Dict[str, Any]:
        """
        Forecast future delivery times using time series analysis

        Args:
            historical_data: Historical delivery data
            periods: Number of periods to forecast

        Returns:
            Dictionary with forecast results
        """
        if not PROPHET_AVAILABLE:
            return {"error": "Prophet not available for forecasting"}

        try:
            # Prepare time series data
            ts_data = self._prepare_time_series_data(historical_data)

            if ts_data.empty:
                return {"error": "Insufficient time series data"}

            # Initialize Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )

            # Fit model
            model.fit(ts_data)

            # Create future dataframe
            future = model.make_future_dataframe(periods=periods)

            # Generate forecast
            forecast = model.predict(future)

            # Extract relevant columns
            forecast_results = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)

            return {
                'forecast': forecast_results.to_dict('records'),
                'model': model,
                'forecast_periods': periods,
                'method': 'Prophet Time Series'
            }

        except Exception as e:
            logger.error(f"Error in time series forecasting: {e}")
            return {"error": str(e)}

    def _prepare_features(self, data: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Prepare features and target for ML training

        Args:
            data: Input data

        Returns:
            Tuple of (X, y) or (None, None) if insufficient data
        """
        # Check if required columns exist
        missing_features = [col for col in self.feature_columns if col not in data.columns]
        if missing_features:
            logger.warning(f"Missing feature columns: {missing_features}")

        available_features = [col for col in self.feature_columns if col in data.columns]

        if not available_features or self.target_column not in data.columns:
            return None, None

        # Select features and target
        X = data[available_features].copy()
        y = data[self.target_column].copy()

        # Handle missing values
        X = X.fillna(X.median())
        y = y.fillna(y.median())

        # Remove outliers (optional)
        if len(y) > 10:
            q1, q3 = y.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            mask = (y >= lower_bound) & (y <= upper_bound)
            X = X[mask]
            y = y[mask]

        return X, y

    def _prepare_time_series_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for time series forecasting

        Args:
            data: Historical data

        Returns:
            Time series DataFrame for Prophet
        """
        if 'order date (DateOrders)' not in data.columns or self.target_column not in data.columns:
            return pd.DataFrame()

        # Aggregate by date
        ts_data = data.groupby(data['order date (DateOrders)'].dt.date)[self.target_column].agg([
            'mean', 'count'
        ]).reset_index()

        ts_data.columns = ['ds', 'y', 'count']

        # Filter dates with sufficient data
        ts_data = ts_data[ts_data['count'] >= 5].copy()

        # Convert date to datetime
        ts_data['ds'] = pd.to_datetime(ts_data['ds'])

        return ts_data[['ds', 'y']]

    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """
        Extract feature importance from trained model

        Args:
            model: Trained ML model
            feature_names: List of feature names

        Returns:
            Dictionary of feature importance scores
        """
        try:
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(feature_names, model.feature_importances_))
                return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            elif hasattr(model, 'coef_'):
                # For linear models
                importance_dict = dict(zip(feature_names, np.abs(model.coef_)))
                return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")

        return None

    def predict_delivery_time(self, new_data: pd.DataFrame,
                            model_name: str = 'random_forest') -> np.ndarray:
        """
        Predict delivery times for new data

        Args:
            new_data: New data for prediction
            model_name: Name of trained model to use

        Returns:
            Array of predicted delivery times
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")

        X, _ = self._prepare_features(new_data)
        if X is None:
            raise ValueError("Cannot prepare features for prediction")

        X_scaled = self.scaler.transform(X)
        predictions = self.models[model_name].predict(X_scaled)

        return predictions

    def save_results(self, results: Dict[str, Any],
                    output_path: Optional[Path] = None) -> None:
        """
        Save prediction results

        Args:
            results: Prediction results
            output_path: Output file path
        """
        output_path = output_path or DELIVERY_RESULTS_FILE
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert results to DataFrame
        if 'forecast' in results and results['forecast']:
            forecast_df = pd.DataFrame(results['forecast'])
            forecast_df.to_csv(output_path, index=False)
            logger.info(f"Delivery prediction results saved to {output_path}")
        elif 'metrics' in results:
            # Save model performance metrics
            metrics_data = []
            for model_name, model_results in results.items():
                if 'metrics' in model_results:
                    metrics = model_results['metrics']
                    metrics_data.append({
                        'model': model_name,
                        **metrics
                    })

            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                metrics_df.to_csv(output_path, index=False)
                logger.info(f"Model performance metrics saved to {output_path}")

    def get_model_comparison(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Compare performance of different models

        Args:
            results: Training results from train_prediction_models

        Returns:
            DataFrame with model comparison
        """
        comparison_data = []

        for model_name, model_results in results.items():
            if 'metrics' in model_results:
                metrics = model_results['metrics']
                comparison_data.append({
                    'model': model_name,
                    'mae': metrics.get('mae', 0),
                    'rmse': metrics.get('rmse', 0),
                    'r2_score': metrics.get('r2_score', 0),
                    'cv_mae': metrics.get('cv_mae', 0)
                })

        return pd.DataFrame(comparison_data).round(4)