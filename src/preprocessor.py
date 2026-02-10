"""
Data preprocessing module for SCM optimization project
Handles data cleaning, feature engineering, and preparation for analysis
Adapted for DataCo Supply Chain Dataset
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path

from .config import PROCESSED_DATA_FILE

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Handles data cleaning and preprocessing for SCM analysis
    Specifically designed for DataCo Supply Chain Dataset
    """

    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Initialize preprocessor

        Args:
            data: Input DataFrame to preprocess
        """
        self.original_data = data.copy() if data is not None else None
        self.processed_data: Optional[pd.DataFrame] = None
        self.preprocessing_steps: List[str] = []

    def preprocess_data(self, data: Optional[pd.DataFrame] = None,
                       steps: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Main preprocessing pipeline

        Args:
            data: DataFrame to preprocess (if None, uses self.original_data)
            steps: List of preprocessing steps to apply

        Returns:
            Preprocessed DataFrame
        """
        if data is not None:
            self.original_data = data.copy()

        if self.original_data is None:
            raise ValueError("No data provided for preprocessing")

        self.processed_data = self.original_data.copy()
        self.preprocessing_steps = []

        # Default preprocessing steps for DataCo dataset
        default_steps = [
            'handle_missing_values',
            'clean_data_types',
            'create_datetime_features',
            'create_delivery_features',
            'create_geographical_features',
            'create_cost_features',
            'create_risk_features',
            'encode_categorical'
        ]

        steps = steps or default_steps

        for step in steps:
            if hasattr(self, f'_{step}'):
                logger.info(f"Applying preprocessing step: {step}")
                try:
                    getattr(self, f'_{step}')()
                    self.preprocessing_steps.append(step)
                    print(f"    [OK] {step} completed")
                except Exception as e:
                    print(f"    [WARNING] {step} failed: {e}")
                    logger.warning(f"Step {step} failed: {e}")
            else:
                logger.warning(f"Unknown preprocessing step: {step}")

        logger.info(f"Preprocessing completed with {len(self.preprocessing_steps)} steps")
        return self.processed_data

    def _handle_missing_values(self) -> None:
        """
        Handle missing values dengan strategi:
        1. Drop kolom jika > 50% missing (tidak berguna)
        2. Drop rows jika kolom PENTING (tanggal, delivery) kosong
        3. Isi dengan median/mean untuk numerik, mode untuk kategorikal
        """
        original_rows = len(self.processed_data)
        
        # === STEP 1: Drop kolom dengan > 50% missing ===
        cols_to_drop = []
        for col in self.processed_data.columns:
            missing_pct = self.processed_data[col].isnull().mean() * 100
            if missing_pct > 50:
                cols_to_drop.append(col)
        
        if cols_to_drop:
            self.processed_data.drop(cols_to_drop, axis=1, inplace=True)
            print(f"       Dropped columns (>50% null): {cols_to_drop}")
        
        # === STEP 2: Drop rows jika kolom PENTING kosong ===
        critical_columns = [
            'order date (DateOrders)',
            'shipping date (DateOrders)',
            'Days for shipping (real)',
            'Days for shipment (scheduled)',
            'Late_delivery_risk',
            'Customer City',
            'Product Price'
        ]
        
        existing_critical = [col for col in critical_columns if col in self.processed_data.columns]
        rows_before = len(self.processed_data)
        self.processed_data.dropna(subset=existing_critical, inplace=True)
        rows_dropped = rows_before - len(self.processed_data)
        
        if rows_dropped > 0:
            print(f"       Dropped {rows_dropped:,} rows (missing critical data)")
        
        # === STEP 3: Fill remaining missing values ===
        # Numeric: fill with median
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.processed_data[col].isnull().any():
                median_val = self.processed_data[col].median()
                null_count = self.processed_data[col].isnull().sum()
                self.processed_data[col].fillna(median_val, inplace=True)
                print(f"       Filled {col}: {null_count} nulls -> median ({median_val:.2f})")
        
        # Categorical: fill with mode or 'Unknown'
        categorical_cols = self.processed_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.processed_data[col].isnull().any():
                mode_val = self.processed_data[col].mode()
                fill_val = mode_val.iloc[0] if not mode_val.empty else 'Unknown'
                null_count = self.processed_data[col].isnull().sum()
                self.processed_data[col] = self.processed_data[col].fillna(fill_val)
                print(f"       Filled {col}: {null_count} nulls -> '{fill_val}'")

    def _clean_data_types(self) -> None:
        """Ensure correct data types"""
        # Convert zipcodes to string
        zipcode_cols = ['Customer Zipcode', 'Order Zipcode']
        for col in zipcode_cols:
            if col in self.processed_data.columns:
                self.processed_data[col] = self.processed_data[col].astype(str)

        # Convert IDs to string
        id_cols = ['Order Id', 'Customer Id', 'Order Customer Id', 'Product Card Id', 
                   'Order Item Id', 'Category Id', 'Department Id']
        for col in id_cols:
            if col in self.processed_data.columns:
                self.processed_data[col] = self.processed_data[col].astype(str)

        # Ensure numeric columns are numeric
        numeric_cols = ['Product Price', 'Order Item Quantity', 'Sales', 
                       'Order Item Total', 'Order Profit Per Order', 
                       'Benefit per order', 'Sales per customer',
                       'Days for shipping (real)', 'Days for shipment (scheduled)']
        for col in numeric_cols:
            if col in self.processed_data.columns:
                self.processed_data[col] = pd.to_numeric(self.processed_data[col], errors='coerce')

    def _create_datetime_features(self) -> None:
        """Create datetime-based features"""
        # Parse order date
        if 'order date (DateOrders)' in self.processed_data.columns:
            order_date = pd.to_datetime(self.processed_data['order date (DateOrders)'], errors='coerce')
            self.processed_data['order_year'] = order_date.dt.year
            self.processed_data['order_month'] = order_date.dt.month
            self.processed_data['order_day'] = order_date.dt.day
            self.processed_data['order_dayofweek'] = order_date.dt.dayofweek
            self.processed_data['order_hour'] = order_date.dt.hour
            self.processed_data['order_quarter'] = order_date.dt.quarter
            self.processed_data['is_weekend'] = order_date.dt.dayofweek.isin([5, 6]).astype(int)

        # Parse shipping date
        if 'shipping date (DateOrders)' in self.processed_data.columns:
            shipping_date = pd.to_datetime(self.processed_data['shipping date (DateOrders)'], errors='coerce')
            self.processed_data['shipping_year'] = shipping_date.dt.year
            self.processed_data['shipping_month'] = shipping_date.dt.month
            self.processed_data['shipping_day'] = shipping_date.dt.day
            self.processed_data['shipping_dayofweek'] = shipping_date.dt.dayofweek

    def _create_delivery_features(self) -> None:
        """Create delivery-related features"""
        # Calculate delivery delay (real - scheduled)
        if all(col in self.processed_data.columns for col in 
               ['Days for shipping (real)', 'Days for shipment (scheduled)']):
            self.processed_data['delivery_delay'] = (
                self.processed_data['Days for shipping (real)'] - 
                self.processed_data['Days for shipment (scheduled)']
            )
            
            # Positive delay means late, negative means early
            self.processed_data['is_delayed'] = (self.processed_data['delivery_delay'] > 0).astype(int)
            
            # Delay category
            self.processed_data['delay_category'] = pd.cut(
                self.processed_data['delivery_delay'],
                bins=[-float('inf'), -2, 0, 2, float('inf')],
                labels=['Very Early', 'On Time', 'Slightly Late', 'Very Late']
            )

        # Delivery efficiency (scheduled / real)
        if all(col in self.processed_data.columns for col in 
               ['Days for shipping (real)', 'Days for shipment (scheduled)']):
            # Avoid division by zero
            real_days = self.processed_data['Days for shipping (real)'].replace(0, 1)
            self.processed_data['delivery_efficiency'] = (
                self.processed_data['Days for shipment (scheduled)'] / real_days
            )

    def _create_geographical_features(self) -> None:
        """Create geographical features"""
        # Create location strings
        if all(col in self.processed_data.columns for col in 
               ['Customer City', 'Customer State', 'Customer Country']):
            self.processed_data['customer_location'] = (
                self.processed_data['Customer City'].astype(str) + ', ' +
                self.processed_data['Customer State'].astype(str) + ', ' +
                self.processed_data['Customer Country'].astype(str)
            )

        if all(col in self.processed_data.columns for col in 
               ['Order City', 'Order State', 'Order Country']):
            self.processed_data['order_location'] = (
                self.processed_data['Order City'].astype(str) + ', ' +
                self.processed_data['Order State'].astype(str) + ', ' +
                self.processed_data['Order Country'].astype(str)
            )

        # Same city/country flag
        if 'Customer City' in self.processed_data.columns and 'Order City' in self.processed_data.columns:
            self.processed_data['same_city'] = (
                self.processed_data['Customer City'] == self.processed_data['Order City']
            ).astype(int)

        if 'Customer Country' in self.processed_data.columns and 'Order Country' in self.processed_data.columns:
            self.processed_data['same_country'] = (
                self.processed_data['Customer Country'] == self.processed_data['Order Country']
            ).astype(int)

    def _create_cost_features(self) -> None:
        """Create cost-related features"""
        # Profit margin percentage
        if 'Order Profit Per Order' in self.processed_data.columns and 'Sales' in self.processed_data.columns:
            sales = self.processed_data['Sales'].replace(0, 1)
            self.processed_data['profit_margin_pct'] = (
                self.processed_data['Order Profit Per Order'] / sales * 100
            )

        # Revenue per item
        if 'Order Item Total' in self.processed_data.columns and 'Order Item Quantity' in self.processed_data.columns:
            qty = self.processed_data['Order Item Quantity'].replace(0, 1)
            self.processed_data['revenue_per_item'] = (
                self.processed_data['Order Item Total'] / qty
            )

        # Is profitable order
        if 'Order Profit Per Order' in self.processed_data.columns:
            self.processed_data['is_profitable'] = (
                self.processed_data['Order Profit Per Order'] > 0
            ).astype(int)

        # Discount impact
        if 'Order Item Discount' in self.processed_data.columns and 'Order Item Total' in self.processed_data.columns:
            total = self.processed_data['Order Item Total'].replace(0, 1)
            self.processed_data['discount_impact_pct'] = (
                self.processed_data['Order Item Discount'] / total * 100
            )

    def _create_risk_features(self) -> None:
        """Create risk assessment features"""
        # High value order flag
        if 'Order Item Total' in self.processed_data.columns:
            threshold = self.processed_data['Order Item Total'].quantile(0.75)
            self.processed_data['is_high_value_order'] = (
                self.processed_data['Order Item Total'] > threshold
            ).astype(int)

        # Rush order flag (short scheduled shipping time)
        if 'Days for shipment (scheduled)' in self.processed_data.columns:
            self.processed_data['is_rush_order'] = (
                self.processed_data['Days for shipment (scheduled)'] <= 2
            ).astype(int)

        # Combined risk score
        risk_factors = []
        if 'is_delayed' in self.processed_data.columns:
            risk_factors.append('is_delayed')
        if 'is_high_value_order' in self.processed_data.columns:
            risk_factors.append('is_high_value_order')
        if 'is_rush_order' in self.processed_data.columns:
            risk_factors.append('is_rush_order')

        if risk_factors:
            self.processed_data['risk_score'] = self.processed_data[risk_factors].sum(axis=1)

    def _encode_categorical(self) -> None:
        """Encode categorical variables for ML"""
        categorical_cols = ['Type', 'Delivery Status', 'Category Name',
                          'Customer Segment', 'Department Name', 'Shipping Mode',
                          'Order Status', 'Market', 'Order Region']

        for col in categorical_cols:
            if col in self.processed_data.columns:
                # Create dummy variables
                dummies = pd.get_dummies(self.processed_data[col], prefix=col.replace(' ', '_'), drop_first=True)
                self.processed_data = pd.concat([self.processed_data, dummies], axis=1)

    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing steps applied"""
        if self.processed_data is None:
            return {"error": "No preprocessing applied"}

        original_shape = self.original_data.shape if self.original_data is not None else None
        processed_shape = self.processed_data.shape

        return {
            'steps_applied': self.preprocessing_steps,
            'original_shape': original_shape,
            'processed_shape': processed_shape,
            'new_columns_added': processed_shape[1] - (original_shape[1] if original_shape else 0),
            'columns_with_nulls': self.processed_data.isnull().sum()[self.processed_data.isnull().sum() > 0].to_dict(),
            'numeric_columns': len(self.processed_data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(self.processed_data.select_dtypes(include=['object']).columns)
        }

    def get_feature_columns(self) -> Dict[str, List[str]]:
        """Get categorized list of feature columns for ML"""
        if self.processed_data is None:
            return {"error": "No data processed"}

        return {
            'numeric_features': list(self.processed_data.select_dtypes(include=[np.number]).columns),
            'categorical_features': list(self.processed_data.select_dtypes(include=['object']).columns),
            'datetime_features': [col for col in self.processed_data.columns 
                                 if any(x in col for x in ['year', 'month', 'day', 'hour', 'dayofweek', 'quarter'])],
            'target_columns': ['Late_delivery_risk', 'is_delayed', 'delivery_delay', 'Delivery Status'],
            'id_columns': ['Order Id', 'Customer Id', 'Order Customer Id', 'Product Card Id']
        }

    def save_processed_data(self, output_path: Optional[Path] = None) -> None:
        """Save processed data"""
        output_path = output_path or PROCESSED_DATA_FILE

        if self.processed_data is None:
            raise ValueError("No processed data to save")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.processed_data.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
        print(f"[OK] Data saved to {output_path}")