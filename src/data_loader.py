"""
Data loading and validation module for SCM optimization project
Handles CSV loading, data validation, and basic preprocessing
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

from .config import RAW_DATA_FILE, PROCESSED_DATA_FILE

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Handles data loading and validation for SCM dataset
    """

    def __init__(self, file_path: Optional[Path] = None):
        """
        Initialize DataLoader

        Args:
            file_path: Path to data file. If None, uses default from config
        """
        self.file_path = file_path or RAW_DATA_FILE
        self.data: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}

    def load_data(self, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file with error handling

        Args:
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If data validation fails
        """
        try:
            if not self.file_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.file_path}")

            # Default CSV reading parameters
            default_params = {
                'encoding': 'latin-1',  # DataCo dataset uses latin-1 encoding
                'low_memory': False,
                'parse_dates': self._get_date_columns(),
                'dtype': self._get_dtypes()
            }
            default_params.update(kwargs)

            self.data = pd.read_csv(self.file_path, **default_params)
            logger.info(f"Successfully loaded {len(self.data)} rows from {self.file_path}")

            # Validate data
            self._validate_data()

            # Store metadata
            self._store_metadata()

            return self.data

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def _get_date_columns(self) -> list:
        """Get list of date columns to parse"""
        return [
            'order date (DateOrders)',
            'shipping date (DateOrders)'
        ]

    def _get_dtypes(self) -> Dict[str, str]:
        """Get data types for specific columns"""
        return {
            'Order Id': 'str',
            'Order Customer Id': 'str',
            'Order Item Id': 'str',
            'Product Card Id': 'str',
            'Customer Id': 'str',
            'Customer Zipcode': 'str',
            'Order Zipcode': 'str',
            'Product Price': 'float64',
            'Order Item Quantity': 'int64',
            'Sales': 'float64',
            'Order Item Total': 'float64',
            'Order Profit Per Order': 'float64',
            'Days for shipping (real)': 'int64',
            'Days for shipment (scheduled)': 'int64',
            'Late_delivery_risk': 'int64'
        }

    def _validate_data(self) -> None:
        """
        Validate loaded data for required columns and data quality

        Raises:
            ValueError: If validation fails
        """
        if self.data is None:
            raise ValueError("No data loaded")

        # Key columns for SCM analysis
        required_columns = [
            'Order Id', 'Order Item Id', 'order date (DateOrders)',
            'shipping date (DateOrders)', 'Days for shipping (real)',
            'Days for shipment (scheduled)', 'Late_delivery_risk',
            'Customer City', 'Customer Country',
            'Product Price', 'Order Item Quantity'
        ]

        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            logger.warning(f"Missing columns (non-critical): {missing_columns}")

        # Check for empty data
        if len(self.data) == 0:
            raise ValueError("Dataset is empty")

        # Check for duplicate order IDs
        if self.data['Order Id'].duplicated().any():
            logger.warning("Found duplicate Order IDs")

        logger.info("Data validation passed")

    def _store_metadata(self) -> None:
        """Store dataset metadata"""
        if self.data is not None:
            date_range = None
            if 'order date (DateOrders)' in self.data.columns:
                date_range = {
                    'order_date_min': self.data['order date (DateOrders)'].min(),
                    'order_date_max': self.data['order date (DateOrders)'].max(),
                    'shipping_date_min': self.data['shipping date (DateOrders)'].min() if 'shipping date (DateOrders)' in self.data.columns else None,
                    'shipping_date_max': self.data['shipping date (DateOrders)'].max() if 'shipping date (DateOrders)' in self.data.columns else None
                }

            self.metadata = {
                'num_rows': len(self.data),
                'num_columns': len(self.data.columns),
                'columns': list(self.data.columns),
                'dtypes': {str(k): str(v) for k, v in self.data.dtypes.to_dict().items()},
                'memory_usage': self.data.memory_usage(deep=True).sum(),
                'date_range': date_range
            }

    def get_data_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the loaded data

        Returns:
            Dictionary containing data information
        """
        if self.data is None:
            return {"error": "No data loaded"}

        return {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'null_counts': self.data.isnull().sum().to_dict(),
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024 / 1024,
            'numeric_summary': self.data.describe().to_dict() if len(self.data.select_dtypes(include=[np.number]).columns) > 0 else None,
            'metadata': self.metadata
        }

    def save_processed_data(self, output_path: Optional[Path] = None) -> None:
        """
        Save processed data to CSV

        Args:
            output_path: Output file path. If None, uses default from config
        """
        output_path = output_path or PROCESSED_DATA_FILE

        if self.data is None:
            raise ValueError("No data to save")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")

    def get_sample_data(self, n: int = 5) -> pd.DataFrame:
        """
        Get sample rows from dataset

        Args:
            n: Number of sample rows

        Returns:
            Sample DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded")

        return self.data.head(n)