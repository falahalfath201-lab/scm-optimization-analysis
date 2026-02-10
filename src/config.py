"""
Configuration file for SCM Optimization Project
Contains constants, file paths, and model parameters
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DATA_DIR = DATA_DIR / "results"

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Source directories
SRC_DIR = PROJECT_ROOT / "src"

# File paths
RAW_DATA_FILE = RAW_DATA_DIR / "DataCoSupplyChainDataset.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "cleaned_data.csv"
ROUTE_RESULTS_FILE = RESULTS_DATA_DIR / "route_optimization_results.csv"
DELIVERY_RESULTS_FILE = RESULTS_DATA_DIR / "delivery_prediction_results.csv"
WAREHOUSE_RESULTS_FILE = RESULTS_DATA_DIR / "warehouse_analysis_results.csv"

# Model parameters
RANDOM_SEED = 42

# Route optimization parameters
MAX_VEHICLES = 25
VEHICLE_CAPACITY = 1000
MAX_ROUTE_TIME = 480  # minutes

# ML model parameters
TEST_SIZE = 0.2
CV_FOLDS = 5

# Time series parameters
FORECAST_HORIZON = 30  # days
SEASONALITY_MODE = 'multiplicative'

# Risk assessment thresholds
HIGH_RISK_THRESHOLD = 0.8
MEDIUM_RISK_THRESHOLD = 0.6

# Cost optimization parameters
FUEL_COST_PER_KM = 0.5
LABOR_COST_PER_HOUR = 25
VEHICLE_FIXED_COST = 100

# Warehouse location parameters
MAX_FACILITIES = 20
FACILITY_FIXED_COST = 50000  # Cost per facility per month
FACILITY_CAPACITY = 5000  # Orders per month

# Geographical parameters
EARTH_RADIUS_KM = 6371
DEFAULT_CRS = "EPSG:4326"  # WGS84

# API Keys (load from .env file)
OSRM_BASE_URL = "http://router.project-osrm.org/route/v1/driving/"
OPENROUTESERVICE_API_KEY = os.getenv("OPENROUTESERVICE_API_KEY", "")

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Performance thresholds
MIN_DELIVERY_TIME_ACCURACY = 0.85
MIN_COST_SAVINGS_PERCENTAGE = 5.0
MAX_ROUTE_OPTIMIZATION_TIME = 300  # seconds

# Visualization settings
PLOT_WIDTH = 1200
PLOT_HEIGHT = 800
COLOR_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Export settings
EXPORT_FORMATS = ['csv', 'json', 'html']
FIGURE_FORMATS = ['png', 'svg', 'pdf']