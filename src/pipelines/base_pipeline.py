"""
Base Pipeline Class
Provides common functionality for all analysis pipelines (DRY principle)
"""

import pandas as pd
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from ..data_loader import DataLoader
from ..preprocessor import DataPreprocessor
from ..utils.visualizer import Visualizer
from ..utils.metrics import calculate_delivery_metrics, calculate_cost_metrics


logger = logging.getLogger(__name__)


class BasePipeline(ABC):
    """
    Abstract base class for all analysis pipelines
    
    Implements Template Method pattern:
    - Common steps (load, preprocess, baseline) are implemented here
    - Analysis-specific steps must be implemented by subclasses
    
    Usage:
        class MyPipeline(BasePipeline):
            def analyze(self, df): ...
            def save_results(self, results): ...
    """
    
    def __init__(self, name: str, use_cache: bool = True):
        """
        Initialize base pipeline
        
        Args:
            name: Pipeline name (used for logging and file naming)
            use_cache: Whether to use cached processed data
        """
        self.name = name
        self.use_cache = use_cache
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Components
        self.loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.visualizer = Visualizer()
        
        # Data containers
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.baseline_metrics: Optional[Dict[str, Any]] = None
        self.results: Optional[Dict[str, Any]] = None
        
        # Paths
        self.output_dir = Path(f'outputs/reports')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized {self.name} pipeline")
    
    def load_data(self, data_file: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from file
        
        Args:
            data_file: Optional path to data file (not used, kept for signature compatibility)
        
        Returns:
            Raw DataFrame
        """
        print(f"\n[1/5] Loading data...")
        self.raw_data = self.loader.load_data()
        print(f"   Loaded {len(self.raw_data):,} rows")
        return self.raw_data
    
    def preprocess(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Preprocess data
        
        Args:
            df: Optional DataFrame to preprocess (uses self.raw_data if None)
        
        Returns:
            Processed DataFrame
        """
        print(f"\n[2/5] Preprocessing data...")
        
        if df is None:
            df = self.raw_data
        
        if df is None:
            raise ValueError("No data to preprocess. Call load_data() first.")
        
        self.processed_data = self.preprocessor.preprocess_data(df)
        print(f"   Processed {len(self.processed_data):,} rows")
        
        return self.processed_data
    
    def calculate_baseline(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Calculate baseline metrics
        
        Args:
            df: Optional DataFrame (uses self.processed_data if None)
        
        Returns:
            Dictionary of baseline metrics
        """
        print(f"\n[3/5] Calculating baseline metrics...")
        
        if df is None:
            df = self.processed_data
        
        if df is None:
            raise ValueError("No data for baseline. Call preprocess() first.")
        
        self.baseline_metrics = {
            'delivery_metrics': calculate_delivery_metrics(df),
            'cost_metrics': calculate_cost_metrics(df),
            'total_orders': len(df),
            'date_range': {
                'start': df['order date (DateOrders)'].min() if 'order date (DateOrders)' in df.columns else None,
                'end': df['order date (DateOrders)'].max() if 'order date (DateOrders)' in df.columns else None
            }
        }
        
        print(f"   Baseline calculated:")
        print(f"      Total Orders: {self.baseline_metrics['total_orders']:,}")
        print(f"      Avg Delivery: {self.baseline_metrics['delivery_metrics']['mean_delivery_days']:.2f} days")
        print(f"      Total Revenue: ${self.baseline_metrics['cost_metrics']['total_revenue']:,.2f}")
        
        return self.baseline_metrics
    
    @abstractmethod
    def analyze(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Perform analysis (must be implemented by subclass)
        
        Args:
            df: Optional DataFrame (uses self.processed_data if None)
        
        Returns:
            Analysis results dictionary
        """
        pass
    
    @abstractmethod
    def save_results(self, results: Optional[Dict[str, Any]] = None) -> None:
        """
        Save analysis results (must be implemented by subclass)
        
        Args:
            results: Optional results dict (uses self.results if None)
        """
        pass
    
    def generate_report(self, results: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate text report (can be overridden by subclass)
        
        Args:
            results: Optional results dict (uses self.results if None)
        
        Returns:
            Report text
        """
        if results is None:
            results = self.results
        
        if results is None:
            return "No results to report"
        
        # Default report
        report = f"""
{'='*70}
{self.name.upper()} ANALYSIS REPORT
{'='*70}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

BASELINE METRICS:
{'-'*70}
Total Orders: {self.baseline_metrics['total_orders']:,}
Avg Delivery Time: {self.baseline_metrics['delivery_metrics']['mean_delivery_days']:.2f} days
On-Time Rate: {self.baseline_metrics['delivery_metrics'].get('on_time_rate', 0):.1f}%
Total Revenue: ${self.baseline_metrics['cost_metrics']['total_revenue']:,.2f}
Total Profit: ${self.baseline_metrics['cost_metrics']['total_profit']:,.2f}

ANALYSIS RESULTS:
{'-'*70}
"""
        # Add custom results (subclasses should override for better formatting)
        for key, value in results.items():
            if not isinstance(value, (dict, list)):
                report += f"{key}: {value}\n"
        
        report += "\n" + "="*70 + "\n"
        
        return report
    
    def save_report(self, report: str, filename: Optional[str] = None) -> None:
        """
        Save report to file
        
        Args:
            report: Report text
            filename: Optional filename (auto-generated if None)
        """
        if filename is None:
            filename = f"{self.name.lower().replace(' ', '_')}_report_{self.timestamp}.txt"
        
        report_path = self.output_dir / filename
        report_path.write_text(report, encoding='utf-8')
        print(f"   Report saved: {report_path}")
    
    def run(self, shared_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Any]:
        """
        Execute complete pipeline (Template Method)
        
        Args:
            shared_data: Optional pre-loaded/preprocessed data from master pipeline
                        Keys: 'raw', 'processed', 'baseline'
        
        Returns:
            Complete results dictionary
        """
        print(f"\n{'='*70}")
        print(f"{self.name.upper()} PIPELINE")
        print(f"{'='*70}")
        
        try:
            # Use shared data if available (from master pipeline)
            if shared_data:
                print("\n[SHARED DATA MODE] Using pre-loaded data")
                self.raw_data = shared_data.get('raw')
                self.processed_data = shared_data.get('processed')
                self.baseline_metrics = shared_data.get('baseline')
                
                if self.processed_data is not None:
                    print(f"   Using {len(self.processed_data):,} pre-processed rows")
            else:
                # Standalone mode: load and process data
                print("\n[STANDALONE MODE] Loading data independently")
                self.load_data()
                self.preprocess()
                self.calculate_baseline()
            
            # Analysis (specific to each pipeline)
            print(f"\n[4/5] Running {self.name} analysis...")
            self.results = self.analyze(self.processed_data)
            
            # Save results
            print(f"\n[5/5] Saving results...")
            self.save_results(self.results)
            
            # Generate and save report
            report = self.generate_report(self.results)
            self.save_report(report)
            
            print(f"\n{'='*70}")
            print(f"[OK] {self.name} pipeline completed successfully")
            print(f"{'='*70}\n")
            
            return {
                'pipeline': self.name,
                'status': 'success',
                'results': self.results,
                'baseline': self.baseline_metrics,
                'timestamp': self.timestamp
            }
        
        except Exception as e:
            logger.error(f"{self.name} pipeline failed: {e}", exc_info=True)
            print(f"\n[ERROR] {self.name} pipeline failed: {e}\n")
            
            return {
                'pipeline': self.name,
                'status': 'failed',
                'error': str(e),
                'timestamp': self.timestamp
            }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
