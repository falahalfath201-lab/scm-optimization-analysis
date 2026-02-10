"""
Master Pipeline
Orchestrates multiple analysis pipelines with shared data loading
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import logging

from ..data_loader import DataLoader
from ..preprocessor import DataPreprocessor
from ..utils.metrics import calculate_delivery_metrics, calculate_cost_metrics

from .cost_optimization_pipeline import CostOptimizationPipeline
from .route_optimization_pipeline import RouteOptimizationPipeline
from .delivery_prediction_pipeline import DeliveryPredictionPipeline
from .risk_analysis_pipeline import RiskAnalysisPipeline
from .warehouse_location_pipeline import WarehouseLocationPipeline


logger = logging.getLogger(__name__)


class MasterPipeline:
    """
    Master pipeline orchestrator
    
    Features:
    - Shared data loading (load once, use multiple times)
    - Selective pipeline execution
    - Combined reporting
    - Resource optimization
    """
    
    # Available pipelines
    AVAILABLE_PIPELINES = {
        'cost_optimization': CostOptimizationPipeline,
        'route_optimization': RouteOptimizationPipeline,
        'delivery_prediction': DeliveryPredictionPipeline,
        'risk_analysis': RiskAnalysisPipeline,
        'warehouse_location': WarehouseLocationPipeline
    }
    
    def __init__(self):
        """Initialize master pipeline"""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        
        # Shared data (loaded once)
        self.shared_data: Optional[Dict[str, Any]] = None
        self.pipeline_results: Dict[str, Dict[str, Any]] = {}
        
        # Output
        self.output_dir = Path('outputs/reports')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Master Pipeline initialized")
    
    def prepare_shared_data(self, data_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Load and preprocess data once for all pipelines
        
        Args:
            data_file: Optional path to data file (not used, kept for signature compatibility)
        
        Returns:
            Shared data dict with 'raw', 'processed', 'baseline'
        """
        print("\n" + "="*70)
        print("MASTER PIPELINE - PREPARING SHARED DATA")
        print("="*70)
        
        # 1. Load data
        print("\n[1/3] Loading data...")
        raw_data = self.loader.load_data()
        print(f"   Loaded {len(raw_data):,} rows")
        
        # 2. Preprocess
        print("\n[2/3] Preprocessing data...")
        processed_data = self.preprocessor.preprocess_data(raw_data)
        print(f"   Processed {len(processed_data):,} rows")
        
        # 3. Calculate baseline
        print("\n[3/3] Calculating baseline metrics...")
        baseline_metrics = {
            'delivery_metrics': calculate_delivery_metrics(processed_data),
            'cost_metrics': calculate_cost_metrics(processed_data),
            'total_orders': len(processed_data),
            'date_range': {
                'start': processed_data['order date (DateOrders)'].min() if 'order date (DateOrders)' in processed_data.columns else None,
                'end': processed_data['order date (DateOrders)'].max() if 'order date (DateOrders)' in processed_data.columns else None
            }
        }
        
        print(f"\n   Baseline Summary:")
        print(f"      Total Orders: {baseline_metrics['total_orders']:,}")
        print(f"      Avg Delivery: {baseline_metrics['delivery_metrics']['mean_delivery_days']:.2f} days")
        print(f"      Total Revenue: ${baseline_metrics['cost_metrics']['total_revenue']:,.2f}")
        print(f"      Late Rate: {100 - baseline_metrics['delivery_metrics'].get('on_time_rate', 0):.1f}%")
        
        self.shared_data = {
            'raw': raw_data,
            'processed': processed_data,
            'baseline': baseline_metrics
        }
        
        print("\n[OK] Shared data prepared")
        print("="*70)
        
        return self.shared_data
    
    def run_pipeline(self, pipeline_name: str) -> Dict[str, Any]:
        """
        Run a single pipeline with shared data
        
        Args:
            pipeline_name: Name of pipeline to run
        
        Returns:
            Pipeline result dict
        """
        if pipeline_name not in self.AVAILABLE_PIPELINES:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")
        
        # Ensure shared data is prepared
        if self.shared_data is None:
            self.prepare_shared_data()
        
        # Create and run pipeline
        PipelineClass = self.AVAILABLE_PIPELINES[pipeline_name]
        pipeline = PipelineClass()
        
        result = pipeline.run(shared_data=self.shared_data)
        
        # Store result
        self.pipeline_results[pipeline_name] = result
        
        return result
    
    def run_selected(self, pipeline_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Run selected pipelines
        
        Args:
            pipeline_names: List of pipeline names to run
        
        Returns:
            Dict of pipeline results
        """
        print("\n" + "="*70)
        print(f"MASTER PIPELINE - RUNNING {len(pipeline_names)} SELECTED ANALYSES")
        print("="*70)
        print(f"Selected: {', '.join(pipeline_names)}\n")
        
        # Prepare shared data once
        if self.shared_data is None:
            self.prepare_shared_data()
        
        # Run each pipeline
        results = {}
        for i, pipeline_name in enumerate(pipeline_names, 1):
            print(f"\n{'~'*70}")
            print(f"PIPELINE {i}/{len(pipeline_names)}: {pipeline_name.upper()}")
            print(f"{'~'*70}")
            
            try:
                result = self.run_pipeline(pipeline_name)
                results[pipeline_name] = result
            except Exception as e:
                logger.error(f"Pipeline {pipeline_name} failed: {e}", exc_info=True)
                print(f"[ERROR] Pipeline {pipeline_name} failed: {e}")
                results[pipeline_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return results
    
    def run_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Run all available pipelines
        
        Returns:
            Dict of all pipeline results
        """
        return self.run_selected(list(self.AVAILABLE_PIPELINES.keys()))
    
    def generate_master_report(self) -> str:
        """
        Generate comprehensive master report combining all results
        
        Returns:
            Master report text
        """
        report = f"""
{'='*70}
SUPPLY CHAIN OPTIMIZATION - MASTER REPORT
{'='*70}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Report ID: {self.timestamp}

EXECUTIVE SUMMARY:
{'-'*70}
"""
        
        if self.shared_data and 'baseline' in self.shared_data:
            baseline = self.shared_data['baseline']
            report += f"""
Total Orders Analyzed: {baseline['total_orders']:,}
Analysis Period: {baseline['date_range']['start']} to {baseline['date_range']['end']}

Key Metrics:
   Avg Delivery Time: {baseline['delivery_metrics']['mean_delivery_days']:.2f} days
   On-Time Delivery Rate: {baseline['delivery_metrics'].get('on_time_rate', 0):.1f}%
   Total Revenue: ${baseline['cost_metrics']['total_revenue']:,.2f}
   Total Profit: ${baseline['cost_metrics']['total_profit']:,.2f}
   Profit Margin: {baseline['cost_metrics']['profit_margin']:.2f}%
"""
        
        report += f"\n\nPIPELINES EXECUTED:\n{'-'*70}\n"
        
        success_count = sum(1 for r in self.pipeline_results.values() if r.get('status') == 'success')
        total_count = len(self.pipeline_results)
        
        report += f"Total: {total_count} | Successful: {success_count} | Failed: {total_count - success_count}\n\n"
        
        for pipeline_name, result in self.pipeline_results.items():
            status_icon = "[OK]" if result.get('status') == 'success' else "[FAIL]"
            report += f"   {status_icon} {pipeline_name.replace('_', ' ').title()}\n"
        
        # Key findings from each pipeline
        report += f"\n\nKEY FINDINGS:\n{'-'*70}\n"
        
        for pipeline_name, result in self.pipeline_results.items():
            if result.get('status') == 'success':
                report += f"\n{pipeline_name.replace('_', ' ').title()}:\n"
                
                # Extract key metrics based on pipeline type
                if pipeline_name == 'cost_optimization' and 'results' in result:
                    res = result['results']
                    if 'optimization' in res:
                        opt = res['optimization']
                        report += f"   - Potential Savings: ${opt.get('savings', 0):,.2f} ({opt.get('savings_pct', 0):.1f}%)\n"
                
                elif pipeline_name == 'delivery_prediction' and 'results' in result:
                    res = result['results']
                    if 'models' in res and res['models'].get('status') == 'success':
                        best = res['models']['best_model']
                        report += f"   - Best Prediction Model: {best}\n"
                
                elif pipeline_name == 'risk_analysis' and 'results' in result:
                    res = result['results']
                    if 'best_model' in res:
                        report += f"   - Best Risk Model: {res['best_model']}\n"
                
                elif pipeline_name == 'warehouse_location' and 'results' in result:
                    res = result['results']
                    if 'scenarios' in res and res['scenarios'].get('status') == 'success':
                        best_n = res['scenarios'].get('best_n_facilities')
                        report += f"   - Optimal Warehouse Count: {best_n}\n"
        
        report += f"\n\nRECOMMENDATIONS:\n{'-'*70}\n"
        report += "1. Review detailed pipeline reports for specific recommendations\n"
        report += "2. Implement cost optimization strategies for maximum savings\n"
        report += "3. Focus on high-risk orders identified by risk analysis\n"
        report += "4. Consider warehouse location optimization for long-term strategy\n"
        
        report += "\n" + "="*70 + "\n"
        report += "END OF MASTER REPORT\n"
        report += "="*70 + "\n"
        
        return report
    
    def save_master_report(self, report: Optional[str] = None) -> None:
        """
        Save master report to file
        
        Args:
            report: Optional report text (generates if None)
        """
        if report is None:
            report = self.generate_master_report()
        
        report_file = self.output_dir / f"master_report_{self.timestamp}.txt"
        report_file.write_text(report, encoding='utf-8')
        print(f"\n[OK] Master report saved: {report_file}")
    
    def __repr__(self) -> str:
        return f"MasterPipeline(pipelines={len(self.AVAILABLE_PIPELINES)}, executed={len(self.pipeline_results)})"
