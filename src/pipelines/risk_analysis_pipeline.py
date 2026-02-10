"""
Risk Analysis Pipeline
Analyzes and predicts late delivery risks
"""

import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path

from .base_pipeline import BasePipeline
from ..analysis.late_delivery_risk import LateDeliveryRiskModel


class RiskAnalysisPipeline(BasePipeline):
    """Pipeline for late delivery risk analysis"""
    
    def __init__(self):
        super().__init__(name="Risk Analysis")
        self.analyzer = LateDeliveryRiskModel()
        self.results_dir = Path('data/results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run risk analysis"""
        if df is None:
            df = self.processed_data
        
        print("   Training risk prediction models...")
        
        # Train models using the train method
        results = self.analyzer.train(df, test_size=0.2)
        
        # Add feature importance if available
        if 'best_model' in results and results['best_model'] in results.get('models', {}):
            best_model_obj = results['models'][results['best_model']]
            risk_factors = self.analyzer.analyze_risk_factors()
            if risk_factors and 'feature_importance' in risk_factors:
                results['feature_importance'] = risk_factors['feature_importance']
        
        return results
    
    def save_results(self, results: Optional[Dict[str, Any]] = None) -> None:
        """Save risk analysis results"""
        if results is None:
            results = self.results
        
        results_file = self.results_dir / f'risk_analysis_results_{self.timestamp}.csv'
        
        # Save model comparison
        if 'model_comparison' in results:
            comp_data = []
            for model_name, metrics in results['model_comparison'].items():
                comp_data.append({
                    'Model': model_name,
                    'Accuracy': metrics.get('accuracy', 0),
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0),
                    'F1': metrics.get('f1', 0),
                    'ROC-AUC': metrics.get('roc_auc', 0)
                })
            
            if comp_data:
                pd.DataFrame(comp_data).to_csv(results_file, index=False)
                print(f"   Saved results: {results_file}")
        
        # Feature importance visualization
        if 'feature_importance' in results:
            self.visualizer.plot_feature_importance(
                results['feature_importance'],
                title="Late Delivery Risk - Feature Importance",
                filename="risk_feature_importance.png",
                subdirectory="risk_analysis"
            )
        
        print("   [OK] Risk analysis results saved")
    
    def generate_report(self, results: Optional[Dict[str, Any]] = None) -> str:
        """Generate risk analysis report"""
        if results is None:
            results = self.results
        
        report = f"""
{'='*70}
RISK ANALYSIS REPORT
{'='*70}
Generated: {self.timestamp}

BASELINE METRICS:
{'-'*70}
Total Orders: {self.baseline_metrics['total_orders']:,}
Late Delivery Rate: {100 - self.baseline_metrics['delivery_metrics'].get('on_time_rate', 0):.1f}%

MODEL PERFORMANCE:
{'-'*70}
"""
        
        if 'best_model' in results:
            report += f"Best Model: {results['best_model']}\n\n"
        
        if 'model_comparison' in results:
            report += f"{'Model':<25} {'Accuracy':>10} {'F1':>10} {'ROC-AUC':>10}\n"
            report += "-"*60 + "\n"
            
            for model_name, metrics in results['model_comparison'].items():
                marker = "[BEST]" if model_name == results.get('best_model') else "      "
                report += f"{marker} {model_name:<19} {metrics.get('accuracy', 0):>10.4f} {metrics.get('f1', 0):>10.4f} {metrics.get('roc_auc', 0):>10.4f}\n"
        
        report += "\n" + "="*70 + "\n"
        
        return report
