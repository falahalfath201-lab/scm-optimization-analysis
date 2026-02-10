"""
Delivery Prediction Pipeline
Predicts delivery times using ML models
"""

import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path

from .base_pipeline import BasePipeline
from ..analysis.delivery_prediction import DeliveryPredictor


class DeliveryPredictionPipeline(BasePipeline):
    """Pipeline for delivery time prediction"""
    
    def __init__(self):
        super().__init__(name="Delivery Prediction")
        self.predictor = DeliveryPredictor()
        self.results_dir = Path('data/results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run delivery prediction analysis"""
        if df is None:
            df = self.processed_data
        
        results = {}
        
        # 1. Analyze patterns
        print("   [1/3] Analyzing delivery patterns...")
        patterns = self.predictor.analyze_delivery_patterns(df)
        results['patterns'] = patterns
        
        # 2. Train models
        print("   [2/3] Training prediction models...")
        model_results = self.predictor.train_models(df, sample_size=50000)
        results['models'] = model_results
        
        # 3. Feature importance
        print("   [3/3] Extracting feature importance...")
        if model_results.get('status') == 'success':
            best_model_name = model_results['best_model']
            if best_model_name in model_results['trained_models']:
                best_model = model_results['trained_models'][best_model_name]
                importance = self.predictor.get_feature_importance(best_model['model'])
                results['feature_importance'] = importance
        
        return results
    
    def save_results(self, results: Optional[Dict[str, Any]] = None) -> None:
        """Save delivery prediction results"""
        if results is None:
            results = self.results
        
        # Save model performance
        results_file = self.results_dir / f'delivery_prediction_results_{self.timestamp}.csv'
        
        if 'models' in results and results['models'].get('status') == 'success':
            model_data = []
            for name, model_info in results['models'].get('trained_models', {}).items():
                if 'metrics' in model_info:
                    metrics = model_info['metrics']
                    model_data.append({
                        'Model': name,
                        'MAE': metrics.get('mae', 0),
                        'RMSE': metrics.get('rmse', 0),
                        'R2': metrics.get('r2_score', 0)
                    })
            
            if model_data:
                pd.DataFrame(model_data).to_csv(results_file, index=False)
                print(f"   Saved results: {results_file}")
        
        # Visualizations
        if 'feature_importance' in results:
            self.visualizer.plot_feature_importance(
                results['feature_importance'],
                title="Delivery Time - Feature Importance",
                filename="feature_importance.png",
                subdirectory="delivery_prediction"
            )
        
        print("   [OK] Delivery prediction results saved")
    
    def generate_report(self, results: Optional[Dict[str, Any]] = None) -> str:
        """Generate delivery prediction report"""
        if results is None:
            results = self.results
        
        report = f"""
{'='*70}
DELIVERY PREDICTION ANALYSIS REPORT
{'='*70}
Generated: {self.timestamp}

BASELINE METRICS:
{'-'*70}
Total Orders: {self.baseline_metrics['total_orders']:,}
Avg Delivery Time: {self.baseline_metrics['delivery_metrics']['mean_delivery_days']:.2f} days
On-Time Rate: {self.baseline_metrics['delivery_metrics'].get('on_time_rate', 0):.1f}%

MODEL PERFORMANCE:
{'-'*70}
"""
        
        if 'models' in results and results['models'].get('status') == 'success':
            best_model = results['models']['best_model']
            report += f"Best Model: {best_model}\n\n"
            report += f"{'Model':<25} {'MAE':>10} {'RMSE':>10} {'R2':>10}\n"
            report += "-"*60 + "\n"
            
            for name, model_info in results['models'].get('trained_models', {}).items():
                if 'metrics' in model_info:
                    m = model_info['metrics']
                    marker = "[BEST]" if name == best_model else "      "
                    report += f"{marker} {name:<19} {m.get('mae', 0):>10.3f} {m.get('rmse', 0):>10.3f} {m.get('r2_score', 0):>10.3f}\n"
        
        report += "\n" + "="*70 + "\n"
        
        return report
