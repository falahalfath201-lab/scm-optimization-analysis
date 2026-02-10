"""
Cost Optimization Pipeline
Analyzes and optimizes shipping costs
"""

import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path

from .base_pipeline import BasePipeline
from ..analysis.cost_optimization import CostOptimizer
from ..utils.metrics import calculate_cost_savings, format_metric


class CostOptimizationPipeline(BasePipeline):
    """Pipeline for cost optimization analysis"""
    
    def __init__(self):
        super().__init__(name="Cost Optimization")
        self.optimizer = CostOptimizer()
        self.results_dir = Path('data/results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run cost optimization analysis"""
        if df is None:
            df = self.processed_data
        
        results = {}
        
        # 1. Analyze current costs
        print("   [1/4] Analyzing current costs...")
        current_costs = self.optimizer.analyze_current_costs(df)
        results['current_costs'] = current_costs
        
        # 2. Optimize shipping modes
        print("   [2/4] Optimizing shipping modes...")
        optimized = self.optimizer.optimize_shipping_mode(df)
        results['optimization'] = optimized
        
        # 3. Cost-benefit analysis
        print("   [3/4] Running cost-benefit analysis...")
        cba = self.optimizer.cost_benefit_analysis(df)
        results['cost_benefit'] = cba
        
        # 4. Get summary
        print("   [4/4] Generating summary...")
        summary = self.optimizer.get_optimization_summary(df)
        results['summary'] = summary
        
        return results
    
    def save_results(self, results: Optional[Dict[str, Any]] = None) -> None:
        """Save cost optimization results"""
        if results is None:
            results = self.results
        
        # Save numerical results to CSV
        results_file = self.results_dir / f'cost_optimization_results_{self.timestamp}.csv'
        
        # Create summary DataFrame
        summary_data = []
        
        if 'optimization' in results:
            opt = results['optimization']
            summary_data.append({
                'Metric': 'Current Cost',
                'Value': opt['current_cost'],
                'Category': 'Cost'
            })
            summary_data.append({
                'Metric': 'Optimized Cost',
                'Value': opt['optimized_cost'],
                'Category': 'Cost'
            })
            summary_data.append({
                'Metric': 'Savings',
                'Value': opt['savings'],
                'Category': 'Savings'
            })
            summary_data.append({
                'Metric': 'Savings Percentage',
                'Value': opt['savings_pct'],
                'Category': 'Savings'
            })
        
        pd.DataFrame(summary_data).to_csv(results_file, index=False)
        print(f"   Saved results: {results_file}")
        
        # Generate visualizations
        print("   Generating visualizations...")
        
        if 'cost_benefit' in results:
            cba = results['cost_benefit']
            if 'strategies' in cba:
                # Cost comparison chart
                cost_data = {name: data['total_cost'] 
                           for name, data in cba['strategies'].items()}
                self.visualizer.plot_cost_comparison(
                    cost_data,
                    title="Cost Strategy Comparison",
                    filename="cost_strategy_comparison.png",
                    subdirectory="cost_optimization"
                )
        
        print("   [OK] Cost optimization results saved")
    
    def generate_report(self, results: Optional[Dict[str, Any]] = None) -> str:
        """Generate detailed cost optimization report"""
        if results is None:
            results = self.results
        
        report = f"""
{'='*70}
COST OPTIMIZATION ANALYSIS REPORT
{'='*70}
Generated: {self.timestamp}

BASELINE METRICS:
{'-'*70}
Total Orders: {self.baseline_metrics['total_orders']:,}
Total Revenue: ${self.baseline_metrics['cost_metrics']['total_revenue']:,.2f}
Total Profit: ${self.baseline_metrics['cost_metrics']['total_profit']:,.2f}
Profit Margin: {self.baseline_metrics['cost_metrics']['profit_margin']:.2f}%

OPTIMIZATION RESULTS:
{'-'*70}
"""
        
        if 'optimization' in results:
            opt = results['optimization']
            report += f"""
Current Shipping Cost: ${opt['current_cost']:,.2f}
Optimized Shipping Cost: ${opt['optimized_cost']:,.2f}
Savings: ${opt['savings']:,.2f} ({opt['savings_pct']:.1f}%)
Expected Service Level: {opt['expected_service_level']*100:.1f}%

Recommended Mode Distribution:
"""
            for mode, count in opt.get('mode_distribution', {}).items():
                pct = (count / self.baseline_metrics['total_orders']) * 100
                report += f"   {mode}: {count:,} orders ({pct:.1f}%)\n"
        
        if 'cost_benefit' in results:
            cba = results['cost_benefit']
            report += f"\n\nCOST-BENEFIT ANALYSIS:\n{'-'*70}\n"
            report += f"Best Strategy: {cba.get('best_strategy_name', 'N/A')}\n\n"
            report += f"{'Strategy':<30} {'Total Cost':>15}\n"
            report += "-"*50 + "\n"
            
            for name, data in cba.get('strategies', {}).items():
                marker = "[BEST]" if name == cba.get('best_strategy') else "      "
                report += f"{marker} {name:<24} ${data['total_cost']:>12,.2f}\n"
        
        report += "\n" + "="*70 + "\n"
        
        return report
