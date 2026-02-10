"""
Warehouse Location Pipeline
Optimizes warehouse locations
"""

import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path

from .base_pipeline import BasePipeline
from ..analysis.warehouse_location import WarehouseOptimizer


class WarehouseLocationPipeline(BasePipeline):
    """Pipeline for warehouse location optimization"""
    
    def __init__(self):
        super().__init__(name="Warehouse Location")
        self.optimizer = WarehouseOptimizer()
        self.results_dir = Path('data/results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run warehouse location optimization"""
        if df is None:
            df = self.processed_data
        
        results = {}
        
        # 1. Find optimal locations
        print("   [1/3] Finding optimal warehouse locations...")
        locations_result = self.optimizer.optimize_locations(
            df, n_facilities=5, method='kmeans', sample_size=30000
        )
        results['optimal_locations'] = locations_result
        
        # 2. Compare scenarios
        print("   [2/3] Comparing different scenarios...")
        scenarios_result = self.optimizer.compare_scenarios(
            df, facility_counts=[3, 5, 7, 10], sample_size=20000
        )
        results['scenarios'] = scenarios_result
        
        # 3. Coverage analysis
        print("   [3/3] Analyzing coverage...")
        if locations_result.get('status') == 'success':
            coverage = self.optimizer.analyze_coverage(
                df, locations_result['locations'], radius_km=100, sample_size=30000
            )
            results['coverage'] = coverage
        
        return results
    
    def save_results(self, results: Optional[Dict[str, Any]] = None) -> None:
        """Save warehouse location results"""
        if results is None:
            results = self.results
        
        results_file = self.results_dir / f'warehouse_location_results_{self.timestamp}.csv'
        
        # Save scenario comparison
        if 'scenarios' in results and results['scenarios'].get('status') == 'success':
            scenario_data = []
            for n, data in results['scenarios'].get('results', {}).items():
                scenario_data.append({
                    'Facilities': n,
                    'Avg_Distance_km': data.get('avg_distance', 0),
                    'Total_Cost': data.get('total_cost', 0)
                })
            
            if scenario_data:
                pd.DataFrame(scenario_data).to_csv(results_file, index=False)
                print(f"   Saved results: {results_file}")
        
        # Visualization - warehouse locations
        if 'optimal_locations' in results and results['optimal_locations'].get('status') == 'success':
            locations = results['optimal_locations']['locations']
            labels = [f"WH-{i}" for i in range(len(locations))]
            
            self.visualizer.plot_warehouse_locations(
                locations,
                labels=labels,
                title="Optimal Warehouse Locations",
                filename="warehouse_locations.png",
                subdirectory="warehouse_location"
            )
        
        print("   [OK] Warehouse location results saved")
    
    def generate_report(self, results: Optional[Dict[str, Any]] = None) -> str:
        """Generate warehouse location report"""
        if results is None:
            results = self.results
        
        report = f"""
{'='*70}
WAREHOUSE LOCATION OPTIMIZATION REPORT
{'='*70}
Generated: {self.timestamp}

BASELINE METRICS:
{'-'*70}
Total Orders: {self.baseline_metrics['total_orders']:,}

OPTIMAL LOCATIONS:
{'-'*70}
"""
        
        if 'optimal_locations' in results and results['optimal_locations'].get('status') == 'success':
            opt = results['optimal_locations']
            report += f"Number of Facilities: {len(opt['locations'])}\n"
            report += f"Avg Distance: {opt.get('avg_distance', 0):.2f} km\n"
            report += f"Total Distance: {opt.get('total_distance', 0):,.2f} km\n\n"
        
        if 'scenarios' in results and results['scenarios'].get('status') == 'success':
            report += "\nSCENARIO COMPARISON:\n" + "-"*70 + "\n"
            report += f"{'Facilities':>12} {'Avg Dist (km)':>15} {'Total Cost':>15}\n"
            report += "-"*50 + "\n"
            
            for n, data in results['scenarios'].get('results', {}).items():
                report += f"{n:>12} {data.get('avg_distance', 0):>15.2f} ${data.get('total_cost', 0):>14,.2f}\n"
            
            best_n = results['scenarios'].get('best_n_facilities')
            if best_n:
                report += f"\nBest Scenario: {best_n} facilities\n"
        
        report += "\n" + "="*70 + "\n"
        
        return report
