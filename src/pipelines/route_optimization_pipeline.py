"""
Route Optimization Pipeline
Optimizes delivery routes using TSP/VRP with detailed route visualization
"""

import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt

from .base_pipeline import BasePipeline
from ..analysis.route_optimization import RouteOptimizer
from ..utils.visualizer import Visualizer


class RouteOptimizationPipeline(BasePipeline):
    """Pipeline for route optimization analysis"""
    
    def __init__(self):
        super().__init__(name="Route Optimization")
        self.optimizer = RouteOptimizer()
        self.visualizer = Visualizer()
        self.results_dir = Path('data/results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run route optimization analysis"""
        if df is None:
            df = self.processed_data
        
        print("   Optimizing routes for top regions...")
        
        # Get top regions
        top_regions = df['Customer State'].value_counts().head(3)
        results = {'regions': {}}
        
        for i, (region, count) in enumerate(top_regions.items(), 1):
            print(f"      [{i}/3] Optimizing {region} ({count:,} orders)...")
            
            region_result = self.optimizer.optimize_region(
                df=df,
                region_column='Customer State',
                region_value=region,
                max_locations=15,  # Reduced for better visualization
                problem_type='vrp'  # Force VRP for multi-vehicle routes
            )
            
            # Extract detailed route information
            region_result = self._extract_route_details(region_result, region)
            results['regions'][region] = region_result
        
        return results
    
    def save_results(self, results: Optional[Dict[str, Any]] = None) -> None:
        """Save route optimization results"""
        if results is None:
            results = self.results
        
        results_file = self.results_dir / f'route_optimization_results_{self.timestamp}.csv'
        
        # Save summary
        summary_data = []
        for region, data in results.get('regions', {}).items():
            if 'region' in data:
                summary_data.append({
                    'Region': region,
                    'Orders': data['region']['total_orders'],
                    'Locations': data['region']['unique_locations']
                })
        
        if summary_data:
            pd.DataFrame(summary_data).to_csv(results_file, index=False)
            print(f"   Saved results: {results_file}")
        
        # Generate visualizations
        self._create_visualizations(results)
        
        print("   [OK] Route optimization results saved")
    
    def _extract_route_details(self, result: Dict[str, Any], region: str) -> Dict[str, Any]:
        """Extract detailed route information for visualization and reporting"""
        vehicle_routes = []
        depot_coords = result.get('region', {}).get('depot', (0, 0))
        
        # Get best strategy routes (use cluster_tsp as default, or benchmark winner)
        best_strategy = 'cluster_tsp'
        if 'benchmark' in result and 'best_strategy' in result['benchmark']:
            best_strategy = result['benchmark']['best_strategy']
        
        # Extract routes from strategies
        if 'strategies' in result and best_strategy in result['strategies']:
            strategy_result = result['strategies'][best_strategy]
            if 'routes' in strategy_result and isinstance(strategy_result['routes'], list):
                for route in strategy_result['routes']:
                    if 'route_coords' in route and len(route['route_coords']) > 1:
                        vehicle_routes.append({
                            'vehicle_id': route.get('vehicle_id', 0),
                            'stops': route['route_coords'],
                            'distance_km': route.get('distance_km', 0),
                            'time_min': route.get('total_time_min', route.get('time_min', 0)),
                            'num_stops': route.get('num_stops', len(route['route_coords']) - 1)
                        })
        
        result['vehicle_routes'] = vehicle_routes
        result['depot_coords'] = depot_coords
        result['best_strategy'] = best_strategy
        return result
    
    def _create_visualizations(self, results: Dict[str, Any]) -> None:
        """Create route optimization visualizations"""
        print("   Creating route visualizations...")
        
        for region, data in results.get('regions', {}).items():
            # Plot route map for each region
            routes_data = data.get('vehicle_routes', [])
            
            if routes_data:
                self._plot_detailed_routes(
                    routes=routes_data,
                    region=region,
                    depot=data.get('depot_coords')
                )
        
        # Create summary comparison chart
        self._plot_route_summary(results)
    
    def _plot_detailed_routes(self, routes: List[Dict], region: str, depot: Tuple[float, float]) -> None:
        """Plot detailed route map with vehicle routes"""
        import seaborn as sns
        
        if not routes:
            return
        
        fig, ax = plt.subplots(figsize=(14, 10))
        colors = sns.color_palette("husl", len(routes))
        
        # Plot each vehicle's route
        for i, route in enumerate(routes):
            stops = route['stops']
            if len(stops) < 2:
                continue
            
            lats, lons = zip(*stops)
            vehicle_id = route['vehicle_id']
            
            # Plot route line with numbered stops
            ax.plot(lons, lats, 'o-', color=colors[i], linewidth=2.5,
                   markersize=10, label=f"Vehicle {vehicle_id+1} ({route['num_stops']} stops)",
                   alpha=0.8, markeredgecolor='white', markeredgewidth=1.5)
            
            # Add stop numbers
            for j, (lat, lon) in enumerate(stops):
                if j == 0:  # Depot
                    ax.plot(lon, lat, '*', color=colors[i], markersize=25,
                           markeredgecolor='black', markeredgewidth=2, zorder=10)
                    ax.text(lon, lat, 'DEPOT', fontsize=9, fontweight='bold',
                           ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.3',
                           facecolor='yellow', alpha=0.7))
                else:
                    ax.text(lon, lat, str(j), fontsize=8, fontweight='bold',
                           ha='center', va='center',
                           bbox=dict(boxstyle='circle,pad=0.1', facecolor='white',
                           edgecolor=colors[i], linewidth=2))
        
        ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
        ax.set_title(f'Vehicle Routes - {region}\n(Multi-Vehicle Route Optimization)',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        self.visualizer.save_figure(
            filename=f'route_map_{region}_{self.timestamp}.png',
            subdirectory='route_optimization'
        )
    
    def _plot_route_summary(self, results: Dict[str, Any]) -> None:
        """Plot summary comparison of route optimization across regions"""
        regions = []
        orders = []
        locations = []
        
        for region, data in results.get('regions', {}).items():
            if 'region' in data:
                regions.append(region)
                orders.append(data['region']['total_orders'])
                locations.append(data['region']['unique_locations'])
        
        if not regions:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Orders by region
        colors = plt.cm.viridis(range(len(regions)))
        ax1.bar(regions, orders, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Number of Orders', fontsize=12)
        ax1.set_title('Orders by Region', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (r, o) in enumerate(zip(regions, orders)):
            ax1.text(i, o, f'{o:,}', ha='center', va='bottom', fontweight='bold')
        
        # Unique locations by region
        ax2.bar(regions, locations, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Unique Locations', fontsize=12)
        ax2.set_title('Delivery Locations by Region', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (r, loc) in enumerate(zip(regions, locations)):
            ax2.text(i, loc, f'{loc}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        self.visualizer.save_figure(
            filename=f'route_summary_{self.timestamp}.png',
            subdirectory='route_optimization'
        )
    
    def generate_report(self, results: Optional[Dict[str, Any]] = None) -> str:
        """Generate route optimization report with detailed routes"""
        if results is None:
            results = self.results
        
        report = f"""
{'='*70}
ROUTE OPTIMIZATION ANALYSIS REPORT
{'='*70}
Generated: {self.timestamp}

BASELINE METRICS:
{'-'*70}
Total Orders: {self.baseline_metrics['total_orders']:,}

ROUTE OPTIMIZATION RESULTS:
{'-'*70}
"""
        
        for region, data in results.get('regions', {}).items():
            if 'region' in data:
                r = data['region']
                report += f"\n{'='*70}\n"
                report += f"Region: {region}\n"
                report += f"{'='*70}\n"
                report += f"Total Orders: {r['total_orders']:,}\n"
                report += f"Unique Locations: {r['unique_locations']}\n"
                report += f"Depot Coordinates: {data.get('depot_coords', 'N/A')}\n\n"
                
                # Vehicle routes details
                vehicle_routes = data.get('vehicle_routes', [])
                if vehicle_routes:
                    report += f"Number of Vehicles: {len(vehicle_routes)}\n\n"
                    
                    for route in vehicle_routes:
                        vehicle_id = route['vehicle_id']
                        report += f"{'-'*70}\n"
                        report += f"VEHICLE {vehicle_id + 1}\n"
                        report += f"{'-'*70}\n"
                        report += f"Total Stops: {route['num_stops']}\n"
                        report += f"Route Distance: {route['distance_km']:.2f} km\n"
                        report += f"Estimated Time: {route['time_min']:.2f} minutes\n\n"
                        
                        # Route sequence
                        report += f"Route Sequence:\n"
                        stops = route['stops']
                        for i, (lat, lon) in enumerate(stops):
                            if i == 0:
                                report += f"   Start: DEPOT at ({lat:.4f}, {lon:.4f})\n"
                            else:
                                report += f"   Stop {i}: Location ({lat:.4f}, {lon:.4f})\n"
                        
                        if len(stops) > 1:
                            report += f"   Return: DEPOT at ({stops[0][0]:.4f}, {stops[0][1]:.4f})\n"
                        report += "\n"
        
        report += "\n" + "="*70 + "\n"
        
        return report
