"""
Route Optimization Module - Unified Interface (Facade Pattern)

This is the main entry point for all route optimization tasks.
Internally uses TSPOptimizer and VRPOptimizer.

Usage:
    from src.analysis.route_optimization import RouteOptimizer
    
    optimizer = RouteOptimizer()
    
    # TSP - Single vehicle
    tsp_result = optimizer.solve_tsp(locations, method='2opt')
    
    # VRP - Multiple vehicles  
    vrp_result = optimizer.solve_vrp(locations, num_vehicles=3, strategy='balanced')
    
    # Benchmark all methods
    benchmark = optimizer.benchmark(locations)
    
    # Region-based optimization
    result = optimizer.optimize_region(df, region='CA', problem_type='vrp')
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging

from .tsp_optimizer import TSPOptimizer, print_benchmark_report as print_tsp_report
from .vrp_optimizer import VRPOptimizer, print_vrp_report
from ..utils.distance_calculator import DistanceCalculator
from ..config import MAX_VEHICLES

logger = logging.getLogger(__name__)


class RouteOptimizer:
    """
    Unified Route Optimizer - Facade for TSP and VRP optimization
    
    Provides a single interface for:
    - TSP (Traveling Salesman Problem) - single vehicle
    - VRP (Vehicle Routing Problem) - multiple vehicles
    - Benchmarking and comparison
    """
    
    def __init__(self, 
                 num_vehicles: int = 3,
                 speed_kmh: float = 40,
                 max_time_per_vehicle_min: float = 480):
        """
        Initialize Route Optimizer
        
        Args:
            num_vehicles: Number of vehicles for VRP (default 3)
            speed_kmh: Average travel speed in km/h (default 40)
            max_time_per_vehicle_min: Max route time per vehicle in minutes (default 480 = 8 hours)
        """
        self.num_vehicles = num_vehicles
        self.speed_kmh = speed_kmh
        self.max_time_min = max_time_per_vehicle_min
        
        # Initialize internal optimizers
        self._tsp = TSPOptimizer()
        self._vrp = VRPOptimizer(
            num_vehicles=num_vehicles,
            speed_kmh=speed_kmh,
            max_time_per_vehicle_min=max_time_per_vehicle_min
        )
        
        self.calculator = DistanceCalculator()
        self.results_history: List[Dict] = []
    
    # =========================================================================
    # TSP Methods
    # =========================================================================
    
    def solve_tsp(self, locations: List[Tuple[float, float]],
                  depot_index: int = 0,
                  method: str = '2opt') -> Dict[str, Any]:
        """
        Solve TSP (single vehicle routing)
        
        Args:
            locations: List of (lat, lon) coordinates
            depot_index: Starting point index
            method: 'nearest_neighbor', '2opt', 'ortools', or 'all'
            
        Returns:
            TSP solution with route and distance
        """
        result = self._tsp.optimize(locations, depot_index, method)
        result['problem_type'] = 'TSP'
        self.results_history.append(result)
        return result
    
    def benchmark_tsp(self, locations: List[Tuple[float, float]],
                      depot_index: int = 0) -> Dict[str, Any]:
        """
        Benchmark all TSP methods
        
        Returns comparison of Nearest Neighbor, 2-Opt, and OR-Tools
        """
        return self.solve_tsp(locations, depot_index, method='all')
    
    # =========================================================================
    # VRP Methods
    # =========================================================================
    
    def solve_vrp(self, locations: List[Tuple[float, float]],
                  depot_index: int = 0,
                  num_vehicles: Optional[int] = None,
                  strategy: str = 'cluster_tsp') -> Dict[str, Any]:
        """
        Solve VRP (multi-vehicle routing)
        
        Args:
            locations: List of (lat, lon) coordinates
            depot_index: Depot location index
            num_vehicles: Override default vehicle count
            strategy: 'cluster_tsp', 'balanced', or 'min_total'
            
        Returns:
            VRP solution with routes per vehicle
        """
        if num_vehicles and num_vehicles != self._vrp.num_vehicles:
            self._vrp = VRPOptimizer(
                num_vehicles=num_vehicles,
                speed_kmh=self.speed_kmh,
                max_time_per_vehicle_min=self.max_time_min
            )
        
        result = self._vrp.optimize(locations, depot_index, strategy=strategy)
        result['problem_type'] = 'VRP'
        self.results_history.append(result)
        return result
    
    def benchmark_vrp(self, locations: List[Tuple[float, float]],
                      depot_index: int = 0,
                      num_vehicles: Optional[int] = None) -> Dict[str, Any]:
        """
        Benchmark all VRP strategies
        
        Returns comparison of Cluster-TSP, Balanced, and Min-Total
        """
        return self.solve_vrp(locations, depot_index, num_vehicles, strategy='all')
    
    # =========================================================================
    # Region-based Optimization
    # =========================================================================
    
    def optimize_region(self, df: pd.DataFrame,
                       region_column: str = 'Customer State',
                       region_value: str = 'CA',
                       lat_col: str = 'Latitude',
                       lon_col: str = 'Longitude',
                       max_locations: int = 50,
                       problem_type: str = 'auto') -> Dict[str, Any]:
        """
        Optimize routes for a specific region
        
        Args:
            df: DataFrame with location data
            region_column: Column to filter by
            region_value: Region to optimize
            lat_col: Latitude column name
            lon_col: Longitude column name
            max_locations: Maximum locations to include
            problem_type: 'tsp', 'vrp', or 'auto' (auto selects based on location count)
            
        Returns:
            Optimization results with benchmark
        """
        # Filter region
        region_df = df[df[region_column] == region_value].copy()
        
        if len(region_df) == 0:
            return {"error": f"No data for {region_column}={region_value}"}
        
        # Get unique locations
        unique_locs = region_df[[lat_col, lon_col]].drop_duplicates()
        
        if len(unique_locs) > max_locations:
            unique_locs = unique_locs.head(max_locations)
        
        # Depot at centroid
        centroid = (unique_locs[lat_col].mean(), unique_locs[lon_col].mean())
        locations = [centroid] + list(zip(unique_locs[lat_col], unique_locs[lon_col]))
        
        # Auto-select problem type
        if problem_type == 'auto':
            # Use TSP for small problems, VRP for larger
            problem_type = 'tsp' if len(locations) <= 15 else 'vrp'
        
        # Solve
        if problem_type == 'tsp':
            result = self.benchmark_tsp(locations, depot_index=0)
        else:
            result = self._vrp.optimize_region(
                df, region_column, region_value, lat_col, lon_col, max_locations, strategy='all'
            )
        
        result['region'] = {
            'column': region_column,
            'value': region_value,
            'total_orders': len(region_df),
            'unique_locations': len(locations) - 1,
            'depot': centroid
        }
        
        return result
    
    # =========================================================================
    # Comparison & Analysis
    # =========================================================================
    
    def compare_tsp_vs_vrp(self, locations: List[Tuple[float, float]],
                           depot_index: int = 0,
                           num_vehicles: int = 3) -> Dict[str, Any]:
        """
        Compare TSP (1 vehicle) vs VRP (multiple vehicles)
        
        Useful for deciding whether to use multiple vehicles
        """
        # TSP with best method
        tsp_result = self.solve_tsp(locations, depot_index, method='2opt')
        tsp_dist = tsp_result['methods']['2opt']['total_distance_km']
        tsp_time = (tsp_dist / self.speed_kmh) * 60  # minutes
        
        # VRP with cluster strategy
        vrp_result = self.solve_vrp(locations, depot_index, num_vehicles, strategy='cluster_tsp')
        vrp_dist = vrp_result['total_distance_km']
        vrp_total_time = vrp_result['total_time_min']
        vrp_max_time = vrp_result['max_route_time_min']
        
        return {
            'comparison': {
                'tsp': {
                    'vehicles': 1,
                    'total_distance_km': tsp_dist,
                    'total_time_min': round(tsp_time, 2),
                    'max_route_time_min': round(tsp_time, 2)
                },
                'vrp': {
                    'vehicles': num_vehicles,
                    'total_distance_km': vrp_dist,
                    'total_time_min': vrp_total_time,
                    'max_route_time_min': vrp_max_time
                }
            },
            'analysis': {
                'time_saved_min': round(tsp_time - vrp_max_time, 2),
                'time_saved_pct': round((1 - vrp_max_time / tsp_time) * 100, 2) if tsp_time > 0 else 0,
                'extra_distance_km': round(vrp_dist - tsp_dist, 2),
                'extra_distance_pct': round((vrp_dist / tsp_dist - 1) * 100, 2) if tsp_dist > 0 else 0,
                'recommendation': 'VRP' if vrp_max_time < tsp_time * 0.7 else 'TSP'
            },
            'tsp_details': tsp_result,
            'vrp_details': vrp_result
        }
    
    def get_available_regions(self, df: pd.DataFrame,
                              region_column: str = 'Customer State',
                              top_n: int = 10) -> pd.Series:
        """Get top regions by order count"""
        return df[region_column].value_counts().head(top_n)
    
    # =========================================================================
    # Printing & Reporting
    # =========================================================================
    
    def print_tsp_report(self, results: Dict[str, Any]) -> None:
        """Print TSP benchmark report"""
        print_tsp_report(results)
    
    def print_vrp_report(self, results: Dict[str, Any]) -> None:
        """Print VRP optimization report"""
        print_vrp_report(results)
    
    def print_comparison(self, results: Dict[str, Any]) -> None:
        """Print TSP vs VRP comparison"""
        print("\n" + "=" * 60)
        print("TSP vs VRP COMPARISON")
        print("=" * 60)
        
        comp = results['comparison']
        analysis = results['analysis']
        
        print(f"\n{'Metric':<25} {'TSP (1 vehicle)':<20} {'VRP ({} vehicles)':<20}".format(comp['vrp']['vehicles']))
        print("-" * 65)
        print(f"{'Total Distance':<25} {comp['tsp']['total_distance_km']:<20.2f} {comp['vrp']['total_distance_km']:<20.2f} km")
        print(f"{'Total Time':<25} {comp['tsp']['total_time_min']:<20.2f} {comp['vrp']['total_time_min']:<20.2f} min")
        print(f"{'Max Route Time':<25} {comp['tsp']['max_route_time_min']:<20.2f} {comp['vrp']['max_route_time_min']:<20.2f} min")
        
        print(f"\n[DATA] Analysis:")
        print(f"   Time Saved (parallel): {analysis['time_saved_min']:.1f} min ({analysis['time_saved_pct']:.1f}%)")
        print(f"   Extra Distance: {analysis['extra_distance_km']:.1f} km ({analysis['extra_distance_pct']:.1f}%)")
        print(f"   [OK] Recommendation: {analysis['recommendation']}")
        
        print("=" * 60)
