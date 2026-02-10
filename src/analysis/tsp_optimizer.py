"""
TSP (Traveling Salesman Problem) Optimizer
Solves single-vehicle routing with multiple methods for benchmarking
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import time
import logging

from ..utils.distance_calculator import DistanceCalculator

logger = logging.getLogger(__name__)


class TSPOptimizer:
    """
    TSP Optimizer with multiple solving methods for benchmarking
    
    Methods:
    1. Nearest Neighbor (baseline) - Greedy heuristic
    2. 2-Opt Improvement - Local search improvement on NN
    3. OR-Tools - Google's optimization solver
    """
    
    def __init__(self):
        self.calculator = DistanceCalculator()
        self.results_history: List[Dict] = []
    
    def optimize(self, locations: List[Tuple[float, float]], 
                 depot_index: int = 0,
                 method: str = 'all') -> Dict[str, Any]:
        """
        Run TSP optimization
        
        Args:
            locations: List of (lat, lon) coordinates
            depot_index: Starting point index (default 0)
            method: 'nearest_neighbor', '2opt', 'ortools', or 'all'
            
        Returns:
            Optimization results with benchmarks
        """
        n = len(locations)
        if n < 2:
            return {"error": "Need at least 2 locations"}
        
        # Create distance matrix
        dist_matrix = self._create_distance_matrix(locations)
        
        results = {
            'num_locations': n,
            'depot_index': depot_index,
            'methods': {}
        }
        
        methods_to_run = ['nearest_neighbor', '2opt', 'ortools'] if method == 'all' else [method]
        
        for m in methods_to_run:
            if m == 'nearest_neighbor':
                results['methods']['nearest_neighbor'] = self._nearest_neighbor(dist_matrix, depot_index)
            elif m == '2opt':
                results['methods']['2opt'] = self._two_opt(dist_matrix, depot_index)
            elif m == 'ortools':
                results['methods']['ortools'] = self._ortools_tsp(dist_matrix, depot_index)
        
        # Calculate benchmark comparison
        if len(results['methods']) > 1:
            results['benchmark'] = self._calculate_benchmark(results['methods'])
        
        self.results_history.append(results)
        return results
    
    def _create_distance_matrix(self, locations: List[Tuple[float, float]]) -> np.ndarray:
        """Create distance matrix in km"""
        n = len(locations)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                dist = self.calculator.haversine_distance(locations[i], locations[j])
                matrix[i][j] = dist
                matrix[j][i] = dist
        
        return matrix
    
    def _nearest_neighbor(self, dist_matrix: np.ndarray, start: int = 0) -> Dict[str, Any]:
        """
        Nearest Neighbor heuristic (BASELINE)
        Always picks the closest unvisited city
        
        Time Complexity: O(n²)
        """
        start_time = time.time()
        n = len(dist_matrix)
        
        visited = [False] * n
        route = [start]
        visited[start] = True
        total_distance = 0
        current = start
        
        for _ in range(n - 1):
            best_next = -1
            best_dist = float('inf')
            
            for j in range(n):
                if not visited[j] and dist_matrix[current][j] < best_dist:
                    best_dist = dist_matrix[current][j]
                    best_next = j
            
            if best_next >= 0:
                visited[best_next] = True
                route.append(best_next)
                total_distance += best_dist
                current = best_next
        
        # Return to start
        total_distance += dist_matrix[current][start]
        route.append(start)
        
        elapsed = time.time() - start_time
        
        return {
            'method': 'Nearest Neighbor',
            'route': route,
            'total_distance_km': round(total_distance, 2),
            'computation_time_ms': round(elapsed * 1000, 2),
            'is_baseline': True
        }
    
    def _two_opt(self, dist_matrix: np.ndarray, start: int = 0) -> Dict[str, Any]:
        """
        2-Opt improvement heuristic
        Starts with NN solution, then improves by reversing segments
        
        Time Complexity: O(n² × iterations)
        """
        start_time = time.time()
        
        # Start with Nearest Neighbor solution
        nn_result = self._nearest_neighbor(dist_matrix, start)
        route = nn_result['route'][:-1]  # Remove return to depot
        n = len(route)
        
        def route_distance(r):
            dist = sum(dist_matrix[r[i]][r[i+1]] for i in range(len(r)-1))
            dist += dist_matrix[r[-1]][r[0]]  # Return to start
            return dist
        
        improved = True
        best_distance = route_distance(route)
        iterations = 0
        max_iterations = 1000
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    # Reverse segment between i and j
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_distance = route_distance(new_route)
                    
                    if new_distance < best_distance - 0.001:  # Small epsilon
                        route = new_route
                        best_distance = new_distance
                        improved = True
                        break
                if improved:
                    break
        
        route.append(route[0])  # Return to depot
        elapsed = time.time() - start_time
        
        return {
            'method': '2-Opt',
            'route': route,
            'total_distance_km': round(best_distance, 2),
            'computation_time_ms': round(elapsed * 1000, 2),
            'iterations': iterations,
            'is_baseline': False
        }
    
    def _ortools_tsp(self, dist_matrix: np.ndarray, start: int = 0) -> Dict[str, Any]:
        """
        Google OR-Tools TSP solver
        Uses advanced metaheuristics for near-optimal solution
        """
        start_time = time.time()
        n = len(dist_matrix)
        
        # Convert to integer (meters) for OR-Tools
        int_matrix = (dist_matrix * 1000).astype(int).tolist()
        
        # Create routing model
        manager = pywrapcp.RoutingIndexManager(n, 1, start)
        routing = pywrapcp.RoutingModel(manager)
        
        def distance_callback(from_idx, to_idx):
            from_node = manager.IndexToNode(from_idx)
            to_node = manager.IndexToNode(to_idx)
            return int_matrix[from_node][to_node]
        
        transit_cb = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)
        
        # Search parameters
        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_params.time_limit.seconds = 10
        
        solution = routing.SolveWithParameters(search_params)
        elapsed = time.time() - start_time
        
        if solution:
            route = []
            index = routing.Start(0)
            total_distance = 0
            
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                prev = index
                index = solution.Value(routing.NextVar(index))
                total_distance += routing.GetArcCostForVehicle(prev, index, 0)
            
            route.append(route[0])  # Return to depot
            
            return {
                'method': 'OR-Tools',
                'route': route,
                'total_distance_km': round(total_distance / 1000, 2),
                'computation_time_ms': round(elapsed * 1000, 2),
                'is_baseline': False
            }
        else:
            return {
                'method': 'OR-Tools',
                'error': 'No solution found',
                'computation_time_ms': round(elapsed * 1000, 2)
            }
    
    def _calculate_benchmark(self, methods: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate benchmark comparison between methods"""
        baseline = None
        comparisons = []
        
        # Find baseline (Nearest Neighbor)
        for name, result in methods.items():
            if result.get('is_baseline'):
                baseline = result
                break
        
        if not baseline:
            baseline = list(methods.values())[0]
        
        baseline_dist = baseline['total_distance_km']
        
        for name, result in methods.items():
            if 'total_distance_km' in result:
                dist = result['total_distance_km']
                improvement = ((baseline_dist - dist) / baseline_dist) * 100
                comparisons.append({
                    'method': result['method'],
                    'distance_km': dist,
                    'improvement_pct': round(improvement, 2),
                    'time_ms': result.get('computation_time_ms', 0)
                })
        
        # Sort by distance
        comparisons.sort(key=lambda x: x['distance_km'])
        best = comparisons[0]
        
        return {
            'baseline_method': baseline['method'],
            'baseline_distance_km': baseline_dist,
            'best_method': best['method'],
            'best_distance_km': best['distance_km'],
            'best_improvement_pct': best['improvement_pct'],
            'comparisons': comparisons
        }
    
    def optimize_region(self, df: pd.DataFrame, 
                       region_column: str,
                       region_value: str,
                       lat_col: str = 'Latitude',
                       lon_col: str = 'Longitude',
                       max_locations: int = 50) -> Dict[str, Any]:
        """
        Optimize TSP for a specific region
        
        Args:
            df: DataFrame with location data
            region_column: Column to filter by (e.g., 'Customer State')
            region_value: Value to filter (e.g., 'CA')
            lat_col: Latitude column name
            lon_col: Longitude column name
            max_locations: Maximum locations to optimize
            
        Returns:
            Optimization results with benchmark
        """
        # Filter by region
        region_df = df[df[region_column] == region_value].copy()
        
        if len(region_df) == 0:
            return {"error": f"No data found for {region_column}={region_value}"}
        
        # Get unique locations
        unique_locs = region_df[[lat_col, lon_col]].drop_duplicates()
        
        if len(unique_locs) > max_locations:
            unique_locs = unique_locs.head(max_locations)
        
        # Convert to list of tuples
        locations = list(zip(unique_locs[lat_col], unique_locs[lon_col]))
        
        # Add depot at centroid
        centroid = (unique_locs[lat_col].mean(), unique_locs[lon_col].mean())
        locations = [centroid] + locations
        
        # Run optimization
        results = self.optimize(locations, depot_index=0, method='all')
        results['region'] = {
            'column': region_column,
            'value': region_value,
            'total_orders': len(region_df),
            'unique_locations': len(locations) - 1,
            'depot': centroid,
            'depot_location': centroid  # Backward compatibility
        }
        
        return results


def print_benchmark_report(results: Dict[str, Any]) -> None:
    """Pretty print benchmark results"""
    print("\n" + "=" * 60)
    print("TSP OPTIMIZATION BENCHMARK REPORT")
    print("=" * 60)
    
    if 'region' in results:
        r = results['region']
        print(f"\n[PIN] Region: {r['value']} ({r['column']})")
        print(f"   Orders: {r['total_orders']:,}")
        print(f"   Unique Locations: {r['unique_locations']}")
        # Support both 'depot' and 'depot_location' keys for backward compatibility
        depot = r.get('depot_location') or r.get('depot')
        if depot:
            print(f"   Depot: ({depot[0]:.4f}, {depot[1]:.4f})")
    
    print(f"\n[DATA] RESULTS BY METHOD:")
    print("-" * 60)
    
    for name, method in results.get('methods', {}).items():
        if 'error' in method:
            print(f"   [ERROR] {method['method']}: {method['error']}")
        else:
            baseline_tag = " (BASELINE)" if method.get('is_baseline') else ""
            print(f"   {'[O]' if method.get('is_baseline') else '[+]'} {method['method']}{baseline_tag}")
            print(f"      Distance: {method['total_distance_km']:.2f} km")
            print(f"      Time: {method['computation_time_ms']:.2f} ms")
            if 'iterations' in method:
                print(f"      Iterations: {method['iterations']}")
    
    if 'benchmark' in results:
        b = results['benchmark']
        print(f"\n[CHART] BENCHMARK SUMMARY:")
        print("-" * 60)
        print(f"   Baseline: {b['baseline_method']} = {b['baseline_distance_km']:.2f} km")
        print(f"   Best:     {b['best_method']} = {b['best_distance_km']:.2f} km")
        print(f"   Improvement: {b['best_improvement_pct']:.2f}%")
        
        print(f"\n   Ranking:")
        for i, c in enumerate(b['comparisons'], 1):
            marker = "[1st]" if i == 1 else "[2nd]" if i == 2 else "[3rd]" if i == 3 else "  "
            print(f"   {marker} {i}. {c['method']}: {c['distance_km']:.2f} km "
                  f"({c['improvement_pct']:+.2f}%) [{c['time_ms']:.1f}ms]")
    
    print("\n" + "=" * 60)
